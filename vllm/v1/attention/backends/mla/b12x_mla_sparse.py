# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x sparse-MLA backend for SM120 / SM121 (consumer Blackwell).

Counterpart to ``SparseMLASm120Backend`` (FlashInfer V32 v2). Same envelope --
``fp8_ds_mla`` KV cache (656 B/token), head_size = 576, paged block_size = 64,
V32-family models with an ``index_topk`` config (DeepSeek V3.2, GLM-5.1, Kimi
K2.5) -- but the decode/extend kernels come from b12x's unified SM120 backend
via the ``b12x.integration.mla`` front door (``sparse_mla_decode_forward`` /
``sparse_mla_extend_forward``). On SM120+ CUDA those front-door functions route
to ``b12x/attention/mla/unified_sm120`` automatically (GLM_NSA q_head_dim==576
contract). Selecting this backend also selects b12x's sparse indexer/top-k path.

Scratch philosophy (eager PLAN -> BIND -> KERNEL; no workspace/arena, ever):
b12x workspaces/arenas are sglang-only and forbidden here. We build a caller-
owned-scratch ``plan_sparse_mla_scratch`` PLAN once per mode (decode / extend),
and each forward maps a vLLM ``current_workspace_manager()`` scratch tensor into
a plain ``B12XSparseMLAScratch`` views CONTAINER via ``plan.bind(...)`` -- a pure
narrow()+view() mapping that allocates nothing and constructs no workspace. The
binding holds views (never a ``B12XAttentionWorkspace``); the unified SM120
sparse-MLA decode/extend kernels duck-type the container's
``tmp_output`` / ``tmp_lse`` / ``output_buffer`` / ``final_lse`` /
``num_chunks_ptr`` / ``set_split_chunk_config`` fields, so the binding is a
drop-in with no kernel-signature change. q-concat and the scratch are borrowed in
ONE ``get_simultaneous`` call so they never alias.
"""

import inspect
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import get_mla_dims
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# Split-K tile width. Mirrors SparseMLASm120's _DECODE_SPLIT_TILE: the number of
# split-K chunks is ceil(topk / tile). This bounds the chunk dim of the borrowed
# mid_out/mid_lse scratch and the workspace ``max_chunks_per_row`` cap; b12x's
# wave-balanced planner picks num_splits <= this cap.
_DECODE_SPLIT_TILE = 64
_HEAD_ALIGNMENT = 8
_BF16_BYTES = 2
_EXTEND_PREWARM_DONE: set[
    tuple[int | None, int, int, int, int, int, bool, str, bool]
] = set()
_CKV_GATHER_WORKSPACES: dict[tuple[str, int | None], torch.Tensor] = {}
_SPARSE_DECODE_CKV_WORKSPACES: dict[tuple[str, int | None], torch.Tensor] = {}
_SPARSE_DECODE_SELECTED_INDICES: dict[
    tuple[str, int | None, int], torch.Tensor
] = {}
_SPARSE_DECODE_PATCH_SLOTS: dict[
    tuple[str, int | None, int], torch.Tensor
] = {}
_SPARSE_DECODE_UNION_STATES: dict[
    tuple[str, int | None, int, int, int, int, int, int],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
] = {}
_SPARSE_DECODE_P2P_FALLBACKS: dict[
    tuple[str, int | None, str, int, int, int, int], str
] = {}
_KV_FP8_ROPE_REQUESTED = os.getenv("KV_FP8_ROPE", "0") == "1"


_IS_GLM_MOE_DSA_CACHE: bool | None = None


def _is_glm_moe_dsa_model() -> bool:
    """Return true only for GLM or its in-process MTP draft model.

    Robust to being called before the vLLM config context is established (e.g.
    during KV-cache shape resolution / cudagraph compilation in a worker, where
    get_current_vllm_config() raises): fall back to the explicit KV_FP8_ROPE
    request and re-resolve once the config becomes available. Correctness is
    preserved because the fallback is only reached when the user set
    KV_FP8_ROPE=1 for their GLM model; KV_FP8_ROPE=0 short-circuits earlier.
    """
    global _IS_GLM_MOE_DSA_CACHE
    if _IS_GLM_MOE_DSA_CACHE is not None:
        return _IS_GLM_MOE_DSA_CACHE
    from vllm.config import get_current_vllm_config

    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return _KV_FP8_ROPE_REQUESTED
    model_config = vllm_config.model_config
    if model_config is None:
        return False
    model_type = getattr(model_config.hf_config, "model_type", None)
    if model_type == "glm_moe_dsa":
        _IS_GLM_MOE_DSA_CACHE = True
        return True
    speculative_config = getattr(vllm_config, "speculative_config", None)
    target_model_config = getattr(speculative_config, "target_model_config", None)
    target_model_type = (
        getattr(target_model_config.hf_config, "model_type", None)
        if target_model_config is not None
        else None
    )
    result = model_type == "deepseek_mtp" and target_model_type == "glm_moe_dsa"
    _IS_GLM_MOE_DSA_CACHE = result
    return result


def _kv_fp8_rope_enabled() -> bool:
    """Strict public gate plus literal GLM architecture selection."""
    return _KV_FP8_ROPE_REQUESTED and _is_glm_moe_dsa_model()


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d", name, value, default)
        return default
    return parsed


def _validate_sparse_decode_graph_capacity(
    enabled: bool,
    max_reqs: int,
    rows_per_req: int,
    max_cudagraph_rows: int | None,
) -> None:
    """Reject graph ceilings that cannot cover the configured sparse batch."""
    graph_rows = int(max_cudagraph_rows or 0)
    if not enabled or graph_rows == 0:
        return
    required_rows = int(max_reqs) * int(rows_per_req)
    if graph_rows < required_rows:
        raise ValueError(
            "Sparse CKV decode requires max cudagraph capture size >= "
            f"{required_rows} ({max_reqs} requests x {rows_per_req} rows); "
            f"configured {graph_rows}"
        )


def _ckv_prefetch_supports_format(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")


def _ckv_prefetch_target_indices(
    layer_idx: int,
    depth: int,
    layer_caches: list[torch.Tensor | None],
    pending_layers: dict[int, tuple[torch.cuda.Event, int]],
) -> list[int]:
    targets: list[int] = []
    for distance in range(1, depth + 1):
        target_idx = layer_idx + distance
        if target_idx in pending_layers:
            continue
        if target_idx >= len(layer_caches) or layer_caches[target_idx] is None:
            break
        targets.append(target_idx)
    return targets


def _get_ckv_gather_workspace(device: torch.device, nbytes: int) -> torch.Tensor:
    key = (device.type, device.index)
    workspace = _CKV_GATHER_WORKSPACES.get(key)
    if workspace is None:
        workspace = torch.empty((nbytes,), dtype=torch.uint8, device=device)
        _CKV_GATHER_WORKSPACES[key] = workspace
    elif workspace.numel() < nbytes:
        raise RuntimeError(
            "CKV gather workspace cannot grow after attention layers retain "
            f"aliases: existing={workspace.numel()} requested={nbytes}"
        )
    return workspace[:nbytes]


def _get_sparse_decode_ckv_workspace(
    device: torch.device, nbytes: int
) -> torch.Tensor:
    key = (device.type, device.index)
    workspace = _SPARSE_DECODE_CKV_WORKSPACES.get(key)
    if workspace is None:
        workspace = torch.empty((nbytes,), dtype=torch.uint8, device=device)
        _SPARSE_DECODE_CKV_WORKSPACES[key] = workspace
    elif workspace.numel() < nbytes:
        raise RuntimeError(
            "Sparse decode CKV workspace cannot grow after attention layers "
            f"retain aliases: existing={workspace.numel()} requested={nbytes}"
        )
    return workspace[:nbytes]


def _get_sparse_decode_selected_indices(
    device: torch.device, width: int
) -> torch.Tensor:
    key = (device.type, device.index, int(width))
    indices = _SPARSE_DECODE_SELECTED_INDICES.get(key)
    if indices is None:
        indices = torch.arange(width, dtype=torch.int32, device=device).view(1, width)
        _SPARSE_DECODE_SELECTED_INDICES[key] = indices
    return indices


def _get_sparse_decode_patch_slots(
    device: torch.device, rows: int
) -> torch.Tensor:
    key = (device.type, device.index, int(rows))
    slots = _SPARSE_DECODE_PATCH_SLOTS.get(key)
    if slots is None:
        slots = torch.empty((rows,), dtype=torch.int64, device=device)
        _SPARSE_DECODE_PATCH_SLOTS[key] = slots
    return slots


def _get_sparse_decode_union_state(
    device: torch.device,
    rows_per_req: int,
    topk: int,
    union_world_size: int,
    destination_count: int,
    dcp_world_size: int,
    max_reqs: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    key = (
        device.type,
        device.index,
        int(rows_per_req),
        int(topk),
        int(union_world_size),
        int(destination_count),
        int(dcp_world_size),
        int(max_reqs),
    )
    state = _SPARSE_DECODE_UNION_STATES.get(key)
    if state is None:
        all_rank_rows_per_req = union_world_size * rows_per_req
        per_req_capacity = all_rank_rows_per_req * topk
        hash_capacity = 1 << (2 * per_req_capacity - 1).bit_length()
        max_rows = max_reqs * rows_per_req
        union_indices = torch.empty(
            (destination_count, max_reqs, per_req_capacity),
            dtype=torch.int32,
            device=device,
        )
        remap = torch.empty((max_rows, topk), dtype=torch.int32, device=device)
        all_rank_topk = torch.empty(
            (dcp_world_size * max_rows, topk), dtype=torch.int32, device=device
        )
        all_rank_topk_by_req = torch.empty(
            (max_reqs, dcp_world_size, rows_per_req, topk),
            dtype=torch.int32,
            device=device,
        )
        all_rank_remap = torch.empty(
            (max_reqs, destination_count, all_rank_rows_per_req, topk),
            dtype=torch.int32,
            device=device,
        )
        union_count = torch.empty(
            (destination_count, max_reqs), dtype=torch.int32, device=device
        )
        hash_keys = torch.empty(
            (destination_count, max_reqs, hash_capacity),
            dtype=torch.int32,
            device=device,
        )
        hash_values = torch.empty_like(hash_keys)
        local_slots = torch.empty(
            (destination_count, max_reqs, per_req_capacity),
            dtype=torch.int32,
            device=device,
        )
        state = (
            union_indices,
            remap,
            union_count,
            hash_keys,
            hash_values,
            local_slots,
            all_rank_topk,
            all_rank_topk_by_req,
            all_rank_remap,
        )
        _SPARSE_DECODE_UNION_STATES[key] = state
    return state


def _sparse_decode_prefetch_targets(
    layer_idx: int,
    prefetch_depth: int,
    emits_topk_by_layer: list[bool | None],
) -> list[int]:
    """Return consecutive Shared layers following a Full/indexer layer."""
    targets: list[int] = []
    for distance in range(1, prefetch_depth + 1):
        target_idx = layer_idx + distance
        if target_idx >= len(emits_topk_by_layer):
            break
        emits_topk = emits_topk_by_layer[target_idx]
        if emits_topk is None or emits_topk:
            break
        targets.append(target_idx)
    return targets


def _dcp_all_gather_current_stream(
    group,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
        raise ValueError("CKV all-gather tensors must be contiguous")
    if (
        output_tensor.shape[0] != input_tensor.shape[0] * group.world_size
        or output_tensor.shape[1:] != input_tensor.shape[1:]
    ):
        raise ValueError("CKV all-gather tensors have incompatible shapes")

    communicator = getattr(group, "device_communicator", None)
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is not None and not getattr(pynccl_comm, "disabled", False):
        pynccl_comm.all_gather(output_tensor, input_tensor)
        return

    device_group = getattr(group, "device_group", None)
    if device_group is None:
        device_group = getattr(communicator, "device_group", None)
    if device_group is not None:
        dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=device_group,
            async_op=False,
        )
        return

    gathered = group.all_gather(input_tensor, dim=0)
    output_tensor.copy_(gathered)


def _dcp_all_reduce_current_stream(group, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError("Sparse decode CKV all-reduce tensor must be contiguous")

    communicator = getattr(group, "device_communicator", None)
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is not None and not getattr(pynccl_comm, "disabled", False):
        pynccl_comm.all_reduce(tensor, out_tensor=tensor)
        return

    device_group = getattr(group, "device_group", None)
    if device_group is None:
        device_group = getattr(communicator, "device_group", None)
    if device_group is not None:
        dist.all_reduce(tensor, group=device_group, async_op=False)
        return

    reduced = group.all_reduce(tensor)
    tensor.copy_(reduced)


@triton.jit
def _pack_sparse_decode_ckv_local_kernel(
    kv_cache_ptr,
    local_slots_ptr,
    packed_ptr,
    slots_stride0,
    slots_stride1,
    packed_stride0,
    packed_stride1,
    packed_stride2,
    RECORD_BYTES: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    row = tl.program_id(0)
    selected = tl.program_id(1)
    byte = tl.arange(0, BLOCK_BYTES)
    local_slot = tl.load(
        local_slots_ptr + row * slots_stride0 + selected * slots_stride1
    )
    valid = local_slot >= 0
    value = tl.load(
        kv_cache_ptr + local_slot * RECORD_BYTES + byte,
        mask=valid & (byte < RECORD_BYTES),
        other=0,
    )
    tl.store(
        packed_ptr
        + row * packed_stride0
        + selected * packed_stride1
        + byte * packed_stride2,
        value,
        mask=byte < RECORD_BYTES,
    )


def _pack_sparse_decode_ckv_local(
    kv_cache: torch.Tensor,
    local_slots: torch.Tensor,
    packed: torch.Tensor,
) -> None:
    """Pack rank-owned native CKV records without changing top-k order."""
    if kv_cache.dtype != torch.uint8 or packed.dtype != torch.uint8:
        raise TypeError("Sparse decode CKV pack requires uint8 record storage")
    if local_slots.dtype != torch.int32:
        raise TypeError("Sparse decode CKV local slots must be int32")
    if local_slots.ndim != 2 or packed.ndim != 3:
        raise ValueError(
            "Sparse decode CKV pack expects rank-2 slots and rank-3 output"
        )
    if tuple(packed.shape[:2]) != tuple(local_slots.shape):
        raise ValueError("Sparse decode CKV slots/output shapes do not match")
    if not kv_cache.is_contiguous() or not packed.is_contiguous():
        raise ValueError("Sparse decode CKV pack tensors must be contiguous")
    record_bytes = int(packed.shape[2])
    if int(kv_cache.shape[-1]) != record_bytes:
        raise ValueError("Sparse decode CKV record widths do not match")
    block_bytes = triton.next_power_of_2(record_bytes)
    _pack_sparse_decode_ckv_local_kernel[
        (local_slots.shape[0], local_slots.shape[1])
    ](
        kv_cache,
        local_slots,
        packed,
        local_slots.stride(0),
        local_slots.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        RECORD_BYTES=record_bytes,
        BLOCK_BYTES=block_bytes,
    )


@triton.jit
def _find_sparse_decode_current_token_slot_kernel(
    topk_indices_ptr,
    req_id_ptr,
    global_seq_len_ptr,
    output_slot_ptr,
    topk_stride0,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    req = tl.load(req_id_ptr + row)
    selected = tl.load(
        topk_indices_ptr + req * topk_stride0 + offsets,
        mask=offsets < TOPK,
        other=-1,
    )
    global_seq_len = tl.load(global_seq_len_ptr + row)
    current_token = global_seq_len - 1
    matched = tl.max(tl.where(selected == current_token, offsets, -1), axis=0)
    flattened_slot = tl.where(matched >= 0, req * TOPK + matched, -1)
    tl.store(output_slot_ptr + row, flattened_slot)


def _find_sparse_decode_current_token_slot(
    topk_indices: torch.Tensor,
    req_id_per_token: torch.Tensor,
    global_seq_len: torch.Tensor,
    output_slot: torch.Tensor,
) -> None:
    if topk_indices.ndim != 2:
        raise ValueError("Sparse decode current-token patch expects request rows")
    if topk_indices.dtype != torch.int32:
        raise TypeError("Sparse decode current-token top-k must be int32")
    if req_id_per_token.ndim != 1:
        raise TypeError("Sparse decode current-token request ids must be rank 1")
    if global_seq_len.ndim != 1 or global_seq_len.dtype != torch.int32:
        raise TypeError("Sparse decode current-token lengths must be int32")
    if req_id_per_token.shape != global_seq_len.shape:
        raise TypeError("Sparse decode request ids must match the lengths")
    if output_slot.shape != global_seq_len.shape or output_slot.dtype != torch.int64:
        raise TypeError("Sparse decode current-token slots must match the lengths")
    topk = int(topk_indices.shape[1])
    _find_sparse_decode_current_token_slot_kernel[(global_seq_len.numel(),)](
        topk_indices,
        req_id_per_token,
        global_seq_len,
        output_slot,
        topk_indices.stride(0),
        TOPK=topk,
        BLOCK_N=triton.next_power_of_2(topk),
    )


@triton.jit
def _offset_sparse_decode_remap_kernel(
    remap_ptr,
    numel,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    positions = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = positions < numel
    values = tl.load(remap_ptr + positions, mask=mask, other=-1)
    values = tl.where(values >= 0, values + offset, values)
    tl.store(remap_ptr + positions, values, mask=mask)


def _offset_sparse_decode_remap(remap: torch.Tensor, offset: int) -> None:
    """Move one request's valid remap slots into its pooled segment."""
    if remap.dtype != torch.int32 or not remap.is_contiguous():
        raise TypeError("Sparse decode remap must be contiguous int32")
    numel = int(remap.numel())
    if numel == 0 or offset == 0:
        return
    block_size = 256
    _offset_sparse_decode_remap_kernel[(_cdiv(numel, block_size),)](
        remap,
        numel,
        int(offset),
        BLOCK_SIZE=block_size,
    )


@triton.jit
def _mask_page_table_after_nsa_len_kernel(
    page_table_ptr,
    nsa_len_ptr,
    page_stride0,
    page_stride1,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs < width
    nsa_len = tl.load(nsa_len_ptr + row)
    tl.store(
        page_table_ptr + row * page_stride0 + offs * page_stride1,
        -1,
        mask=valid & (offs >= nsa_len),
    )


def _mask_page_table_after_nsa_len(
    page_table: torch.Tensor,
    nsa_cache_seqlens: torch.Tensor,
) -> None:
    width = page_table.shape[1]
    if width == 0 or page_table.shape[0] == 0:
        return
    block_n = 128
    _mask_page_table_after_nsa_len_kernel[
        (page_table.shape[0], triton.cdiv(width, block_n))
    ](
        page_table,
        nsa_cache_seqlens,
        page_table.stride(0),
        page_table.stride(1),
        width,
        BLOCK_N=block_n,
    )


def _global_causal_lens_for_ckv_gather(
    global_seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_id_per_token: torch.Tensor,
    num_actual_toks: int,
) -> torch.Tensor:
    """Return each query token's causal length in the gathered global cache."""
    num_reqs = global_seq_lens.shape[0]
    qsl = query_start_loc[: num_reqs + 1].to(torch.int32)
    req_ids = req_id_per_token[:num_actual_toks].to(torch.int64)
    chunk_start = qsl[:-1][req_ids]
    chunk_len = (qsl[1:] - qsl[:-1])[req_ids]
    full_seq = global_seq_lens[req_ids].to(torch.int32)
    token_idx = torch.arange(
        num_actual_toks,
        device=global_seq_lens.device,
        dtype=torch.int32,
    )
    return full_seq - chunk_len + (token_idx - chunk_start) + 1


@triton.jit
def _map_global_topk_to_gathered_ckv_kernel(
    req_id_ptr,
    token_indices_ptr,
    rank_req_starts_ptr,
    rank_req_lens_ptr,
    out_ptr,
    valid_count_ptr,
    starts_stride0,
    starts_stride1,
    lens_stride0,
    lens_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
    padded_rank_tokens,
    DCP_SIZE: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < NUM_TOPK_TOKENS

    req = tl.load(req_id_ptr + row)
    tok = tl.load(
        token_indices_ptr + row * ti_stride0 + cols * ti_stride1,
        mask=col_mask,
        other=-1,
    )
    owner = (tok // DCP_INTERLEAVE) % DCP_SIZE
    local_idx = (
        tok // (DCP_SIZE * DCP_INTERLEAVE)
    ) * DCP_INTERLEAVE + tok % DCP_INTERLEAVE
    req_start = tl.load(
        rank_req_starts_ptr + owner * starts_stride0 + req * starts_stride1,
        mask=col_mask & (tok >= 0),
        other=0,
    )
    req_len = tl.load(
        rank_req_lens_ptr + owner * lens_stride0 + req * lens_stride1,
        mask=col_mask & (tok >= 0),
        other=0,
    )
    valid = col_mask & (tok >= 0) & (local_idx >= 0) & (local_idx < req_len)
    gathered_slot = owner * padded_rank_tokens + req_start + local_idx

    valid_i32 = valid.to(tl.int32)
    local_offset = tl.cumsum(valid_i32) - valid_i32
    tile_valid_count = tl.sum(valid_i32)
    # Tiles race to reserve output ranges via atomic_add, so the surviving
    # slots land in an unspecified order within a row. Attention is a softmax
    # over the selected set, so order does not affect the result.
    output_base = tl.atomic_add(valid_count_ptr + row, tile_valid_count)
    output_col = output_base + local_offset
    tl.store(
        out_ptr + row * out_stride0 + output_col * out_stride1,
        gathered_slot,
        mask=valid,
    )


def _map_global_topk_to_gathered_ckv(
    req_ids: torch.Tensor,
    token_indices: torch.Tensor,
    rank_req_starts: torch.Tensor,
    rank_req_lens: torch.Tensor,
    out: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    dcp_size: int,
    cp_kv_cache_interleave_size: int,
    padded_rank_tokens: int,
) -> None:
    if token_indices.shape != out.shape:
        raise ValueError("CKV gather index output shape does not match top-k input")
    if rank_req_starts.shape != rank_req_lens.shape:
        raise ValueError("CKV gather request starts/lens shapes do not match")
    if rank_req_starts.shape[0] != dcp_size:
        raise ValueError("CKV gather request metadata does not match DCP size")
    if any(
        tensor.dtype != torch.int32
        for tensor in (
            req_ids,
            token_indices,
            rank_req_starts,
            rank_req_lens,
            out,
            valid_counts,
        )
    ):
        raise TypeError("CKV gather index metadata must be int32")
    if token_indices.shape[1] % 128 != 0:
        raise ValueError("CKV gather top-k width must be divisible by 128")

    out.fill_(-1)
    valid_counts.zero_()
    _map_global_topk_to_gathered_ckv_kernel[
        (token_indices.shape[0], token_indices.shape[1] // 128)
    ](
        req_ids,
        token_indices,
        rank_req_starts,
        rank_req_lens,
        out,
        valid_counts,
        rank_req_starts.stride(0),
        rank_req_starts.stride(1),
        rank_req_lens.stride(0),
        rank_req_lens.stride(1),
        token_indices.stride(0),
        token_indices.stride(1),
        out.stride(0),
        out.stride(1),
        padded_rank_tokens,
        DCP_SIZE=dcp_size,
        DCP_INTERLEAVE=cp_kv_cache_interleave_size,
        NUM_TOPK_TOKENS=token_indices.shape[1],
        BLOCK_N=128,
    )


class B12xMLASparseBackend(AttentionBackend):
    """b12x unified sparse-MLA backend (SM120 / SM121).

    Same envelope as ``SparseMLASm120Backend`` (head 576, fp8_ds_mla, block 64,
    index_topk) but driven by b12x's unified decode/extend kernels.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
        "fp8",  # aliases for fp8_ds_mla on this backend
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Must equal DeepseekV32IndexerBackend.get_supported_kernel_block_sizes
        # on CUDA (= [64]); the unified b12x decode/extend kernels dispatch
        # page_block_size == 64 natively (matches the fp8_ds_mla layout).
        return [64]

    @staticmethod
    def get_name() -> str:
        return "B12X_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["B12xMLASparseImpl"]:
        return B12xMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["B12xMLASparseMetadataBuilder"]:
        return B12xMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # GLM_NSA contract: q_head_dim = kv_lora_rank (512) + qk_rope_head_dim
        # (64) = 576. The unified decode raises on any other q_head_dim.
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Consumer Blackwell SM120 / SM121. The unified b12x kernels gate on
        # get_sm_version(device) >= 120 internally.
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # Require an indexer-equipped (index_topk) model, same as SPARSE_MLA_SM120.
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            if not hasattr(hf_text_config, "index_topk"):
                return "B12X_MLA_SPARSE requires a model with index_topk config"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # = 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # V32 fp8_ds_mla packed: 656 B/token (512 NoPE + 16 inline FP32
            # scales + 128 BF16 RoPE). Mirrors the FlashMLA / SPARSE_MLA_SM120
            # layout; b12x's GLM_NSA decode reads the same record.
            return (num_blocks, block_size, 656)
        if cache_dtype_str == "nvfp4_ds_mla":
            # NVFP4 MLA latent: 256 B E2M1 NoPE data + 32 B E4M3 group-16
            # scales. The stock record has 16 B pad + 128 B BF16 RoPE (432 B).
            # KV_FP8_ROPE=1 reuses the pad for one FP32 amax scale and stores
            # 64 E4M3 bytes at the original RoPE offset (368 B total).
            return (
                num_blocks,
                block_size,
                368 if _kv_fp8_rope_enabled() else 432,
            )
        return (num_blocks, block_size, head_size)


@dataclass
class B12xMLASparseMetadata(AttentionMetadata):
    """Attention metadata for the B12X_MLA_SPARSE backend."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    num_decode_tokens: int
    num_prefill_tokens: int
    # Decode/prefill request counts and the prefill max seq len, part of the
    # MLAAttention.forward_impl metadata contract. B12X routes every token
    # through the top-k MQA path (supports_mha_prefill = False), so
    # prefill_max_seq_len only feeds the (dead) dense-MHA routing check.
    num_decodes: int
    num_prefills: int
    prefill_max_seq_len: int

    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    # DCP keeps global logical top-k ids until forward_mqa maps the entries
    # owned by this rank to local physical slots. These buffers are unnecessary
    # for the direct native-slot path when DCP is disabled.
    req_id_per_token: torch.Tensor | None
    page_table_1: torch.Tensor | None
    nsa_cache_seqlens: torch.Tensor | None
    # Per-request computed KV length (decode cache_seqlens_int32).
    seq_lens: torch.Tensor
    cache_seq_lens_per_req: torch.Tensor
    # Per-token causal KV length consumed directly by the sparse MLA kernel.
    # For pure decode this equals ``seq_lens`` (one token per request).
    cache_seq_lens_per_token: torch.Tensor

    # Transient full-CKV prefill gather metadata.
    ckv_page_table_1: torch.Tensor | None = None
    ckv_nsa_cache_seqlens: torch.Tensor | None = None
    dcp_rank_req_starts: torch.Tensor | None = None
    dcp_rank_req_lens: torch.Tensor | None = None
    dcp_local_cu_seq_lens: torch.Tensor | None = None
    global_cache_seq_lens_per_req: torch.Tensor | None = None
    dcp_local_total_tokens: int = 0
    dcp_padded_total_tokens: int = 0
    dcp_ckv_gather_eligible: bool = False

    block_size: int = 64
    topk_tokens: int = 2048


class B12xMLASparseMetadataBuilder(AttentionMetadataBuilder[B12xMLASparseMetadata]):
    """Builder for B12X_MLA_SPARSE attention metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_exact_metadata_reuse: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device

        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_seqs = vllm_config.scheduler_config.max_num_seqs
        from vllm import envs as envs_mod

        ckv_gather_requested = envs_mod.VLLM_B12X_MLA_CKV_GATHER
        sparse_decode_ckv_requested = (
            envs_mod.VLLM_B12X_MLA_SPARSE_DECODE_CKV_GATHER
        )
        ckv_metadata_requested = (
            ckv_gather_requested or sparse_decode_ckv_requested
        )
        self.ckv_metadata_requested = ckv_metadata_requested
        # Max-batched-token scratch buffers so cudagraph capture sees stable
        # allocations (sliced per build()).
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        self.cache_seq_lens_per_req_buffer = torch.empty(
            (max_seqs,), dtype=torch.int32, device=device
        )
        if self.dcp_world_size > 1:
            self.req_id_per_token_buffer = torch.empty(
                (max_tokens,), dtype=torch.int32, device=device
            )
            self.page_table_1_buffer = torch.empty(
                (max_tokens, self.topk_tokens), dtype=torch.int32, device=device
            )
            self.nsa_cache_seqlens_buffer = torch.empty(
                (max_tokens,), dtype=torch.int32, device=device
            )
            self.req_ids_arange = torch.arange(
                max_tokens, dtype=torch.int32, device=device
            )
            if ckv_gather_requested:
                self.ckv_page_table_1_buffer = torch.empty(
                    (max_tokens, self.topk_tokens), dtype=torch.int32, device=device
                )
                self.ckv_nsa_cache_seqlens_buffer = torch.empty(
                    (max_tokens,), dtype=torch.int32, device=device
                )
                self.dcp_rank_req_lens_buffer = torch.empty(
                    (self.dcp_world_size, max_seqs), dtype=torch.int32, device=device
                )
                self.dcp_rank_req_starts_buffer = torch.empty(
                    (self.dcp_world_size, max_seqs), dtype=torch.int32, device=device
                )
                self.dcp_local_cu_seq_lens_buffer = torch.empty(
                    (max_seqs + 1,), dtype=torch.int32, device=device
                )
            else:
                self.ckv_page_table_1_buffer = None
                self.ckv_nsa_cache_seqlens_buffer = None
                self.dcp_rank_req_lens_buffer = None
                self.dcp_rank_req_starts_buffer = None
                self.dcp_local_cu_seq_lens_buffer = None
        else:
            self.req_id_per_token_buffer = None
            self.page_table_1_buffer = None
            self.nsa_cache_seqlens_buffer = None
            self.req_ids_arange = None
            self.ckv_page_table_1_buffer = None
            self.ckv_nsa_cache_seqlens_buffer = None
            self.dcp_rank_req_lens_buffer = None
            self.dcp_rank_req_starts_buffer = None
            self.dcp_local_cu_seq_lens_buffer = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12xMLASparseMetadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens
        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:
            num_decodes = cm.num_reqs
            num_prefills = 0
            num_decode_tokens = num_tokens
            num_prefill_tokens = 0
        elif cm.batch_topology is not None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                cm.batch_topology.split_decodes_and_prefills(
                    cm,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=True,
                )
            )
        else:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    cm,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=True,
                )
            )
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        use_dcp = self.dcp_world_size > 1
        seq_lens_for_req = (
            cm.dcp_local_seq_lens
            if use_dcp and cm.dcp_local_seq_lens is not None
            else cm.seq_lens
        )
        req_id_per_token_tensor = None
        dcp_rank_req_lens = None
        dcp_rank_req_starts = None
        dcp_local_cu_seq_lens = None
        dcp_local_total_tokens = 0
        dcp_padded_total_tokens = 0
        dcp_ckv_gather_eligible = False

        from vllm import envs as envs_mod

        if envs_mod.VLLM_B12X_MLA_CKV_GATHER:
            # Reset the cross-layer prefetch pipeline once per step so the
            # first layer always sync-gathers. Pending target-layer events are
            # class state that must not leak across scheduler steps. Cache
            # pointers remain registered because their addresses are stable.
            pending_events = getattr(
                B12xMLASparseImpl, "_shared_gather_events", {}
            )
            if pending_events:
                logger.warning_once(
                    "Draining %d unconsumed CKV prefetch event(s) before "
                    "reusing the workspace ring.",
                    len(pending_events),
                )
                for event, _ in pending_events.values():
                    event.synchronize()
            B12xMLASparseImpl._shared_gather_events = {}

        if (
            use_dcp
            and envs_mod.VLLM_B12X_MLA_CKV_GATHER
            and num_decode_tokens == 0
            and num_prefill_tokens == num_tokens
            and cm.max_query_len > envs_mod.VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS
        ):
            assert self.dcp_rank_req_lens_buffer is not None
            assert self.dcp_rank_req_starts_buffer is not None
            assert self.dcp_local_cu_seq_lens_buffer is not None
            global_seq_lens = cm.seq_lens[: cm.num_reqs]
            all_rank_lens = get_dcp_local_seq_lens(
                global_seq_lens,
                self.dcp_world_size,
                dcp_rank=None,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            ).transpose(0, 1)
            dcp_rank_req_lens = self.dcp_rank_req_lens_buffer[
                : self.dcp_world_size, : cm.num_reqs
            ]
            dcp_rank_req_lens.copy_(all_rank_lens)
            dcp_rank_req_starts = self.dcp_rank_req_starts_buffer[
                : self.dcp_world_size, : cm.num_reqs
            ]
            dcp_rank_req_starts[:, 0].zero_()
            if cm.num_reqs > 1:
                torch.cumsum(
                    dcp_rank_req_lens[:, :-1],
                    dim=1,
                    out=dcp_rank_req_starts[:, 1:],
                )

            dcp_local_cu_seq_lens = self.dcp_local_cu_seq_lens_buffer[: cm.num_reqs + 1]
            dcp_local_cu_seq_lens[0].zero_()
            torch.cumsum(
                dcp_rank_req_lens[self.dcp_rank],
                dim=0,
                out=dcp_local_cu_seq_lens[1:],
            )
            rank_totals = dcp_rank_req_lens.sum(dim=1).tolist()
            dcp_local_total_tokens = int(rank_totals[self.dcp_rank])
            dcp_padded_total_tokens = (
                _cdiv(
                    max(int(total) for total in rank_totals),
                    self.kv_cache_spec.block_size,
                )
                * self.kv_cache_spec.block_size
            )
            dcp_ckv_gather_eligible = dcp_padded_total_tokens > 0

        # Per-token causal KV length. In pure decode the common metadata already
        # has exactly the graph-stable tensor both b12x consumers need, so bind it
        # directly instead of staging two identical D2D copies.
        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:
            if use_dcp:
                assert self.req_ids_arange is not None
                req_id_per_token_tensor = self.req_ids_arange[:num_tokens]
                self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                    seq_lens_for_req[:num_tokens], non_blocking=True
                )
                self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                    seq_lens_for_req[: cm.num_reqs], non_blocking=True
                )
                cache_seq_lens_per_token = self.cache_seq_lens_per_token_buffer[
                    :num_tokens
                ]
                cache_seq_lens_per_req = self.cache_seq_lens_per_req_buffer[
                    : cm.num_reqs
                ]
            else:
                cache_seq_lens_per_token = seq_lens_for_req[:num_tokens]
                cache_seq_lens_per_req = seq_lens_for_req[: cm.num_reqs]
        else:
            if cm.batch_topology is not None:
                starts = cm.batch_topology.query_start_loc_np[: cm.num_reqs + 1]
                query_lens = cm.batch_topology.query_lens_np
                req_id_per_token_np = cm.batch_topology.req_id_per_token_np
            else:
                starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
                query_lens = np.diff(starts)
                req_id_per_token_np = np.repeat(
                    np.arange(cm.num_reqs, dtype=np.int32), query_lens
                )
            num_query_tokens = int(starts[-1])
            if num_query_tokens > num_tokens:
                raise RuntimeError(
                    "B12X sparse MLA metadata received query_start_loc with "
                    f"{num_query_tokens} tokens, exceeding padded capacity "
                    f"{num_tokens}"
                )

            req_ids = None
            if use_dcp:
                req_ids = np.zeros((num_tokens,), dtype=np.int32)
                if num_query_tokens:
                    req_ids[:num_query_tokens] = req_id_per_token_np

            if not use_dcp and cm.positions is not None and cm.positions.ndim == 1:
                # Async scheduling intentionally exposes only an optimistic CPU
                # sequence-length bound. That bound can lag when a finished slot is
                # recycled for a shorter request, so it is not a valid causal mask
                # for multi-token verification. Positions are authoritative on the
                # GPU and give the exact per-token KV length for DCP1.
                per_token_lens_t = cm.positions[:num_tokens].to(torch.int32) + 1
            else:
                # DCP needs rank-local lengths rather than global positions. Avoid
                # the blocking lazy seq_lens D2H copy and convert the scheduler's
                # conservative CPU lengths to each rank's local interleaving.
                seq_lens_cpu_src = (
                    cm.seq_lens_cpu_upper_bound
                    if cm.seq_lens_cpu_upper_bound is not None
                    else cm.seq_lens_cpu
                )
                seq_lens_cpu = seq_lens_cpu_src.numpy().astype(np.int32, copy=False)
                per_token_lens = np.zeros((num_tokens,), dtype=np.int32)
                for req_id, q_len in enumerate(query_lens):
                    if q_len <= 0:
                        continue
                    start = int(starts[req_id])
                    end = int(starts[req_id + 1])
                    context_len = int(seq_lens_cpu[req_id]) - int(q_len)
                    if use_dcp:
                        global_per_token_lens = torch.arange(
                            context_len + 1,
                            context_len + int(q_len) + 1,
                            dtype=torch.int32,
                        )
                        per_token_lens[start:end] = get_dcp_local_seq_lens(
                            global_per_token_lens,
                            self.dcp_world_size,
                            self.dcp_rank,
                            self.cp_kv_cache_interleave_size,
                        ).numpy()
                    else:
                        per_token_lens[start:end] = np.arange(
                            context_len + 1,
                            context_len + int(q_len) + 1,
                            dtype=np.int32,
                        )

                per_token_lens_t = torch.from_numpy(per_token_lens)
                if per_token_lens_t.device.type == "cpu":
                    per_token_lens_t = per_token_lens_t.pin_memory()
            if req_ids is not None:
                assert self.req_id_per_token_buffer is not None
                req_ids_t = torch.from_numpy(req_ids)
                if req_ids_t.device.type == "cpu":
                    req_ids_t = req_ids_t.pin_memory()
                self.req_id_per_token_buffer[:num_tokens].copy_(
                    req_ids_t, non_blocking=True
                )
                req_id_per_token_tensor = self.req_id_per_token_buffer[:num_tokens]
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                per_token_lens_t, non_blocking=True
            )
            self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                seq_lens_for_req[: cm.num_reqs], non_blocking=True
            )
            cache_seq_lens_per_token = self.cache_seq_lens_per_token_buffer[:num_tokens]
            cache_seq_lens_per_req = self.cache_seq_lens_per_req_buffer[: cm.num_reqs]

        return B12xMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=num_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            prefill_max_seq_len=cm.max_seq_len if num_prefills > 0 else 0,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token_tensor,
            page_table_1=(
                self.page_table_1_buffer[:num_tokens]
                if self.page_table_1_buffer is not None
                else None
            ),
            nsa_cache_seqlens=(
                self.nsa_cache_seqlens_buffer[:num_tokens]
                if self.nsa_cache_seqlens_buffer is not None
                else None
            ),
            seq_lens=cache_seq_lens_per_req,
            cache_seq_lens_per_req=cache_seq_lens_per_req,
            cache_seq_lens_per_token=cache_seq_lens_per_token,
            ckv_page_table_1=(
                self.ckv_page_table_1_buffer[:num_tokens]
                if self.ckv_page_table_1_buffer is not None
                else None
            ),
            ckv_nsa_cache_seqlens=(
                self.ckv_nsa_cache_seqlens_buffer[:num_tokens]
                if self.ckv_nsa_cache_seqlens_buffer is not None
                else None
            ),
            dcp_rank_req_starts=dcp_rank_req_starts,
            dcp_rank_req_lens=dcp_rank_req_lens,
            dcp_local_cu_seq_lens=dcp_local_cu_seq_lens,
            global_cache_seq_lens_per_req=(
                cm.seq_lens[: cm.num_reqs]
                if use_dcp and self.ckv_metadata_requested
                else None
            ),
            dcp_local_total_tokens=dcp_local_total_tokens,
            dcp_padded_total_tokens=dcp_padded_total_tokens,
            dcp_ckv_gather_eligible=dcp_ckv_gather_eligible,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class B12xMLASparseImpl(MLAAttentionImpl[B12xMLASparseMetadata]):
    """b12x unified sparse-MLA implementation (decode + extend/prefill)."""

    is_sparse: bool = True
    can_return_lse_for_decode: bool = True
    # B12X handles decode and extend inside its own top-k MQA kernels; the
    # generic dense-MHA prefill path assumes cache layouts it never validated.
    supports_mha_prefill: bool = False
    supports_dcp_project_before_merge: bool = True
    supports_dcp_gather_query_in_workspace: bool = True
    supports_dcp_project_before_merge_in_workspace: bool = True
    supports_dcp_reduce_scatter_output_in_workspace: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "B12X_MLA_SPARSE does not support alibi_slopes / sliding_window "
                "/ logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X_MLA_SPARSE only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self._kv_fp8_rope = bool(
            self.kv_cache_dtype == "nvfp4_ds_mla" and _kv_fp8_rope_enabled()
        )
        if _KV_FP8_ROPE_REQUESTED and not _is_glm_moe_dsa_model():
            logger.warning(
                "KV_FP8_ROPE=1 ignored: compact MLA records are restricted to "
                "model_type=glm_moe_dsa and its associated MTP draft"
            )
        if _kv_fp8_rope_enabled() and self.kv_cache_dtype != "nvfp4_ds_mla":
            logger.warning(
                "KV_FP8_ROPE=1 has no effect for kv_cache_dtype=%s; the compact "
                "record is GLM nvfp4_ds_mla-only",
                self.kv_cache_dtype,
            )
        if self._kv_fp8_rope:
            try:
                from b12x.attention.mla.kv_cache import (
                    concat_and_cache_nvfp4_mla_fp8_rope,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "KV_FP8_ROPE=1 requires a b12x build with "
                    "concat_and_cache_nvfp4_mla_fp8_rope package API support"
                ) from exc
            self._concat_and_cache_nvfp4_mla_fp8_rope = (
                concat_and_cache_nvfp4_mla_fp8_rope
            )

        # MLA dims (absorbed: Q post-projection is [T, H, kv_lora_rank + rope]).
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.v_head_dim: int = mla_args.get("v_head_dim", 512)
        # GLM_NSA contract: q_head_dim = kv_lora_rank (512) + qk_rope (64) = 576.
        self.q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.force_contiguous_mla_bmm_input = True
        self.force_contiguous_mla_bmm_weight = True
        self.force_contiguous_mla_bmm_output = True

        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )
        self._sparse_decode_emits_topk = indexer is not None
        assert self.topk_indices_buffer is not None, (
            "B12X_MLA_SPARSE requires sparse-MLA top-k indices "
            "(model with index_topk in its config)."
        )
        self.topk_tokens = int(self.topk_indices_buffer.shape[-1])

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.dcp_workspace_non_dbo = not bool(parallel_config.enable_dbo)
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.tp_world_size = int(parallel_config.tensor_parallel_size)
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        self.need_to_return_lse_for_decode = self.dcp_world_size > 1

        expects_physical_slots = self.dcp_world_size == 1
        if (
            indexer is not None
            and bool(indexer.output_physical_slots) != expects_physical_slots
        ):
            expected = "physical" if expects_physical_slots else "logical"
            raise RuntimeError(
                f"B12X_MLA_SPARSE requires {expected} sparse-indexer output "
                f"when dcp_world_size={self.dcp_world_size}"
            )

        scheduler_config = vllm_config.scheduler_config
        self.device = torch.device(f"cuda:{torch.accelerator.current_device_index()}")
        max_batched = int(scheduler_config.max_num_batched_tokens)
        max_num_seqs = int(scheduler_config.max_num_seqs)
        self.block_size = 64
        # NVFP4 MLA record selection: ScaleFormat.NVFP4_E4M3 (2) rides every
        # decode/extend call so the CuTeDSL kernels specialize on the packed
        # E2M1+E4M3 record instead of the 656 B fp8_ds_mla record. KV_FP8_ROPE
        # only changes its RoPE tail; the latent format and outer-scale
        # correction are deliberately untouched.
        self._b12x_scale_format = 2 if self.kv_cache_dtype == "nvfp4_ds_mla" else None
        self._kv_record_bytes = (
            (368 if self._kv_fp8_rope else 432)
            if self.kv_cache_dtype == "nvfp4_ds_mla"
            else 656
        )
        logger.info(
            "B12X GLM MLA KV format: KV_FP8_ROPE=%d kv_gmem_stride=%d "
            "kv_cache_dtype=%s",
            int(self._kv_fp8_rope),
            self._kv_record_bytes,
            self.kv_cache_dtype,
        )
        # MLAAttention all-gathers the local query-head shard before entering a
        # DCP backend. The kernel must therefore plan for, and return, the full
        # gathered head set; the outer layer reduces/scatters it back afterward.
        self._input_num_heads = self.num_heads * self.dcp_world_size

        # Split-K cap: ceil(topk / tile). Bounds the borrowed mid_out/mid_lse
        # chunk dim and the workspace max_chunks_per_row.
        self._num_splits_cap = max(1, _cdiv(self.topk_tokens, _DECODE_SPLIT_TILE))
        self._kernel_num_heads = (
            _cdiv(self._input_num_heads, _HEAD_ALIGNMENT) * _HEAD_ALIGNMENT
        )
        self._pad_heads = self._kernel_num_heads != self._input_num_heads

        self.spec_decode_max_q = _env_int("VLLM_B12X_MLA_SPEC_DECODE_MAX_Q", 8)
        # The decode kernel handles independent one-token query rows. MTP
        # verification has multiple query rows per request, and later rows must
        # attend to earlier draft rows in the same verifier batch. Route those
        # batches through the extend path unless explicitly overridden.
        self.spec_extend_as_decode = (
            os.getenv("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "0") != "0"
        )

        # Decode query rows per request (1, plus speculative draft tokens).
        q_per_req = 1
        spec = getattr(vllm_config, "speculative_config", None)
        if spec is not None and getattr(spec, "num_speculative_tokens", None):
            q_per_req = 1 + int(spec.num_speculative_tokens)
        self._sparse_decode_rows_per_req = q_per_req
        self._sparse_decode_max_reqs = min(
            max_num_seqs,
            _env_int("VLLM_B12X_MLA_SPARSE_DECODE_MAX_SEQS", 2),
        )
        self._sparse_decode_max_rows = (
            self._sparse_decode_max_reqs * self._sparse_decode_rows_per_req
        )
        if self.spec_extend_as_decode:
            q_per_req = max(q_per_req, self.spec_decode_max_q)
        self._decode_max_rows = min(max_num_seqs * q_per_req, max_batched)
        if self._decode_max_rows < max_num_seqs:
            self._decode_max_rows = max_num_seqs

        self._max_batched = int(max_batched)

        # Lazily import b12x only on this opt-in path.
        from b12x.integration.mla import (
            sparse_mla_decode_forward,
            sparse_mla_extend_forward,
        )
        from b12x.integration.sparse_mla_scratch import (
            B12XSparseMLAScratchCaps,
            plan_sparse_mla_scratch,
        )

        self._sparse_mla_decode_forward = sparse_mla_decode_forward
        self._sparse_mla_extend_forward = sparse_mla_extend_forward

        if self._b12x_scale_format is not None:
            required_kwargs = {"latent_scale", "scale_format"}
            unsupported_forwards = [
                mode
                for mode, forward in (
                    ("decode", sparse_mla_decode_forward),
                    ("extend", sparse_mla_extend_forward),
                )
                if not required_kwargs.issubset(inspect.signature(forward).parameters)
            ]
            if unsupported_forwards:
                raise RuntimeError(
                    "B12X_MLA_SPARSE with kv_cache_dtype='nvfp4_ds_mla' "
                    "requires a b12x build with NVFP4 sparse-MLA API support; "
                    "unsupported forwards: " + ", ".join(unsupported_forwards)
                )

        # Eager PLAN -> BIND -> KERNEL (no b12x workspace/arena, ever). We build a
        # caller-owned-scratch PLAN once per mode; each forward maps a vLLM
        # workspace-manager scratch tensor into a plain B12XSparseMLAScratch views
        # CONTAINER via plan.bind(). The unified SM120 sparse-MLA decode/extend
        # kernels duck-type the container's tmp_output/tmp_lse/output_buffer/
        # final_lse fields. The planner fixes the split count for each captured
        # graph and the merge specializes on that count, so no device-side control
        # scalar initialization is needed. final_lse is pre-materialized as a view
        # so the legacy lazy torch.empty(final_lse) never fires during capture.
        def _make_plan(
            mode: str, max_q_rows: int, num_q_heads: int, max_batch: int
        ) -> Any:
            return plan_sparse_mla_scratch(
                B12XSparseMLAScratchCaps(
                    device=self.device,
                    num_q_heads=int(num_q_heads),
                    max_q_rows=int(max_q_rows),
                    max_width=self.topk_tokens,
                    dtype=torch.bfloat16,
                    kv_dtype=torch.uint8,
                    head_dim=self.q_head_dim,
                    v_head_dim=self.kv_lora_rank,
                    mode=mode,
                    max_batch=int(max_batch),
                    max_chunks_per_row=self._num_splits_cap,
                    page_size=self.block_size,
                )
            )

        self._decode_plan = _make_plan(
            "decode",
            self._decode_max_rows,
            self._kernel_num_heads,
            self._decode_max_rows,
        )
        self._extend_plan = _make_plan(
            "extend", max_batched, self._kernel_num_heads, max_num_seqs
        )
        from vllm import envs as envs_mod

        sparse_decode_ckv_requested = (
            envs_mod.VLLM_B12X_MLA_SPARSE_DECODE_CKV_GATHER
        )
        self._sparse_decode_ckv_enabled = sparse_decode_ckv_requested and (
            self.dcp_world_size > 1
            and self.num_heads % _HEAD_ALIGNMENT == 0
            and self.topk_tokens % self.block_size == 0
            and self.kv_cache_dtype == "nvfp4_ds_mla"
            and self._kv_record_bytes == 432
            and not self._kv_fp8_rope
        )
        if sparse_decode_ckv_requested and not self._sparse_decode_ckv_enabled:
            logger.warning_once(
                "Ignoring VLLM_B12X_MLA_SPARSE_DECODE_CKV_GATHER for "
                "unsupported topology/format: dcp=%d local_heads=%d topk=%d "
                "block=%d dtype=%s record_bytes=%d KV_FP8_ROPE=%d",
                self.dcp_world_size,
                self.num_heads,
                self.topk_tokens,
                self.block_size,
                self.kv_cache_dtype,
                self._kv_record_bytes,
                int(self._kv_fp8_rope),
            )
        _validate_sparse_decode_graph_capacity(
            self._sparse_decode_ckv_enabled,
            self._sparse_decode_max_reqs,
            self._sparse_decode_rows_per_req,
            vllm_config.compilation_config.max_cudagraph_capture_size,
        )
        self._sparse_decode_plan = (
            _make_plan(
                "decode",
                self._sparse_decode_rows_per_req,
                self.num_heads,
                self._sparse_decode_rows_per_req,
            )
            if self._sparse_decode_ckv_enabled
            else None
        )
        self._sparse_decode_batch_plan = (
            _make_plan(
                "decode",
                self._sparse_decode_max_rows,
                self.num_heads,
                self._sparse_decode_max_rows,
            )
            if self._sparse_decode_ckv_enabled
            and self._sparse_decode_max_reqs > 1
            else self._sparse_decode_plan
        )
        # One caller-owned uint8 scratch tensor covers either path (the larger
        # layout); the per-mode materializer carves its views from the prefix.
        self._scratch_nbytes = max(
            int(self._decode_plan.layout.nbytes),
            int(self._extend_plan.layout.nbytes),
            int(self._sparse_decode_plan.layout.nbytes)
            if self._sparse_decode_plan is not None
            else 0,
            int(self._sparse_decode_batch_plan.layout.nbytes)
            if self._sparse_decode_batch_plan is not None
            else 0,
        )

        # Pre-touch q-concat + the attention scratch TOGETHER so the workspace
        # manager grows during warmup (before lock_workspace() runs
        # post-cudagraph-capture) and so the two always come from ONE
        # get_simultaneous call -> distinct, non-overlapping offsets. The manager
        # packs every call from offset 0, so borrowing q and the scratch the kernel
        # writes in separate calls would alias them.
        workspace_specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            (
                (max_batched, self._kernel_num_heads, self.q_head_dim),
                torch.bfloat16,
            )
        ]
        if self._pad_heads:
            workspace_specs.append(
                (
                    (max_batched, self._input_num_heads, self.kv_lora_rank),
                    torch.bfloat16,
                )
            )
        workspace_specs.append(((self._scratch_nbytes,), torch.uint8))
        self._workspace_specs = tuple(workspace_specs)
        self._borrow_workspaces()
        self._prewarm_extend_kernels_once(max_batched)

        # CKV gather setup (Fix B).
        ckv_gather_requested = envs_mod.VLLM_B12X_MLA_CKV_GATHER
        self._ckv_gather_enabled = ckv_gather_requested and (
            self.dcp_world_size > 1
            and self.num_heads % _HEAD_ALIGNMENT == 0
            and self.dcp_workspace_non_dbo
        )
        if ckv_gather_requested and not self._ckv_gather_enabled:
            logger.warning_once(
                "Ignoring VLLM_B12X_MLA_CKV_GATHER on unsupported "
                "topology: dcp=%d local_heads=%d DBO=%s",
                self.dcp_world_size,
                self.num_heads,
                not self.dcp_workspace_non_dbo,
            )
        self._ckv_kernel_num_heads = self.num_heads
        self._ckv_gather_max_tokens = envs_mod.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS
        self._ckv_gather_min_tokens = envs_mod.VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS
        self._ckv_prefetch_supported = (
            self._ckv_gather_enabled
            and _ckv_prefetch_supports_format(self.kv_cache_dtype)
        )
        self._ckv_prefetch_depth = (
            max(1, int(envs_mod.VLLM_B12X_MLA_CKV_PREFETCH_DEPTH))
            if self._ckv_prefetch_supported
            else 1
        )
        self._ckv_workspace_slots = self._ckv_prefetch_depth + 1
        self._ckv_local_capacity = (
            _cdiv(
                _cdiv(self._ckv_gather_max_tokens, max(1, self.dcp_world_size))
                + max_num_seqs * self.cp_kv_cache_interleave_size,
                self.block_size,
            )
            * self.block_size
        )
        self._ckv_workspace_nbytes = (
            (1 + self._ckv_workspace_slots * self.dcp_world_size)
            * self._ckv_local_capacity
            * self._kv_record_bytes
            if self._ckv_gather_enabled
            else 0
        )
        self._ckv_workspace = (
            _get_ckv_gather_workspace(self.device, self._ckv_workspace_nbytes)
            if self._ckv_gather_enabled
            else None
        )
        if self._sparse_decode_ckv_enabled and not self._ckv_gather_enabled:
            raise RuntimeError(
                "Sparse decode CKV gather currently requires full CKV gather "
                "to provision the shared side-stream/event infrastructure"
            )

        # Sparse decode gathers only selected records. The NCCL fallback builds
        # one global union; direct scatter builds one destination-specific union
        # while retaining only the local destination's sparse workspace.
        configured_prefetch_depth = int(
            envs_mod.VLLM_B12X_MLA_CKV_PREFETCH_DEPTH
        )
        requested_sparse_transport = (
            envs_mod.VLLM_B12X_MLA_SPARSE_DECODE_CKV_TRANSPORT
        )
        if requested_sparse_transport not in (
            "nccl",
            "p2p",
            "p2p_scatter",
            "p2p_ce_scatter",
        ):
            logger.warning_once(
                "Unknown sparse CKV transport %r; falling back to NCCL",
                requested_sparse_transport,
            )
            requested_sparse_transport = "nccl"
        self._sparse_decode_destination_scatter = bool(
            requested_sparse_transport in ("p2p_scatter", "p2p_ce_scatter")
        )
        self._sparse_decode_union_world_size = (
            1 if self._sparse_decode_destination_scatter else self.dcp_world_size
        )
        self._sparse_decode_destination_count = (
            self.dcp_world_size if self._sparse_decode_destination_scatter else 1
        )
        self._sparse_decode_per_req_capacity = (
            _cdiv(
                self.topk_tokens
                * self._sparse_decode_rows_per_req
                * self._sparse_decode_union_world_size,
                self.block_size,
            )
            * self.block_size
        )
        self._sparse_decode_padded_topk = (
            self._sparse_decode_max_reqs * self._sparse_decode_per_req_capacity
        )
        self._sparse_decode_prefetch_depth = (
            max(0, configured_prefetch_depth)
            if self._sparse_decode_ckv_enabled
            else 0
        )
        self._sparse_decode_workspace_slots = max(
            1, self._sparse_decode_prefetch_depth + 1
        )
        self._sparse_decode_workspace_nbytes = (
            self._sparse_decode_workspace_slots
            * self._sparse_decode_padded_topk
            * self._kv_record_bytes
            if self._sparse_decode_ckv_enabled
            else 0
        )

        self._sparse_decode_ckv_transport = requested_sparse_transport
        self._sparse_decode_p2p = None
        sparse_p2p_primary_capacity = _env_int(
            "VLLM_B12X_MLA_SPARSE_DECODE_CKV_PRIMARY_CAPACITY",
            max(
                1,
                _cdiv(
                    self._sparse_decode_padded_topk,
                    self.dcp_world_size,
                ),
            ),
        )
        sparse_p2p_key = (
            self.device.type,
            self.device.index,
            requested_sparse_transport,
            self._sparse_decode_workspace_slots,
            self._sparse_decode_padded_topk,
            self._kv_record_bytes,
            sparse_p2p_primary_capacity,
        )
        cached_p2p_fallback = _SPARSE_DECODE_P2P_FALLBACKS.get(sparse_p2p_key)
        if self._sparse_decode_ckv_enabled and cached_p2p_fallback == "stock":
            self._sparse_decode_ckv_enabled = False
            self._sparse_decode_ckv_transport = "stock"
        elif self._sparse_decode_ckv_enabled and cached_p2p_fallback == "nccl":
            self._sparse_decode_ckv_transport = "nccl"
        elif self._sparse_decode_ckv_enabled and requested_sparse_transport in (
            "p2p",
            "p2p_scatter",
            "p2p_ce_scatter",
        ):
            try:
                from vllm.distributed.parallel_state import (
                    get_dcp_ckv_prefetch_group,
                )
                from vllm.v1.attention.backends.mla.b12x_sparse_ckv_p2p import (
                    get_b12x_sparse_ckv_p2p,
                )

                p2p_group = get_dcp_ckv_prefetch_group().device_group
                if p2p_group is None:
                    raise RuntimeError("DCP CKV P2P group has no device group")
                self._sparse_decode_p2p = get_b12x_sparse_ckv_p2p(
                    process_group=p2p_group,
                    device=self.device,
                    workspace_slots=self._sparse_decode_workspace_slots,
                    topk=self._sparse_decode_padded_topk,
                    record_bytes=self._kv_record_bytes,
                    primary_capacity=sparse_p2p_primary_capacity,
                    allocate_direct_workspace=(
                        requested_sparse_transport != "p2p_ce_scatter"
                    ),
                )
                logger.info_once(
                    "Using B12X CUDA-IPC P2P sparse decode CKV transport=%s "
                    "(DCP%d, %d bytes/record, union_capacity=%d)",
                    requested_sparse_transport,
                    self.dcp_world_size,
                    self._kv_record_bytes,
                    self._sparse_decode_padded_topk,
                )
            except Exception as exc:
                if self._sparse_decode_destination_scatter:
                    logger.warning_once(
                        "Destination-specific sparse CKV P2P initialization "
                        "failed (%s); disabling the sparse decode fast path "
                        "and falling back to stock DCP decode",
                        exc,
                    )
                    _SPARSE_DECODE_P2P_FALLBACKS[sparse_p2p_key] = "stock"
                    self._sparse_decode_ckv_enabled = False
                    self._sparse_decode_ckv_transport = "stock"
                else:
                    logger.warning_once(
                        "Sparse CKV P2P initialization failed (%s); falling "
                        "back to NCCL",
                        exc,
                    )
                    _SPARSE_DECODE_P2P_FALLBACKS[sparse_p2p_key] = "nccl"
                    self._sparse_decode_ckv_transport = "nccl"

        # Resolve P2P availability before allocating sparse-only state. A
        # destination-scatter failure falls back to stock DCP decode, so these
        # buffers would otherwise consume KV headroom without ever being read.
        self._sparse_decode_workspace = (
            _get_sparse_decode_ckv_workspace(
                self.device, self._sparse_decode_workspace_nbytes
            )
            if self._sparse_decode_ckv_enabled
            else None
        )
        sparse_union_state = (
            _get_sparse_decode_union_state(
                self.device,
                self._sparse_decode_rows_per_req,
                self.topk_tokens,
                self._sparse_decode_union_world_size,
                self._sparse_decode_destination_count,
                self.dcp_world_size,
                self._sparse_decode_max_reqs,
            )
            if self._sparse_decode_ckv_enabled
            else (None, None, None, None, None, None, None, None, None)
        )
        (
            self._sparse_decode_union_indices,
            self._sparse_decode_selected_indices,
            self._sparse_decode_union_count,
            self._sparse_decode_hash_keys,
            self._sparse_decode_hash_values,
            self._sparse_decode_local_slots,
            self._sparse_decode_all_rank_topk,
            self._sparse_decode_all_rank_topk_by_req,
            self._sparse_decode_all_rank_remap,
        ) = sparse_union_state
        self._sparse_decode_req_ids = (
            torch.arange(
                self._sparse_decode_max_reqs,
                dtype=torch.int32,
                device=self.device,
            )
            if self._sparse_decode_ckv_enabled
            else None
        )
        self._sparse_decode_patch_slot = (
            _get_sparse_decode_patch_slots(
                self.device, self._sparse_decode_max_rows
            )
            if self._sparse_decode_ckv_enabled
            else None
        )

        # Separate extend plan for the gathered-cache path: full local heads
        # (no head all-gather), global seq lens.
        if self._ckv_gather_enabled or self._sparse_decode_ckv_enabled:
            self._ckv_extend_plan = _make_plan(
                "extend", max_batched, self._ckv_kernel_num_heads, max_num_seqs
            )
            self._scratch_nbytes = max(
                self._scratch_nbytes,
                int(self._ckv_extend_plan.layout.nbytes),
            )
        else:
            self._ckv_extend_plan = None

        # Layer prefetch uses a side stream and a bounded ring shared across
        # the per-layer implementation instances.
        if self._ckv_gather_enabled:
            shared_stream = getattr(
                B12xMLASparseImpl, "_shared_gather_stream", None
            )
            if shared_stream is None:
                shared_stream = torch.cuda.Stream(device=self.device)
                B12xMLASparseImpl._shared_gather_stream = shared_stream
            self._ckv_gather_stream = shared_stream
            self._ckv_current_chunk_kv_c: torch.Tensor | None = None
            self._ckv_current_chunk_kpe: torch.Tensor | None = None
            B12xMLASparseImpl._all_layer_kv_caches: list[torch.Tensor | None] = []
            B12xMLASparseImpl._all_layer_sparse_impls: list[
                B12xMLASparseImpl | None
            ] = []
            B12xMLASparseImpl._shared_gather_events: dict[
                int, tuple[torch.cuda.Event, int]
            ] = {}
            B12xMLASparseImpl._sparse_decode_gather_events: dict[
                int, tuple[torch.cuda.Event | None, int]
            ] = {}
            if not self._ckv_prefetch_supported:
                logger.warning_once(
                    "CKV gather prefetch disabled for kv_cache_dtype=%s "
                    "(KV_FP8_ROPE=%s); falling back to synchronous gather.",
                    self.kv_cache_dtype,
                    int(self._kv_fp8_rope),
                )
            else:
                logger.info_once(
                    "Using native CKV layer prefetch with depth=%d and "
                    "%d workspace slots.",
                    self._ckv_prefetch_depth,
                    self._ckv_workspace_slots,
                )
            if self._sparse_decode_prefetch_depth > 0:
                logger.info_once(
                    "Using depth-%d selected-record CKV decode prefetch",
                    self._sparse_decode_prefetch_depth,
                )
        else:
            self._ckv_gather_stream = None
            self._ckv_current_chunk_kv_c = None
            self._ckv_current_chunk_kpe = None

        # Q arrives BF16; the unified kernel quantizes inside.
        self.supports_quant_query_input = False

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Write the post-RoPE key using the selected runtime cache format.

        The disabled branch delegates to the shipped implementation unchanged,
        including its stock 432-byte NVFP4 writer. The enabled branch calls the
        b12x package writer API, which does not replace or perturb the stock
        writer used by KV_FP8_ROPE=0.
        """
        if not self._kv_fp8_rope:
            return super().do_kv_cache_update(
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )
        if kv_cache.numel() == 0:
            return
        if kv_cache_dtype != "nvfp4_ds_mla":
            raise RuntimeError(
                f"KV_FP8_ROPE writer reached a non-NVFP4 cache: {kv_cache_dtype!r}"
            )
        k_pe_flat = k_pe.squeeze(1)
        self._concat_and_cache_nvfp4_mla_fp8_rope(
            kv_c_normed,
            k_pe_flat,
            kv_cache,
            slot_mapping.flatten(),
            k_scale,
        )

    def _borrow_workspaces(self) -> list[torch.Tensor]:
        return current_workspace_manager().get_simultaneous(*self._workspace_specs)

    def _borrow_workspace_parts(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        workspace_tensors = self._borrow_workspaces()
        expected_count = 3 if self._pad_heads else 2
        if len(workspace_tensors) != expected_count:
            raise RuntimeError(
                "B12X DCP prefill borrowed an unexpected workspace count: "
                f"{len(workspace_tensors)} != {expected_count}"
            )
        q_workspace = workspace_tensors[0]
        dense_out_workspace = workspace_tensors[1] if self._pad_heads else None
        scratch_storage = workspace_tensors[-1]
        expected_q_shape = (
            self._max_batched,
            self._kernel_num_heads,
            self.q_head_dim,
        )
        if (
            tuple(q_workspace.shape) != expected_q_shape
            or q_workspace.dtype != torch.bfloat16
            or q_workspace.device != self.device
            or not q_workspace.is_contiguous()
        ):
            raise RuntimeError(
                "B12X DCP prefill borrowed an invalid query workspace: "
                f"shape={tuple(q_workspace.shape)}, dtype={q_workspace.dtype}, "
                f"device={q_workspace.device}"
            )
        if dense_out_workspace is not None and (
            tuple(dense_out_workspace.shape)
            != (self._max_batched, self._input_num_heads, self.kv_lora_rank)
            or dense_out_workspace.dtype != torch.bfloat16
            or dense_out_workspace.device != self.device
            or not dense_out_workspace.is_contiguous()
        ):
            raise RuntimeError("B12X DCP prefill borrowed an invalid dense output")
        if (
            tuple(scratch_storage.shape) != (self._scratch_nbytes,)
            or scratch_storage.dtype != torch.uint8
            or scratch_storage.device != self.device
            or not scratch_storage.is_contiguous()
        ):
            raise RuntimeError("B12X DCP prefill borrowed an invalid raw scratch")
        return q_workspace, dense_out_workspace, scratch_storage

    def _validate_dcp_prefill_workspace_contract(self, num_tokens: int) -> None:
        supported_topologies = {
            (4, 4),
            (6, 2),
            (6, 3),
            (6, 6),
            (8, 2),
            (8, 4),
            (8, 8),
        }
        if (
            not 1025 <= num_tokens <= self._max_batched
            or not self.dcp_workspace_non_dbo
            or (self.tp_world_size, self.dcp_world_size) not in supported_topologies
            or self.dcp_world_size <= 1
            or self.num_heads <= 0
            or self._input_num_heads != self.num_heads * self.dcp_world_size
            or self._kernel_num_heads < self._input_num_heads
            or self._kernel_num_heads % _HEAD_ALIGNMENT != 0
            or self.q_head_dim != 576
            or self.kv_lora_rank != 512
            or self.v_head_dim != 256
        ):
            raise RuntimeError(
                "The DCP prefill workspace path received an unsupported "
                "topology or geometry: "
                f"tokens={num_tokens}/{self._max_batched}, "
                f"TP/DCP={self.tp_world_size}/{self.dcp_world_size}, "
                f"local/input/kernel heads={self.num_heads}/"
                f"{self._input_num_heads}/{self._kernel_num_heads}, "
                f"pad_heads={self._pad_heads}, dimensions="
                f"{self.q_head_dim}/{self.kv_lora_rank}/{self.v_head_dim}"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("The DCP prefill workspace path is eager-only")

    def dcp_all_gather_query_in_workspace(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Gather local DCP query heads through the borrowed MLA workspaces."""
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            if ql_nope.ndim != 3 or q_pe.ndim != 3:
                raise ValueError("DCP workspace tuple queries must be rank-3")
            num_tokens, local_heads, nope_dim = ql_nope.shape
            if tuple(q_pe.shape) != (num_tokens, local_heads, 64) or nope_dim != 512:
                raise ValueError("DCP workspace requires noPE/RoPE dimensions 512/64")
            if ql_nope.dtype != torch.bfloat16 or q_pe.dtype != torch.bfloat16:
                raise TypeError("DCP workspace queries must be BF16")
            tuple_input = True
            query_device = ql_nope.device
        else:
            if q.ndim != 3:
                raise ValueError("DCP workspace query must be rank-3")
            num_tokens, local_heads, head_dim = q.shape
            if head_dim != self.q_head_dim or not q.is_contiguous():
                raise ValueError("DCP workspace tensor query has invalid layout")
            tuple_input = False
            query_device = q.device

        self._validate_dcp_prefill_workspace_contract(int(num_tokens))
        if local_heads != self.num_heads or query_device != self.device:
            raise ValueError("DCP workspace query does not match the MLA plan")

        from vllm.distributed.parallel_state import get_dcp_group

        dcp_group = get_dcp_group()
        process_group = dcp_group.device_group
        if (
            dcp_group.world_size != self.dcp_world_size
            or dcp_group.rank_in_group != self.dcp_rank
        ):
            raise RuntimeError("DCP workspace group does not match the MLA plan")

        q_workspace, _, scratch_storage = self._borrow_workspace_parts()
        q_begin = q_workspace.data_ptr()
        q_end = q_begin + q_workspace.numel() * q_workspace.element_size()
        scratch_begin = scratch_storage.data_ptr()
        scratch_end = scratch_begin + scratch_storage.numel()
        if q_begin < scratch_end and scratch_begin < q_end:
            raise RuntimeError("DCP query and scratch workspaces overlap")

        world_size = self.dcp_world_size
        head_dim = self.q_head_dim
        bytes_per_chunk_row = world_size * local_heads * head_dim * _BF16_BYTES
        chunk_capacity = scratch_storage.numel() // bytes_per_chunk_row
        if chunk_capacity <= 0:
            raise RuntimeError("DCP scratch cannot hold one gathered query row")

        q_workspace_flat = q_workspace.view(-1)
        chunk_start = 0
        while chunk_start < num_tokens:
            chunk_rows = min(chunk_capacity, num_tokens - chunk_start)
            if tuple_input:
                local_offset = chunk_start * self._kernel_num_heads * head_dim
                local_numel = chunk_rows * local_heads * head_dim
                local_chunk = q_workspace_flat.narrow(
                    0, local_offset, local_numel
                ).view(chunk_rows, local_heads, head_dim)
                ops.concat_mla_q(
                    ql_nope.narrow(0, chunk_start, chunk_rows),
                    q_pe.narrow(0, chunk_start, chunk_rows),
                    local_chunk,
                )
            else:
                local_chunk = cast(torch.Tensor, q).narrow(0, chunk_start, chunk_rows)

            gather_numel = world_size * chunk_rows * local_heads * head_dim
            gathered = (
                scratch_storage.narrow(0, 0, gather_numel * _BF16_BYTES)
                .view(torch.bfloat16)
                .view(world_size * chunk_rows, local_heads, head_dim)
            )
            dist.all_gather_into_tensor(
                gathered,
                local_chunk,
                group=process_group,
                async_op=False,
            )
            rank_major = gathered.view(world_size, chunk_rows, local_heads, head_dim)
            destination = q_workspace.narrow(0, chunk_start, chunk_rows)
            for source_rank in range(world_size):
                destination.narrow(1, source_rank * local_heads, local_heads).copy_(
                    rank_major[source_rank]
                )
            chunk_start += chunk_rows

        global_query = q_workspace[:num_tokens, : self._input_num_heads]
        expected_stride = (self._kernel_num_heads * head_dim, head_dim, 1)
        if (
            tuple(global_query.shape) != (num_tokens, self._input_num_heads, head_dim)
            or tuple(global_query.stride()) != expected_stride
        ):
            raise RuntimeError("DCP workspace produced an invalid query view")
        return global_query

    def dcp_project_before_merge_in_workspace(
        self,
        attn_out: torch.Tensor,
        lse: torch.Tensor,
        w_uv: torch.Tensor,
    ) -> torch.Tensor:
        """Project DCP partials from 512 to 256 in borrowed MLA storage."""
        num_tokens = int(attn_out.shape[0])
        self._validate_dcp_prefill_workspace_contract(num_tokens)
        if (
            tuple(attn_out.shape)
            != (num_tokens, self._input_num_heads, self.kv_lora_rank)
            or not attn_out.is_contiguous()
            or attn_out.dtype != torch.bfloat16
            or tuple(w_uv.shape)
            != (self._input_num_heads, self.kv_lora_rank, self.v_head_dim)
            or not w_uv.is_contiguous()
            or w_uv.dtype != torch.bfloat16
            or tuple(lse.shape) != (num_tokens, self._input_num_heads)
            or lse.dtype != torch.float32
        ):
            raise ValueError(
                "DCP workspace projection received an invalid tensor layout"
            )

        q_workspace, dense_out_workspace, scratch_storage = (
            self._borrow_workspace_parts()
        )
        input_numel = self._input_num_heads * num_tokens * self.kv_lora_rank
        projected_numel = self._input_num_heads * num_tokens * self.v_head_dim
        projected_nbytes = projected_numel * _BF16_BYTES
        if (
            q_workspace.numel() < input_numel
            or scratch_storage.numel() < projected_nbytes
        ):
            raise RuntimeError("DCP projection workspace is too small")
        expected_attn_storage = (
            dense_out_workspace if self._pad_heads else scratch_storage
        )
        assert expected_attn_storage is not None
        if attn_out.untyped_storage().data_ptr() != (
            expected_attn_storage.untyped_storage().data_ptr()
        ):
            raise RuntimeError(
                "DCP attention output is not backed by the expected MLA workspace"
            )

        projection_input = q_workspace.view(-1)[:input_numel].view(
            self._input_num_heads, num_tokens, self.kv_lora_rank
        )
        projected_head_major = (
            scratch_storage[:projected_nbytes]
            .view(torch.bfloat16)
            .view(self._input_num_heads, num_tokens, self.v_head_dim)
        )
        projection_input.copy_(attn_out.transpose(0, 1))
        torch.bmm(projection_input, w_uv, out=projected_head_major)
        return projected_head_major.transpose(0, 1)

    def dcp_reduce_scatter_output_in_workspace(
        self,
        corrected_attn_out: torch.Tensor,
    ) -> torch.Tensor:
        """Expose the dead query prefix as DCP reduce-scatter output."""
        num_tokens = int(corrected_attn_out.shape[0])
        self._validate_dcp_prefill_workspace_contract(num_tokens)
        input_head_major = corrected_attn_out.movedim(0, 1)
        if (
            tuple(corrected_attn_out.shape)
            != (num_tokens, self._input_num_heads, self.v_head_dim)
            or corrected_attn_out.dtype != torch.bfloat16
            or not input_head_major.is_contiguous()
        ):
            raise ValueError("DCP reduce-scatter input has an invalid layout")

        q_workspace, _, scratch_storage = self._borrow_workspace_parts()
        if (
            corrected_attn_out.untyped_storage().data_ptr()
            != scratch_storage.untyped_storage().data_ptr()
        ):
            raise RuntimeError(
                "DCP corrected input is not backed by the MLA scratch workspace"
            )
        output_numel = self.num_heads * num_tokens * self.v_head_dim
        output_head_major = q_workspace.view(-1)[:output_numel].view(
            self.num_heads, num_tokens, self.v_head_dim
        )
        output = output_head_major.transpose(0, 1)
        if not output_head_major.is_contiguous():
            raise RuntimeError("DCP reduce-scatter output is not contiguous")
        return output

    def _validate_ckv_workspace(self, ckv_workspace: torch.Tensor) -> None:
        if not self._ckv_gather_enabled:
            raise RuntimeError("CKV gather workspace requested while disabled")
        if (
            tuple(ckv_workspace.shape) != (self._ckv_workspace_nbytes,)
            or ckv_workspace.dtype != torch.uint8
            or ckv_workspace.device != self.device
            or not ckv_workspace.is_contiguous()
        ):
            raise RuntimeError("B12X CKV gather borrowed an invalid workspace")

    def _ckv_workspace_views(
        self, ckv_workspace: torch.Tensor, buf_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_ckv_workspace(ckv_workspace)
        if not 0 <= int(buf_idx) < self._ckv_workspace_slots:
            raise ValueError(
                f"CKV gather buffer index {buf_idx} is outside "
                f"[0, {self._ckv_workspace_slots})"
            )
        records = ckv_workspace.view(-1, self._kv_record_bytes)
        local_buffer = records[: self._ckv_local_capacity]
        gathered_base = (
            self._ckv_local_capacity
            + buf_idx * self.dcp_world_size * self._ckv_local_capacity
        )
        gathered_buffer = records[
            gathered_base : gathered_base
            + self.dcp_world_size * self._ckv_local_capacity
        ]
        return local_buffer, gathered_buffer

    def dcp_prefill_ckv_gather_eligible(
        self,
        attn_metadata: B12xMLASparseMetadata,
        num_tokens: int,
    ) -> bool:
        if not self._ckv_gather_enabled:
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        if (
            not attn_metadata.dcp_ckv_gather_eligible
            or attn_metadata.num_decode_tokens != 0
            or attn_metadata.num_prefill_tokens != attn_metadata.num_actual_tokens
            or int(num_tokens) != attn_metadata.num_actual_tokens
            or int(num_tokens) <= self._ckv_gather_min_tokens
            or attn_metadata.dcp_padded_total_tokens > self._ckv_local_capacity
            or attn_metadata.dcp_local_total_tokens
            > attn_metadata.dcp_padded_total_tokens
        ):
            return False
        return all(
            tensor is not None
            for tensor in (
                attn_metadata.req_id_per_token,
                attn_metadata.ckv_page_table_1,
                attn_metadata.ckv_nsa_cache_seqlens,
                attn_metadata.dcp_rank_req_starts,
                attn_metadata.dcp_rank_req_lens,
                attn_metadata.dcp_local_cu_seq_lens,
            )
        )

    def dcp_sparse_decode_ckv_gather_eligible(
        self,
        attn_metadata: B12xMLASparseMetadata,
        num_tokens: int,
    ) -> bool:
        """Gate sparse decode to a bounded batch with uniform MTP row counts."""
        num_reqs = int(attn_metadata.num_reqs)
        rows = int(num_tokens)
        if (
            self._sparse_decode_ckv_enabled
            and self.dcp_world_size > 1
            and num_reqs > self._sparse_decode_max_reqs
        ):
            logger.info_once(
                "Sparse selected-record CKV decode falling back to stock DCP "
                "decode: active requests=%d exceeds configured maximum=%d",
                num_reqs,
                self._sparse_decode_max_reqs,
            )
        return bool(
            self._sparse_decode_ckv_enabled
            and self.dcp_world_size > 1
            and 1 <= num_reqs <= self._sparse_decode_max_reqs
            and 1 <= rows <= self._sparse_decode_max_rows
            and rows % num_reqs == 0
            and rows // num_reqs <= self._sparse_decode_rows_per_req
            and attn_metadata.num_actual_tokens == rows
            and attn_metadata.max_query_len == rows // num_reqs
            and attn_metadata.req_id_per_token is not None
            and attn_metadata.page_table_1 is not None
            and attn_metadata.nsa_cache_seqlens is not None
            and attn_metadata.global_cache_seq_lens_per_req is not None
        )

    def _dcp_gather_sparse_decode_ckv(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        topk_indices: torch.Tensor,
        buf_idx: int = 0,
        stream: torch.cuda.Stream | None = None,
        build_union: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_rows = int(topk_indices.shape[0])
        num_reqs = int(attn_metadata.num_reqs)
        rows_per_req = num_rows // num_reqs
        per_req_capacity = self._sparse_decode_per_req_capacity
        active_capacity = num_reqs * per_req_capacity
        if not self.dcp_sparse_decode_ckv_gather_eligible(
            attn_metadata, num_rows
        ):
            raise RuntimeError(
                "Sparse decode CKV gather called for an ineligible batch"
            )
        if self._sparse_decode_workspace is None:
            raise RuntimeError("Sparse decode CKV workspace was not initialized")
        if self._sparse_decode_selected_indices is None:
            raise RuntimeError("Sparse decode CKV indices were not initialized")
        if self._sparse_decode_union_indices is None:
            raise RuntimeError("Sparse decode CKV union was not initialized")
        if self._sparse_decode_union_count is None:
            raise RuntimeError("Sparse decode CKV union count was not initialized")
        if self._sparse_decode_hash_keys is None:
            raise RuntimeError("Sparse decode CKV hash keys were not initialized")
        if self._sparse_decode_hash_values is None:
            raise RuntimeError("Sparse decode CKV hash values were not initialized")
        if self._sparse_decode_local_slots is None:
            raise RuntimeError("Sparse decode CKV local slots were not initialized")
        if self._sparse_decode_all_rank_topk is None:
            raise RuntimeError("Sparse decode CKV rank top-k was not initialized")
        if self._sparse_decode_all_rank_topk_by_req is None:
            raise RuntimeError("Sparse decode CKV request top-k was not initialized")
        if self._sparse_decode_all_rank_remap is None:
            raise RuntimeError("Sparse decode CKV rank remap was not initialized")
        if self._sparse_decode_req_ids is None:
            raise RuntimeError("Sparse decode CKV request ids were not initialized")
        if not 0 <= int(buf_idx) < self._sparse_decode_workspace_slots:
            raise ValueError(
                f"Sparse decode CKV buffer index {buf_idx} is outside "
                f"[0, {self._sparse_decode_workspace_slots})"
            )

        assert attn_metadata.req_id_per_token is not None
        assert attn_metadata.page_table_1 is not None
        assert attn_metadata.nsa_cache_seqlens is not None
        assert attn_metadata.global_cache_seq_lens_per_req is not None
        workspace = self._sparse_decode_workspace.view(
            self._sparse_decode_workspace_slots,
            self._sparse_decode_padded_topk,
            self._kv_record_bytes,
        )
        packed = workspace[buf_idx : buf_idx + 1, :active_capacity]
        local_slots_by_destination = self._sparse_decode_local_slots[
            : self._sparse_decode_destination_count, :num_reqs
        ]
        union_indices_by_destination = self._sparse_decode_union_indices[
            : self._sparse_decode_destination_count, :num_reqs
        ]
        local_destination = (
            self.dcp_rank if self._sparse_decode_destination_scatter else 0
        )
        union_indices_by_req = union_indices_by_destination[local_destination]
        local_slots = (
            local_slots_by_destination
            if self._sparse_decode_destination_scatter
            else local_slots_by_destination.reshape(1, active_capacity)
        )
        union_indices = union_indices_by_req.reshape(1, active_capacity)
        caller_stream = torch.cuda.current_stream()
        use_p2p = bool(
            self._sparse_decode_ckv_transport
            in ("p2p", "p2p_scatter", "p2p_ce_scatter")
            and self._sparse_decode_p2p is not None
        )
        use_p2p_scatter = bool(
            self._sparse_decode_ckv_transport
            in ("p2p_scatter", "p2p_ce_scatter")
            and self._sparse_decode_p2p is not None
        )
        use_p2p_ce_scatter = bool(
            self._sparse_decode_ckv_transport == "p2p_ce_scatter"
            and self._sparse_decode_p2p is not None
        )
        if self._sparse_decode_destination_scatter and not use_p2p_scatter:
            raise RuntimeError(
                "Destination-specific sparse CKV state requires direct P2P scatter"
            )
        if use_p2p and stream is not None:
            if self._ckv_gather_stream is None:
                raise RuntimeError("Sparse CKV P2P requires a gather stream")
            execution_stream = self._ckv_gather_stream
            execution_stream.wait_stream(caller_stream)
        elif stream is not None:
            stream.wait_stream(caller_stream)
            execution_stream = stream
        else:
            execution_stream = caller_stream

        with torch.cuda.stream(execution_stream):
            if build_union:
                from vllm.v1.attention.backends.mla.b12x_sparse_ckv_p2p import (
                    build_b12x_sparse_ckv_union_remap,
                )

                local_topk = topk_indices[
                    :num_rows, : self.topk_tokens
                ].contiguous()
                all_rank_rows = self.dcp_world_size * num_rows
                all_rank_topk = self._sparse_decode_all_rank_topk[:all_rank_rows]
                from vllm.distributed.parallel_state import get_dcp_group

                _dcp_all_gather_current_stream(
                    get_dcp_group(), local_topk, all_rank_topk
                )
                # Speculative decoding may schedule fewer than the configured
                # maximum number of MTP rows near request boundaries.
                all_rank_topk_by_req = self._sparse_decode_all_rank_topk_by_req[
                    :num_reqs, :, :rows_per_req
                ]
                all_rank_topk_by_req.copy_(
                    all_rank_topk.view(
                        self.dcp_world_size,
                        num_reqs,
                        rows_per_req,
                        self.topk_tokens,
                    ).permute(1, 0, 2, 3),
                    non_blocking=True,
                )
                for req_index in range(num_reqs):
                    for destination in range(self._sparse_decode_destination_count):
                        if self._sparse_decode_destination_scatter:
                            destination_topk = all_rank_topk_by_req[
                                req_index, destination
                            ]
                            local_row_start = 0
                        else:
                            destination_topk = all_rank_topk_by_req[
                                req_index
                            ].view(-1, self.topk_tokens)
                            local_row_start = self.dcp_rank * rows_per_req
                        destination_rows = int(destination_topk.shape[0])
                        destination_remap = self._sparse_decode_all_rank_remap[
                            req_index, destination, :destination_rows
                        ]
                        build_b12x_sparse_ckv_union_remap(
                            destination_topk,
                            union_indices_by_destination[destination, req_index],
                            destination_remap,
                            self._sparse_decode_union_count[
                                destination, req_index : req_index + 1
                            ],
                            self._sparse_decode_hash_keys[destination, req_index],
                            self._sparse_decode_hash_values[destination, req_index],
                        )
                        if destination == local_destination:
                            selected_start = req_index * rows_per_req
                            selected = self._sparse_decode_selected_indices[
                                selected_start : selected_start + rows_per_req,
                                : self.topk_tokens,
                            ]
                            selected.copy_(
                                destination_remap[
                                    local_row_start : local_row_start + rows_per_req
                                ],
                                non_blocking=True,
                            )
                            if req_index:
                                _offset_sparse_decode_remap(
                                    selected, req_index * per_req_capacity
                                )

            for destination in range(self._sparse_decode_destination_count):
                triton_filter_and_convert_dcp_index(
                    self._sparse_decode_req_ids[:num_reqs],
                    attn_metadata.block_table,
                    union_indices_by_destination[destination],
                    dcp_size=self.dcp_world_size,
                    dcp_rank=self.dcp_rank,
                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=per_req_capacity,
                    compact_valid_to_front=False,
                    out=local_slots_by_destination[destination],
                )
            if use_p2p:
                if use_p2p_ce_scatter:
                    transfer = self._sparse_decode_p2p.transfer_scatter_ce
                elif use_p2p_scatter:
                    transfer = self._sparse_decode_p2p.transfer_scatter
                else:
                    transfer = self._sparse_decode_p2p.transfer
                transfer(
                    kv_cache,
                    local_slots,
                    union_indices,
                    packed,
                    workspace_slot=buf_idx,
                    interleave=self.cp_kv_cache_interleave_size,
                )
            else:
                _pack_sparse_decode_ckv_local(kv_cache, local_slots, packed)

                from vllm.distributed.parallel_state import (
                    get_dcp_ckv_prefetch_group,
                    get_dcp_group,
                )

                dcp_group = (
                    get_dcp_ckv_prefetch_group()
                    if stream is not None
                    else get_dcp_group()
                )
                _dcp_all_reduce_current_stream(dcp_group, packed)

        selected_indices = self._sparse_decode_selected_indices[
            :num_rows, : self.topk_tokens
        ]
        nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[:num_rows]
        if stream is None:
            global_causal_lens = _global_causal_lens_for_ckv_gather(
                attn_metadata.global_cache_seq_lens_per_req,
                attn_metadata.query_start_loc,
                attn_metadata.req_id_per_token,
                num_rows,
            )
            nsa_cache_seqlens.copy_(global_causal_lens, non_blocking=True)
            nsa_cache_seqlens.clamp_(max=self.topk_tokens)
            _mask_page_table_after_nsa_len(selected_indices, nsa_cache_seqlens)
        gathered_cache = packed.view(-1, self.block_size, self._kv_record_bytes)
        return gathered_cache, selected_indices, nsa_cache_seqlens

    def _append_current_token_to_sparse_decode_gathered(
        self,
        gathered_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        union_indices: torch.Tensor,
        global_causal_lens: torch.Tensor,
        layer,
    ) -> None:
        """Patch a prefetched Shared-layer cache with all current MTP tokens."""
        if self._ckv_current_chunk_kv_c is None:
            raise RuntimeError("Sparse decode prefetch is missing current CKV")
        if self._ckv_current_chunk_kpe is None:
            raise RuntimeError("Sparse decode prefetch is missing current RoPE")
        if self._sparse_decode_patch_slot is None:
            raise RuntimeError("Sparse decode patch slot was not initialized")
        if attn_metadata.req_id_per_token is None:
            raise RuntimeError("Sparse decode patch is missing request ids")

        _find_sparse_decode_current_token_slot(
            union_indices,
            attn_metadata.req_id_per_token[: global_causal_lens.numel()],
            global_causal_lens,
            self._sparse_decode_patch_slot[: global_causal_lens.numel()],
        )
        num_rows = int(global_causal_lens.numel())
        kv_c = self._ckv_current_chunk_kv_c[:num_rows]
        k_pe = self._ckv_current_chunk_kpe[:num_rows]
        if k_pe.ndim == 3:
            k_pe = k_pe.squeeze(1)
        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            gathered_cache,
            self._sparse_decode_patch_slot[:num_rows],
            self.kv_cache_dtype,
            getattr(layer, "_k_scale", None),
        )

    def _dcp_gather_ckv(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        ckv_workspace: torch.Tensor,
        buf_idx: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        if not self.dcp_prefill_ckv_gather_eligible(
            attn_metadata, attn_metadata.num_actual_tokens
        ):
            raise RuntimeError("CKV gather called for an ineligible attention batch")
        if (
            kv_cache.dtype != torch.uint8
            or kv_cache.ndim != 3
            or tuple(kv_cache.shape[1:]) != (self.block_size, self._kv_record_bytes)
            or not kv_cache.is_contiguous()
        ):
            raise ValueError(
                "CKV gather requires contiguous native paged KV cache pages"
            )

        assert attn_metadata.dcp_local_cu_seq_lens is not None
        padded_tokens = attn_metadata.dcp_padded_total_tokens
        local_tokens = attn_metadata.dcp_local_total_tokens
        local_buffer, gathered_buffer = self._ckv_workspace_views(
            ckv_workspace, buf_idx
        )
        if stream is not None:
            # The side stream must observe the default stream's prior writes
            # to the paged KV cache (this and earlier steps' do_kv_cache_update)
            # before gathering history off it; there is otherwise no ordering
            # between the two streams in this direction. Enqueued before L's
            # attention/MLP, so the prefetch still overlaps that compute.
            stream.wait_stream(torch.cuda.current_stream())
            stream_ctx = torch.cuda.stream(stream)
        else:
            stream_ctx = torch.cuda.stream(torch.cuda.current_stream())
        with stream_ctx:
            if local_tokens:
                ops.cp_gather_cache(
                    src_cache=kv_cache,
                    dst=local_buffer[:local_tokens],
                    block_table=attn_metadata.block_table,
                    cu_seq_lens=attn_metadata.dcp_local_cu_seq_lens,
                    batch_size=attn_metadata.num_reqs,
                )
            if local_tokens < padded_tokens:
                local_buffer[local_tokens:padded_tokens].zero_()

            from vllm.distributed.parallel_state import (
                get_dcp_ckv_prefetch_group,
                get_dcp_group,
            )

            # The prefetch (side stream) uses a dedicated communicator so it
            # cannot collide with the indexer's DCP merge on the default
            # stream; the synchronous path shares the default stream with the
            # merge and is safe on the main DCP communicator.
            dcp_group = (
                get_dcp_ckv_prefetch_group() if stream is not None else get_dcp_group()
            )
            _dcp_all_gather_current_stream(
                dcp_group,
                local_buffer[:padded_tokens].view(-1),
                gathered_buffer[: self.dcp_world_size * padded_tokens].view(-1),
            )
        # Keep the cache geometry stable across requests. CuTe/B12X caches the
        # compiled prefill launch, while ``padded_tokens`` grows with context;
        # exposing a differently sized first dimension on every request can
        # reuse a launch specialized for an earlier, smaller cache. The live
        # records remain packed in the prefix and selected indices are still
        # based on ``padded_tokens``, so the unused capacity is unreachable.
        gathered_tokens = gathered_buffer[
            : self.dcp_world_size * self._ckv_local_capacity
        ]
        return gathered_tokens.view(-1, self.block_size, self._kv_record_bytes)

    def set_ckv_current_chunk_kv(
        self, kv_c_normed: torch.Tensor, k_pe: torch.Tensor
    ) -> None:
        self._ckv_current_chunk_kv_c = kv_c_normed
        self._ckv_current_chunk_kpe = k_pe

    def _resolve_layer_index(self, layer) -> int | None:
        """Global layer index used to key the cross-layer prefetch registry.

        ``MLAAttention`` exposes ``layer_name`` (a dotted prefix), not
        ``layer_idx``; without this fallback the prefetch pipeline never
        engages and every layer gathers synchronously on the critical path.
        """
        layer_idx = getattr(layer, "layer_idx", None)
        if layer_idx is not None:
            try:
                return int(layer_idx)
            except (TypeError, ValueError):
                return None
        layer_name = getattr(layer, "layer_name", None)
        if not layer_name:
            return None
        from vllm.model_executor.models.utils import extract_layer_index

        try:
            return extract_layer_index(layer_name)
        except (ValueError, AssertionError, IndexError):
            return None

    def _append_current_chunk_to_gathered(
        self,
        gathered_buffer: torch.Tensor,
        attn_metadata: "B12xMLASparseMetadata",
        layer,
        num_actual_toks: int,
    ) -> None:
        """Write the current chunk's BF16 KV into the gathered buffer for
        all DCP ranks.  Every rank already holds the full BF16 latent; the
        normal ``do_kv_cache_update`` only writes this rank's interleaved
        subset to the paged cache.  The prefetch gathered history from the
        next layer's cache *before* that layer's ``do_kv_cache_update``
        ran, so the current chunk is missing.  This method writes all
        tokens — not just the local rank's share — into the correct slots
        of the rank-ordered gathered buffer.
        """
        if self._ckv_current_chunk_kv_c is None or num_actual_toks == 0:
            return
        kv_c = self._ckv_current_chunk_kv_c[:num_actual_toks]
        k_pe_flat = self._ckv_current_chunk_kpe[:num_actual_toks]
        if k_pe_flat.ndim == 3:
            k_pe_flat = k_pe_flat.squeeze(1)

        num_reqs = attn_metadata.num_reqs
        interleave = self.cp_kv_cache_interleave_size
        global_seq_lens = attn_metadata.global_cache_seq_lens_per_req
        if global_seq_lens is None:
            return
        global_seq_lens = global_seq_lens[:num_reqs]
        req_ids = attn_metadata.req_id_per_token[:num_actual_toks].to(torch.int64)
        global_seq_per_token = global_seq_lens[req_ids].to(torch.int32)

        # Map each current-chunk token to its absolute position within its
        # own sequence. ``num_actual_toks`` spans the whole batch, so the
        # position must be computed per request: the current chunk holds the
        # last ``chunk_len_r`` positions of request r, and this token is the
        # ``(t - query_start_loc[r])``-th of them. Using the batch-global
        # ``t``/``num_actual_toks`` is only correct for a single request and
        # otherwise misplaces every request but the last.
        query_start_loc = attn_metadata.query_start_loc[: num_reqs + 1].to(torch.int32)
        req_chunk_start = query_start_loc[:-1][req_ids]
        req_chunk_len = (query_start_loc[1:] - query_start_loc[:-1])[req_ids]
        t = torch.arange(num_actual_toks, device=self.device, dtype=torch.int32)
        global_pos = global_seq_per_token - req_chunk_len + (t - req_chunk_start)
        owner = ((global_pos // interleave) % self.dcp_world_size).to(torch.int64)
        local_pos = (
            global_pos // (self.dcp_world_size * interleave) * interleave
            + global_pos % interleave
        ).to(torch.int64)

        rank_req_starts = attn_metadata.dcp_rank_req_starts
        flat_idx = owner * num_reqs + req_ids
        # ``dcp_rank_req_starts`` is a ``[dcp, :num_reqs]`` slice of a
        # ``[dcp, max_seqs]`` buffer, so it is non-contiguous whenever
        # ``num_reqs < max_seqs``; ``reshape`` compacts to row-length
        # ``num_reqs`` to match ``flat_idx``. ``view`` would raise here.
        rank_start = rank_req_starts.reshape(-1)[flat_idx].to(torch.int64)

        padded_tokens = attn_metadata.dcp_padded_total_tokens
        slots = owner * int(padded_tokens) + rank_start + local_pos

        k_scale = getattr(layer, "_k_scale", None)
        if self._kv_fp8_rope:
            self._concat_and_cache_nvfp4_mla_fp8_rope(
                kv_c,
                k_pe_flat,
                gathered_buffer,
                slots,
                k_scale,
            )
        elif self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla"):
            ops.concat_and_cache_mla(
                kv_c,
                k_pe_flat,
                gathered_buffer,
                slots,
                self.kv_cache_dtype,
                k_scale,
            )
        else:
            raise RuntimeError(
                "CKV gather prefetch append is not yet supported for "
                f"kv_cache_dtype={self.kv_cache_dtype!r}; disable prefetch "
                "or use fp8_ds_mla / nvfp4_ds_mla."
            )

    def _sync_warmup(self) -> None:
        if self.device.type == "cuda":
            torch.accelerator.synchronize(self.device)
        if self.dcp_world_size <= 1:
            return
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            get_dcp_group().barrier()
        except Exception:
            return
        finally:
            if self.device.type == "cuda":
                torch.accelerator.synchronize(self.device)

    def _b12x_kernel_format_kwargs(self, latent_scale: float = 1.0) -> dict[str, Any]:
        if self._b12x_scale_format is None:
            return {}
        return {
            "latent_scale": float(latent_scale),
            "scale_format": self._b12x_scale_format,
        }

    def _prewarm_extend_kernels_once(self, max_batched: int) -> None:
        if self.device.type != "cuda":
            return
        key = (
            self.device.index,
            self.q_head_dim,
            self.kv_lora_rank,
            self._kernel_num_heads,
            int(self.topk_tokens),
            int(self.block_size),
            bool(self.need_to_return_lse_for_decode),
            self.kv_cache_dtype,
            bool(self._kv_fp8_rope),
        )
        if key in _EXTEND_PREWARM_DONE:
            return
        _EXTEND_PREWARM_DONE.add(key)
        kernel_format_kwargs = self._b12x_kernel_format_kwargs()

        rows_to_warm = (1, 2, 4, max(1, int(max_batched)))
        seen_rows: set[int] = set()
        # GLM fp8_ds_mla cache records are 656 B/token; the real KV cache is
        # laid out (num_blocks, block_size, 656) (see the allocator at the
        # block-shape branch above), so a page's stride(0) = block_size*656.
        # The prewarm dummy must match that layout -- (1, block_size, 656) --
        # so _cache_block_stride_bytes sees stride >= page_size*656. The prior
        # (block_size, 1, 656) shape put block_size in dim 0, giving stride(0)
        # = 656 < page_size*656, which tripped the SM120 stride assertion
        # whenever this prewarm ran (i.e. spec + cudagraphs, the first config
        # to reach here; verifier-only and eager-snap both skipped it).
        # One page is enough: prewarm top-k indices all point at slot zero.
        kv_cache = torch.zeros(
            (1, self.block_size, self._kv_record_bytes),
            dtype=torch.uint8,
            device=self.device,
        )
        for rows in rows_to_warm:
            rows = int(rows)
            if rows in seen_rows:
                continue
            seen_rows.add(rows)
            q = torch.zeros(
                (rows, self._kernel_num_heads, self.q_head_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
            selected_indices = torch.zeros(
                (rows, self.topk_tokens), dtype=torch.int32, device=self.device
            )
            cache_seqlens = torch.full(
                (1,), self.block_size, dtype=torch.int32, device=self.device
            )
            nsa_cache_seqlens = torch.ones(
                (rows,), dtype=torch.int32, device=self.device
            )
            scratch_storage = torch.empty(
                (self._scratch_nbytes,), dtype=torch.uint8, device=self.device
            )
            binding = self._extend_plan.bind(
                scratch=scratch_storage,
                q=q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            if self.need_to_return_lse_for_decode:
                self._sparse_mla_extend_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    return_lse=True,
                    lse_scale="natural",
                    **kernel_format_kwargs,
                )
            else:
                self._sparse_mla_extend_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    **kernel_format_kwargs,
                )
            self._sync_warmup()

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Stored by MultiHeadLatentAttentionWrapper as a host float. Avoid
        # reading device state or allocating per-call CUDA state; the CuTe
        # launch receives this as a runtime scalar.
        latent_scale = float(getattr(layer, "_nvfp4_mla_outer_scale", 1.0))
        kernel_format_kwargs = self._b12x_kernel_format_kwargs(latent_scale)
        query_rows = q[0].shape[0] if isinstance(q, tuple) else q.shape[0]
        use_ckv_gather = self.dcp_prefill_ckv_gather_eligible(
            attn_metadata, int(query_rows)
        )
        use_sparse_decode_ckv_gather = (
            self.dcp_sparse_decode_ckv_gather_eligible(
                attn_metadata, int(query_rows)
            )
        )
        use_local_query_heads = use_ckv_gather or use_sparse_decode_ckv_gather
        workspace_tensors = self._borrow_workspaces()
        q_workspace = workspace_tensors[0]
        dense_out_workspace = workspace_tensors[1] if self._pad_heads else None
        ckv_workspace = self._ckv_workspace
        scratch_storage = workspace_tensors[-1]
        expected_input_heads = (
            self.num_heads if use_local_query_heads else self._input_num_heads
        )
        if use_local_query_heads:
            local_q_numel = (
                self._max_batched * self._ckv_kernel_num_heads * self.q_head_dim
            )
            q_buffer = q_workspace.view(-1)[:local_q_numel].view(
                self._max_batched,
                self._ckv_kernel_num_heads,
                self.q_head_dim,
            )
        else:
            q_buffer = q_workspace
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            num_actual_toks = ql_nope.shape[0]
            num_input_heads = ql_nope.shape[1]
            if num_input_heads != expected_input_heads:
                raise ValueError(
                    "B12X_MLA_SPARSE query heads do not match the planned "
                    f"head count: {num_input_heads} != {expected_input_heads}."
                )
            q_buffer = q_buffer[:num_actual_toks]
            q_all = q_buffer[:, :num_input_heads]
            ops.concat_mla_q(ql_nope, q_pe, q_all)
        else:
            num_actual_toks = q.shape[0]
            num_input_heads = q.shape[1]
            if num_input_heads != expected_input_heads:
                raise ValueError(
                    "B12X_MLA_SPARSE query heads do not match the planned "
                    f"head count: {num_input_heads} != {expected_input_heads}."
                )
            q_buffer = q_buffer[:num_actual_toks]
            q_all = q_buffer[:, :num_input_heads]
            exact_workspace_alias = (
                tuple(q.shape) == tuple(q_all.shape)
                and tuple(q.stride()) == tuple(q_all.stride())
                and q.dtype == q_all.dtype
                and q.device == q_all.device
                and q.untyped_storage().data_ptr() == q_all.untyped_storage().data_ptr()
                and q.storage_offset() == q_all.storage_offset()
            )
            if not exact_workspace_alias:
                q_all.copy_(q.contiguous())

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        per_token_cache = attn_metadata.cache_seq_lens_per_token[:num_actual_toks]
        sparse_decode_global_causal_lens = None
        if use_sparse_decode_ckv_gather:
            assert attn_metadata.global_cache_seq_lens_per_req is not None
            assert attn_metadata.req_id_per_token is not None
            sparse_decode_global_causal_lens = (
                _global_causal_lens_for_ckv_gather(
                    attn_metadata.global_cache_seq_lens_per_req,
                    attn_metadata.query_start_loc,
                    attn_metadata.req_id_per_token,
                    num_actual_toks,
                )
            )
        if use_ckv_gather:
            assert attn_metadata.req_id_per_token is not None
            assert attn_metadata.ckv_page_table_1 is not None
            assert attn_metadata.ckv_nsa_cache_seqlens is not None
            assert attn_metadata.dcp_rank_req_starts is not None
            assert attn_metadata.dcp_rank_req_lens is not None
            selected_indices = attn_metadata.ckv_page_table_1[
                :num_actual_toks, : topk_indices.shape[1]
            ]
            nsa_cache_seqlens = attn_metadata.ckv_nsa_cache_seqlens[:num_actual_toks]
            _map_global_topk_to_gathered_ckv(
                attn_metadata.req_id_per_token[:num_actual_toks],
                topk_indices,
                attn_metadata.dcp_rank_req_starts,
                attn_metadata.dcp_rank_req_lens,
                selected_indices,
                nsa_cache_seqlens,
                dcp_size=self.dcp_world_size,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                padded_rank_tokens=attn_metadata.dcp_padded_total_tokens,
            )
            # The gathered buffer holds the *global* sequence and the extend
            # kernel attends it in full (cache_seqlens = global lengths), so
            # the count of selected entries must be capped by the global
            # per-token causal length. ``per_token_cache`` is the *local*
            # per-rank length (~global/dcp); using it drops the other ranks'
            # selected tokens whenever it is the smaller bound
            # (local_len < min(topk, global_len)) -- i.e. for short contexts,
            # which shows up as small-context-only garbage.
            assert attn_metadata.global_cache_seq_lens_per_req is not None
            global_causal_len = _global_causal_lens_for_ckv_gather(
                attn_metadata.global_cache_seq_lens_per_req,
                attn_metadata.query_start_loc,
                attn_metadata.req_id_per_token,
                num_actual_toks,
            )
            torch.minimum(
                nsa_cache_seqlens,
                global_causal_len,
                out=nsa_cache_seqlens,
            )
            _mask_page_table_after_nsa_len(selected_indices, nsa_cache_seqlens)
        elif use_sparse_decode_ckv_gather:
            # Preserve the global logical top-k until selected native records
            # have been materialized into the sparse workspace below.
            selected_indices = topk_indices
            nsa_cache_seqlens = per_token_cache
        elif self.dcp_world_size > 1:
            # The indexer globally merges logical top-k ids across DCP ranks.
            # Compact just this rank's winners into local physical cache slots;
            # the outer MLA layer combines the rank-local outputs using LSE.
            assert attn_metadata.req_id_per_token is not None
            assert attn_metadata.page_table_1 is not None
            assert attn_metadata.nsa_cache_seqlens is not None
            selected_indices = attn_metadata.page_table_1[
                :num_actual_toks, : topk_indices.shape[1]
            ]
            nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[:num_actual_toks]
            # Zero-copy: the kernel scatters directly into the persistent
            # CUDA-graph-stable views consumed by the b12x planned kernels.
            triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                out=selected_indices,
                valid_counts=nsa_cache_seqlens,
            )
            torch.minimum(
                nsa_cache_seqlens,
                per_token_cache,
                out=nsa_cache_seqlens,
            )
            _mask_page_table_after_nsa_len(selected_indices, nsa_cache_seqlens)
        else:
            # Without DCP, the b12x indexer writes flat physical cache slots
            # directly into the shared top-k buffer.
            selected_indices = topk_indices
            nsa_cache_seqlens = per_token_cache

        # KV cache -> paged rank-3 uint8. B12X unified SM120 kernels consume
        # flat slot ids in selected_indices, but compute raw byte offsets as:
        #   block = slot // page_size, local = slot % page_size
        # so the cache tensor itself must expose a per-block stride of
        # block_size * record_bytes. The older split path used a token-flat
        # (num_slots, 1, bytes) view; that makes stride(0) one record and breaks
        # the unified block-stride contract.
        kv_u8 = kv_c_and_k_pe_cache.view(torch.uint8)
        if kv_u8.ndim == 3 and kv_u8.shape[1] == self.block_size:
            kv_cache = kv_u8
        elif kv_u8.ndim == 3 and kv_u8.shape[1] == 1:
            if kv_u8.shape[0] % self.block_size != 0:
                raise ValueError(
                    "B12X_MLA_SPARSE flat KV cache rows must be divisible by "
                    f"block_size={self.block_size}; got {kv_u8.shape[0]}"
                )
            kv_cache = kv_u8.reshape(-1, self.block_size, kv_u8.shape[-1])
        else:
            raise ValueError(
                f"B12X_MLA_SPARSE expected {self.kv_cache_dtype} KV cache as "
                f"(blocks,{self.block_size},bytes) or (slots,1,bytes), got "
                f"{tuple(kv_u8.shape)}"
            )
        if not kv_cache.is_contiguous():
            raise ValueError(
                "B12X_MLA_SPARSE requires a contiguous native paged KV cache; "
                f"got stride={tuple(kv_cache.stride())}"
            )
        if use_sparse_decode_ckv_gather:
            layer_idx = self._resolve_layer_index(layer)
            if layer_idx is not None:
                while len(B12xMLASparseImpl._all_layer_kv_caches) <= layer_idx:
                    B12xMLASparseImpl._all_layer_kv_caches.append(None)
                while len(B12xMLASparseImpl._all_layer_sparse_impls) <= layer_idx:
                    B12xMLASparseImpl._all_layer_sparse_impls.append(None)
                B12xMLASparseImpl._all_layer_kv_caches[layer_idx] = kv_cache
                B12xMLASparseImpl._all_layer_sparse_impls[layer_idx] = self

            pending_gather = (
                B12xMLASparseImpl._sparse_decode_gather_events.pop(layer_idx, None)
                if layer_idx is not None
                else None
            )
            if pending_gather is not None:
                gather_event, gather_buf_idx = pending_gather
                # During CUDA-graph capture the Full layer queues all three
                # Shared-layer transfers on the captured main stream. Stream
                # FIFO already provides the dependency, so no event is needed.
                if gather_event is not None:
                    torch.cuda.current_stream().wait_event(gather_event)
                if self._sparse_decode_workspace is None:
                    raise RuntimeError(
                        "Sparse decode CKV workspace was not initialized"
                    )
                if self._sparse_decode_union_indices is None:
                    raise RuntimeError("Sparse decode CKV union was not initialized")
                if self._sparse_decode_selected_indices is None:
                    raise RuntimeError("Sparse decode CKV remap was not initialized")
                if sparse_decode_global_causal_lens is None:
                    raise RuntimeError(
                        "Sparse decode CKV gather is missing global causal lengths"
                    )
                sparse_workspace = self._sparse_decode_workspace.view(
                    self._sparse_decode_workspace_slots,
                    self._sparse_decode_padded_topk,
                    self._kv_record_bytes,
                )
                active_num_reqs = int(attn_metadata.num_reqs)
                active_capacity = (
                    active_num_reqs * self._sparse_decode_per_req_capacity
                )
                kv_cache = sparse_workspace[
                    gather_buf_idx : gather_buf_idx + 1, :active_capacity
                ].view(-1, self.block_size, self._kv_record_bytes)
                self._append_current_token_to_sparse_decode_gathered(
                    kv_cache,
                    attn_metadata,
                    (
                        self._sparse_decode_union_indices[
                            self.dcp_rank, :active_num_reqs
                        ]
                        if self._sparse_decode_destination_scatter
                        else self._sparse_decode_union_indices[
                            0, :active_num_reqs
                        ]
                    ),
                    sparse_decode_global_causal_lens,
                    layer,
                )
                selected_indices = self._sparse_decode_selected_indices[
                    :num_actual_toks, : self.topk_tokens
                ]
                nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[
                    :num_actual_toks
                ]
                nsa_cache_seqlens.copy_(
                    sparse_decode_global_causal_lens, non_blocking=True
                )
                nsa_cache_seqlens.clamp_(max=self.topk_tokens)
                _mask_page_table_after_nsa_len(
                    selected_indices, nsa_cache_seqlens
                )
            else:
                sync_buf_idx = (
                    layer_idx % self._sparse_decode_workspace_slots
                    if layer_idx is not None
                    else 0
                )
                kv_cache, selected_indices, nsa_cache_seqlens = (
                    self._dcp_gather_sparse_decode_ckv(
                        kv_cache,
                        attn_metadata,
                        topk_indices,
                        buf_idx=sync_buf_idx,
                    )
                )

            logger.info_once(
                "Using lossless selected-record CKV union gather for C<=%d/MTP "
                "DCP decode (rows_per_req<=%d topk=%d pooled_capacity=%d "
                "record_bytes=%d shared_prefetch_depth=%d transport=%s "
                "destination_scatter=%s)",
                self._sparse_decode_max_reqs,
                self._sparse_decode_rows_per_req,
                self.topk_tokens,
                self._sparse_decode_padded_topk,
                self._kv_record_bytes,
                self._sparse_decode_prefetch_depth,
                self._sparse_decode_ckv_transport,
                self._sparse_decode_destination_scatter,
            )

            if (
                self._sparse_decode_emits_topk
                and layer_idx is not None
                and self._ckv_gather_stream is not None
            ):
                capturing = torch.cuda.is_current_stream_capturing()
                emits_topk_by_layer = [
                    None if impl is None else impl._sparse_decode_emits_topk
                    for impl in B12xMLASparseImpl._all_layer_sparse_impls
                ]
                target_indices = _sparse_decode_prefetch_targets(
                    layer_idx,
                    self._sparse_decode_prefetch_depth,
                    emits_topk_by_layer,
                )
                for target_idx in target_indices:
                    if target_idx in B12xMLASparseImpl._sparse_decode_gather_events:
                        continue
                    target_impl = B12xMLASparseImpl._all_layer_sparse_impls[
                        target_idx
                    ]
                    target_kv = B12xMLASparseImpl._all_layer_kv_caches[target_idx]
                    if target_impl is None or target_kv is None:
                        break
                    target_buf_idx = (
                        target_idx % self._sparse_decode_workspace_slots
                    )
                    target_impl._dcp_gather_sparse_decode_ckv(
                        target_kv,
                        attn_metadata,
                        topk_indices,
                        buf_idx=target_buf_idx,
                        # Multi-stream event plumbing is not graph-safe here.
                        # Capture the same depth-3 schedule on the main stream;
                        # eager execution keeps the overlapping side stream.
                        stream=None if capturing else self._ckv_gather_stream,
                        build_union=False,
                    )
                    target_event = None
                    if not capturing:
                        target_event = torch.cuda.Event(blocking=False)
                        target_event.record(self._ckv_gather_stream)
                    B12xMLASparseImpl._sparse_decode_gather_events[target_idx] = (
                        target_event,
                        target_buf_idx,
                    )
        if use_ckv_gather:
            if ckv_workspace is None:
                raise RuntimeError("CKV gather workspace was not borrowed")
            layer_idx = self._resolve_layer_index(layer)
            if layer_idx is not None:
                while len(B12xMLASparseImpl._all_layer_kv_caches) <= layer_idx:
                    B12xMLASparseImpl._all_layer_kv_caches.append(None)
                B12xMLASparseImpl._all_layer_kv_caches[layer_idx] = kv_cache
            pending = (
                B12xMLASparseImpl._shared_gather_events.pop(layer_idx, None)
                if layer_idx is not None
                else None
            )
            if pending is not None:
                gather_event, current_buf_idx = pending
                gather_event.wait()
                _, gathered_buffer = self._ckv_workspace_views(
                    ckv_workspace, current_buf_idx
                )
                kv_cache = gathered_buffer[
                    : self.dcp_world_size * self._ckv_local_capacity
                ].view(-1, self.block_size, self._kv_record_bytes)
                self._append_current_chunk_to_gathered(
                    kv_cache, attn_metadata, layer, num_actual_toks
                )
            else:
                current_buf_idx = (
                    layer_idx % self._ckv_workspace_slots
                    if layer_idx is not None
                    else 0
                )
                kv_cache = self._dcp_gather_ckv(
                    kv_cache,
                    attn_metadata,
                    ckv_workspace,
                    buf_idx=current_buf_idx,
                )
            logger.info_once(
                "Using transient full-CKV gather for B12X sparse MLA prefill "
                "(capacity=%d logical tokens)",
                self._ckv_gather_max_tokens,
            )
            if self._ckv_prefetch_supported and layer_idx is not None:
                targets = _ckv_prefetch_target_indices(
                    layer_idx,
                    self._ckv_prefetch_depth,
                    B12xMLASparseImpl._all_layer_kv_caches,
                    B12xMLASparseImpl._shared_gather_events,
                )
                if layer_idx == 0 and not targets:
                    logger.info_once(
                        "CKV layer prefetch registry is warming; the first "
                        "eligible request uses synchronous gathers."
                    )
                for target_idx in targets:
                    target_kv = B12xMLASparseImpl._all_layer_kv_caches[target_idx]
                    assert target_kv is not None
                    target_buf_idx = target_idx % self._ckv_workspace_slots
                    self._dcp_gather_ckv(
                        target_kv,
                        attn_metadata,
                        ckv_workspace,
                        buf_idx=target_buf_idx,
                        stream=self._ckv_gather_stream,
                    )
                    target_event = torch.cuda.Event(blocking=False)
                    target_event.record(self._ckv_gather_stream)
                    B12xMLASparseImpl._shared_gather_events[target_idx] = (
                        target_event,
                        target_buf_idx,
                    )

        use_decode_kernel = use_sparse_decode_ckv_gather or (
            attn_metadata.max_query_len <= 1
        ) or (
            self.spec_extend_as_decode
            and attn_metadata.max_query_len <= self.spec_decode_max_q
            and num_actual_toks <= attn_metadata.num_reqs * self.spec_decode_max_q
            and num_actual_toks <= self._decode_max_rows
        )
        if use_decode_kernel:
            cache_seqlens = (
                sparse_decode_global_causal_lens
                if use_sparse_decode_ckv_gather
                else attn_metadata.cache_seq_lens_per_req
                if attn_metadata.max_query_len <= 1
                else attn_metadata.cache_seq_lens_per_token[:num_actual_toks]
            )
            if cache_seqlens is None:
                raise RuntimeError(
                    "Sparse decode CKV gather is missing global sequence lengths"
                )
            decode_q = q_all
            if self._pad_heads and not use_sparse_decode_ckv_gather:
                decode_q = q_buffer[:, : self._kernel_num_heads]
                decode_q[:, self._input_num_heads :, :].zero_()
            # Eager bind maps caller-owned scratch into views. forced_num_splits
            # pins the planner choice for this captured graph; the merge kernel is
            # specialized on that count and needs no device-side control fill.
            if use_sparse_decode_ckv_gather:
                decode_plan = (
                    self._sparse_decode_plan
                    if int(attn_metadata.num_reqs) == 1
                    else self._sparse_decode_batch_plan
                )
            else:
                decode_plan = self._decode_plan
            if decode_plan is None:
                raise RuntimeError("Sparse decode CKV plan was not initialized")
            binding = decode_plan.bind(
                scratch=scratch_storage,
                q=decode_q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            if self.need_to_return_lse_for_decode and not use_sparse_decode_ckv_gather:
                out, lse = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_decode_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        forced_num_splits=self._num_splits_cap,
                        return_lse=True,
                        lse_scale="natural",
                        **kernel_format_kwargs,
                    ),
                )
                if self._pad_heads:
                    assert dense_out_workspace is not None
                    dense_out = dense_out_workspace[:num_actual_toks]
                    dense_out.copy_(out[:, : self._input_num_heads, :])
                    out = dense_out
                    lse = lse[:, : self._input_num_heads]
                return out, lse
            out = cast(
                torch.Tensor,
                self._sparse_mla_decode_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    forced_num_splits=self._num_splits_cap,
                    **kernel_format_kwargs,
                ),
            )
            if self._pad_heads and not use_sparse_decode_ckv_gather:
                assert dense_out_workspace is not None
                dense_out = dense_out_workspace[:num_actual_toks]
                dense_out.copy_(out[:, : self._input_num_heads, :])
                out = dense_out
            return out, None
        else:
            # Extend / prefill -> single-pass unified prefill (no split-K
            # scratch needed; only output_buffer is read). b12x supports 8-head
            # granularity, so only a non-aligned local tail is padded here.
            if use_ckv_gather:
                if attn_metadata.global_cache_seq_lens_per_req is None:
                    raise RuntimeError("CKV gather is missing global sequence lengths")
                cache_seqlens = attn_metadata.global_cache_seq_lens_per_req
            else:
                cache_seqlens = attn_metadata.cache_seq_lens_per_req
            prefill_q = q_all
            if self._pad_heads and not use_ckv_gather:
                prefill_q = q_buffer[:, : self._kernel_num_heads]
                prefill_q[:, self._input_num_heads :, :].zero_()

            extend_plan = self._ckv_extend_plan if use_ckv_gather else self._extend_plan
            if extend_plan is None:
                raise RuntimeError("CKV gather extend plan was not initialized")
            binding = extend_plan.bind(
                scratch=scratch_storage,
                q=prefill_q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            lse = None
            if self.need_to_return_lse_for_decode and not use_ckv_gather:
                out, lse = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_extend_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        return_lse=True,
                        lse_scale="natural",
                        **kernel_format_kwargs,
                    ),
                )
            else:
                out = cast(
                    torch.Tensor,
                    self._sparse_mla_extend_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        **kernel_format_kwargs,
                    ),
                )
            if self._pad_heads and not use_ckv_gather:
                assert dense_out_workspace is not None
                dense_out = dense_out_workspace[:num_actual_toks]
                dense_out.copy_(out[:, : self._input_num_heads, :])
                out = dense_out
                if lse is not None:
                    lse = lse[:, : self._input_num_heads]
        return out, lse
