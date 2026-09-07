# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own GLM mHC token rows within an eligible GB10 eager model forward."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

_LOG = logging.getLogger(__name__)
_REPORTS = 0


def device_is_supported(properties: Any) -> bool:
    """Return whether a device matches the bounded 48-SM GB10 geometry."""
    return (
        (properties.major, properties.minor) == (12, 1)
        and properties.multi_processor_count == 48
        and "GB10" in properties.name
    )


def configure(model: Any, config: Any, enabled: bool) -> None:
    """Agree on device/configuration admission before any graph is captured.

    Args:
        model: GLM base model whose eager prefills may own token rows.
        config: Serving configuration shared by all TP ranks.
        enabled: Explicit opt-in from VLLM_GLM53_MHC_PREFILL_SHARD.

    Raises:
        RuntimeError: TP ranks disagree or an enabled device/config is unsupported.
    """
    import torch

    from vllm.distributed import get_tp_group

    group = get_tp_group()
    error = None
    if enabled:
        try:
            captures = config.compilation_config.cudagraph_capture_sizes or []
            if (
                group.world_size != 4
                or config.scheduler_config.max_num_batched_tokens != 8192
                or any(size >= 8192 for size in captures)
            ):
                raise RuntimeError(
                    "GLM mHC row ownership requires TP4, an 8192-token ceiling, "
                    "and graph capture sizes below 8192"
                )
            parallel = config.parallel_config
            if (
                parallel.tensor_parallel_size != 4
                or parallel.decode_context_parallel_size != 4
                or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.enable_expert_parallel
                or parallel.enable_eplb
                or parallel.use_sequence_parallel_moe
                or config.model_config.dtype != torch.bfloat16
                or model.config.hidden_size != 4096
            ):
                raise RuntimeError(
                    "GLM mHC row ownership requires BF16 H4096 and "
                    "TP4/DCP4/PP1/DP1/PCP1 without expert or sequence parallelism"
                )
            properties = torch.cuda.get_device_properties(
                torch.accelerator.current_device_index()
            )
            if not device_is_supported(properties):
                raise RuntimeError("GLM mHC row ownership requires a 48-SM GB10 GPU")
        except (AttributeError, RuntimeError, AssertionError) as exc:
            error = str(exc)
    votes: list[Any] = [None] * group.world_size
    torch.distributed.all_gather_object(votes, (enabled, error), group=group.cpu_group)
    if any(vote[0] != enabled for vote in votes):
        raise RuntimeError("All TP ranks must agree on VLLM_GLM53_MHC_PREFILL_SHARD")
    errors = [vote[1] for vote in votes if vote[1] is not None]
    if errors:
        raise RuntimeError("GLM mHC configuration admission failed: " + repr(errors))
    model._mhc_prefill_enabled = enabled
    model._mhc_prefill_parallel_config = config.parallel_config


def pure_prefill_metadata(metadata: Any, names: tuple[str, ...], rows: int) -> bool:
    """Use host counts only; missing or ambiguous metadata cannot opt in."""
    if not isinstance(metadata, dict) or not names or rows != 8192:
        return False
    fields = (
        "num_decodes",
        "num_decode_tokens",
        "num_spec_decodes",
        "num_spec_decode_tokens",
    )
    for name in names:
        item = metadata.get(name)
        if item is None:
            return False
        if (
            type(getattr(item, "num_prefills", None)) is not int
            or item.num_prefills <= 0
        ):
            return False
        if (
            type(getattr(item, "num_prefill_tokens", None)) is not int
            or item.num_prefill_tokens != rows
        ):
            return False
        if any(
            type(getattr(item, field, None)) is not int or getattr(item, field) != 0
            for field in fields
        ):
            return False
    return True


def validate_moe_deferral(runner: Any) -> None:
    config = runner.moe_config
    parallel = config.moe_parallel_config
    if (
        config.tp_size != 4
        or config.dp_size != 1
        or config.ep_size != 1
        or config.pcp_size != 1
        or config.is_sequence_parallel
        or config.skip_final_all_reduce
        or parallel.use_all2all_kernels
        or runner._fused_output_is_reduced
        or runner.routed_output_transform is not None
        or runner.routed_input_transform is not None
        or type(runner.router).__name__ == "ZeroExpertRouter"
    ):
        raise RuntimeError(
            "mHC prefill requires one unreduced conventional TP4 MoE output"
        )


def validate_model(model: Any) -> tuple[str, ...]:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper

    from .attention import Glm5NextMLAAttention
    from .kda import Glm5NextLinearAttention

    config = model._mhc_prefill_parallel_config
    if (
        config.tensor_parallel_size != 4
        or config.decode_context_parallel_size != 4
        or config.pipeline_parallel_size != 1
        or config.data_parallel_size != 1
        or config.prefill_context_parallel_size != 1
        or config.enable_expert_parallel
        or config.enable_eplb
        or model.is_sequence_parallel
    ):
        raise RuntimeError("mHC prefill requires TP4/DCP4/PP1/DP1/PCP1 without EP/EPLB")
    names = []
    for layer in model._active_layers:
        if not layer.mhc or layer.is_mtp_layer or layer._b12x_mhc is None:
            raise RuntimeError("mHC prefill requires B12X mHC on every base layer")
        attn = layer.self_attn
        if type(attn) is Glm5NextLinearAttention:
            names.append(attn.prefix)
        elif type(attn) is Glm5NextMLAAttention:
            if type(attn.mla_attn) is not MultiHeadLatentAttentionWrapper:
                raise RuntimeError(
                    "Unsupported outer MLA wrapper for mHC row ownership"
                )
        else:
            raise RuntimeError(
                "Unsupported GLM attention implementation for mHC row ownership"
            )
        projection = attn.o_proj
        if (
            type(projection) is not RowParallelLinear
            or projection.tp_size != 4
            or not projection.reduce_results
            or projection.bias is not None
        ):
            raise RuntimeError(
                "Unsupported attention projection reduction for mHC row ownership"
            )
        if layer._mlp_is_moe:
            if type(layer.mlp.experts) is not MoERunner:
                raise RuntimeError("Unsupported MoE runner for mHC row ownership")
            validate_moe_deferral(layer.mlp.experts)
        elif (
            type(layer.mlp.down_proj) is not RowParallelLinear
            or layer.mlp.down_proj.tp_size != 4
            or not layer.mlp.down_proj.reduce_results
            or layer.mlp.down_proj.bias is not None
        ):
            raise RuntimeError("Unsupported dense FFN reduction for mHC row ownership")
    if not names:
        raise RuntimeError("mHC prefill requires explicit GDN metadata owners")
    return tuple(names)


@dataclass
class PrefillOwnership:
    """Hold per-forward row ownership and return caller-owned collective outputs.

    Each collective allocates its own output. Final and auxiliary tensors escape
    the model forward; shared reusable buffers would need explicit consumer
    lifetime ownership. Stream records protect allocator reuse after release.
    """

    comm: Any
    rank: int
    rows: int = 8192
    rs_count: int = 0
    ag_count: int = 0

    def local_view(self, tensor: Any) -> Any:
        if tensor.shape[0] != self.rows:
            raise RuntimeError("mHC full-to-owner row count mismatch")
        q = self.rows // 4
        return tensor.narrow(0, self.rank * q, q)

    def _check(self, tensor: Any, expected_rows: int) -> None:
        import torch

        if not self.comm.available or self.comm.disabled:
            raise RuntimeError("mHC PyNccl communicator became unavailable")
        if (
            tensor.shape[0] != expected_rows
            or not tensor.is_cuda
            or tensor.device != self.comm.device
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                "mHC collective requires contiguous BF16 owner/full rows"
            )

    def reduce_scatter(self, partial: Any) -> Any:
        from vllm.utils.torch_utils import current_stream

        self._check(partial, self.rows)
        if tuple(partial.shape) != (self.rows, 4096):
            raise RuntimeError("mHC reduce-scatter expects a full hidden TP partial")
        output = partial.new_empty((self.rows // 4, 4096))
        stream = current_stream()
        self.comm.reduce_scatter(output, partial, stream=stream)
        partial.record_stream(stream)
        output.record_stream(stream)
        self.rs_count += 1
        return output

    def all_gather(self, owned: Any) -> Any:
        from vllm.utils.torch_utils import current_stream

        self._check(owned, self.rows // 4)
        output = owned.new_empty((self.rows, *owned.shape[1:]))
        stream = current_stream()
        self.comm.all_gather(output, owned, stream=stream)
        owned.record_stream(stream)
        output.record_stream(stream)
        self.ag_count += 1
        return output

    def finish(self, layers: int, auxiliary_gathers: int) -> None:
        global _REPORTS
        if (
            self.rs_count != 2 * layers
            or self.ag_count != 2 * layers + auxiliary_gathers
        ):
            raise RuntimeError(
                "mHC collective accounting mismatch: "
                f"RS={self.rs_count} AG={self.ag_count}"
            )
        if _REPORTS < 8:
            _LOG.info(
                "GLM_MHC_PREFILL rank=%d rows=%d owner_rows=%d rs=%d ag=%d aux=%d",
                self.rank,
                self.rows,
                self.rows // 4,
                self.rs_count,
                self.ag_count,
                auxiliary_gathers,
            )
            _REPORTS += 1


def maybe_create(model: Any, hidden: Any, positions: Any) -> PrefillOwnership | None:
    if not getattr(model, "_mhc_prefill_enabled", False) or tuple(hidden.shape) != (
        8192,
        4096,
    ):
        return None
    import torch

    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None
    from vllm.distributed import get_tp_group
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    context = get_forward_context()
    if (
        context.cudagraph_runtime_mode.name != "NONE"
        or context.ubatch_slices is not None
    ):
        return None
    group = get_tp_group()
    if group.world_size != 4:
        raise RuntimeError("mHC prefill TP group is not four ranks")
    comm = getattr(group.device_communicator, "pynccl_comm", None)
    error = None
    try:
        if comm is None or not comm.available or comm.disabled or comm.world_size != 4:
            raise RuntimeError(
                "mHC prefill requires the enabled TP PyNccl communicator"
            )
        if comm.rank != group.rank_in_group or comm.device != hidden.device:
            raise RuntimeError("mHC communicator rank/device ownership mismatch")
        names = validate_model(model)
    except (AttributeError, RuntimeError) as exc:
        names = ()
        error = str(exc)
    eligible = (
        error is None
        and positions.shape[0] == 8192
        and hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and pure_prefill_metadata(context.attn_metadata, names, 8192)
    )
    # Small host vote before changing ownership; no GPU metadata synchronization.
    # Rank-local capability or metadata differences cannot silently choose
    # incompatible reduction paths after one rank has produced partial output.
    votes: list[Any] = [None] * 4
    torch.distributed.all_gather_object(votes, (eligible, error), group=group.cpu_group)
    errors = [item[1] for item in votes if item[1] is not None]
    if errors:
        raise RuntimeError("mHC prefill capability vote failed: " + repr(errors))
    if not all(item[0] for item in votes):
        return None
    return PrefillOwnership(comm=comm, rank=group.rank_in_group)
