# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X direct-copy transport for sparse MLA CKV records."""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist
from b12x.distributed._cuda_ipc import CudaRTLibrary
from b12x.distributed.pcie_dma import FLAG_SLOTS, FLAG_STRIDE, _load_extension
from b12x.distributed.pcie_oneshot import _broadcast_gather_object
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

_ALIGNMENT = 256
_DEFAULT_BARRIER_TIMEOUT_CYCLES = 8_000_000_000
_CHANNELS: dict[
    tuple[int | None, int, int, int, int, int, bool, int], B12xSparseCkvP2P
] = {}


@dataclass(frozen=True)
class _SparseSharedBuffer:
    local_ptr: int
    peer_ptrs: tuple[int, ...]
    remote_ptrs: tuple[int, ...]


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


@lru_cache(maxsize=1)
def _load_sparse_extension():
    source = Path(__file__).with_suffix(".cu")
    return load(
        name="vllm_b12x_sparse_ckv_p2p_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def build_b12x_sparse_ckv_union_remap(
    indices: torch.Tensor,
    union_indices: torch.Tensor,
    remap: torch.Tensor,
    union_count: torch.Tensor,
    hash_keys: torch.Tensor,
    hash_values: torch.Tensor,
) -> None:
    """Build a capture-safe deterministic union shared by MTP rows."""
    _load_sparse_extension().build_union_remap(
        indices,
        union_indices,
        remap,
        union_count,
        hash_keys,
        hash_values,
    )


class B12xSparseCkvP2P:
    """Single-stream, byte-exact direct-copy channel with skew overflow."""

    def __init__(
        self,
        *,
        process_group: ProcessGroup,
        device: torch.device,
        workspace_slots: int,
        topk: int,
        record_bytes: int,
        primary_capacity: int | None = None,
        allocate_direct_workspace: bool = True,
        barrier_timeout_cycles: int | None = None,
    ) -> None:
        self.group = process_group
        self.device = device
        self.rank = dist.get_rank(group=process_group)
        self.world_size = dist.get_world_size(group=process_group)
        self.workspace_slots = int(workspace_slots)
        self.topk = int(topk)
        self.record_bytes = int(record_bytes)
        self.allocate_direct_workspace = bool(allocate_direct_workspace)
        self.barrier_timeout_cycles = int(
            barrier_timeout_cycles
            if barrier_timeout_cycles is not None
            else os.getenv(
                "VLLM_B12X_SPARSE_CKV_BARRIER_TIMEOUT_CYCLES",
                str(_DEFAULT_BARRIER_TIMEOUT_CYCLES),
            )
        )
        if self.barrier_timeout_cycles <= 0:
            raise ValueError("sparse CKV barrier timeout must be positive")
        self.slot_bytes = self.topk * self.record_bytes
        if self.world_size < 2:
            raise ValueError("sparse CKV P2P needs at least two DCP ranks")
        if 2 * self.world_size > FLAG_SLOTS:
            raise ValueError("sparse CKV P2P exceeds the B12X flag-slot budget")
        balanced_capacity = (self.topk + self.world_size - 1) // self.world_size
        self.primary_capacity = int(primary_capacity or balanced_capacity)
        if not 0 < self.primary_capacity <= self.topk:
            raise ValueError("primary capacity must be in (0, topk]")
        self.overflow_capacity = self.topk - self.primary_capacity

        self.primary_positions_offset = 16
        self.primary_records_offset = _align_up(
            self.primary_positions_offset + self.primary_capacity * 4, 16
        )
        self.primary_bytes = _align_up(
            self.primary_records_offset + self.primary_capacity * self.record_bytes
        )
        self.overflow_positions_offset = 0
        self.overflow_records_offset = _align_up(max(1, self.overflow_capacity) * 4, 16)
        self.overflow_bytes = _align_up(
            self.overflow_records_offset
            + max(1, self.overflow_capacity) * self.record_bytes
        )
        self.flags_bytes = _align_up(FLAG_SLOTS * FLAG_STRIDE)
        self.primary_base_offset = self.flags_bytes
        self.overflow_base_offset = (
            self.primary_base_offset + self.world_size * self.primary_bytes
        )
        self.direct_base_offset = _align_up(
            self.overflow_base_offset + self.overflow_bytes
        )
        self.direct_bytes = (
            self.workspace_slots * self.slot_bytes
            if self.allocate_direct_workspace
            else 0
        )
        self.ce_staging_primary_base_offset = _align_up(
            self.direct_base_offset + self.direct_bytes
        )
        self.ce_staging_primary_bytes = self.world_size * self.primary_bytes
        self.ce_staging_overflow_base_offset = _align_up(
            self.ce_staging_primary_base_offset + self.ce_staging_primary_bytes
        )
        self.ce_staging_overflow_bytes = self.world_size * self.overflow_bytes

        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(device.index or 0)
        self._dma_ext = _load_extension()
        self._sparse_ext = _load_sparse_extension()
        self._closed = False
        self._validate_peer_access()
        self._slab = self._allocate_shared_buffer_rank_consistent(
            self.ce_staging_overflow_base_offset + self.ce_staging_overflow_bytes
        )
        self._send_counters = torch.zeros(
            2 * self.world_size, dtype=torch.int32, device=device
        )
        self._wait_counters = torch.zeros(
            2 * self.world_size, dtype=torch.int32, device=device
        )
        overflow_pointers = [
            pointer + self.overflow_base_offset for pointer in self._slab.peer_ptrs
        ]
        self._peer_overflow_ptrs = torch.tensor(
            overflow_pointers, dtype=torch.int64, device=device
        )
        direct_pointers = [
            [
                pointer + self.direct_base_offset + slot * self.slot_bytes
                for pointer in self._slab.peer_ptrs
            ]
            for slot in range(self.workspace_slots)
        ]
        self._peer_direct_ptrs = torch.tensor(
            direct_pointers, dtype=torch.int64, device=device
        )
        ce_overflow_pointers = [
            [
                pointer
                + self.ce_staging_overflow_base_offset
                + destination * self.overflow_bytes
                for pointer in self._slab.peer_ptrs
            ]
            for destination in range(self.world_size)
        ]
        self._peer_ce_overflow_ptrs = torch.tensor(
            ce_overflow_pointers, dtype=torch.int64, device=device
        )
        self._barrier_publish_ptrs = torch.tensor(
            [
                [
                    self._flag_ptr(destination, self.rank)
                    for destination in range(self.world_size)
                ],
                [
                    self._flag_ptr(destination, self.world_size + self.rank)
                    for destination in range(self.world_size)
                ],
            ],
            dtype=torch.int64,
            device=device,
        )
        self._barrier_wait_ptrs = torch.tensor(
            [
                [
                    self._flag_ptr(self.rank, source)
                    for source in range(self.world_size)
                ],
                [
                    self._flag_ptr(self.rank, self.world_size + source)
                    for source in range(self.world_size)
                ],
            ],
            dtype=torch.int64,
            device=device,
        )

    def _all_ranks_succeeded(self, local_success: bool) -> bool:
        status = torch.tensor(
            [1 if local_success else 0], dtype=torch.int32, device=self.device
        )
        dist.all_reduce(status, op=dist.ReduceOp.MIN, group=self.group)
        return bool(status.item())

    def _validate_peer_access(self) -> None:
        device_index = self.device.index
        if device_index is None:
            raise ValueError("sparse CKV P2P requires an explicit CUDA device")
        local_device = torch.tensor(
            [device_index], dtype=torch.int32, device=self.device
        )
        peer_devices = [torch.empty_like(local_device) for _ in range(self.world_size)]
        dist.all_gather(peer_devices, local_device, group=self.group)
        local_success = True
        for peer in peer_devices:
            peer_index = int(peer.item())
            if peer_index == device_index:
                continue
            try:
                local_success = local_success and torch.cuda.can_device_access_peer(
                    device_index, peer_index
                )
            except Exception:
                local_success = False
        if not self._all_ranks_succeeded(local_success):
            raise RuntimeError(
                "sparse CKV P2P requires full CUDA peer access across the DCP group"
            )

    def _allocate_shared_buffer_rank_consistent(
        self, size_in_bytes: int
    ) -> _SparseSharedBuffer:
        local_ptr: int | None = None
        local_handle: bytes | None = None
        local_error: Exception | None = None
        try:
            local_ptr = self._ipc.cudaMalloc(size_in_bytes)
        except Exception as exc:
            local_error = exc
        if not self._all_ranks_succeeded(local_error is None):
            if local_ptr is not None:
                with suppress(Exception):
                    self._ipc.cudaFree(local_ptr)
            raise RuntimeError(
                "sparse CKV P2P slab allocation failed on at least one DCP rank"
            ) from local_error

        assert local_ptr is not None
        try:
            self._ipc.cudaMemset(local_ptr, 0, size_in_bytes)
            local_handle = self._ipc.cudaIpcGetMemHandleBytes(local_ptr)
        except Exception as exc:
            local_error = exc
        if not self._all_ranks_succeeded(local_error is None):
            with suppress(Exception):
                self._ipc.cudaFree(local_ptr)
            raise RuntimeError(
                "sparse CKV P2P IPC-handle creation failed on at least one DCP rank"
            ) from local_error

        assert local_handle is not None
        handles = _broadcast_gather_object(local_handle, self.group)
        peer_ptrs: list[int] = []
        remote_ptrs: list[int] = []
        local_error = None
        try:
            for index, handle in enumerate(handles):
                if index == self.rank:
                    peer_ptrs.append(local_ptr)
                else:
                    remote_ptr = self._ipc.cudaIpcOpenMemHandleBytes(handle)
                    peer_ptrs.append(remote_ptr)
                    remote_ptrs.append(remote_ptr)
        except Exception as exc:
            local_error = exc
        if not self._all_ranks_succeeded(local_error is None):
            for ptr in remote_ptrs:
                with suppress(Exception):
                    self._ipc.cudaIpcCloseMemHandle(ptr)
            with suppress(Exception):
                self._ipc.cudaFree(local_ptr)
            raise RuntimeError(
                "sparse CKV P2P IPC-handle open failed on at least one DCP rank"
            ) from local_error
        if len(peer_ptrs) != self.world_size:
            raise RuntimeError("sparse CKV P2P did not open every DCP peer")
        return _SparseSharedBuffer(
            local_ptr=local_ptr,
            peer_ptrs=tuple(peer_ptrs),
            remote_ptrs=tuple(remote_ptrs),
        )

    def _flag_ptr(self, rank: int, slot: int) -> int:
        return self._slab.peer_ptrs[rank] + slot * FLAG_STRIDE

    @staticmethod
    def _counter_ptr(counters: torch.Tensor, slot: int) -> int:
        return counters.data_ptr() + slot * counters.element_size()

    def _primary_ptr(self, destination: int, source: int) -> int:
        return (
            self._slab.peer_ptrs[destination]
            + self.primary_base_offset
            + source * self.primary_bytes
        )

    def _local_primary_ptr(self) -> int:
        return (
            self._slab.local_ptr
            + self.primary_base_offset
            + self.rank * self.primary_bytes
        )

    def _local_overflow_ptr(self) -> int:
        return self._slab.local_ptr + self.overflow_base_offset

    def _local_direct_ptr(self, workspace_slot: int) -> int:
        return (
            self._slab.local_ptr
            + self.direct_base_offset
            + int(workspace_slot) * self.slot_bytes
        )

    def _local_ce_primary_ptr(self, destination: int) -> int:
        return (
            self._slab.local_ptr
            + self.ce_staging_primary_base_offset
            + int(destination) * self.primary_bytes
        )

    def _local_ce_overflow_ptr(self, destination: int) -> int:
        return (
            self._slab.local_ptr
            + self.ce_staging_overflow_base_offset
            + int(destination) * self.overflow_bytes
        )

    def build_union_remap(
        self,
        indices: torch.Tensor,
        union_indices: torch.Tensor,
        remap: torch.Tensor,
        union_count: torch.Tensor,
        hash_keys: torch.Tensor,
        hash_values: torch.Tensor,
    ) -> None:
        """Build a deterministic per-sequence union preserving row order."""
        self._sparse_ext.build_union_remap(
            indices,
            union_indices,
            remap,
            union_count,
            hash_keys,
            hash_values,
        )

    def transfer_scatter(
        self,
        kv_cache: torch.Tensor,
        local_slots: torch.Tensor,
        global_indices: torch.Tensor,
        output: torch.Tensor,
        *,
        workspace_slot: int,
        interleave: int,
    ) -> torch.Tensor:
        """Scatter owner records into peer-final slots, then copy locally."""
        if not self.allocate_direct_workspace:
            raise RuntimeError("direct sparse CKV workspace was not allocated")
        if self._closed:
            raise RuntimeError("B12X sparse CKV P2P channel is closed")
        if not 0 <= int(workspace_slot) < self.workspace_slots:
            raise ValueError("workspace_slot is outside the P2P workspace")
        if kv_cache.dtype != torch.uint8 or not kv_cache.is_contiguous():
            raise ValueError("kv_cache must be contiguous uint8")
        if local_slots.dtype != torch.int32:
            raise ValueError("local_slots must be int32")
        if local_slots.ndim != 3 or local_slots.shape[0] != self.world_size:
            raise ValueError("P2P scatter expects [destination, request, slot] maps")
        if local_slots.stride(2) != 1 or local_slots.stride(1) != local_slots.shape[2]:
            raise ValueError("P2P scatter request/slot dimensions must be packed")
        active_topk = int(local_slots.shape[1] * local_slots.shape[2])
        if not 0 < active_topk <= self.topk:
            raise ValueError("active sparse-pool width is outside channel capacity")
        if global_indices.dtype != torch.int32 or not global_indices.is_contiguous():
            raise ValueError("global_indices must be contiguous int32")
        if tuple(global_indices.shape) != (1, active_topk):
            raise ValueError("global indices must match the active sparse-pool width")
        if int(interleave) <= 0:
            raise ValueError("interleave must be positive")
        if output.dtype != torch.uint8 or not output.is_contiguous():
            raise ValueError("output must be contiguous uint8")
        active_slot_bytes = active_topk * self.record_bytes
        if output.numel() < active_slot_bytes:
            raise ValueError("output is smaller than the active sparse CKV pool")

        self._sparse_ext.scatter_records(
            kv_cache,
            local_slots,
            self._peer_direct_ptrs[int(workspace_slot)],
            self.record_bytes,
        )
        self._sparse_ext.barrier_all_peers(
            self._barrier_publish_ptrs[0],
            self._barrier_wait_ptrs[0],
            self._send_counters[: self.world_size],
            self._wait_counters[: self.world_size],
            self.barrier_timeout_cycles,
        )

        self._dma_ext.dma_copy(
            output.data_ptr(),
            self._local_direct_ptr(int(workspace_slot)),
            active_slot_bytes,
        )

        # The copy makes the caller-owned output independent from the shared
        # peer slab. A second phase prevents any rank from reusing this slot
        # before every destination has completed that copy.
        self._sparse_ext.barrier_all_peers(
            self._barrier_publish_ptrs[1],
            self._barrier_wait_ptrs[1],
            self._send_counters[self.world_size : 2 * self.world_size],
            self._wait_counters[self.world_size : 2 * self.world_size],
            self.barrier_timeout_cycles,
        )
        return output

    def transfer_scatter_ce(
        self,
        kv_cache: torch.Tensor,
        local_slots: torch.Tensor,
        global_indices: torch.Tensor,
        output: torch.Tensor,
        *,
        workspace_slot: int,
        interleave: int,
    ) -> torch.Tensor:
        """Stage owner records locally, then use copy engines for peer traffic."""
        if self._closed:
            raise RuntimeError("B12X sparse CKV P2P channel is closed")
        if not 0 <= int(workspace_slot) < self.workspace_slots:
            raise ValueError("workspace_slot is outside the P2P workspace")
        if kv_cache.dtype != torch.uint8 or not kv_cache.is_contiguous():
            raise ValueError("kv_cache must be contiguous uint8")
        if local_slots.dtype != torch.int32:
            raise ValueError("local_slots must be int32")
        if local_slots.ndim != 3 or local_slots.shape[0] != self.world_size:
            raise ValueError("P2P CE scatter expects [destination, request, slot] maps")
        if local_slots.stride(2) != 1 or local_slots.stride(1) != local_slots.shape[2]:
            raise ValueError("P2P CE scatter request/slot dimensions must be packed")
        active_topk = int(local_slots.shape[1] * local_slots.shape[2])
        if not 0 < active_topk <= self.topk:
            raise ValueError("active sparse-pool width is outside channel capacity")
        if global_indices.dtype != torch.int32 or not global_indices.is_contiguous():
            raise ValueError("global_indices must be contiguous int32")
        if tuple(global_indices.shape) != (1, active_topk):
            raise ValueError("global indices must match the active sparse-pool width")
        if int(interleave) <= 0:
            raise ValueError("interleave must be positive")
        if output.dtype != torch.uint8 or not output.is_contiguous():
            raise ValueError("output must be contiguous uint8")
        active_slot_bytes = active_topk * self.record_bytes
        if output.numel() < active_slot_bytes:
            raise ValueError("output is smaller than the active sparse CKV pool")
        scaled_primary_capacity = (
            self.primary_capacity * active_topk + self.topk - 1
        ) // self.topk
        active_primary_capacity = min(
            scaled_primary_capacity,
            (active_topk + self.world_size - 1) // self.world_size,
        )
        active_overflow_capacity = active_topk - active_primary_capacity
        if active_overflow_capacity > self.overflow_capacity:
            raise ValueError("active sparse CKV overflow exceeds channel capacity")
        active_primary_bytes = _align_up(
            self.primary_records_offset
            + active_primary_capacity * self.record_bytes
        )

        for destination in range(self.world_size):
            self._sparse_ext.pack_compact_records(
                kv_cache,
                local_slots[destination],
                self._local_ce_primary_ptr(destination),
                self._local_ce_overflow_ptr(destination),
                self.record_bytes,
                active_primary_capacity,
                self.primary_positions_offset,
                self.primary_records_offset,
                self.overflow_positions_offset,
                self.overflow_records_offset,
            )
        for destination in range(self.world_size):
            self._dma_ext.dma_copy(
                self._primary_ptr(destination, self.rank),
                self._local_ce_primary_ptr(destination),
                active_primary_bytes,
            )
        self._sparse_ext.barrier_all_peers(
            self._barrier_publish_ptrs[0],
            self._barrier_wait_ptrs[0],
            self._send_counters[: self.world_size],
            self._wait_counters[: self.world_size],
            self.barrier_timeout_cycles,
        )

        output_bytes = output.reshape(-1)[:active_slot_bytes]
        output_bytes.zero_()
        self._sparse_ext.unpack_compact_records(
            self._slab.local_ptr + self.primary_base_offset,
            self.primary_bytes,
            self._peer_ce_overflow_ptrs[self.rank],
            output_bytes,
            active_topk,
            self.record_bytes,
            active_primary_capacity,
            self.primary_positions_offset,
            self.primary_records_offset,
            self.overflow_positions_offset,
            self.overflow_records_offset,
        )
        self._sparse_ext.barrier_all_peers(
            self._barrier_publish_ptrs[1],
            self._barrier_wait_ptrs[1],
            self._send_counters[self.world_size : 2 * self.world_size],
            self._wait_counters[self.world_size : 2 * self.world_size],
            self.barrier_timeout_cycles,
        )
        return output

    def transfer(
        self,
        kv_cache: torch.Tensor,
        local_slots: torch.Tensor,
        global_indices: torch.Tensor,
        output: torch.Tensor,
        *,
        workspace_slot: int,
        interleave: int,
    ) -> torch.Tensor:
        """Broadcast compact owner records and peer-pull only owner skew."""
        if self._closed:
            raise RuntimeError("B12X sparse CKV P2P channel is closed")
        if not 0 <= int(workspace_slot) < self.workspace_slots:
            raise ValueError("workspace_slot is outside the P2P workspace")
        if kv_cache.dtype != torch.uint8 or not kv_cache.is_contiguous():
            raise ValueError("kv_cache must be contiguous uint8")
        if local_slots.dtype != torch.int32 or not local_slots.is_contiguous():
            raise ValueError("local_slots must be contiguous int32")
        if local_slots.ndim != 2 or local_slots.shape[0] != 1:
            raise ValueError("P2P transport expects one flattened sparse-pool row")
        active_topk = int(local_slots.shape[1])
        if not 0 < active_topk <= self.topk:
            raise ValueError("active sparse-pool width is outside channel capacity")
        if global_indices.dtype != torch.int32 or not global_indices.is_contiguous():
            raise ValueError("global_indices must be contiguous int32")
        if tuple(global_indices.shape) != (1, active_topk):
            raise ValueError("global indices must match the active sparse-pool width")
        if int(interleave) <= 0:
            raise ValueError("interleave must be positive")
        if output.dtype != torch.uint8 or not output.is_contiguous():
            raise ValueError("output must be contiguous uint8")
        active_slot_bytes = active_topk * self.record_bytes
        if output.numel() < active_slot_bytes:
            raise ValueError("output is smaller than the active sparse CKV pool")

        local_primary = self._local_primary_ptr()
        self._sparse_ext.pack_compact_records(
            kv_cache,
            local_slots,
            local_primary,
            self._local_overflow_ptr(),
            self.record_bytes,
            self.primary_capacity,
            self.primary_positions_offset,
            self.primary_records_offset,
            self.overflow_positions_offset,
            self.overflow_records_offset,
        )
        for destination in range(self.world_size):
            self._dma_ext.dma_copy(
                self._primary_ptr(destination, self.rank),
                local_primary,
                self.primary_bytes,
            )
            self._dma_ext.dma_set_flag(
                self._flag_ptr(destination, self.rank),
                self._counter_ptr(self._send_counters, destination),
            )
        for source in range(self.world_size):
            self._dma_ext.dma_wait_flag(
                self._flag_ptr(self.rank, source),
                self._counter_ptr(self._wait_counters, source),
            )

        output_bytes = output.reshape(-1)[:active_slot_bytes]
        output_bytes.zero_()
        self._sparse_ext.unpack_compact_records(
            self._slab.local_ptr + self.primary_base_offset,
            self.primary_bytes,
            self._peer_overflow_ptrs,
            output_bytes,
            active_topk,
            self.record_bytes,
            self.primary_capacity,
            self.primary_positions_offset,
            self.primary_records_offset,
            self.overflow_positions_offset,
            self.overflow_records_offset,
        )

        for source in range(self.world_size):
            counter_slot = self.world_size + source
            self._dma_ext.dma_set_flag(
                self._flag_ptr(source, self.world_size + self.rank),
                self._counter_ptr(self._send_counters, counter_slot),
            )
        for destination in range(self.world_size):
            counter_slot = self.world_size + destination
            self._dma_ext.dma_wait_flag(
                self._flag_ptr(self.rank, self.world_size + destination),
                self._counter_ptr(self._wait_counters, counter_slot),
            )
        return output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize(self.device)
        for ptr in self._slab.remote_ptrs:
            with suppress(Exception):
                self._ipc.cudaIpcCloseMemHandle(ptr)
        with suppress(Exception):
            self._ipc.cudaFree(self._slab.local_ptr)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def get_b12x_sparse_ckv_p2p(
    *,
    process_group: ProcessGroup,
    device: torch.device,
    workspace_slots: int,
    topk: int,
    record_bytes: int,
    primary_capacity: int | None = None,
    allocate_direct_workspace: bool = True,
    barrier_timeout_cycles: int | None = None,
) -> B12xSparseCkvP2P:
    """Return one process-wide channel shared by every sparse MLA layer."""
    balanced_capacity = (int(topk) + dist.get_world_size(process_group) - 1) // (
        dist.get_world_size(process_group)
    )
    capacity = int(primary_capacity or balanced_capacity)
    timeout_cycles = int(
        barrier_timeout_cycles
        if barrier_timeout_cycles is not None
        else os.getenv(
            "VLLM_B12X_SPARSE_CKV_BARRIER_TIMEOUT_CYCLES",
            str(_DEFAULT_BARRIER_TIMEOUT_CYCLES),
        )
    )
    key = (
        device.index,
        id(process_group),
        int(workspace_slots),
        int(topk),
        int(record_bytes),
        capacity,
        bool(allocate_direct_workspace),
        timeout_cycles,
    )
    channel = _CHANNELS.get(key)
    if channel is None:
        channel = B12xSparseCkvP2P(
            process_group=process_group,
            device=device,
            workspace_slots=workspace_slots,
            topk=topk,
            record_bytes=record_bytes,
            primary_capacity=capacity,
            allocate_direct_workspace=allocate_direct_workspace,
            barrier_timeout_cycles=timeout_cycles,
        )
        _CHANNELS[key] = channel
    return channel
