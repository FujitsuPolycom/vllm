# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xMLASparseImpl,
    _ckv_prefetch_supports_format,
    _ckv_prefetch_target_indices,
    _sparse_decode_union_world_size,
    _validate_sparse_decode_graph_capacity,
)


def test_ckv_prefetch_supports_native_432_byte_format() -> None:
    assert _ckv_prefetch_supports_format("nvfp4_ds_mla")
    assert _ckv_prefetch_supports_format("fp8_ds_mla")
    assert not _ckv_prefetch_supports_format("auto")


def test_ckv_prefetch_targets_contiguous_future_layers() -> None:
    caches = [torch.empty(0) for _ in range(6)]

    assert _ckv_prefetch_target_indices(1, 3, caches, {}) == [2, 3, 4]


def test_ckv_prefetch_targets_skip_pending_and_stop_at_gap() -> None:
    caches = [torch.empty(0), torch.empty(0), torch.empty(0), None, torch.empty(0)]
    pending = {2: (None, 0)}  # type: ignore[dict-item]

    assert _ckv_prefetch_target_indices(1, 3, caches, pending) == []


def test_ckv_workspace_reuses_local_staging_across_ring_slots() -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._ckv_gather_enabled = True
    impl._ckv_workspace_slots = 4
    impl._ckv_local_capacity = 8
    impl._kv_record_bytes = 432
    impl.dcp_world_size = 4
    impl.device = torch.device("cpu")
    impl._ckv_workspace_nbytes = (
        1 + impl._ckv_workspace_slots * impl.dcp_world_size
    ) * impl._ckv_local_capacity * impl._kv_record_bytes
    workspace = torch.empty(impl._ckv_workspace_nbytes, dtype=torch.uint8)

    local_0, gathered_0 = impl._ckv_workspace_views(workspace, 0)
    local_3, gathered_3 = impl._ckv_workspace_views(workspace, 3)

    assert local_0.data_ptr() == local_3.data_ptr()
    assert gathered_0.shape == gathered_3.shape == (32, 432)
    assert gathered_0.data_ptr() != gathered_3.data_ptr()


def test_replicated_indexer_removes_dcp_from_sparse_union_width() -> None:
    assert _sparse_decode_union_world_size(True, 4) == 1
    assert _sparse_decode_union_world_size(True, 8) == 1
    assert _sparse_decode_union_world_size(False, 4) == 4
    assert _sparse_decode_union_world_size(False, 8) == 8


def test_sparse_decode_graph_capacity_covers_mtp_rows() -> None:
    _validate_sparse_decode_graph_capacity(True, 8, 4, 32)
    _validate_sparse_decode_graph_capacity(True, 8, 4, 64)
    _validate_sparse_decode_graph_capacity(True, 1, 4, 4)


def test_sparse_decode_graph_capacity_rejects_undersized_graph() -> None:
    try:
        _validate_sparse_decode_graph_capacity(True, 8, 4, 16)
    except ValueError as exc:
        assert "requires max cudagraph capture size >= 32" in str(exc)
    else:
        raise AssertionError("undersized graph ceiling should fail")


def test_sparse_decode_graph_capacity_accepts_eager_or_disabled_path() -> None:
    _validate_sparse_decode_graph_capacity(True, 8, 4, 0)
    _validate_sparse_decode_graph_capacity(True, 8, 4, None)
    _validate_sparse_decode_graph_capacity(False, 8, 4, 1)


if __name__ == "__main__":
    test_ckv_prefetch_supports_native_432_byte_format()
    test_ckv_prefetch_targets_contiguous_future_layers()
    test_ckv_prefetch_targets_skip_pending_and_stop_at_gap()
    test_ckv_workspace_reuses_local_staging_across_ring_slots()
    test_replicated_indexer_removes_dcp_from_sparse_union_width()
    test_sparse_decode_graph_capacity_covers_mtp_rows()
    test_sparse_decode_graph_capacity_rejects_undersized_graph()
    test_sparse_decode_graph_capacity_accepts_eager_or_disabled_path()
    print("native-432 CKV prefetch policy validation passed")
