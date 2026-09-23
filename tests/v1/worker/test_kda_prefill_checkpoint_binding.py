# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA checkpoint binding and convolution-history wrapper contracts."""

from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.cpu_test


def test_disabled_coalescing_uses_one_checkpoint_caps_without_added_keyword(
    monkeypatch,
):
    from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as module

    class OneCheckpointCaps:
        def __init__(
            self,
            *,
            device,
            max_tokens,
            max_seqs,
            max_state_slots,
            heads,
            head_dim,
            model_dtype,
            state_dtype,
            qk_l2norm,
            checkpoint_export,
            null_state_index,
            metadata_validation,
        ):
            self.metadata_validation = metadata_validation
            self.max_tokens = max_tokens

    api = NS(Caps=OneCheckpointCaps, plan=lambda caps: caps)
    monkeypatch.setattr(module, "current_platform", NS(current_device=lambda: "cpu"))
    layer = NS(
        _b12x_prefill_api=api,
        _b12x_prefill_checkpoint_capacity=1,
        _b12x_prefill_max_tokens=8192,
        _b12x_prefill_max_seqs=16,
        local_num_heads=16,
        head_dim=128,
        model_config=NS(dtype=torch.bfloat16),
        get_state_dtype=lambda: (torch.float32, torch.float32),
    )
    plan = module.KimiGatedDeltaNetAttention._make_b12x_kda_prefill_plan(layer, 8)
    assert isinstance(plan, OneCheckpointCaps)
    assert plan.metadata_validation == "trusted"


def test_tp2_four_checkpoint_plan_uses_32_local_heads(monkeypatch):
    from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as module

    observed = {}

    class FourCheckpointCaps:
        __dataclass_fields__ = {"max_checkpoints": object()}

        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(module, "current_platform", NS(current_device=lambda: "cuda:0"))
    layer = NS(
        _b12x_prefill_api=NS(
            Caps=FourCheckpointCaps,
            plan=lambda caps: caps,
        ),
        _b12x_prefill_checkpoint_capacity=4,
        _b12x_prefill_max_tokens=8192,
        _b12x_prefill_max_seqs=8,
        local_num_heads=32,
        head_dim=128,
        model_config=NS(dtype=torch.bfloat16),
        get_state_dtype=lambda: (torch.float32, torch.float32),
    )
    module.KimiGatedDeltaNetAttention._make_b12x_kda_prefill_plan(layer, 128)
    assert observed["heads"] == 32
    assert observed["head_dim"] == 128
    assert observed["max_checkpoints"] == 4
    assert observed["checkpoint_export"] is True
    assert observed["metadata_validation"] == "transactional"


def test_one_checkpoint_binding_retains_vector_metadata_without_mhc_state():
    from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as module

    observed = {}

    def bind(plan, **kwargs):
        observed.update(kwargs)
        return NS(error_code=torch.zeros(1, dtype=torch.int32))

    layer = NS(
        _b12x_prefill_api=NS(bind=bind, run=lambda *args, **kwargs: None),
        _b12x_prefill_plan=object(),
        _b12x_prefill_checkpoint_capacity=1,
        _b12x_prefill_max_tokens=32,
        _b12x_prefill_max_seqs=2,
        _b12x_prefill_initial_indices=torch.zeros(2, dtype=torch.int32),
        _b12x_prefill_null_indices=torch.zeros(2, dtype=torch.int32),
        _b12x_prefill_zero_offsets=torch.zeros(2, dtype=torch.int32),
        _b12x_prefill_num_seqs=torch.zeros(1, dtype=torch.int32),
        _b12x_prefill_num_tokens=torch.zeros(1, dtype=torch.int32),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1, 128),
        head_dim=128,
        gate_lower_bound=-5.0,
    )
    rows = torch.empty(16, 1, 128, dtype=torch.bfloat16)
    result = module.KimiGatedDeltaNetAttention._run_b12x_kda_prefill(
        layer,
        scratch=torch.empty(1024, dtype=torch.uint8),
        q=rows,
        k=rows,
        v=rows,
        raw_g=rows,
        raw_beta=torch.zeros(16, 1, dtype=torch.bfloat16),
        cu_seqlens=torch.tensor([0, 16], dtype=torch.int32),
        state_indices=torch.tensor([1], dtype=torch.int32),
        has_initial_state=torch.tensor([True]),
        checkpoint=None,
        recurrent_state=torch.empty(4, 1, 128, 128),
        output=torch.empty_like(rows),
    )
    error, bound = result
    assert error is None
    assert bound is None
    assert observed["checkpoint_state_indices"].shape == (1,)
    assert observed["checkpoint_offsets"].shape == (1,)
    assert observed["initial_state_indices"].tolist() == [1]


def test_four_checkpoint_binding_normalizes_cache_hit_fallback_and_clears_rows():
    from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as module

    observed = []

    def bind(plan, **kwargs):
        observed.append(
            (
                kwargs["checkpoint_state_indices"].clone(),
                kwargs["checkpoint_offsets"].clone(),
            )
        )
        return NS(error_code=torch.zeros(1, dtype=torch.int32))

    layer = NS(
        _b12x_prefill_api=NS(bind=bind, run=lambda *args, **kwargs: None),
        _b12x_prefill_plan=object(),
        _b12x_prefill_checkpoint_capacity=4,
        _b12x_prefill_max_tokens=32,
        _b12x_prefill_max_seqs=3,
        _b12x_prefill_initial_indices=torch.zeros(3, dtype=torch.int32),
        _b12x_prefill_null_indices=torch.full((3, 4), 99, dtype=torch.int32),
        _b12x_prefill_zero_offsets=torch.full((3, 4), 99, dtype=torch.int32),
        _b12x_prefill_num_seqs=torch.zeros(1, dtype=torch.int32),
        _b12x_prefill_num_tokens=torch.zeros(1, dtype=torch.int32),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1, 128),
        head_dim=128,
        gate_lower_bound=-5.0,
    )

    def run(requests, checkpoint):
        rows = torch.empty(requests * 4, 1, 128, dtype=torch.bfloat16)
        module.KimiGatedDeltaNetAttention._run_b12x_kda_prefill(
            layer,
            scratch=torch.empty(1024, dtype=torch.uint8),
            q=rows,
            k=rows,
            v=rows,
            raw_g=rows,
            raw_beta=torch.zeros(requests * 4, 1, dtype=torch.bfloat16),
            cu_seqlens=torch.arange(0, requests * 4 + 1, 4, dtype=torch.int32),
            state_indices=torch.arange(1, requests + 1, dtype=torch.int32),
            has_initial_state=torch.ones(requests, dtype=torch.bool),
            checkpoint=checkpoint,
            recurrent_state=torch.empty(8, 1, 128, 128),
            output=torch.empty_like(rows),
        )

    run(
        2,
        NS(
            checkpoint_offsets=torch.tensor([515, 515], dtype=torch.int32),
            state_indices=torch.tensor([4, 5], dtype=torch.int32),
        ),
    )
    run(1, None)
    run(
        2,
        NS(
            checkpoint_offsets=torch.tensor(
                [[16, 32, 48, 64], [8, 24, 40, 56]], dtype=torch.int32
            ),
            state_indices=torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=torch.int32),
        ),
    )

    assert observed[0][0].tolist() == [[4, 0, 0, 0], [5, 0, 0, 0]]
    assert observed[0][1].tolist() == [[515, 0, 0, 0], [515, 0, 0, 0]]
    assert observed[1][0].tolist() == [[0, 0, 0, 0]]
    assert observed[1][1].tolist() == [[0, 0, 0, 0]]
    assert observed[2][0].tolist() == [[1, 2, 3, 4], [4, 5, 6, 7]]
    assert observed[2][1].tolist() == [[16, 32, 48, 64], [8, 24, 40, 56]]


def test_four_checkpoint_binding_expands_irregular_packed_csr_for_conv_history():
    from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as module

    observed = {}

    def bind(plan, **kwargs):
        observed.update(kwargs)
        return NS(error_code=torch.zeros(1, dtype=torch.int32))

    layer = NS(
        _b12x_prefill_api=NS(bind=bind, run=lambda *args, **kwargs: None),
        _b12x_prefill_plan=object(),
        _b12x_prefill_checkpoint_capacity=4,
        _b12x_prefill_max_tokens=32,
        _b12x_prefill_max_seqs=3,
        _b12x_prefill_initial_indices=torch.zeros(3, dtype=torch.int32),
        _b12x_prefill_null_indices=torch.full((3, 4), 99, dtype=torch.int32),
        _b12x_prefill_zero_offsets=torch.full((3, 4), 99, dtype=torch.int32),
        _b12x_prefill_num_seqs=torch.zeros(1, dtype=torch.int32),
        _b12x_prefill_num_tokens=torch.zeros(1, dtype=torch.int32),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1, 128),
        head_dim=128,
        gate_lower_bound=-5.0,
    )
    rows = torch.empty(12, 1, 128, dtype=torch.bfloat16)
    error, bound = module.KimiGatedDeltaNetAttention._run_b12x_kda_prefill(
        layer,
        scratch=torch.empty(1024, dtype=torch.uint8),
        q=rows,
        k=rows,
        v=rows,
        raw_g=rows,
        raw_beta=torch.zeros(12, 1, dtype=torch.bfloat16),
        cu_seqlens=torch.tensor([0, 4, 8, 12], dtype=torch.int32),
        state_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
        has_initial_state=torch.ones(3, dtype=torch.bool),
        checkpoint=NS(
            checkpoint_offsets=torch.tensor([512, 1024, 512], dtype=torch.int32),
            state_indices=torch.tensor([10, 11, 12], dtype=torch.int32),
            checkpoint_indptr=torch.tensor([0, 2, 2, 3], dtype=torch.int32),
            checkpoint_query_indices=torch.tensor([0, 0, 2], dtype=torch.int64),
        ),
        recurrent_state=torch.empty(16, 1, 128, 128),
        output=torch.empty_like(rows),
    )

    assert error is not None
    assert observed["checkpoint_state_indices"].tolist() == [
        [10, 11, 0, 0],
        [0, 0, 0, 0],
        [12, 0, 0, 0],
    ]
    assert observed["checkpoint_offsets"].tolist() == [
        [512, 1024, 0, 0],
        [0, 0, 0, 0],
        [512, 0, 0, 0],
    ]
    assert bound.checkpoint_indptr is None
    assert bound.checkpoint_query_indices is None
    assert bound.state_indices.shape == (3, 4)

    kernel = {}

    class CaptureKernel:
        def __getitem__(self, grid):
            kernel["grid"] = grid

            def launch(*args):
                kernel["args"] = args

            return launch

    original_kernel = module._store_cache_checkpoints_kernel
    module._store_cache_checkpoints_kernel = CaptureKernel()
    layer.conv1d = NS(weight=torch.empty(12, 1, 4))
    try:
        module.KimiGatedDeltaNetAttention._store_kda_conv_checkpoint(
            layer,
            mixed_qkv=torch.empty(12, 12),
            conv_state=torch.empty(16, 12, 6),
            recurrent_state=torch.empty(16, 1, 128, 128),
            query_start_loc=torch.tensor([0, 4, 8, 12], dtype=torch.int32),
            checkpoint=bound,
            error_code=error,
        )
    finally:
        module._store_cache_checkpoints_kernel = original_kernel
    assert kernel["grid"][0] == 12
    assert kernel["args"][22] == 4
    assert kernel["args"][21] is None


@pytest.mark.parametrize(
    "offsets,indices",
    [
        (torch.tensor([16]), torch.tensor([1])),
        (torch.tensor([[16, 32]]), torch.tensor([[1, 2]])),
        (torch.tensor([16, 32]), torch.tensor([1])),
    ],
)
def test_four_checkpoint_binding_rejects_ambiguous_fallback_shapes(offsets, indices):
    from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as module

    layer = NS(
        _b12x_prefill_api=NS(
            bind=lambda *args, **kwargs: NS(error_code=torch.zeros(1)),
            run=lambda *args, **kwargs: None,
        ),
        _b12x_prefill_plan=object(),
        _b12x_prefill_checkpoint_capacity=4,
        _b12x_prefill_max_tokens=16,
        _b12x_prefill_max_seqs=2,
        _b12x_prefill_initial_indices=torch.zeros(2, dtype=torch.int32),
        _b12x_prefill_null_indices=torch.zeros(2, 4, dtype=torch.int32),
        _b12x_prefill_zero_offsets=torch.zeros(2, 4, dtype=torch.int32),
        _b12x_prefill_num_seqs=torch.zeros(1, dtype=torch.int32),
        _b12x_prefill_num_tokens=torch.zeros(1, dtype=torch.int32),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1, 128),
        head_dim=128,
        gate_lower_bound=-5.0,
    )
    rows = torch.empty(8, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Checkpoint|fallback"):
        module.KimiGatedDeltaNetAttention._run_b12x_kda_prefill(
            layer,
            scratch=torch.empty(1024, dtype=torch.uint8),
            q=rows,
            k=rows,
            v=rows,
            raw_g=rows,
            raw_beta=torch.zeros(8, 1, dtype=torch.bfloat16),
            cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
            state_indices=torch.tensor([1, 2], dtype=torch.int32),
            has_initial_state=torch.ones(2, dtype=torch.bool),
            checkpoint=NS(checkpoint_offsets=offsets, state_indices=indices),
            recurrent_state=torch.empty(8, 1, 128, 128),
            output=torch.empty_like(rows),
        )


def test_checkpoint_convolution_stores_causal_history_not_speculative_capacity(
    monkeypatch,
):
    from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as module

    observed = {}

    class CaptureKernel:
        def __getitem__(self, grid):
            observed["grid"] = grid

            def launch(*args):
                observed["args"] = args

            return launch

    monkeypatch.setattr(module, "_store_cache_checkpoints_kernel", CaptureKernel())
    layer = NS(conv1d=NS(weight=torch.empty(6, 1, 4)))
    checkpoint = NS(
        checkpoint_offsets=torch.tensor([[16, 32]], dtype=torch.int32),
        state_indices=torch.tensor([[2, 3]], dtype=torch.int32),
    )
    error = torch.zeros(1, dtype=torch.int32)
    module.KimiGatedDeltaNetAttention._store_kda_conv_checkpoint(
        layer,
        mixed_qkv=torch.empty(32, 6),
        conv_state=torch.empty(8, 6, 6),
        recurrent_state=torch.empty(8, 1, 128, 128),
        query_start_loc=torch.tensor([0, 32], dtype=torch.int32),
        checkpoint=checkpoint,
        error_code=error,
    )
    assert observed["grid"][0] == 2
    assert observed["args"][15] == 3
    assert observed["args"][21] is None
    assert observed["args"][22] == 2
    assert observed["args"][23] is error and observed["args"][24] is True


@pytest.mark.parametrize("capacity", [1, 2])
def test_prefill_warmup_binds_the_planned_checkpoint_capacity(monkeypatch, capacity):
    from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as module

    shape = (2, capacity) if capacity == 2 else (2,)
    observed = {}

    def bind(plan, **kwargs):
        observed.update(kwargs)
        return object()

    caps = NS(
        device=torch.device("cpu"),
        heads=1,
        head_dim=128,
        chunk_tokens=16,
        model_dtype=torch.bfloat16,
        max_state_slots=4,
    )
    layer = NS(
        _b12x_prefill_plan=NS(caps=caps),
        _b12x_prefill_api=NS(bind=bind, prewarm=lambda binding: None),
        _b12x_prefill_null_indices=torch.zeros(shape, dtype=torch.int32),
        _b12x_prefill_zero_offsets=torch.zeros(shape, dtype=torch.int32),
        _b12x_prefill_num_seqs=torch.zeros(1, dtype=torch.int32),
        _b12x_prefill_num_tokens=torch.zeros(1, dtype=torch.int32),
        _b12x_prefill_max_tokens=8192,
        _b12x_prefill_max_seqs=2,
        _b12x_prefill_checkpoint_capacity=capacity,
        local_num_heads=1,
        head_dim=128,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1, 128),
        kv_cache=(torch.empty(4, 384, 6), torch.empty(4, 1, 128, 128)),
    )
    monkeypatch.setattr(
        module,
        "get_b12x_scratch_buffers",
        lambda plan: [torch.empty(4096, dtype=torch.uint8)],
    )
    unit = module._B12xKdaPrefillWarmup().get_b12x_warmup_unit(
        layer, (16,), torch.bfloat16
    )
    unit.compile()
    expected = (1, 2) if capacity == 2 else (1,)
    assert observed["checkpoint_state_indices"].shape == expected
    assert observed["checkpoint_offsets"].shape == expected
