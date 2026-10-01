# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp decode GEMM selection on Hopper and Blackwell.

Dispatch follows Kimi-K3 and uses the local ``(N, K)`` shape and token count.
Plans contain measured CUDA graph capture sizes; other token counts use the
standard linear implementation.
"""

import functools

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import LayerNameType, direct_register_custom_op

QWEN4_EXP_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=4),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=4),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=4, vector_width=4),
    },
    # QSA fused QKV/gate projection, TP=4.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # QSA indexer Q/K projection, replicated in a TP=4 deployment.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=4),
        16: SkinnyGemmConfig(16, 128, 2, k_unroll=4, vector_width=4),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 32, 4, k_unroll=2),
    },
    # HC merged down/injection projection, replicated in a TP=4 deployment.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 128, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, static_k=10240),
        4: SkinnyGemmConfig(4, 128, 2, static_k=10240),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4),
    },
}

# H200 plans selected by exhaustive CUDA graph replay measurements over
# M={1, 2, 4, 8, 16}. Only points that beat the standard linear implementation
# in both hot-cache and L2-flush measurements are retained; other token counts
# keep the standard implementation and its GEMM heuristics.
QWEN4_EXP_SM90_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=4, static_k=2560),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=2, static_k=1536),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=1536),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=6, vector_width=4),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 3, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 1, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # QSA fused QKV/gate projection, TP=4.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=5),
    },
    # QSA indexer Q/K projection, replicated in a TP=4 deployment.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 256, 2, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 2, vector_width=4, static_k=2560),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=5, vector_width=4),
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=2),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=2, static_k=2560),
    },
    # HC merged down/injection projection, replicated in a TP=4 deployment.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=5),
        2: SkinnyGemmConfig(2, 256, 3, static_k=10240),
        4: SkinnyGemmConfig(4, 256, 3, static_k=10240),
        8: SkinnyGemmConfig(8, 256, 3, static_k=10240),
    },
    # Final HC down projection, replicated in a TP=4 deployment.
    (320, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=10),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=10),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=10),
    },
}


# GB10 (SM121) plans selected by CUDA graph replay measurements with weights
# read from memory. Every shape was measured at M = 1, 2, 4 and 8, and most
# also at M = 3 and 16. A point is retained when its output error stays within
# twice the standard linear implementation's BF16 error and it is at least 5%
# faster; any other M uses the standard linear implementation. Comments name
# the measured projection and give both times in microseconds.
QWEN4_EXP_SM121_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # tp4_mtp_shared_up.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 160, 1, k_unroll=2, vector_width=8),  # 9.34 -> 6.89
        2: SkinnyGemmConfig(2, 160, 2, k_unroll=1, vector_width=8),  # 18.54 -> 6.89
        3: SkinnyGemmConfig(3, 160, 1, k_unroll=2, vector_width=8),  # 9.32 -> 6.88
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=4, vector_width=4),  # 9.31 -> 6.93
        8: SkinnyGemmConfig(8, 160, 1, k_unroll=2, vector_width=8),  # 9.4 -> 7.03
        16: SkinnyGemmConfig(16, 64, 1, k_unroll=1, vector_width=8),  # 9.47 -> 7.78
    },
    # hc_down.
    (320, 10240): {
        1: SkinnyGemmConfig(1, 256, 4, k_unroll=4, vector_width=8),  # 32.04 -> 28.38
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, vector_width=8),  # 31.17 -> 28.39
        3: SkinnyGemmConfig(3, 256, 1, k_unroll=4, vector_width=8),  # 31.25 -> 28.17
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=4, vector_width=8),  # 31.25 -> 27.99
        8: SkinnyGemmConfig(8, 256, 1, k_unroll=1, vector_width=8),  # 31.72 -> 28.33
        16: SkinnyGemmConfig(16, 128, 2, k_unroll=1, vector_width=8),  # 32.78 -> 30.14
    },
    # hc_down_inject_324.
    (324, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=8),  # 30.91 -> 28.3
        2: SkinnyGemmConfig(2, 256, 2, k_unroll=4, vector_width=8),  # 31.87 -> 28.22
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=4, vector_width=8),  # 31.84 -> 27.84
        8: SkinnyGemmConfig(8, 160, 1, k_unroll=4, vector_width=8),  # 31.85 -> 28.39
    },
    # hc_down_inject.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=8),  # 36.58 -> 29.53
        2: SkinnyGemmConfig(2, 256, 2, k_unroll=4, vector_width=8),  # 32.54 -> 29.42
        3: SkinnyGemmConfig(3, 256, 1, k_unroll=4, vector_width=8),  # 32.61 -> 29.29
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=4, vector_width=8),  # 33.07 -> 28.99
        8: SkinnyGemmConfig(8, 160, 1, k_unroll=4, vector_width=8),  # 32.7 -> 29.35
        16: SkinnyGemmConfig(16, 64, 1, k_unroll=1, vector_width=8),  # 36.39 -> 31.25
    },
    # router.
    (512, 2560): {
        1: SkinnyGemmConfig(1, 160, 1, k_unroll=2, vector_width=8),  # 12.99 -> 11.72
        2: SkinnyGemmConfig(2, 160, 1, k_unroll=4, vector_width=8),  # 14.77 -> 11.6
        3: SkinnyGemmConfig(3, 160, 1, k_unroll=4, vector_width=8),  # 14.78 -> 11.62
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=4, vector_width=8),  # 14.81 -> 11.64
        8: SkinnyGemmConfig(8, 160, 1, k_unroll=2, vector_width=8),  # 14.84 -> 11.71
        16: SkinnyGemmConfig(16, 160, 1, k_unroll=1, vector_width=8),  # 14.9 -> 12.76
    },
    # tp2_mtp_shared_up.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 160, 1, k_unroll=2, vector_width=8),  # 16.05 -> 14.65
        2: SkinnyGemmConfig(2, 160, 2, k_unroll=4, vector_width=8),  # 17.55 -> 14.62
        3: SkinnyGemmConfig(3, 160, 1, k_unroll=4, vector_width=8),  # 17.51 -> 14.56
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=4, vector_width=8),  # 17.53 -> 14.54
        8: SkinnyGemmConfig(8, 160, 1, k_unroll=2, vector_width=8),  # 17.5 -> 14.54
        16: SkinnyGemmConfig(16, 160, 1, k_unroll=1, vector_width=8),  # 17.62 -> 15.48
    },
    # tp2_mtp_shared_down.
    (2560, 320): {
        1: SkinnyGemmConfig(1, 160, 2, k_unroll=2, vector_width=2),  # 9.2 -> 7.3
        2: SkinnyGemmConfig(2, 160, 2, k_unroll=1, vector_width=2),  # 8.52 -> 7.34
        3: SkinnyGemmConfig(3, 160, 2, k_unroll=1, vector_width=2),  # 8.65 -> 7.64
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=4, vector_width=2),  # 8.65 -> 7.85
    },
    # tp4_mtp_o.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=2, vector_width=2),  # 43.58 -> 30.21
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=1, vector_width=4),  # 33.46 -> 30.45
        3: SkinnyGemmConfig(3, 64, 1, k_unroll=4, vector_width=8),  # 33.5 -> 30.4
        4: SkinnyGemmConfig(4, 64, 1, k_unroll=4, vector_width=8),  # 33.55 -> 30.43
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=2, vector_width=8),  # 33.46 -> 30.76
    },
    # mtp_fc.
    (2560, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2),  # 82.55 -> 54.13
        2: SkinnyGemmConfig(2, 160, 1, k_unroll=4, vector_width=8),  # 58.22 -> 54.14
        3: SkinnyGemmConfig(3, 160, 1, k_unroll=4, vector_width=8),  # 58.27 -> 53.59
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=4, vector_width=8),  # 58.19 -> 53.61
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),  # 58.24 -> 53.85
        16: SkinnyGemmConfig(16, 160, 2, k_unroll=2, vector_width=8),  # 59.27 -> 55.24
    },
    # tp2_mtp_o.
    (2560, 3072): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=1, vector_width=4),  # 98.02 -> 64.46
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=1, vector_width=4),  # 69.09 -> 64.48
        3: SkinnyGemmConfig(3, 256, 1, k_unroll=4, vector_width=4),  # 69.15 -> 64.34
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=8),  # 69.09 -> 64.19
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=2, vector_width=8),  # 69.12 -> 64.27
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=1, vector_width=8),  # 69.3 -> 65.16
    },
    # tp4_ple_kv.
    (3200, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2),  # 102.87 -> 66.1
        2: SkinnyGemmConfig(2, 160, 1, k_unroll=4, vector_width=8),  # 71.66 -> 66.07
        4: SkinnyGemmConfig(4, 160, 2, k_unroll=4, vector_width=8),  # 71.79 -> 66.56
        8: SkinnyGemmConfig(8, 160, 1, k_unroll=2, vector_width=8),  # 72.68 -> 65.23
    },
    # tp4_mtp_qkv.
    (3328, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2),  # 108.27 -> 69.47
        2: SkinnyGemmConfig(2, 160, 1, k_unroll=4, vector_width=8),  # 75.37 -> 69.49
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=4, vector_width=8),  # 75.38 -> 68.86
        8: SkinnyGemmConfig(8, 160, 1, k_unroll=2, vector_width=8),  # 75.57 -> 68.57
    },
    # tp4_mtp_qkv_3584.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2),  # 112.51 -> 74.67
        2: SkinnyGemmConfig(2, 160, 1, k_unroll=4, vector_width=8),  # 80.88 -> 74.62
        3: SkinnyGemmConfig(3, 160, 1, k_unroll=4, vector_width=8),  # 80.83 -> 73.98
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=4, vector_width=8),  # 80.84 -> 74.02
        8: SkinnyGemmConfig(8, 160, 1, k_unroll=2, vector_width=8),  # 80.92 -> 73.66
        16: SkinnyGemmConfig(16, 160, 2, k_unroll=2, vector_width=8),  # 84.35 -> 74.5
    },
    # tp2_ple_kv.
    (6400, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2),  # 200.53 -> 133.33
        2: SkinnyGemmConfig(2, 160, 1, k_unroll=4, vector_width=8),  # 143.13 -> 133.63
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=4, vector_width=8),  # 146.39 -> 132.6
        8: SkinnyGemmConfig(8, 160, 2, k_unroll=4, vector_width=8),  # 144.18 -> 133.07
    },
    # tp2_mtp_qkv.
    (6656, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2),  # 207.42 -> 139.34
        2: SkinnyGemmConfig(2, 160, 1, k_unroll=4, vector_width=8),  # 148.83 -> 139.6
        3: SkinnyGemmConfig(3, 160, 1, k_unroll=4, vector_width=8),  # 149.09 -> 138.35
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=4, vector_width=8),  # 155.28 -> 138.28
        8: SkinnyGemmConfig(8, 160, 1, k_unroll=2, vector_width=8),  # 153.36 -> 138.38
        16: SkinnyGemmConfig(16, 160, 1, k_unroll=1, vector_width=8),  # 154.76 -> 140.55
    },
    # hc_up.
    (10240, 320): {
        1: SkinnyGemmConfig(1, 160, 1, k_unroll=1, vector_width=2),  # 29.28 -> 26.42
        2: SkinnyGemmConfig(2, 160, 1, k_unroll=1, vector_width=2),  # 29.77 -> 26.5
        3: SkinnyGemmConfig(3, 160, 2, k_unroll=4, vector_width=2),  # 32.83 -> 27.41
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=4, vector_width=2),  # 30.01 -> 27.74
    },
    # ple_kv_12800.
    (12800, 2560): {
        1: SkinnyGemmConfig(1, 160, 1, k_unroll=4, vector_width=4),  # 386.93 -> 269.53
        2: SkinnyGemmConfig(2, 160, 1, k_unroll=4, vector_width=4),  # 288.23 -> 270.73
        3: SkinnyGemmConfig(3, 160, 2, k_unroll=4, vector_width=8),  # 284.93 -> 268.46
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=4, vector_width=8),  # 287.39 -> 267.69
        8: SkinnyGemmConfig(8, 160, 2, k_unroll=4, vector_width=8),  # 287.28 -> 269.13
        16: SkinnyGemmConfig(16, 160, 2, k_unroll=2, vector_width=8),  # 292.65 -> 267.02
    },
    # tp4_lm_head.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 160, 1, k_unroll=4, vector_width=8),  # 1851.9 -> 1322.73
        8: SkinnyGemmConfig(8, 160, 4, k_unroll=4, vector_width=8),  # 1382.93 -> 1298.92
        16: SkinnyGemmConfig(16, 160, 2, k_unroll=2, vector_width=8),  # 1400.66 -> 1291.21
    },
    # tp2_lm_head.
    (124160, 2560): {
        1: SkinnyGemmConfig(1, 160, 1, k_unroll=2, vector_width=8),  # 3745.61 -> 2587.92
        8: SkinnyGemmConfig(8, 160, 1, k_unroll=2, vector_width=8),  # 2724.8 -> 2562.12
        16: SkinnyGemmConfig(16, 160, 2, k_unroll=2, vector_width=8),  # 2746.5 -> 2550.38
    },
}


def _is_sm121() -> bool:
    return current_platform.is_device_capability((12, 1))


def applies_with_b12x() -> bool:
    """Whether the low-latency hook may run beside B12X.

    On SM121 the plans cover BF16 projections that B12X leaves to torch.
    """
    return _is_sm121()


def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))


def _is_sm90() -> bool:
    return current_platform.is_device_capability((9, 0))


def _gemm_plans() -> dict[tuple[int, int], dict[int, SkinnyGemmConfig]]:
    if _is_sm103():
        return QWEN4_EXP_GEMM_PLANS
    if _is_sm90():
        return QWEN4_EXP_SM90_GEMM_PLANS
    if _is_sm121():
        return QWEN4_EXP_SM121_GEMM_PLANS
    return {}


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


class _Qwen4ExpLowLatencyApply:
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.qwen4_exp_low_latency_gemm(x, layer.weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class Qwen4ExpLowLatencyLinearMethod(_Qwen4ExpLowLatencyApply, UnquantizedLinearMethod):
    pass


class Qwen4ExpLowLatencyEmbeddingMethod(
    _Qwen4ExpLowLatencyApply, UnquantizedEmbeddingMethod
):
    pass


def _qwen4_exp_low_latency_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    plan = _gemm_plans().get((weight.shape[0], weight.shape[1]))
    config = None if plan is None else plan.get(x.shape[0])
    if (
        config is not None
        and _runtime_ok(x, weight)
        and shape_dynamic_skinny_gemm.is_available()
    ):
        return shape_dynamic_skinny_gemm(x, weight, config)
    return torch.nn.functional.linear(x, weight)


def _qwen4_exp_low_latency_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen4_exp_low_latency_gemm",
    op_func=_qwen4_exp_low_latency_gemm,
    fake_impl=_qwen4_exp_low_latency_gemm_fake,
)


def enable_qwen4_exp_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    plans = _gemm_plans()
    if dtype != torch.bfloat16 or not plans:
        return
    if not shape_dynamic_skinny_gemm.is_available():
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        plan = plans.get((weight.shape[0], weight.shape[1]))
        if plan is None:
            continue
        if is_linear:
            child.quant_method = Qwen4ExpLowLatencyLinearMethod()
        else:
            child.quant_method = Qwen4ExpLowLatencyEmbeddingMethod()
        warmup_configs.update(plan.values())

    if warmup_configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)


# Largest row count that uses the MXFP8 hyper-connection mixing weights. The
# B12X MXFP8 kernel with BF16 activations runs 16-row M tiles; one tile beats
# the BF16 GEMM, while larger batches (from four concurrent speculative
# requests up to prefill chunks) run as fast or faster on the retained BF16
# weight.
HC_MIXING_MXFP8_MAX_ROWS = 16


def _hc_mixing_linear(
    x: torch.Tensor,
    bf16_weight: torch.Tensor,
    out_features: int,
    layer_name: LayerNameType,
) -> torch.Tensor:
    if x.shape[0] <= HC_MIXING_MXFP8_MAX_ROWS:
        return torch.ops.vllm.b12x_blockscaled_linear(
            x, None, out_features, layer_name
        )
    return torch.nn.functional.linear(x, bf16_weight)


def _hc_mixing_linear_fake(
    x: torch.Tensor,
    bf16_weight: torch.Tensor,
    out_features: int,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], out_features))


# The row-count choice happens inside an opaque op, so the compiled graph has
# no shape guard and one capture serves every batch size.
direct_register_custom_op(
    op_name="qwen4_exp_hc_mixing_linear",
    op_func=_hc_mixing_linear,
    fake_impl=_hc_mixing_linear_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


@functools.cache
def _hc_mixing_method_class() -> type:
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineLinearMethod,
    )

    class HyperConnectionMixingMxfp8Method(Mxfp8OnlineLinearMethod):
        """MXFP8 weights for decode-sized batches, BF16 for larger ones.

        The loaded BF16 weight stays on the layer as a non-persistent buffer;
        the MXFP8 copy adds about half of its bytes.
        """

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            if getattr(layer, "_already_called_process_weights_after_loading", False):
                return
            layer.register_buffer(
                "hc_mixing_bf16_weight", layer.weight.data, persistent=False
            )
            super().process_weights_after_loading(layer)

        def apply(
            self,
            layer: nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            source = x.reshape(-1, x.shape[-1]).contiguous()
            out_features = int(layer.b12x_mxfp8_packed_weight.out_features)
            output = torch.ops.vllm.qwen4_exp_hc_mixing_linear(
                source,
                layer.hc_mixing_bf16_weight,
                out_features,
                layer.b12x_layer_name,
            )
            if bias is not None:
                output = output + bias
            return output.view(*x.shape[:-1], out_features)

    return HyperConnectionMixingMxfp8Method


def quantize_hyperconnection_mixing(module: nn.Module) -> int:
    """Quantize BF16 hyper-connection down/injection projections to MXFP8.

    Enabled by VLLM_QWEN4_EXP_MXFP8_HC. These replicated projections are read
    in full on every rank at each decode step; MXFP8 halves those bytes for
    batches of at most HC_MIXING_MXFP8_MAX_ROWS rows. A projection qualifies
    when its input width is a multiple of 128, so the B12X MXFP8 kernel
    accepts BF16 activations. Weights are quantized once after loading.
    Returns the number of switched projections.
    """
    if not envs.VLLM_QWEN4_EXP_MXFP8_HC:
        return 0
    method = _hc_mixing_method_class()

    switched = 0
    for name, child in module.named_modules():
        weight = getattr(child, "weight", None)
        if (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
            and name.rsplit(".", 1)[-1]
            in ("input_mix_weight_down_block_inject", "input_mix_weight_down")
            and weight is not None
            and weight.dim() == 2
            and weight.shape[1] % 128 == 0
        ):
            child.quant_method = method(use_a16=True)
            switched += 1
    return switched
