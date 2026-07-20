# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.models.deepseek_v2 as deepseek_v2
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV32IndexerCache,
    _replicate_indexer_cache_under_dcp,
)


def _vllm_config(
    num_hidden_layers: int = 78,
    dcp_size: int = 4,
    pcp_size: int = 1,
):
    return SimpleNamespace(
        use_v2_model_runner=True,
        cache_config=SimpleNamespace(block_size=256),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=num_hidden_layers)
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )


def _indexer_cache(layer_id: int = 78):
    cache = object.__new__(DeepseekV32IndexerCache)
    cache.head_dim = 512
    cache.dtype = torch.bfloat16
    cache.prefix = f"model.layers.{layer_id}.self_attn.indexer"
    cache.cache_config = SimpleNamespace(block_size=256)
    return cache


def test_dcp_shard_draft_defaults_to_sharded(monkeypatch):
    monkeypatch.delenv("VLLM_DCP_SHARD_DRAFT", raising=False)

    config = _vllm_config()
    cache = _indexer_cache()
    cache.dcp_replicated = _replicate_indexer_cache_under_dcp(cache.prefix, config)
    spec = cache.get_kv_cache_spec(config)

    assert spec.dcp_replicated is False


def test_dcp_shard_draft_can_restore_replicated_legacy_mode(monkeypatch):
    monkeypatch.setenv("VLLM_DCP_SHARD_DRAFT", "0")

    config = _vllm_config()
    cache = _indexer_cache()
    cache.dcp_replicated = _replicate_indexer_cache_under_dcp(cache.prefix, config)
    spec = cache.get_kv_cache_spec(config)

    assert spec.dcp_replicated is True
    assert spec.block_size == cache.cache_config.block_size


@pytest.mark.parametrize("dcp_size", [2, 4, 6, 8])
def test_target_indexer_replication_equalizes_global_block_coverage(
    monkeypatch, dcp_size: int
):
    monkeypatch.setenv("VLLM_DCP_REPLICATE_INDEXER_CACHE", "1")
    monkeypatch.setattr(deepseek_v2, "use_b12x_sparse_indexer", lambda: True)

    config = _vllm_config(dcp_size=dcp_size)
    cache = _indexer_cache(layer_id=12)
    cache.dcp_replicated = _replicate_indexer_cache_under_dcp(cache.prefix, config)
    spec = cache.get_kv_cache_spec(config)

    assert spec.dcp_replicated is True
    assert spec.block_size == dcp_size * cache.cache_config.block_size


def test_target_indexer_replication_rejects_pcp(monkeypatch):
    monkeypatch.setenv("VLLM_DCP_REPLICATE_INDEXER_CACHE", "1")
    monkeypatch.setattr(deepseek_v2, "use_b12x_sparse_indexer", lambda: True)

    config = _vllm_config(dcp_size=4, pcp_size=2)
    with pytest.raises(NotImplementedError, match="DCP2 through DCP8 with PCP1"):
        _replicate_indexer_cache_under_dcp(
            "model.layers.12.indexer.k_cache", config
        )


def test_target_indexer_replication_requires_v2_model_runner(monkeypatch):
    monkeypatch.setenv("VLLM_DCP_REPLICATE_INDEXER_CACHE", "1")

    config = _vllm_config()
    config.use_v2_model_runner = False
    with pytest.raises(NotImplementedError, match="V2 model runner"):
        _replicate_indexer_cache_under_dcp("model.layers.12.indexer.k_cache", config)
