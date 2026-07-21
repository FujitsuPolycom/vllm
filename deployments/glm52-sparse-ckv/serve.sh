#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-MXFP8-NVFP4-NF3-Hybrid}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5802}"
TP="${TP:-4}"
DCP="${DCP:-4}"
DCP_BACKEND="${DCP_BACKEND:-a2a}"
MTP="${MTP:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-300000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
GRAPH="${GRAPH:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-3426112942}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
QUANTIZATION="${QUANTIZATION:-nvfp4_nf3_hybrid}"
ALLREDUCE_MODE="${ALLREDUCE_MODE:-nccl}"
GLM52_INDEX_TOPK_PATTERN="${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
QUANTIZATION_CONFIG_JSON="${QUANTIZATION_CONFIG_JSON:-}"
if [[ -z "${QUANTIZATION_CONFIG_JSON}" ]]; then
  QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"},"ignore":["re:^model\\.layers\\.0\\.","re:.*\\.self_attn\\.indexer\\.","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}'
fi

[[ "${TP}" =~ ^[0-9]+$ ]] || { echo "TP must be an integer" >&2; exit 2; }
[[ "${DCP}" =~ ^[0-9]+$ ]] || { echo "DCP must be an integer" >&2; exit 2; }
[[ "${MTP}" =~ ^[0-9]+$ ]] || { echo "MTP must be an integer" >&2; exit 2; }
[[ "${MAX_NUM_SEQS}" =~ ^[0-9]+$ ]] || { echo "MAX_NUM_SEQS must be an integer" >&2; exit 2; }
[[ "${GRAPH}" =~ ^[0-9]+$ ]] || { echo "GRAPH must be an integer" >&2; exit 2; }
[[ "${#GLM52_INDEX_TOPK_PATTERN}" -eq 78 ]] || { echo "GLM52_INDEX_TOPK_PATTERN must contain 78 layer entries" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export VLLM_USE_B12X_PCIE_DMA="${B12X_PCIE_DMA:-0}"
export VLLM_PCIE_DMA_FP8="${F8_DMA:-0}"
export B12X_PCIE_DMA_FP8="${F8_DMA:-0}"

allreduce_args=()
case "${ALLREDUCE_MODE}" in
  nccl)
    export VLLM_ENABLE_PCIE_ALLREDUCE=0
    export VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE=0
    export VLLM_ALLREDUCE_USE_SYMM_MEM=0
    allreduce_args=(--disable-custom-all-reduce)
    ;;
  b12x)
    export VLLM_ENABLE_PCIE_ALLREDUCE=1
    export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
    export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
    export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB
    ;;
  *)
    echo "ALLREDUCE_MODE must be nccl or b12x" >&2
    exit 2
    ;;
esac

dcp_args=(--decode-context-parallel-size "${DCP}")
if [[ "${DCP}" != "1" ]]; then
  dcp_args+=(
    --dcp-comm-backend "${DCP_BACKEND}"
    --dcp-kv-cache-interleave-size 1
  )
fi

spec_args=()
if [[ "${MTP}" != "0" ]]; then
  spec_json="$(printf '{\"model\":\"%s\",\"method\":\"mtp\",\"num_speculative_tokens\":%s,\"moe_backend\":\"%s\",\"draft_sample_method\":\"probabilistic\"}' "${MODEL}" "${MTP}" "${MOE_BACKEND}")"
  spec_args=(--speculative-config "${spec_json}")
fi

kv_args=()
if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then
  kv_args=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
fi

hf_overrides="$(printf '{\"use_index_cache\":true,\"index_topk_pattern\":\"%s\"}' "${GLM52_INDEX_TOPK_PATTERN}")"

cmd=(
  vllm serve "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size "${TP}"
  "${allreduce_args[@]}"
  "${dcp_args[@]}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --attention-backend B12X_MLA_SPARSE
  --moe-backend "${MOE_BACKEND}"
  --quantization "${QUANTIZATION}"
  --quantization-config "${QUANTIZATION_CONFIG_JSON}"
  --load-format "${LOAD_FORMAT}"
  -cc.pass_config.fuse_allreduce_rms=True
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  "${kv_args[@]}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
  --max-cudagraph-capture-size "${GRAPH}"
  --async-scheduling
  --enable-chunked-prefill
  --enable-prefix-caching
  --enable-flashinfer-autotune
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --reasoning-parser glm45
  --default-chat-template-kwargs '{"reasoning_effort":"high"}'
  --enable-prompt-tokens-details
  --enable-force-include-usage
  --enable-request-id-headers
  --hf-overrides "${hf_overrides}"
  "${spec_args[@]}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
