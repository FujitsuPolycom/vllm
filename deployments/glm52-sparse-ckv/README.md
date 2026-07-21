# GLM-5.2 Sparse CKV Integration Release

This folder reproduces the validated GLM-5.2 sparse-CKV stack from immutable
commits on the FujitsuPolycom forks. It is intended as an experimental release
and review target, not as a general vLLM distribution.

## Validated Host

- ASUS WRX90E-SAGE SE
- AMD Ryzen Threadripper PRO 9965WX
- 128 GiB system RAM
- 4x RTX PRO 6000 Blackwell 96GB, 400 W each
- TP4, DCP4, MTP3
- Full CUDA peer access; ACS and IOMMU disabled on the validated host

The four GPUs are attached through separate CPU root ports rather than a Gen5
P2P switch. Transport results are therefore specific to this host topology.

The selected-record transports require CUDA IPC peer access between every GPU.
Check topology before launch:

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
./preflight.sh
```

## Pinned Sources

| Component | Pin |
| --- | --- |
| Base image | `voipmonitor/vllm@sha256:41078d5fac36b802e0f3798057a4b383c24296cb8570f6f4fc138c34d8cfa303` |
| vLLM | `FujitsuPolycom/vllm@a81a0811314e23bc6b2baa69f34247aacea71e23` |
| Sparkinfer | `FujitsuPolycom/sparkinfer@a82fc76f9a2321e3cfd3065a9fa78452a27ddf9f` |
| Model | `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid@68babde27a97a4c980c2494e830dd424975cd5a3` |

The Docker build verifies both Git SHAs before installing anything. It also
runs the focused CPU policy and transport tests used for the validated image.
The runtime toolchain is pinned to CUTLASS DSL 4.6.0 and QuACK 0.6.1, matching
the validated image rather than the older packages in the v19 base.
The base vLLM package metadata still declares CUTLASS DSL 4.5.3, so `pip check`
reports the intentional 4.6.0 override; startup and focused tests use 4.6.0.

This is an overlay build, not a source-only CUDA stack build. It intentionally
inherits the pinned `voipmonitor/vllm` image for CUDA, NCCL, FlashInfer,
and InstantTensor, and requires separate access to the pinned Hugging Face
checkpoint. The release installs its own small launcher so the fixed KV budget
and NCCL all-reduce mode cannot be silently dropped by a base-image launcher.

## Included Features

- Lockstep allocation for mixed MLA KV-cache groups
- Replicated sparse-indexer KV with DCP-sharded main CKV
- Native 432-byte CKV history gather and depth-3 Shared-layer prefetch
- Per-sequence MTP union and deduplication
- Selected-record sparse CKV decode for C1-C8
- Direct CUDA-IPC P2P and copy-engine transports
- One-shot bulk transfer for the three Shared layers

The default Compose selects the copy-engine transport because it won the TP4
A/B on the validated host. The direct transport remains available by changing
`VLLM_B12X_MLA_SPARSE_DECODE_TRANSPORT` from `ce` to `direct`.

## Build And Launch

```bash
git clone --branch codex/glm52-sparse-ckv-release-20260721 \
  https://github.com/FujitsuPolycom/vllm.git
cd vllm/deployments/glm52-sparse-ckv

# Skip this download when the pinned model revision is already local.
hf download madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  --revision 68babde27a97a4c980c2494e830dd424975cd5a3 \
  --local-dir /srv/ai/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid

cp .env.example .env
# Edit MODEL_PATH and the cache paths in .env.
./preflight.sh
docker compose config --quiet
docker compose build --pull
docker compose up -d
docker compose logs -f glm52-sparse-ckv
```

Keep the Compose entrypoint set to `serve-glm52-sparse-ckv.sh`. The inherited
v19 launcher does not accept this release's fixed `KV_CACHE_MEMORY_BYTES` or
`ALLREDUCE_MODE` controls; using it changes memory allocation and can OOM.

The first startup performs model loading, JIT compilation, warmup, and CUDA
graph capture. Do not benchmark the first request. The API is ready when this
returns the served model:

```bash
curl -fsS http://127.0.0.1:5802/v1/models
python validate-api.py
```

Confirm that the intended format and features activated:

```bash
docker compose logs glm52-sparse-ckv 2>&1 | \
  grep -E 'kv_gmem_stride=432|replicat|prefetch|sparse.*CKV|transport=ce'
```

The validated profile uses a fixed 3,426,112,942-byte KV allocation per GPU,
`max-model-len=300000`, batch 2048, graph 32, and eight active sequences. The
reported logical KV pool was 302,047 tokens.

The image assembled only from the public pins in this folder also completed a
four-GPU startup, CUDA-graph capture, and the two-run deterministic API probe at
that exact 302,047-token capacity.

## Feature Controls

| Variable | Default | Purpose |
| --- | ---: | --- |
| `VLLM_DCP_REPLICATE_INDEXER_CACHE` | `1` | Replicate the small indexer cache while keeping main CKV DCP-sharded |
| `VLLM_B12X_MLA_CKV_GATHER` / `DCP_CKV_GATHER` | `1` | Gather full CKV for the prefill path |
| `VLLM_B12X_MLA_CKV_PREFETCH_DEPTH` | `3` | Prefetch all three Shared layers after each Full layer |
| `VLLM_B12X_MLA_SPARSE_DECODE_CKV_GATHER` | `1` | Enable selected-record sparse decode |
| `VLLM_B12X_MLA_SPARSE_DECODE_TRANSPORT` | `ce` | Select copy-engine or direct P2P transport |
| `VLLM_B12X_MLA_SPARSE_DECODE_BULK_PREFETCH` | `1` | Transfer S1/S2/S3 together with one synchronization |
| `VLLM_B12X_MLA_SPARSE_DECODE_MAX_SEQS` | `8` | Enable the pooled fast path through C8 |

## Current Boundaries

- Sparse selected-record decode is validated only with
  `KV_CACHE_DTYPE=nvfp4_ds_mla`, `KV_FP8_ROPE=0`, and the native 432-byte
  NVFP4 + BF16-RoPE record.
- TP4/DCP4/MTP3 is the production-tested topology. DCP1 and DCP4 were A/B
  tested; TP6/TP8 and DCP6/DCP8 still need physical validation.
- Both explicit `ce` and `direct` transports fail closed when peer
  initialization is unavailable. `auto` may fall back from CE to direct, but
  this release defaults to strict `ce` and requires full peer access.
- LMCache is not part of this release.
- FP8-RoPE support, calibrated MLA outer scales, and the newer absorbed-QBMM
  memory optimization are intentionally excluded until separately validated.

See [RESULTS.md](RESULTS.md) for the configuration and performance matrix.
Machine-readable source and runtime pins are in
[VERSION_LOCK.json](VERSION_LOCK.json).

## Security

The validated Compose uses host networking and `privileged: true` to preserve
the known-working GPU/P2P environment. Run it only on a trusted, isolated host.
Do not expose the unauthenticated vLLM port directly to an untrusted network.

## Credits

The scheduling design builds on Koush's sparse-CKV, shared-layer lookahead, and
MTP-deduplication work. The transport and kernel integration builds on Luke
Alonso's Sparkinfer/B12X infrastructure. Implementation, testing, benchmarking,
and documentation were developed with OpenAI Codex assistance and manually
reviewed on the validated host.
