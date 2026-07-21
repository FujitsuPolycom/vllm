# Validation Results

## Daily Profile

Validated on ai01 with TP4/DCP4/MTP3, copy-engine sparse decode, replicated
indexer, depth-3 prefetch, 300,000 maximum model length, batch 2048, graph 32,
and the 432-byte NVFP4 + BF16-RoPE cache format.

| Metric | Result |
| --- | ---: |
| Coding peak, five-run median | 105.2 tok/s |
| Coding peak, five-run mean | 104.5 tok/s |
| Coding peak range | 101.6-108.0 tok/s |
| Reported KV capacity | 302,047 tokens |

Cold prefill on the same final image:

| Context | Prefill |
| ---: | ---: |
| 8K | 3,326 tok/s |
| 32K | 3,139 tok/s |
| 64K | 2,966 tok/s |
| 128K | 2,719 tok/s |
| 256K | 2,356 tok/s |

Deterministic temperature-zero outputs matched byte-for-byte across two runs
at 8,095, 64,032, and 127,998 prompt tokens.

## Public Image Gate

On 2026-07-21, the Dockerfile was rebuilt on ai01 using only the public source
pins in this folder. Build-time focused tests reported:

- Sparkinfer transport/policy: 56 passed, 4 GPU-only skipped
- vLLM attention/workspace policy: 120 passed, 18 skipped
- Mixed-DCP KV allocation: 3 passed, 75 deselected

The resulting image then completed a four-GPU TP4/DCP4/MTP3 startup with the
exact daily profile. Runtime logs confirmed:

- `--disable-custom-all-reduce` and the 3,426,112,942-byte KV pin
- 302,047 logical KV tokens
- replicated sparse-indexer KV
- selected-record CKV decode through C8 with MTP3 and depth 3
- one bulk CKV exchange for S1/S2/S3
- successful CUDA graph capture

`validate-api.py` returned `sparse-ckv-ready` identically twice. Historical
performance numbers below were produced by the same pinned source stack before
the public packaging layer was added; the release-image gate verifies startup
and correctness, not a fresh performance rerun.

## Matched 90K Coding Matrix

These arms used the same fixed 3,426,112,942-byte KV allocation, model, graph
size, batch size, temperature zero, reasoning disabled, one warmup, and five
measured coding runs. The shorter model limit lets DCP1 and DCP4 use an
identical request profile.

| Topology | MTP | Sparse decode | Coding median | KV tokens |
| --- | ---: | --- | ---: | ---: |
| DCP1 | 0 | Off/no-op | 56.7 tok/s | 93,888 |
| DCP1 | 3 | Off/no-op | 135.7 tok/s | 92,480 |
| DCP4 | 0 | Off | 44.4 tok/s | 305,920 |
| DCP4 | 0 | CE | 45.4 tok/s | 305,920 |
| DCP4 | 3 | Off | 82.9 tok/s | 302,080 |
| DCP4 | 3 | CE | 99.2 tok/s | 302,080 |

Sparse selected-record decode adds about 2.4% at MTP0 and 19.6% at MTP3 in
this matched coding test. The daily 300K profile is configuration-sensitive and
is reported separately rather than mixed into the DCP gap calculation.

In the matched bulk-prefetch A/B, enabling the one-shot three-layer exchange
improved the 64K+ geometric mean by 8.5%, moved the coding median from 99.63 to
103.79 tok/s, and left the 302,047-token KV capacity unchanged.

## Interpretation

- MTP3 is the intended deployment regime and receives the largest sparse-gather
  benefit because candidate rows share one per-sequence union.
- DCP4 provides roughly 3.3x the logical KV capacity of DCP1 with this fixed
  per-GPU allocation.
- Sparse decode recovers a meaningful portion of the DCP4 decode penalty, but
  the strict DCP4 coding result does not fully match DCP1.
- Results are for one 4-GPU PCIe host. Treat TP8/DCP8 projections as hypotheses
  until measured on that topology.

## Benchmark Reproduction

The development runs used `llm-inference-bench` v0.4.29. The exact benchmark
script used on ai01 has SHA-256
`0591b2cfe7767b8a048cfd57011526ca5fc64ea29415362c9ab1204c6c5d55e9`.
Representative commands for the measured request shapes are:

```bash
BENCH=/opt/llm-inference-bench
PY="$BENCH/.venv/bin/python"

# Sustained decode matrix plus five sequential coding-peak requests.
$PY "$BENCH/llm_decode_bench.py" --host 127.0.0.1 --port 5802 \
  --skip-prefill --contexts 8k,32k,64k --concurrency 1,2,4,8 \
  --duration 8 --max-tokens 512 --temperature 0 \
  --coding-peak --coding-peak-runs 5 --coding-peak-temperature 0 \
  --display-mode screen --output benchmark-results.json

# One-sample cold-prefill ladder used to check long-context scaling.
$PY "$BENCH/llm_decode_bench.py" --host 127.0.0.1 --port 5802 \
  --prefill-only --prefill-contexts 8k,32k,64k,128k,256k \
  --temperature 0 --display-mode screen --output prefill-results.json
```

Coding peak uses the harness's built-in `/mnt/test.py`-compatible coding
prompt and 2,000-token output budget. Results above are historical validation
of the pinned source stack; the Docker build and public-image gate run
correctness tests, not performance benchmarks.
