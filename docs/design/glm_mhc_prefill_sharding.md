# GLM mHC token ownership for GB10 prefill

Status: **research-only**. The implementation moves repeated mHC work onto each
tensor-parallel rank's token quarter during eligible eager prefills. CPU tests
cover admission and actual model/projection/MoE call boundaries. GLM-5.3-Flash
serving checks on four GB10 GPUs with TP4/DCP4 verified eligible request dispatch,
quarter-token geometry, and completed exact-answer/cache-reuse requests with
continuation coalescing enabled and disabled. The four-arm comparison completed
three cold samples per 8K/16K/32K prefill size and four 20-second decode cells per
arm without reported request errors. With coalescing disabled, measured 8K
prompts had no eligible mHC forward; 16K/32K prompts had one/three eligible
8,192-row forwards. These bounded checks do not establish model-quality
equivalence or long-duration reliability.

## Activation and scope

Set `VLLM_GLM53_MHC_PREFILL_SHARD=1` on every worker of an otherwise working GLM
deployment to enable the feature. The default is zero. No SparkRing installer,
host inventory, virtual mesh, or custom transport is required. The implementation
uses the TP group's existing enabled PyNccl communicator and current CUDA stream.

Admission requires 48-SM NVIDIA GB10 GPUs, compute capability 12.1, BF16 model
activations, hidden size 4,096, an 8,192-token batch ceiling, and TP4/DCP4 with
PP1/DP1/PCP1. Expert parallelism, expert load balancing, and sequence-parallel
MoE are excluded. Captured graph sizes must stay below 8,192. Every active base
layer must provide the supported GLM attention/projection/MoE and B12X mHC
implementations.

At model entry, host GDN metadata must describe 8,192 pure-prefill tokens and
zero ordinary or speculative decode rows. Other row counts, ambiguous metadata,
mixed batches, compilation, and CUDA graph capture/replay retain the normal
path. MTP layers cannot receive mHC ownership. All TP ranks agree on activation
at construction and on eligible metadata/capabilities before issuing partial
outputs. Unsupported enabled configurations fail explicitly.

## Native cache geometry for qualification

The 512-token cache layout used by the standalone schedule below requires these
worker environment settings on every rank:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=512
export VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=512
export VLLM_GLM53_MHC_PREFILL_SHARD=1
```

Retain the model and parallel configuration described above. Set
`--block-size 512 --mamba-block-size 512 --mamba-cache-mode align`
and `--recurrent-checkpoint-policy aligned --prefix-cache-retention-interval 0`.
The split-page settings are native platform controls already present in the
baseline implementation. They preserve independent attention and recurrent
pages; requesting 512 only through the CLI does not prevent page harmonization
from choosing a larger token block. The value `auto` selects a different layout
and is not interchangeable with the fixed values in this protocol.

Verify physical target/recurrent blocks of 512 tokens, prefix-lookup alignment
of 512, and DCP4 scheduler alignment of 2,048 in the resolved configuration.
Hold both split-page settings constant across comparison arms. These settings
require neither a SparkCache connector nor continuation coalescing or its B12X
checkpoint extension.

Cache geometry does not itself activate mHC. A scheduled forward must still
meet the 8,192-row pure-prefill admission gate. Enable the diagnostics described
below for an activation check and require request-kind `GLM_MHC_ENQUEUE` records
on all four ranks, followed by successful request completion. Startup/dummy
records and enabled environment flags alone do not establish that the request
used token ownership. Hold the diagnostic setting constant within comparisons.

## Computation and tensor lifetime

The first mHC pre runs on full tokens. Its residual, post, and combine state
become views of the owning rank's contiguous quarter. Attention and FFN inputs
remain full-token tensors.

For each attention or FFN boundary, an explicit per-call argument defers its
normal TP output reduction. NCCL reduce-scatter sums the full TP partial into
the owner's 2,048 rows. B12X mHC runs on those rows; an all-gather restores the
normalized full-token input before the next attention or FFN. Shared and routed
MoE outputs are combined before that single deferred reduction. Already-reduced
or unsupported MoE outputs are rejected. Existing pre-reduction L2-prefetch
callbacks keep their call positions.

The final hidden output and every auxiliary capture are gathered into full token
order before downstream consumers. A base model with 45 layers and no auxiliary
captures uses 90 reduce-scatters and 90 all-gathers per eligible forward; each
auxiliary capture adds one gather. MTP attention/cache work and decode reductions
retain their normal paths.

Collective outputs are allocated per use and returned with caller ownership.
Final and auxiliary tensors can outlive the forward, so one shared reusable
buffer would require an explicit consumer lifetime contract. CUDA stream records
protect allocator reuse after release. This implementation does not claim
allocation-free execution or a fixed model-memory saving.

The feature calls the existing B12X mHC API. It requires no two-checkpoint B12X
extension and does not enable continuation coalescing. Scheduler chunk sizes
still determine whether an individual forward reaches the 8,192-row gate.

CPU execution of the unmodified scheduler and real KV cache manager demonstrates
independent eligibility with one B12X checkpoint, MTP3, physical/hash blocks of
512 tokens, scheduler blocks of 2,048, and sparse retention interval zero:

| Prompt | Scheduled chunks with coalescing off | mHC-owned forwards |
| --- | --- | ---: |
| 8K | 4,096 + 2,048 + 2,048 | 0 |
| 16K | 8,192 + 4,096 + 2,048 + 2,048 | 1 |
| 32K | Three 8,192 chunks + 4,096 + 2,048 + 2,048 | 3 |
| 64K | Seven 8,192 chunks + 4,096 + 2,048 + 2,048 | 7 |

Thus mHC can activate independently for full chunks of long prompts. It does
not accelerate the 8K prompt in this uncoalesced schedule: none of that prompt's
forwards reaches the row gate. Other retention or scheduling settings may change
coverage. These are CPU scheduling/admission results, not GPU timing results.

## Admission and execution diagnostics

`VLLM_GLM53_MHC_PREFILL_DIAGNOSTICS=1` enables host-only diagnostic warnings;
the default is zero. It does not enable mHC ownership or change eligibility.
`GLM_MHC_DIAGNOSTIC` records the configuration flag, fallback reason, observed
hidden shapes, graph mode, and GDN admission counts with their Python types.
Non-host count values are omitted without converting or printing tensors.
Each reason is reported once per request/dummy category; shape rejection also
distinguishes each observed shape. Suppressed logging does not consume reports.
Compiler tracing skips diagnostic mutation.

`GLM_MHC_ENQUEUE` records each admitted forward after all its host calls return:
rank, request/dummy category, collective enqueue counts, and actual mHC output
row counts grouped by operation. For 45 base layers with no auxiliary captures,
the expected witness is 90 reduce-scatters, 90 gathers, one 8,192-row first pre,
44 attention post/pre calls on 2,048 rows, 45 FFN post/pre calls on 2,048 rows,
and one final post on 2,048 rows. The V2 runner marks dummy forwards through a
host-only forward-context field, so startup work cannot consume request reports.

These records prove host dispatch and tensor geometry, not asynchronous GPU
completion or numerical correctness. Successful request completion and the
correctness protocol remain required. Diagnostic logging adds host work; record
its setting with benchmark conditions and disable it for isolated throughput
qualification after activation has been established.

## CPU validation

In a supported vLLM development environment:

```bash
.venv/bin/python -m pytest tests/models/test_glm5next_mhc_prefill.py -q
.venv/bin/python -m pytest tests/models/test_glm5next_model.py \
  -k 'mhc or mtp_compacts' -q
```

The tests execute the actual model, row-parallel projection, MoE runner, and
ownership helper. CPU doubles supply distributed collectives and mHC arithmetic
for a small model chain, checking full final/auxiliary results and local residual
state. They also check hardware/configuration rejection, mixed/graph fallback,
rank disagreement, exactly-once reduction, prefetch callbacks, and output stream
ownership. They do not validate CUDA visibility, NCCL transport, or model quality.

## GPU and full-model qualification protocol

The following work is pending and must use an explicitly authorized test stack.
Keep model revision, weights, B12X revision, TP/DCP settings, cache settings,
sampling parameters, and worker image identical between arms; change only
`VLLM_GLM53_MHC_PREFILL_SHARD`.

1. Complete model loading and automatic warmup with the feature off and on.
   Record all-rank health and the exact source/image/toolchain identities.
2. For cold 8K, 16K, 32K, and 64K prompts, record actual scheduled token counts
   and `GLM_MHC_PREFILL` activation/count logs. Reject missing workers, partial
   collectives, or unexpected auxiliary-output shapes.
3. Compare complete logits or token log-probabilities, exact-answer probes, and
   recurrent/cache state across fresh, continued, repeated, and extended prompts.
   A different collective reduction order is not expected to be bitwise equal;
   numerical and model-quality acceptance criteria must be declared before
   evaluating results.
4. Exercise prefill followed by captured MTP decode, eager small batches, and
   concurrent decode plus prefill. Verify mixed batches use ordinary reductions.
   Check decode and cache correctness separately from prefill throughput.
5. Run the repository's [GSM8K](../../tests/evals/gsm8k/README.md) and
   [MRCR](../../tests/evals/mrcr/README.md) evaluation protocols against each
   server arm. Save raw evaluation outputs and compare accuracy, including
   long-context buckets. No evaluation result is supplied by this source port.
6. Measure cold-prefill TTFT using exact token counts and explicit zero cached
   tokens after warmup. Interleave repeated arms, preserve errors, and report
   variability and any changed clock/power/cache/launch conditions.

Standalone mHC and continuation coalescing must be tested independently before
the combined setting. No GPU result from a different runtime composition
qualifies this source change.
