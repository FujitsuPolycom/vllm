#!/usr/bin/env python3
"""GPU acceptance test for sparse native-432 CKV union and P2P transport.

Run the local union test directly, or the transport test with torchrun:

  python tests/distributed/test_b12x_sparse_ckv_p2p_gpu.py --mode union
  torchrun --standalone --nproc-per-node 4 \
    tests/distributed/test_b12x_sparse_ckv_p2p_gpu.py --mode transport
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

RECORD_BYTES = 432


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("b12x_sparse_ckv_p2p_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import sparse CKV transport from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_rows(rows: int, topk: int, overlap: float, offset: int = 0) -> torch.Tensor:
    shared = round(topk * overlap)
    base = torch.arange(offset, offset + topk, dtype=torch.int32)
    result = [base]
    next_unique = offset + topk
    for row in range(1, rows):
        values = torch.cat(
            (
                base[:shared],
                torch.arange(
                    next_unique,
                    next_unique + topk - shared,
                    dtype=torch.int32,
                ),
            )
        )
        next_unique += topk - shared
        result.append(torch.roll(values, shifts=row * 137))
    return torch.stack(result).contiguous()


def _record_pattern(tokens: torch.Tensor) -> torch.Tensor:
    byte_offsets = torch.arange(
        RECORD_BYTES, dtype=torch.int64, device=tokens.device
    )
    return (
        tokens.to(torch.int64)[:, None] * 17
        + byte_offsets[None, :] * 13
        + 7
    ).remainder(251).to(torch.uint8)


def _assert_selected_records_equal(
    output: torch.Tensor,
    expected: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    allow_inactive_stale: bool,
    label: str,
) -> int:
    """Compare every selected record while treating unselected tail as scratch."""
    if output.shape != expected.shape:
        raise AssertionError(
            f"{label}: output shape {output.shape} != expected {expected.shape}"
        )
    record_mismatch = torch.any(output != expected, dim=-1)
    selected = selected_indices.view_as(record_mismatch) >= 0
    selected_mismatch = record_mismatch & selected
    if torch.any(selected_mismatch):
        byte_mismatch = int(
            torch.count_nonzero(
                (output != expected) & selected.unsqueeze(-1)
            ).item()
        )
        bad_records = int(torch.count_nonzero(selected_mismatch).item())
        raise AssertionError(
            f"{label}: {bad_records} selected records differ "
            f"({byte_mismatch} bytes)"
        )

    inactive_stale = int(
        torch.count_nonzero(record_mismatch & ~selected).item()
    )
    if inactive_stale and not allow_inactive_stale:
        raise AssertionError(
            f"{label}: {inactive_stale} inactive records were not cleared"
        )
    return inactive_stale


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _assert_union(
    extension,
    indices: torch.Tensor,
    *,
    graph_replay: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    entries = indices.numel()
    union_indices = torch.empty(entries, dtype=torch.int32, device=indices.device)
    remap = torch.empty_like(indices)
    union_count = torch.empty(1, dtype=torch.int32, device=indices.device)
    hash_capacity = _next_power_of_two(2 * entries)
    hash_keys = torch.empty(hash_capacity, dtype=torch.int32, device=indices.device)
    hash_values = torch.empty_like(hash_keys)

    def run() -> None:
        extension.build_union_remap(
            indices,
            union_indices,
            remap,
            union_count,
            hash_keys,
            hash_values,
        )

    run()
    torch.cuda.synchronize(indices.device)
    expected_count = int(torch.unique(indices).numel())
    if int(union_count.item()) != expected_count:
        raise AssertionError(
            f"union count mismatch: {int(union_count.item())} != {expected_count}"
        )
    if not torch.equal(union_indices[remap], indices):
        raise AssertionError("union remap does not reconstruct all MTP rows")

    if graph_replay:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(indices.device)
        with torch.cuda.graph(graph):
            run()
        replacement = _make_rows(
            indices.shape[0], indices.shape[1], 0.55, offset=31
        ).to(indices.device)
        indices.copy_(replacement)
        graph.replay()
        torch.cuda.synchronize(indices.device)
        expected_count = int(torch.unique(indices).numel())
        if int(union_count.item()) != expected_count:
            raise AssertionError("captured union did not follow changed input indices")
        if not torch.equal(union_indices[remap], indices):
            raise AssertionError("captured remap failed after input indices changed")

    return union_indices, remap, expected_count


def _run_union(args, module) -> None:
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    extension = module._load_sparse_extension()
    for rows in (1, args.rows):
        indices = _make_rows(rows, args.topk, args.overlap).to(device)
        _, _, union_count = _assert_union(
            extension, indices, graph_replay=args.graph_replay
        )
        print(
            f"PASS union rows={rows} topk={args.topk} "
            f"union={union_count}/{indices.numel()} graph={args.graph_replay}",
            flush=True,
        )


def _max_rank_ms(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _time_call(call, iterations: int, device: torch.device) -> list[float]:
    samples: list[float] = []
    stream = torch.cuda.current_stream(device)
    for _ in range(iterations):
        dist.barrier(device_ids=[device.index])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        call()
        end.record(stream)
        end.synchronize()
        samples.append(_max_rank_ms(start.elapsed_time(end), device))
    return samples


def _run_transport(args, module) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if not 2 <= world_size <= 8:
        raise RuntimeError("sparse CKV acceptance supports DCP world sizes 2 through 8")

    entries_per_request = args.rows * args.topk
    entries = args.requests * entries_per_request
    channel_requests = args.channel_requests or args.requests
    if channel_requests < args.requests:
        raise ValueError("channel_requests must cover the active requests")
    channel_entries = channel_requests * entries_per_request
    extension = module._load_sparse_extension()
    if args.fault_init_rank is not None and rank == args.fault_init_rank:
        original_malloc = module.CudaRTLibrary.cudaMalloc

        def fail_malloc(_self, _size):
            raise RuntimeError("intentional sparse CKV initialization failure")

        module.CudaRTLibrary.cudaMalloc = fail_malloc
    else:
        original_malloc = None
    try:
        channel = module.B12xSparseCkvP2P(
            process_group=dist.group.WORLD,
            device=device,
            workspace_slots=args.workspace_slots,
            topk=channel_entries,
            record_bytes=RECORD_BYTES,
            primary_capacity=args.primary_capacity,
            allocate_direct_workspace=args.transport != "ce",
        )
    except RuntimeError as exc:
        if args.fault_init_rank is None:
            raise
        dist.barrier(device_ids=[device.index])
        if rank == 0:
            print(
                "PASS rank-consistent sparse CKV initialization failure: "
                f"fault_rank={args.fault_init_rank} error={exc}",
                flush=True,
            )
        dist.destroy_process_group()
        return
    finally:
        if original_malloc is not None:
            module.CudaRTLibrary.cudaMalloc = original_malloc
    if args.fault_init_rank is not None:
        channel.close()
        dist.destroy_process_group()
        raise AssertionError("asymmetric initialization failure was not propagated")
    try:
        oversized_slots = torch.full(
            (world_size, 1, channel.topk + 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        dummy_kv = torch.empty((1, 1, RECORD_BYTES), dtype=torch.uint8, device=device)
        dummy_indices = torch.empty((1, 1), dtype=torch.int32, device=device)
        dummy_output = torch.empty(
            (1, 1, RECORD_BYTES), dtype=torch.uint8, device=device
        )
        send_counters_before = channel._send_counters.clone()
        wait_counters_before = channel._wait_counters.clone()
        try:
            transfer = (
                channel.transfer_scatter_ce
                if args.transport == "ce"
                else channel.transfer_scatter
            )
            transfer(
                dummy_kv,
                oversized_slots,
                dummy_indices,
                dummy_output,
                workspace_slot=0,
                interleave=args.interleave,
            )
        except ValueError as exc:
            if "outside channel capacity" not in str(exc):
                raise
        else:
            raise AssertionError("oversized sparse CKV plane was not rejected")
        if not torch.equal(channel._send_counters, send_counters_before):
            raise AssertionError("oversized plane changed P2P send counters")
        if not torch.equal(channel._wait_counters, wait_counters_before):
            raise AssertionError("oversized plane changed P2P wait counters")

        indices = _make_rows(args.rows, args.topk, args.overlap).to(device)
        union_indices, remap, expected_count = _assert_union(
            extension, indices, graph_replay=False
        )

        peer_unions = [torch.empty_like(union_indices) for _ in range(world_size)]
        peer_remaps = [torch.empty_like(remap) for _ in range(world_size)]
        dist.all_gather(peer_unions, union_indices)
        dist.all_gather(peer_remaps, remap)
        if any(not torch.equal(value, union_indices) for value in peer_unions):
            raise AssertionError("union layout differs across DCP ranks")
        if any(not torch.equal(value, remap) for value in peer_remaps):
            raise AssertionError("remap differs across DCP ranks")

        max_token = int(indices.max().item())
        interleave = int(args.interleave)
        global_tokens = torch.arange(
            max_token + 1, dtype=torch.int32, device=device
        )
        global_owners = global_tokens.div(
            interleave, rounding_mode="floor"
        ).remainder(world_size)
        local_tokens = global_tokens[global_owners == rank]
        local_ordinals = (
            global_tokens.div(interleave * world_size, rounding_mode="floor")
            * interleave
            + global_tokens.remainder(interleave)
        )[global_owners == rank]
        local_capacity = int(local_ordinals.max().item()) + 1
        local_token_by_ordinal = torch.empty(
            local_capacity, dtype=torch.int32, device=device
        )
        local_token_by_ordinal[local_ordinals] = local_tokens
        marker_stride = 1_000_000
        local_markers = torch.cat(
            [
                local_token_by_ordinal + request * marker_stride
                for request in range(args.requests)
            ]
        )
        kv_cache = _record_pattern(local_markers).view(-1, 1, RECORD_BYTES)
        valid = union_indices >= 0
        owners = union_indices.div(
            interleave, rounding_mode="floor"
        ).remainder(world_size)
        local_union_ordinals = (
            union_indices.div(
                interleave * world_size, rounding_mode="floor"
            )
            * interleave
            + union_indices.remainder(interleave)
        )
        local_slots_by_request = torch.stack(
            [
                torch.where(
                    valid & (owners == rank),
                    request * local_capacity
                    + local_union_ordinals,
                    torch.full_like(union_indices, -1),
                )
                for request in range(args.requests)
            ]
        )
        local_slots = (
            local_slots_by_request.unsqueeze(0)
            .expand(world_size, -1, -1)
            .contiguous()
        )
        # Every destination must receive exactly one owner record for every
        # entry in each request's deduplicated union. This is the transport
        # invariant that proves MTP duplicates do not become duplicate P2P
        # writes merely because they occur in multiple query rows.
        local_record_writes = (local_slots >= 0).sum(dim=(1, 2), dtype=torch.int64)
        global_record_writes = local_record_writes.clone()
        dist.all_reduce(global_record_writes, op=dist.ReduceOp.SUM)
        expected_writes_per_destination = args.requests * expected_count
        expected_writes = torch.full_like(
            global_record_writes, expected_writes_per_destination
        )
        if not torch.equal(global_record_writes, expected_writes):
            raise AssertionError(
                "deduplicated union does not map to one owner write per record: "
                f"observed={global_record_writes.cpu().tolist()} "
                f"expected={expected_writes.cpu().tolist()}"
            )
        total_record_writes = int(global_record_writes.sum().item())
        candidate_record_writes = (
            world_size * args.requests * args.rows * args.topk
        )
        global_indices = union_indices.repeat(args.requests).view(1, -1)
        output = torch.empty(
            (1, entries, RECORD_BYTES), dtype=torch.uint8, device=device
        )
        local_reference = torch.zeros_like(output)
        expected = torch.zeros_like(output)
        for request in range(args.requests):
            start = request * entries_per_request
            stop = start + entries_per_request
            expected[0, start:stop][valid] = _record_pattern(
                union_indices[valid] + request * marker_stride
            )
            local_reference[0, start:stop][valid & (owners == rank)] = expected[
                0, start:stop
            ][valid & (owners == rank)]

        def p2p_call() -> None:
            transfer = (
                channel.transfer_scatter_ce
                if args.transport == "ce"
                else channel.transfer_scatter
            )
            transfer(
                kv_cache,
                local_slots,
                global_indices,
                output,
                workspace_slot=0,
                interleave=interleave,
            )

        if args.fault_skip_rank is not None:
            if not 0 <= args.fault_skip_rank < world_size:
                raise ValueError("fault_skip_rank must name an active DCP rank")
            if rank == args.fault_skip_rank:
                print(
                    f"FAULT rank {rank} withholding sparse CKV publication",
                    flush=True,
                )
                time.sleep(args.fault_sleep_seconds)
                return
            p2p_call()
            torch.cuda.synchronize(device)
            raise AssertionError(
                "sparse CKV barrier returned despite a missing peer publication"
            )

        allow_inactive_stale = args.transport == "scatter"
        inactive_stale_records = 0
        for _ in range(args.warmup):
            p2p_call()
        torch.cuda.synchronize(device)
        inactive_stale_records = max(
            inactive_stale_records,
            _assert_selected_records_equal(
                output,
                expected,
                global_indices,
                allow_inactive_stale=allow_inactive_stale,
                label="P2P native-432",
            ),
        )
        for request in range(args.requests):
            start = request * entries_per_request
            request_output = output[0, start : start + entries_per_request]
            reconstructed = request_output[remap]
            reference = _record_pattern(
                indices.reshape(-1) + request * marker_stride
            ).view(args.rows, args.topk, RECORD_BYTES)
            if not torch.equal(reconstructed, reference):
                raise AssertionError(
                    f"request {request} remap does not reconstruct native-432 rows"
                )

        nccl_output = local_reference.clone()
        dist.all_reduce(nccl_output, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        if not torch.equal(nccl_output, expected):
            raise AssertionError("NCCL native-432 reference differs from expected")

        if args.graph_replay:
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(device)
            with torch.cuda.graph(graph):
                p2p_call()
            output.zero_()
            graph.replay()
            torch.cuda.synchronize(device)
            inactive_stale_records = max(
                inactive_stale_records,
                _assert_selected_records_equal(
                    output,
                    expected,
                    global_indices,
                    allow_inactive_stale=allow_inactive_stale,
                    label="P2P native-432 CUDA graph replay",
                ),
            )

            # Reuse the same captured graph after changing physical cache
            # contents, selected-record order, and valid union length. This
            # catches stale block-table/selection data hidden by a static-shape
            # replay that merely repeats the captured request.
            original_kv = kv_cache.clone()
            original_slots = local_slots.clone()
            original_indices = global_indices.clone()
            mutated_expected = torch.zeros_like(expected)
            replay_count = max(1, expected_count - 31)
            first_request_indices = original_indices.view(
                args.requests, entries_per_request
            )[0]
            valid_positions = torch.nonzero(
                first_request_indices >= 0, as_tuple=False
            ).flatten()
            if valid_positions.numel() != expected_count:
                raise AssertionError(
                    "union_count does not match valid union positions: "
                    f"{expected_count} != {valid_positions.numel()}"
                )
            permutation = torch.roll(valid_positions, shifts=137)[:replay_count]
            global_indices_by_request = global_indices.view(
                args.requests, entries_per_request
            )
            for request in range(args.requests):
                start = request * entries_per_request
                stop = start + entries_per_request
                old_indices = original_indices.view(
                    args.requests, entries_per_request
                )[request]
                global_indices_by_request[request].fill_(-1)
                global_indices_by_request[request, :replay_count].copy_(
                    old_indices[permutation]
                )
                for destination in range(world_size):
                    local_slots[destination, request].fill_(-1)
                    local_slots[
                        destination, request, :replay_count
                    ].copy_(
                        original_slots[destination, request, permutation]
                    )
                mutated_expected[0, start : start + replay_count].copy_(
                    expected[0, start:stop][permutation].bitwise_xor(0x5A)
                )
            kv_cache.bitwise_xor_(0x5A)
            output.zero_()
            graph.replay()
            torch.cuda.synchronize(device)
            inactive_stale_records = max(
                inactive_stale_records,
                _assert_selected_records_equal(
                    output,
                    mutated_expected,
                    global_indices,
                    allow_inactive_stale=allow_inactive_stale,
                    label="P2P graph replay with changed indices/blocks",
                ),
            )

            kv_cache.copy_(original_kv)
            local_slots.copy_(original_slots)
            global_indices.copy_(original_indices)

            # Simulate prefix sharing: two distinct logical selected records
            # owned by rank 0 alias the same physical CKV slot. The captured
            # graph must observe the changed block mapping on replay.
            alias_expected = expected.clone()
            owner_zero_positions = torch.nonzero(
                valid & (owners == 0), as_tuple=False
            ).flatten()
            if owner_zero_positions.numel() < 2:
                raise AssertionError("alias replay needs two rank-0 records")
            alias_source = int(owner_zero_positions[0].item())
            alias_destination = int(owner_zero_positions[-1].item())
            for request in range(args.requests):
                local_slots[:, request, alias_destination].copy_(
                    local_slots[:, request, alias_source]
                )
                start = request * entries_per_request
                alias_expected[0, start + alias_destination].copy_(
                    alias_expected[0, start + alias_source]
                )
            output.zero_()
            graph.replay()
            torch.cuda.synchronize(device)
            inactive_stale_records = max(
                inactive_stale_records,
                _assert_selected_records_equal(
                    output,
                    alias_expected,
                    global_indices,
                    allow_inactive_stale=allow_inactive_stale,
                    label="P2P graph replay prefix-sharing/block alias",
                ),
            )

            local_slots.copy_(original_slots)
            output.zero_()
            graph.replay()
            torch.cuda.synchronize(device)
            inactive_stale_records = max(
                inactive_stale_records,
                _assert_selected_records_equal(
                    output,
                    expected,
                    global_indices,
                    allow_inactive_stale=allow_inactive_stale,
                    label="P2P graph replay baseline restore",
                ),
            )

        p2p_samples = _time_call(p2p_call, args.iterations, device)

        def nccl_call() -> None:
            nccl_output.copy_(local_reference)
            dist.all_reduce(nccl_output, op=dist.ReduceOp.SUM)

        nccl_samples = _time_call(nccl_call, args.iterations, device)
        if rank == 0:
            p2p_ms = statistics.median(p2p_samples)
            nccl_ms = statistics.median(nccl_samples)
            print(
                f"PASS DCP{world_size} requests={args.requests} "
                f"rows={args.rows} topk={args.topk} transport={args.transport} "
                f"interleave={interleave} "
                f"union={expected_count}/{entries} native432=byte_exact "
                f"record_writes={total_record_writes}/{candidate_record_writes} "
                f"inactive_stale_records={inactive_stale_records} "
                f"remap=exact preflight=bounded graph={args.graph_replay} "
                f"p2p_ms={p2p_ms:.4f} nccl_ms={nccl_ms:.4f} "
                f"speedup={nccl_ms / p2p_ms:.2f}x",
                flush=True,
            )
    finally:
        channel.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("union", "transport"), required=True)
    parser.add_argument("--module", type=Path)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--channel-requests", type=int)
    parser.add_argument("--transport", choices=("scatter", "ce"), default="scatter")
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--interleave", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--overlap", type=float, default=0.90)
    parser.add_argument("--workspace-slots", type=int, default=4)
    parser.add_argument("--primary-capacity", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--graph-replay", action="store_true")
    parser.add_argument("--fault-skip-rank", type=int)
    parser.add_argument("--fault-init-rank", type=int)
    parser.add_argument("--fault-sleep-seconds", type=float, default=15.0)
    args = parser.parse_args()

    if args.module is None:
        from vllm.v1.attention.backends.mla import b12x_sparse_ckv_p2p as module
    else:
        module = _load_module(args.module.resolve())
    if args.mode == "union":
        _run_union(args, module)
    else:
        _run_transport(args, module)


if __name__ == "__main__":
    main()
