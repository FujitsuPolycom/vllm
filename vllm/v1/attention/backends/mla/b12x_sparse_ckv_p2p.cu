// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <cstdio>
#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t hash_token_index(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__global__ void barrier_all_peers_kernel(
    const int64_t* __restrict__ publish_flag_ptrs,
    const int64_t* __restrict__ wait_flag_ptrs,
    int32_t* __restrict__ send_counters,
    int32_t* __restrict__ wait_counters,
    int count,
    uint64_t timeout_cycles) {
  const int peer = static_cast<int>(threadIdx.x);
  if (peer >= count) {
    return;
  }

  const int32_t publish_value = send_counters[peer] + 1;
  send_counters[peer] = publish_value;
  // The record scatter or local copy completed before this kernel in stream
  // order. Publish all peer flags from one launch, then wait for every source.
  __threadfence_system();
  auto* publish_flag = reinterpret_cast<int32_t*>(
      static_cast<uintptr_t>(publish_flag_ptrs[peer]));
  asm volatile(
      "st.relaxed.sys.global.u32 [%1], %0;"
      :
      : "r"(publish_value), "l"(publish_flag));

  const int32_t expected = wait_counters[peer] + 1;
  wait_counters[peer] = expected;
  const auto* wait_flag = reinterpret_cast<const int32_t*>(
      static_cast<uintptr_t>(wait_flag_ptrs[peer]));
  int32_t observed;
  const uint64_t started = clock64();
  uint32_t spins = 0;
  do {
    asm volatile(
        "ld.acquire.sys.global.u32 %0, [%1];"
        : "=r"(observed)
        : "l"(wait_flag));
    if (((++spins & 0x3ffU) == 0U) &&
        (clock64() - started > timeout_cycles)) {
      printf(
          "B12X sparse CKV barrier timed out: peer=%d expected=%d "
          "observed=%d\n",
          peer,
          expected,
          observed);
      asm volatile("trap;");
      return;
    }
  } while (static_cast<int32_t>(observed - expected) < 0);
}

__global__ void build_union_hash_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ hash_keys,
    int32_t* __restrict__ hash_values,
    int64_t input_entries,
    int64_t hash_capacity) {
  const int64_t input_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (input_index >= input_entries) {
    return;
  }

  const int32_t key = indices[input_index];
  if (key < 0) {
    return;
  }

  const uint32_t hash_mask = static_cast<uint32_t>(hash_capacity - 1);
  uint32_t slot = hash_token_index(static_cast<uint32_t>(key)) & hash_mask;
  for (int64_t probe = 0; probe < hash_capacity; ++probe) {
    const int32_t previous = atomicCAS(hash_keys + slot, -1, key);
    if (previous == -1 || previous == key) {
      // The first flattened occurrence is deterministic across ranks even
      // when hash insertion order differs between kernels or launches.
      atomicMin(hash_values + slot, static_cast<int32_t>(input_index));
      return;
    }
    slot = (slot + 1) & hash_mask;
  }
}

__global__ void finalize_union_remap_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ union_indices,
    int32_t* __restrict__ remap,
    int32_t* __restrict__ union_count,
    const int32_t* __restrict__ hash_keys,
    const int32_t* __restrict__ hash_values,
    int64_t input_entries,
    int64_t hash_capacity) {
  const int64_t input_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (input_index >= input_entries) {
    return;
  }

  const int32_t key = indices[input_index];
  if (key < 0) {
    remap[input_index] = -1;
    return;
  }

  const uint32_t hash_mask = static_cast<uint32_t>(hash_capacity - 1);
  uint32_t slot = hash_token_index(static_cast<uint32_t>(key)) & hash_mask;
  for (int64_t probe = 0; probe < hash_capacity; ++probe) {
    const int32_t stored_key = hash_keys[slot];
    if (stored_key == key) {
      const int32_t union_slot = hash_values[slot];
      remap[input_index] = union_slot;
      if (union_slot == input_index) {
        union_indices[union_slot] = key;
        atomicAdd(union_count, 1);
      }
      return;
    }
    if (stored_key == -1) {
      break;
    }
    slot = (slot + 1) & hash_mask;
  }
  remap[input_index] = -1;
}

__global__ void scatter_records_kernel(
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ local_slots,
    const int64_t* __restrict__ peer_payload_ptrs,
    int64_t topk,
    int64_t destination_stride,
    int64_t record_bytes,
    int world_size) {
  constexpr int64_t kVectorBytes = sizeof(uint4);
  const int64_t vectors_per_record = record_bytes / kVectorBytes;
  const int64_t vector_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_vectors = topk * vectors_per_record;
  if (vector_index >= total_vectors) {
    return;
  }

  const int64_t selected = vector_index / vectors_per_record;
  const int destination = static_cast<int>(blockIdx.y);
  if (destination >= world_size) {
    return;
  }
  const int32_t local_slot = local_slots[
      static_cast<int64_t>(destination) * destination_stride + selected];
  if (local_slot < 0) {
    return;
  }
  const int64_t record_vector = vector_index % vectors_per_record;
  const int64_t byte_offset = record_vector * kVectorBytes;
  const auto* source = reinterpret_cast<const uint4*>(
      kv_cache + static_cast<int64_t>(local_slot) * record_bytes + byte_offset);
  const uint4 value = *source;

  // Give each destination its own grid plane. Every plane reads that
  // destination's owner-local slot map and writes final sparse-pool offsets.
  auto* output = reinterpret_cast<uint4*>(
      static_cast<uintptr_t>(peer_payload_ptrs[destination])
      + selected * record_bytes + byte_offset);
  *output = value;
}

__global__ void publish_records_kernel(
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ local_slots,
    uint8_t* __restrict__ local_payload,
    int64_t topk,
    int64_t record_bytes) {
  constexpr int64_t kVectorBytes = sizeof(uint4);
  const int64_t vectors_per_record = record_bytes / kVectorBytes;
  const int64_t vector_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_vectors = topk * vectors_per_record;
  if (vector_index >= total_vectors) {
    return;
  }
  const int64_t selected = vector_index / vectors_per_record;
  const int32_t local_slot = local_slots[selected];
  if (local_slot < 0) {
    return;
  }
  const int64_t byte_offset =
      (vector_index % vectors_per_record) * kVectorBytes;
  const auto* source = reinterpret_cast<const uint4*>(
      kv_cache + static_cast<int64_t>(local_slot) * record_bytes + byte_offset);
  auto* destination = reinterpret_cast<uint4*>(
      local_payload + selected * record_bytes + byte_offset);
  *destination = *source;
}

__global__ void pull_records_kernel(
    const int32_t* __restrict__ global_indices,
    const int64_t* __restrict__ peer_payload_ptrs,
    uint8_t* __restrict__ output,
    int64_t topk,
    int64_t record_bytes,
    int world_size,
    int interleave) {
  constexpr int64_t kVectorBytes = sizeof(uint4);
  const int64_t vectors_per_record = record_bytes / kVectorBytes;
  const int64_t vector_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_vectors = topk * vectors_per_record;
  if (vector_index >= total_vectors) {
    return;
  }
  const int64_t selected = vector_index / vectors_per_record;
  const int64_t byte_offset =
      (vector_index % vectors_per_record) * kVectorBytes;
  const int32_t token = global_indices[selected];
  uint4 value = make_uint4(0, 0, 0, 0);
  if (token >= 0) {
    const int owner = (token / interleave) % world_size;
    const auto* source = reinterpret_cast<const uint4*>(
        static_cast<uintptr_t>(peer_payload_ptrs[owner])
        + selected * record_bytes + byte_offset);
    value = *source;
  }
  auto* destination = reinterpret_cast<uint4*>(
      output + selected * record_bytes + byte_offset);
  *destination = value;
}

__global__ void pack_compact_records_kernel(
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ local_slots,
    uint8_t* __restrict__ primary,
    uint8_t* __restrict__ overflow,
    int64_t topk,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset) {
  constexpr int kWarpSize = 32;
  const int64_t thread_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t selected = thread_index / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  if (selected >= topk) {
    return;
  }
  const int32_t local_slot = local_slots[selected];
  if (local_slot < 0) {
    return;
  }

  int32_t ordinal = 0;
  if (lane == 0) {
    ordinal = atomicAdd(reinterpret_cast<int32_t*>(primary), 1);
  }
  ordinal = __shfl_sync(0xffffffff, ordinal, 0);
  const bool use_primary = ordinal < primary_capacity;
  const int64_t compact_index =
      use_primary ? ordinal : ordinal - primary_capacity;
  uint8_t* payload = use_primary ? primary : overflow;
  const int64_t positions_offset =
      use_primary ? primary_positions_offset : overflow_positions_offset;
  const int64_t records_offset =
      use_primary ? primary_records_offset : overflow_records_offset;
  if (lane == 0) {
    reinterpret_cast<int32_t*>(payload + positions_offset)[compact_index] =
        static_cast<int32_t>(selected);
  }

  const int64_t vectors_per_record = record_bytes / sizeof(uint4);
  if (lane < vectors_per_record) {
    const auto* source = reinterpret_cast<const uint4*>(
        kv_cache + static_cast<int64_t>(local_slot) * record_bytes);
    auto* destination = reinterpret_cast<uint4*>(
        payload + records_offset + compact_index * record_bytes);
    destination[lane] = source[lane];
  }
}

__global__ void unpack_compact_records_kernel(
    const uint8_t* __restrict__ primary_base,
    int64_t primary_stride,
    const int64_t* __restrict__ peer_overflow_ptrs,
    uint8_t* __restrict__ output,
    int64_t topk,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset,
    int world_size) {
  constexpr int kWarpSize = 32;
  const int64_t thread_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t compact_record = thread_index / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int owner = static_cast<int>(compact_record / topk);
  const int64_t ordinal = compact_record % topk;
  if (owner >= world_size) {
    return;
  }

  const uint8_t* primary = primary_base + owner * primary_stride;
  const int32_t total_records = *reinterpret_cast<const int32_t*>(primary);
  if (ordinal >= total_records) {
    return;
  }
  const bool use_primary = ordinal < primary_capacity;
  const int64_t compact_index =
      use_primary ? ordinal : ordinal - primary_capacity;
  const uint8_t* payload =
      use_primary
      ? primary
      : reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(peer_overflow_ptrs[owner]));
  const int64_t positions_offset =
      use_primary ? primary_positions_offset : overflow_positions_offset;
  const int64_t records_offset =
      use_primary ? primary_records_offset : overflow_records_offset;
  const int32_t selected =
      reinterpret_cast<const int32_t*>(payload + positions_offset)[compact_index];

  const int64_t vectors_per_record = record_bytes / sizeof(uint4);
  if (lane < vectors_per_record) {
    const auto* source = reinterpret_cast<const uint4*>(
        payload + records_offset + compact_index * record_bytes);
    auto* destination = reinterpret_cast<uint4*>(
        output + static_cast<int64_t>(selected) * record_bytes);
    destination[lane] = source[lane];
  }
}

__global__ void byte_or_kernel(
    uint4* __restrict__ output,
    const uint4* __restrict__ left,
    const uint4* __restrict__ right,
    int64_t vectors) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= vectors) {
    return;
  }
  const uint4 a = left[index];
  const uint4 b = right[index];
  output[index] = make_uint4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
}

void validate_common(
    const torch::Tensor& kv_cache,
    const torch::Tensor& local_slots,
    int64_t record_bytes) {
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be CUDA");
  TORCH_CHECK(local_slots.is_cuda(), "local_slots must be CUDA");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8, "kv_cache must be uint8");
  TORCH_CHECK(
      local_slots.scalar_type() == torch::kInt32, "local_slots must be int32");
  TORCH_CHECK(kv_cache.is_contiguous(), "kv_cache must be contiguous");
  TORCH_CHECK(local_slots.is_contiguous(), "local_slots must be contiguous");
  TORCH_CHECK(local_slots.dim() == 2, "local_slots must be rank 2");
  TORCH_CHECK(record_bytes > 0, "record_bytes must be positive");
  TORCH_CHECK(
      record_bytes % static_cast<int64_t>(sizeof(uint4)) == 0,
      "record_bytes must be divisible by 16");
  TORCH_CHECK(
      kv_cache.size(-1) == record_bytes,
      "kv_cache record width does not match record_bytes");
}

void build_union_remap(
    torch::Tensor indices,
    torch::Tensor union_indices,
    torch::Tensor remap,
    torch::Tensor union_count,
    torch::Tensor hash_keys,
    torch::Tensor hash_values) {
  TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
  TORCH_CHECK(union_indices.is_cuda(), "union_indices must be CUDA");
  TORCH_CHECK(remap.is_cuda(), "remap must be CUDA");
  TORCH_CHECK(union_count.is_cuda(), "union_count must be CUDA");
  TORCH_CHECK(hash_keys.is_cuda(), "hash_keys must be CUDA");
  TORCH_CHECK(hash_values.is_cuda(), "hash_values must be CUDA");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(
      union_indices.scalar_type() == torch::kInt32,
      "union_indices must be int32");
  TORCH_CHECK(remap.scalar_type() == torch::kInt32, "remap must be int32");
  TORCH_CHECK(
      union_count.scalar_type() == torch::kInt32, "union_count must be int32");
  TORCH_CHECK(hash_keys.scalar_type() == torch::kInt32, "hash_keys must be int32");
  TORCH_CHECK(
      hash_values.scalar_type() == torch::kInt32, "hash_values must be int32");
  TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(union_indices.is_contiguous(), "union_indices must be contiguous");
  TORCH_CHECK(remap.is_contiguous(), "remap must be contiguous");
  TORCH_CHECK(union_count.is_contiguous(), "union_count must be contiguous");
  TORCH_CHECK(hash_keys.is_contiguous(), "hash_keys must be contiguous");
  TORCH_CHECK(hash_values.is_contiguous(), "hash_values must be contiguous");
  TORCH_CHECK(indices.dim() == 2, "indices must be rank 2");
  TORCH_CHECK(remap.sizes() == indices.sizes(), "remap shape must match indices");
  TORCH_CHECK(union_indices.dim() == 1, "union_indices must be rank 1");
  TORCH_CHECK(union_count.numel() == 1, "union_count must contain one element");
  TORCH_CHECK(hash_keys.dim() == 1, "hash_keys must be rank 1");
  TORCH_CHECK(hash_values.sizes() == hash_keys.sizes(), "hash arrays must match");
  const int64_t hash_capacity = hash_keys.numel();
  TORCH_CHECK(hash_capacity > 0, "hash capacity must be positive");
  TORCH_CHECK(
      (hash_capacity & (hash_capacity - 1)) == 0,
      "hash capacity must be a power of two");
  TORCH_CHECK(
      hash_capacity >= 2 * indices.numel(),
      "hash capacity must be at least twice the input size");
  TORCH_CHECK(
      union_indices.numel() >= indices.numel(),
      "deterministic union needs one slot per flattened input entry");

  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_CUDA_CHECK(cudaMemsetAsync(
      union_indices.data_ptr<int32_t>(),
      0xff,
      union_indices.numel() * sizeof(int32_t),
      stream));
  AT_CUDA_CHECK(cudaMemsetAsync(
      remap.data_ptr<int32_t>(),
      0xff,
      remap.numel() * sizeof(int32_t),
      stream));
  AT_CUDA_CHECK(cudaMemsetAsync(
      union_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  AT_CUDA_CHECK(cudaMemsetAsync(
      hash_keys.data_ptr<int32_t>(),
      0xff,
      hash_keys.numel() * sizeof(int32_t),
      stream));
  AT_CUDA_CHECK(cudaMemsetAsync(
      hash_values.data_ptr<int32_t>(),
      0x7f,
      hash_values.numel() * sizeof(int32_t),
      stream));

  constexpr int threads = 256;
  const int blocks = static_cast<int>(
      (indices.numel() + threads - 1) / threads);
  build_union_hash_kernel<<<blocks, threads, 0, stream>>>(
      indices.data_ptr<int32_t>(),
      hash_keys.data_ptr<int32_t>(),
      hash_values.data_ptr<int32_t>(),
      indices.numel(),
      hash_capacity);
  AT_CUDA_CHECK(cudaGetLastError());
  finalize_union_remap_kernel<<<blocks, threads, 0, stream>>>(
      indices.data_ptr<int32_t>(),
      union_indices.data_ptr<int32_t>(),
      remap.data_ptr<int32_t>(),
      union_count.data_ptr<int32_t>(),
      hash_keys.data_ptr<int32_t>(),
      hash_values.data_ptr<int32_t>(),
      indices.numel(),
      hash_capacity);
  AT_CUDA_CHECK(cudaGetLastError());
}

void scatter_records(
    torch::Tensor kv_cache,
    torch::Tensor local_slots,
    torch::Tensor peer_payload_ptrs,
    int64_t record_bytes) {
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be CUDA");
  TORCH_CHECK(local_slots.is_cuda(), "local_slots must be CUDA");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8, "kv_cache must be uint8");
  TORCH_CHECK(
      local_slots.scalar_type() == torch::kInt32, "local_slots must be int32");
  TORCH_CHECK(kv_cache.is_contiguous(), "kv_cache must be contiguous");
  TORCH_CHECK(local_slots.dim() == 3, "local_slots must be rank 3");
  TORCH_CHECK(local_slots.stride(2) == 1, "slot dimension must be contiguous");
  TORCH_CHECK(
      local_slots.stride(1) == local_slots.size(2),
      "request/slot dimensions must be packed");
  TORCH_CHECK(record_bytes > 0, "record_bytes must be positive");
  TORCH_CHECK(
      record_bytes % static_cast<int64_t>(sizeof(uint4)) == 0,
      "record_bytes must be divisible by 16");
  TORCH_CHECK(
      kv_cache.size(-1) == record_bytes,
      "kv_cache record width does not match record_bytes");
  TORCH_CHECK(peer_payload_ptrs.is_cuda(), "peer pointers must be CUDA");
  TORCH_CHECK(
      peer_payload_ptrs.scalar_type() == torch::kInt64,
      "peer pointers must be int64");
  TORCH_CHECK(
      peer_payload_ptrs.is_contiguous(), "peer pointers must be contiguous");
  TORCH_CHECK(
      local_slots.size(0) == peer_payload_ptrs.numel(),
      "scatter requires one local-slot row per destination");

  const int64_t topk = local_slots.size(1) * local_slots.size(2);
  const int64_t destination_stride = local_slots.stride(0);
  const int64_t vectors = topk * record_bytes / sizeof(uint4);
  constexpr int threads = 256;
  const int blocks = static_cast<int>((vectors + threads - 1) / threads);
  const dim3 grid(blocks, static_cast<unsigned int>(peer_payload_ptrs.numel()));
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  scatter_records_kernel<<<grid, threads, 0, stream>>>(
      kv_cache.data_ptr<uint8_t>(),
      local_slots.data_ptr<int32_t>(),
      peer_payload_ptrs.data_ptr<int64_t>(),
      topk,
      destination_stride,
      record_bytes,
      static_cast<int>(peer_payload_ptrs.numel()));
  AT_CUDA_CHECK(cudaGetLastError());
}

void barrier_all_peers(
    torch::Tensor publish_flag_ptrs,
    torch::Tensor wait_flag_ptrs,
    torch::Tensor send_counters,
    torch::Tensor wait_counters,
    int64_t timeout_cycles) {
  TORCH_CHECK(publish_flag_ptrs.is_cuda(), "publish pointers must be CUDA");
  TORCH_CHECK(wait_flag_ptrs.is_cuda(), "wait pointers must be CUDA");
  TORCH_CHECK(send_counters.is_cuda(), "send counters must be CUDA");
  TORCH_CHECK(wait_counters.is_cuda(), "wait counters must be CUDA");
  TORCH_CHECK(
      publish_flag_ptrs.scalar_type() == torch::kInt64,
      "publish pointers must be int64");
  TORCH_CHECK(
      wait_flag_ptrs.scalar_type() == torch::kInt64,
      "wait pointers must be int64");
  TORCH_CHECK(
      send_counters.scalar_type() == torch::kInt32,
      "send counters must be int32");
  TORCH_CHECK(
      wait_counters.scalar_type() == torch::kInt32,
      "wait counters must be int32");
  TORCH_CHECK(publish_flag_ptrs.is_contiguous(), "publish pointers must be contiguous");
  TORCH_CHECK(wait_flag_ptrs.is_contiguous(), "wait pointers must be contiguous");
  TORCH_CHECK(send_counters.is_contiguous(), "send counters must be contiguous");
  TORCH_CHECK(wait_counters.is_contiguous(), "wait counters must be contiguous");
  const int64_t count = publish_flag_ptrs.numel();
  TORCH_CHECK(count > 0 && count <= 32, "peer count must be in [1, 32]");
  TORCH_CHECK(wait_flag_ptrs.numel() == count, "wait pointer count mismatch");
  TORCH_CHECK(send_counters.numel() == count, "send counter count mismatch");
  TORCH_CHECK(wait_counters.numel() == count, "wait counter count mismatch");
  TORCH_CHECK(timeout_cycles > 0, "barrier timeout cycles must be positive");

  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  barrier_all_peers_kernel<<<1, 32, 0, stream>>>(
      publish_flag_ptrs.data_ptr<int64_t>(),
      wait_flag_ptrs.data_ptr<int64_t>(),
      send_counters.data_ptr<int32_t>(),
      wait_counters.data_ptr<int32_t>(),
      static_cast<int>(count),
      static_cast<uint64_t>(timeout_cycles));
  AT_CUDA_CHECK(cudaGetLastError());
}

void publish_records(
    torch::Tensor kv_cache,
    torch::Tensor local_slots,
    int64_t local_payload_ptr,
    int64_t record_bytes) {
  validate_common(kv_cache, local_slots, record_bytes);
  TORCH_CHECK(local_slots.size(0) == 1, "publish requires one slot row");
  TORCH_CHECK(local_payload_ptr != 0, "local payload pointer must be nonzero");
  const int64_t topk = local_slots.numel();
  const int64_t vectors = topk * record_bytes / sizeof(uint4);
  constexpr int threads = 256;
  const int blocks = static_cast<int>((vectors + threads - 1) / threads);
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  publish_records_kernel<<<blocks, threads, 0, stream>>>(
      kv_cache.data_ptr<uint8_t>(),
      local_slots.data_ptr<int32_t>(),
      reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(local_payload_ptr)),
      topk,
      record_bytes);
  AT_CUDA_CHECK(cudaGetLastError());
}

void pull_records(
    torch::Tensor global_indices,
    torch::Tensor peer_payload_ptrs,
    torch::Tensor output,
    int64_t record_bytes,
    int64_t interleave) {
  TORCH_CHECK(global_indices.is_cuda(), "global_indices must be CUDA");
  TORCH_CHECK(peer_payload_ptrs.is_cuda(), "peer pointers must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(
      global_indices.scalar_type() == torch::kInt32,
      "global_indices must be int32");
  TORCH_CHECK(
      peer_payload_ptrs.scalar_type() == torch::kInt64,
      "peer pointers must be int64");
  TORCH_CHECK(output.scalar_type() == torch::kUInt8, "output must be uint8");
  TORCH_CHECK(global_indices.is_contiguous(), "global_indices must be contiguous");
  TORCH_CHECK(
      peer_payload_ptrs.is_contiguous(), "peer pointers must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(global_indices.dim() == 2, "global_indices must be rank 2");
  TORCH_CHECK(global_indices.size(0) == 1, "MTP0 transport requires one row");
  TORCH_CHECK(interleave > 0, "interleave must be positive");
  TORCH_CHECK(
      record_bytes % static_cast<int64_t>(sizeof(uint4)) == 0,
      "record_bytes must be divisible by 16");
  const int64_t topk = global_indices.numel();
  TORCH_CHECK(
      output.numel() >= topk * record_bytes, "output is smaller than top-k");
  const int64_t vectors = topk * record_bytes / sizeof(uint4);
  constexpr int threads = 256;
  const int blocks = static_cast<int>((vectors + threads - 1) / threads);
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  pull_records_kernel<<<blocks, threads, 0, stream>>>(
      global_indices.data_ptr<int32_t>(),
      peer_payload_ptrs.data_ptr<int64_t>(),
      output.data_ptr<uint8_t>(),
      topk,
      record_bytes,
      static_cast<int>(peer_payload_ptrs.numel()),
      static_cast<int>(interleave));
  AT_CUDA_CHECK(cudaGetLastError());
}

void pack_compact_records(
    torch::Tensor kv_cache,
    torch::Tensor local_slots,
    int64_t primary_ptr,
    int64_t overflow_ptr,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset) {
  validate_common(kv_cache, local_slots, record_bytes);
  TORCH_CHECK(primary_ptr != 0, "primary pointer must be nonzero");
  TORCH_CHECK(overflow_ptr != 0, "overflow pointer must be nonzero");
  TORCH_CHECK(primary_capacity > 0, "primary capacity must be positive");
  TORCH_CHECK(
      primary_capacity <= local_slots.numel(),
      "primary capacity cannot exceed top-k");
  const int64_t topk = local_slots.numel();
  constexpr int threads = 256;
  constexpr int warp_size = 32;
  const int blocks = static_cast<int>(
      (topk * warp_size + threads - 1) / threads);
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_CUDA_CHECK(cudaMemsetAsync(
      reinterpret_cast<void*>(static_cast<uintptr_t>(primary_ptr)),
      0,
      sizeof(int32_t),
      stream));
  pack_compact_records_kernel<<<blocks, threads, 0, stream>>>(
      kv_cache.data_ptr<uint8_t>(),
      local_slots.data_ptr<int32_t>(),
      reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(primary_ptr)),
      reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(overflow_ptr)),
      topk,
      record_bytes,
      primary_capacity,
      primary_positions_offset,
      primary_records_offset,
      overflow_positions_offset,
      overflow_records_offset);
  AT_CUDA_CHECK(cudaGetLastError());
}

void unpack_compact_records(
    int64_t primary_base_ptr,
    int64_t primary_stride,
    torch::Tensor peer_overflow_ptrs,
    torch::Tensor output,
    int64_t topk,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset) {
  TORCH_CHECK(primary_base_ptr != 0, "primary base pointer must be nonzero");
  TORCH_CHECK(primary_stride > 0, "primary stride must be positive");
  TORCH_CHECK(peer_overflow_ptrs.is_cuda(), "peer overflow pointers must be CUDA");
  TORCH_CHECK(
      peer_overflow_ptrs.scalar_type() == torch::kInt64,
      "peer overflow pointers must be int64");
  TORCH_CHECK(
      peer_overflow_ptrs.is_contiguous(),
      "peer overflow pointers must be contiguous");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(output.scalar_type() == torch::kUInt8, "output must be uint8");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(topk > 0, "top-k must be positive");
  TORCH_CHECK(record_bytes > 0, "record bytes must be positive");
  TORCH_CHECK(
      record_bytes % static_cast<int64_t>(sizeof(uint4)) == 0,
      "record bytes must be divisible by 16");
  TORCH_CHECK(primary_capacity > 0, "primary capacity must be positive");
  TORCH_CHECK(primary_capacity <= topk, "primary capacity cannot exceed top-k");
  TORCH_CHECK(
      output.numel() >= topk * record_bytes,
      "output is smaller than the sparse CKV payload");
  const int world_size = static_cast<int>(peer_overflow_ptrs.numel());
  constexpr int threads = 256;
  constexpr int warp_size = 32;
  const int64_t warps = static_cast<int64_t>(world_size) * topk;
  const int blocks = static_cast<int>(
      (warps * warp_size + threads - 1) / threads);
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  unpack_compact_records_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(
          static_cast<uintptr_t>(primary_base_ptr)),
      primary_stride,
      peer_overflow_ptrs.data_ptr<int64_t>(),
      output.data_ptr<uint8_t>(),
      topk,
      record_bytes,
      primary_capacity,
      primary_positions_offset,
      primary_records_offset,
      overflow_positions_offset,
      overflow_records_offset,
      world_size);
  AT_CUDA_CHECK(cudaGetLastError());
}

void byte_or(
    int64_t output_ptr,
    int64_t left_ptr,
    int64_t right_ptr,
    int64_t bytes) {
  TORCH_CHECK(output_ptr != 0, "output pointer must be nonzero");
  TORCH_CHECK(left_ptr != 0, "left pointer must be nonzero");
  TORCH_CHECK(right_ptr != 0, "right pointer must be nonzero");
  TORCH_CHECK(bytes > 0, "bytes must be positive");
  TORCH_CHECK(bytes % sizeof(uint4) == 0, "bytes must be divisible by 16");
  const int64_t vectors = bytes / sizeof(uint4);
  constexpr int threads = 256;
  const int blocks = static_cast<int>((vectors + threads - 1) / threads);
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  byte_or_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<uint4*>(static_cast<uintptr_t>(output_ptr)),
      reinterpret_cast<const uint4*>(static_cast<uintptr_t>(left_ptr)),
      reinterpret_cast<const uint4*>(static_cast<uintptr_t>(right_ptr)),
      vectors);
  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "build_union_remap",
      &build_union_remap,
      "Build one compact MTP CKV union and a per-row remap table");
  module.def(
      "scatter_records",
      &scatter_records,
      "Write rank-owned native CKV records directly into every peer slab");
  module.def(
      "barrier_all_peers",
      &barrier_all_peers,
      "Publish and wait for every sparse-CKV peer in one CUDA launch");
  module.def(
      "publish_records",
      &publish_records,
      "Publish rank-owned native CKV records into the local IPC slab");
  module.def(
      "pull_records",
      &pull_records,
      "Pull each selected native CKV record from its owning peer slab");
  module.def(
      "pack_compact_records",
      &pack_compact_records,
      "Compact rank-owned native CKV records into primary and overflow slabs");
  module.def(
      "unpack_compact_records",
      &unpack_compact_records,
      "Reconstruct native CKV records from direct-copy and overflow slabs");
  module.def(
      "byte_or",
      &byte_or,
      "Bitwise-OR two byte payloads into an output payload");
}
