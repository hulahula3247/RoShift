#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

namespace {

constexpr int kBlockSize = 64;
constexpr int kKeep = 32;

void check_half_cuda(const torch::Tensor & tensor, const char * name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be float16");
}

template <int LoadPolicy>
__device__ __forceinline__ half2 load_weight_half2(const half * pointer) {
    if constexpr (LoadPolicy == 1) {
        const uint32_t packed =
            __ldcg(reinterpret_cast<const uint32_t *>(pointer));
        return *reinterpret_cast<const half2 *>(&packed);
    } else if constexpr (LoadPolicy == 2) {
        const uint32_t packed =
            __ldcs(reinterpret_cast<const uint32_t *>(pointer));
        return *reinterpret_cast<const half2 *>(&packed);
    } else {
        return *reinterpret_cast<const half2 *>(pointer);
    }
}

__global__ void select_input_k32_and_zero(
    half * output,
    uint8_t * local_indices,
    const half * score_input,
    const half * score_scale,
    int output_dim) {
    __shared__ uint32_t keys[kBlockSize];
    const int local = threadIdx.x;
    const int index = blockIdx.x * kBlockSize + local;

    for (
        int out = blockIdx.x * kBlockSize + local;
        out < output_dim;
        out += gridDim.x * kBlockSize) {
        output[out] = __float2half(0.0f);
    }

    const float score =
        fabsf(__half2float(score_input[index])) *
        __half2float(score_scale[index]);
    const uint32_t key = __float_as_uint(score) >> 16;
    keys[local] = key;
    __syncthreads();

    int rank = 0;
#pragma unroll
    for (int other = 0; other < kBlockSize; ++other) {
        rank +=
            keys[other] > key || (keys[other] == key && other < local);
    }
    if (rank < kKeep) {
        local_indices[blockIdx.x * kKeep + rank] =
            static_cast<uint8_t>(local);
    }
}

__global__ void select_permuted_input_k32_and_zero(
    half * output,
    uint16_t * global_indices,
    half * selected_values,
    const half * value,
    const half * score_input,
    const half * score_scale,
    const int32_t * permutation,
    int output_dim) {
    __shared__ uint32_t keys[kBlockSize];
    const int local = threadIdx.x;
    const int permuted_index = blockIdx.x * kBlockSize + local;
    const int original_index = permutation[permuted_index];

    for (
        int out = blockIdx.x * kBlockSize + local;
        out < output_dim;
        out += gridDim.x * kBlockSize) {
        output[out] = __float2half(0.0f);
    }

    const float score =
        fabsf(__half2float(score_input[original_index])) *
        __half2float(score_scale[original_index]);
    const uint32_t key = __float_as_uint(score);
    keys[local] = key;
    __syncthreads();

    int rank = 0;
#pragma unroll
    for (int other = 0; other < kBlockSize; ++other) {
        rank +=
            keys[other] > key || (keys[other] == key && other < local);
    }
    if (rank < kKeep) {
        const int selected = blockIdx.x * kKeep + rank;
        global_indices[selected] = static_cast<uint16_t>(original_index);
        selected_values[selected] = value[original_index];
    }
}

__global__ void select_output_k32_and_zero(
    half * output,
    uint8_t * local_indices,
    const half * output_score,
    const half * score_scale) {
    __shared__ uint32_t keys[kBlockSize];
    const int local = threadIdx.x;
    const int index = blockIdx.x * kBlockSize + local;
    output[index] = __float2half(0.0f);

    const float score =
        fabsf(__half2float(output_score[index])) *
        __half2float(score_scale[index]);
    const uint32_t key = __float_as_uint(score) >> 16;
    keys[local] = key;
    __syncthreads();

    int rank = 0;
#pragma unroll
    for (int other = 0; other < kBlockSize; ++other) {
        rank +=
            keys[other] > key || (keys[other] == key && other < local);
    }
    if (rank < kKeep) {
        local_indices[blockIdx.x * kKeep + rank] =
            static_cast<uint8_t>(local);
    }
}

template <int InputChunk, int OutputThreads>
__global__ void permuted_input_sparse_gemv_kernel(
    half * output,
    const uint16_t * global_indices,
    const half * selected_values,
    const half * weight_column_major,
    int output_dim) {
    constexpr int blocks_per_chunk = InputChunk / kBlockSize;
    const int input_blocks = 4096 / kBlockSize;
    const int input_groups = input_blocks / blocks_per_chunk;
    const int output_pairs = output_dim / 2;
    const int output_tiles =
        (output_pairs + OutputThreads - 1) / OutputThreads;
    const int input_group = blockIdx.x / output_tiles;
    const int output_tile = blockIdx.x - input_group * output_tiles;
    const int output_pair =
        output_tile * OutputThreads + threadIdx.x;
    if (input_group >= input_groups || output_pair >= output_pairs) {
        return;
    }

    const int first_block = input_group * blocks_per_chunk;
    const int last_block = first_block + blocks_per_chunk;
    const int out = output_pair * 2;
    float2 accum = make_float2(0.0f, 0.0f);
    for (int input_block = first_block;
         input_block < last_block;
         ++input_block) {
#pragma unroll
        for (int rank = 0; rank < kKeep; ++rank) {
            const int selected = input_block * kKeep + rank;
            const int index =
                static_cast<int>(global_indices[selected]);
            const float x = __half2float(selected_values[selected]);
            const half2 weights = *reinterpret_cast<const half2 *>(
                weight_column_major + index * output_dim + out);
            const float2 weights_f = __half22float2(weights);
            accum.x = fmaf(x, weights_f.x, accum.x);
            accum.y = fmaf(x, weights_f.y, accum.y);
        }
    }
    atomicAdd(
        reinterpret_cast<half2 *>(output) + output_pair,
        __floats2half2_rn(accum.x, accum.y));
}

template <int InputChunk, int OutputThreads, int LoadPolicy>
__global__ void permuted_input_sparse_warp_kernel(
    half * output,
    const uint16_t * global_indices,
    const half * selected_values,
    const half * weight_column_major,
    int output_dim) {
    constexpr int blocks_per_chunk = InputChunk / kBlockSize;
    const int input_blocks = 4096 / kBlockSize;
    const int input_groups = input_blocks / blocks_per_chunk;
    const int output_pairs = output_dim / 2;
    const int output_tiles =
        (output_pairs + OutputThreads - 1) / OutputThreads;
    const int input_group = blockIdx.x / output_tiles;
    const int output_tile = blockIdx.x - input_group * output_tiles;
    const int output_pair =
        output_tile * OutputThreads + threadIdx.x;
    if (input_group >= input_groups || output_pair >= output_pairs) {
        return;
    }

    const int lane = threadIdx.x & 31;
    const unsigned warp_mask = __activemask();
    const int first_block = input_group * blocks_per_chunk;
    const int last_block = first_block + blocks_per_chunk;
    const int out = output_pair * 2;
    float2 accum = make_float2(0.0f, 0.0f);
    for (int input_block = first_block;
         input_block < last_block;
         ++input_block) {
        const int selected = input_block * kKeep + lane;
        const int index_lane =
            static_cast<int>(global_indices[selected]);
        const float value_lane =
            __half2float(selected_values[selected]);
#pragma unroll
        for (int source_lane = 0;
             source_lane < kKeep;
             ++source_lane) {
            const int index = __shfl_sync(
                warp_mask, index_lane, source_lane);
            const float x = __shfl_sync(
                warp_mask, value_lane, source_lane);
            const half2 weights = load_weight_half2<LoadPolicy>(
                weight_column_major + index * output_dim + out);
            const float2 weights_f = __half22float2(weights);
            accum.x = fmaf(x, weights_f.x, accum.x);
            accum.y = fmaf(x, weights_f.y, accum.y);
        }
    }
    atomicAdd(
        reinterpret_cast<half2 *>(output) + output_pair,
        __floats2half2_rn(accum.x, accum.y));
}

// O projection: each block owns one selected B64 input block and one
// contiguous tile of output pairs.
__global__ void input_sparse_o_kernel(
    half * output,
    const uint8_t * local_indices,
    const half * weight_column_major,
    const half * value,
    int output_dim) {
    constexpr int output_threads = 64;
    const int output_pairs = output_dim / 2;
    const int output_tiles =
        (output_pairs + output_threads - 1) / output_threads;
    const int input_block = blockIdx.x / output_tiles;
    const int output_tile = blockIdx.x - input_block * output_tiles;
    const int output_pair =
        output_tile * output_threads + threadIdx.x;
    if (output_pair >= output_pairs) {
        return;
    }

    const int out = output_pair * 2;
    float2 accum = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int rank = 0; rank < kKeep; ++rank) {
        const int local =
            static_cast<int>(local_indices[input_block * kKeep + rank]);
        const int index = input_block * kBlockSize + local;
        const float x = __half2float(value[index]);
        const half2 weights = *reinterpret_cast<const half2 *>(
            weight_column_major + index * output_dim + out);
        const float2 weights_f = __half22float2(weights);
        accum.x = fmaf(x, weights_f.x, accum.x);
        accum.y = fmaf(x, weights_f.y, accum.y);
    }
    atomicAdd(
        reinterpret_cast<half2 *>(output) + output_pair,
        __floats2half2_rn(accum.x, accum.y));
}

template <int Threads, int LoadPolicy>
__global__ void output_sparse_up_kernel(
    half * output,
    const uint8_t * local_indices,
    const half * weight_row_major,
    const half * input,
    int input_dim) {
    using BlockReduce = cub::BlockReduce<float, Threads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;

    const int task = blockIdx.x;
    const int output_block = task / kKeep;
    const int rank = task - output_block * kKeep;
    const int local =
        static_cast<int>(local_indices[output_block * kKeep + rank]);
    const int row = output_block * kBlockSize + local;
    const int pairs = input_dim / 2;

    float sum = 0.0f;
    for (int pair = threadIdx.x; pair < pairs; pair += Threads) {
        const half2 x2 =
            *reinterpret_cast<const half2 *>(input + pair * 2);
        const half2 w2 = load_weight_half2<LoadPolicy>(
            weight_row_major + row * input_dim + pair * 2);
        const float2 x_f = __half22float2(x2);
        const float2 w_f = __half22float2(w2);
        sum = fmaf(x_f.x, w_f.x, sum);
        sum = fmaf(x_f.y, w_f.y, sum);
    }
    sum = BlockReduce(reduce_storage).Sum(sum);
    if (threadIdx.x == 0) {
        output[row] = __float2half(sum);
    }
}

// Down projection: four selected B64 blocks are processed per input tile.
// Every warp broadcasts one selected activation at a time so weight loads
// remain contiguous across output pairs.
template <int LoadPolicy>
__global__ void input_sparse_down_kernel(
    half * output,
    const uint8_t * local_indices,
    const half * weight_column_major,
    const half * value,
    int input_blocks,
    int output_dim) {
    constexpr int input_blocks_per_tile = 4;
    constexpr int output_threads = 128;
    const int output_pairs = output_dim / 2;
    const int output_tiles =
        (output_pairs + output_threads - 1) / output_threads;
    const int input_tile = blockIdx.x / output_tiles;
    const int output_tile = blockIdx.x - input_tile * output_tiles;
    const int output_pair =
        output_tile * output_threads + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const unsigned warp_mask = __activemask();
    const int first_block = input_tile * input_blocks_per_tile;
    const int last_block =
        min(first_block + input_blocks_per_tile, input_blocks);
    if (output_pair >= output_pairs) {
        return;
    }

    float2 accum = make_float2(0.0f, 0.0f);
    const int out = output_pair * 2;
    for (int input_block = first_block;
         input_block < last_block;
         ++input_block) {
        const int local_lane = static_cast<int>(
            local_indices[input_block * kKeep + lane]);
        const int index_lane =
            input_block * kBlockSize + local_lane;
        const float value_lane = __half2float(value[index_lane]);
#pragma unroll
        for (int source_lane = 0; source_lane < kKeep; ++source_lane) {
            const int index =
                __shfl_sync(warp_mask, index_lane, source_lane);
            const float x =
                __shfl_sync(warp_mask, value_lane, source_lane);
            const half2 weights = load_weight_half2<LoadPolicy>(
                weight_column_major + index * output_dim + out);
            const float2 weights_f = __half22float2(weights);
            accum.x = fmaf(x, weights_f.x, accum.x);
            accum.y = fmaf(x, weights_f.y, accum.y);
        }
    }
    atomicAdd(
        reinterpret_cast<half2 *>(output) + output_pair,
        __floats2half2_rn(accum.x, accum.y));
}

}  // namespace

torch::Tensor pairwise_input_cuda(
    torch::Tensor value,
    torch::Tensor weight,
    torch::Tensor score_input,
    torch::Tensor score_scale,
    int64_t block_size,
    int64_t radix_bits) {
    check_half_cuda(value, "value");
    check_half_cuda(weight, "weight");
    check_half_cuda(score_input, "score_input");
    check_half_cuda(score_scale, "score_scale");
    TORCH_CHECK(block_size == kBlockSize, "block_size must be 64");
    TORCH_CHECK(radix_bits == 16, "radix_bits must be 16");
    TORCH_CHECK(value.numel() == 4096, "O input must have D=4096");
    TORCH_CHECK(score_input.numel() == 4096, "score input must have D=4096");
    TORCH_CHECK(score_scale.numel() == 4096, "score scale must have D=4096");
    TORCH_CHECK(
        weight.sizes() == torch::IntArrayRef({4096, 4096}),
        "O weight must be [4096, 4096]");
    TORCH_CHECK(weight.stride(1) > 1, "O weight must be column major");

    constexpr int input_blocks = 4096 / kBlockSize;
    constexpr int output_dim = 4096;
    constexpr int output_tiles = (output_dim / 2) / 64;
    auto output = torch::empty({1, 1, output_dim}, value.options());
    auto indices = torch::empty(
        {input_blocks, kKeep}, value.options().dtype(torch::kUInt8));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    select_input_k32_and_zero<<<input_blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<half *>(output.data_ptr<at::Half>()),
        indices.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(score_input.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(score_scale.data_ptr<at::Half>()),
        output_dim);
    input_sparse_o_kernel<<<input_blocks * output_tiles, 64, 0, stream>>>(
        reinterpret_cast<half *>(output.data_ptr<at::Half>()),
        indices.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(value.data_ptr<at::Half>()),
        output_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor permuted_pairwise_input_cuda(
    torch::Tensor value,
    torch::Tensor weight,
    torch::Tensor score_input,
    torch::Tensor score_scale,
    torch::Tensor permutation) {
    check_half_cuda(value, "value");
    check_half_cuda(weight, "weight");
    check_half_cuda(score_input, "score_input");
    check_half_cuda(score_scale, "score_scale");
    TORCH_CHECK(value.numel() == 4096, "input must have D=4096");
    TORCH_CHECK(score_input.numel() == 4096, "score input must have D=4096");
    TORCH_CHECK(score_scale.numel() == 4096, "score scale must have D=4096");
    TORCH_CHECK(
        weight.size(1) == 4096,
        "weight input dimension must be 4096");
    TORCH_CHECK(
        weight.size(0) == 6144 || weight.size(0) == 14336,
        "permuted pairwise input supports QKV or Gate");
    TORCH_CHECK(weight.stride(1) > 1, "weight must be column major");
    TORCH_CHECK(permutation.is_cuda(), "permutation must be CUDA");
    TORCH_CHECK(
        permutation.scalar_type() == at::kInt,
        "permutation must be int32");
    TORCH_CHECK(permutation.is_contiguous(), "permutation must be contiguous");
    TORCH_CHECK(permutation.numel() == 4096, "permutation must have D=4096");

    constexpr int input_blocks = 4096 / kBlockSize;
    const int output_dim = static_cast<int>(weight.size(0));
    auto output = torch::empty({1, 1, output_dim}, value.options());
    auto indices = torch::empty(
        {input_blocks, kKeep},
        value.options().dtype(torch::kUInt16));
    auto selected_values = torch::empty(
        {input_blocks, kKeep}, value.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    select_permuted_input_k32_and_zero
        <<<input_blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<half *>(output.data_ptr<at::Half>()),
            indices.data_ptr<uint16_t>(),
            reinterpret_cast<half *>(
                selected_values.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(value.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(
                score_input.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(
                score_scale.data_ptr<at::Half>()),
            permutation.data_ptr<int32_t>(),
            output_dim);

    if (output_dim == 6144) {
        constexpr int output_threads = 256;
        constexpr int output_tiles =
            (6144 / 2 + output_threads - 1) / output_threads;
        permuted_input_sparse_gemv_kernel<64, output_threads>
            <<<input_blocks * output_tiles, output_threads, 0, stream>>>(
                reinterpret_cast<half *>(output.data_ptr<at::Half>()),
                indices.data_ptr<uint16_t>(),
                reinterpret_cast<const half *>(
                    selected_values.data_ptr<at::Half>()),
                reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
                output_dim);
    } else {
        constexpr int output_threads = 128;
        constexpr int input_groups = input_blocks / (512 / kBlockSize);
        constexpr int output_tiles =
            (14336 / 2 + output_threads - 1) / output_threads;
        permuted_input_sparse_warp_kernel<512, output_threads, 2>
            <<<input_groups * output_tiles, output_threads, 0, stream>>>(
                reinterpret_cast<half *>(output.data_ptr<at::Half>()),
                indices.data_ptr<uint16_t>(),
                reinterpret_cast<const half *>(
                    selected_values.data_ptr<at::Half>()),
                reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
                output_dim);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> output_sparse_up_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_score,
    torch::Tensor score_scale,
    int64_t block_size,
    int64_t radix_bits,
    int64_t threads,
    int64_t load_policy) {
    check_half_cuda(input, "input");
    check_half_cuda(weight, "weight");
    check_half_cuda(output_score, "output_score");
    check_half_cuda(score_scale, "score_scale");
    TORCH_CHECK(block_size == kBlockSize, "block_size must be 64");
    TORCH_CHECK(radix_bits == 16, "radix_bits must be 16");
    TORCH_CHECK(threads == 256, "Up threads must be 256");
    TORCH_CHECK(load_policy == 2, "Up load policy must be streaming");
    TORCH_CHECK(input.numel() == 4096, "Up input must have D=4096");
    TORCH_CHECK(
        weight.sizes() == torch::IntArrayRef({14336, 4096}),
        "Up weight must be [14336, 4096]");
    TORCH_CHECK(weight.is_contiguous(), "Up weight must be row major");
    TORCH_CHECK(output_score.numel() == 14336, "output score size mismatch");
    TORCH_CHECK(score_scale.numel() == 14336, "score scale size mismatch");

    constexpr int output_dim = 14336;
    constexpr int output_blocks = output_dim / kBlockSize;
    constexpr int selected_rows = output_blocks * kKeep;
    auto output = torch::empty({1, 1, output_dim}, input.options());
    auto indices = torch::empty(
        {output_blocks, kKeep}, input.options().dtype(torch::kUInt8));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    select_output_k32_and_zero<<<output_blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<half *>(output.data_ptr<at::Half>()),
        indices.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(output_score.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(score_scale.data_ptr<at::Half>()));
    output_sparse_up_kernel<256, 2>
        <<<selected_rows, 256, 0, stream>>>(
            reinterpret_cast<half *>(output.data_ptr<at::Half>()),
            indices.data_ptr<uint8_t>(),
            reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
            4096);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, indices};
}

torch::Tensor input_sparse_down_cuda(
    torch::Tensor value,
    torch::Tensor weight,
    torch::Tensor local_indices,
    int64_t block_size,
    int64_t load_policy) {
    check_half_cuda(value, "value");
    check_half_cuda(weight, "weight");
    TORCH_CHECK(block_size == kBlockSize, "block_size must be 64");
    TORCH_CHECK(load_policy == 2, "Down load policy must be streaming");
    TORCH_CHECK(value.numel() == 14336, "Down input must have Dff=14336");
    TORCH_CHECK(
        weight.sizes() == torch::IntArrayRef({4096, 14336}),
        "Down weight must be [4096, 14336]");
    TORCH_CHECK(weight.stride(1) > 1, "Down weight must be column major");
    TORCH_CHECK(local_indices.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(
        local_indices.scalar_type() == at::kByte,
        "indices must be uint8");
    TORCH_CHECK(
        local_indices.sizes() ==
            torch::IntArrayRef({14336 / kBlockSize, kKeep}),
        "indices must have shape [224, 32]");

    constexpr int input_blocks = 14336 / kBlockSize;
    constexpr int input_tiles = input_blocks / 4;
    constexpr int output_dim = 4096;
    constexpr int output_tiles = (output_dim / 2) / 128;
    auto output = torch::empty({1, 1, output_dim}, value.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        output.data_ptr(), 0, output.numel() * sizeof(half), stream));
    input_sparse_down_kernel<2>
        <<<input_tiles * output_tiles, 128, 0, stream>>>(
            reinterpret_cast<half *>(output.data_ptr<at::Half>()),
            local_indices.data_ptr<uint8_t>(),
            reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(value.data_ptr<at::Half>()),
            input_blocks,
            output_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

TORCH_LIBRARY(ours_minimal_block_sparse, module) {
    module.def(
        "pairwise_input(Tensor value, Tensor weight, Tensor score_input, "
        "Tensor score_scale, int block_size, int radix_bits) -> Tensor");
    module.def(
        "permuted_pairwise_input(Tensor value, Tensor weight, "
        "Tensor score_input, Tensor score_scale, Tensor permutation) "
        "-> Tensor");
    module.def(
        "output_sparse_up(Tensor input, Tensor weight, Tensor output_score, "
        "Tensor score_scale, int block_size, int radix_bits, int threads, "
        "int load_policy) -> (Tensor, Tensor)");
    module.def(
        "input_sparse_down(Tensor value, Tensor weight, Tensor indices, "
        "int block_size, int load_policy) -> Tensor");
}

TORCH_LIBRARY_IMPL(ours_minimal_block_sparse, CUDA, module) {
    module.impl("pairwise_input", &pairwise_input_cuda);
    module.impl(
        "permuted_pairwise_input",
        &permuted_pairwise_input_cuda);
    module.impl("output_sparse_up", &output_sparse_up_cuda);
    module.impl("input_sparse_down", &input_sparse_down_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
