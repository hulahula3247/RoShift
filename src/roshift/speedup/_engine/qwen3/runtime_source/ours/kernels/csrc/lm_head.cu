#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <vector>

namespace {

constexpr int kPrefixWarps = 16;
constexpr int kTailWarps = 16;
constexpr int kRowsPerBlock = 256;

void check_half_cuda(const torch::Tensor & tensor, const char * name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be float16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ float atomic_max_float(float * address, float value) {
    int * address_as_int = reinterpret_cast<int *>(address);
    int old = *address_as_int;
    while (__int_as_float(old) < value) {
        const int assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
    return __int_as_float(old);
}

__global__ void initialize_stats(
    float * stats,
    int * candidate_count) {
    if (threadIdx.x == 0) {
        stats[0] = -CUDART_INF_F;
        stats[1] = 0.0f;
        stats[2] = 0.0f;
        stats[3] = 0.0f;
        stats[4] = 0.0f;
        candidate_count[0] = 0;
    }
}

__global__ void initialize_three_group_stats(
    float * stats,
    int * candidate_counts) {
    if (threadIdx.x == 0) {
        stats[0] = -CUDART_INF_F;
        stats[1] = 0.0f;
        stats[2] = 0.0f;
        stats[3] = 0.0f;
        stats[4] = 0.0f;
        stats[5] = -CUDART_INF_F;
        stats[6] = 0.0f;
        stats[7] = 0.0f;
        stats[8] = 0.0f;
        stats[9] = 0.0f;
        candidate_counts[0] = 0;
        candidate_counts[1] = 0;
    }
}

__global__ __launch_bounds__(kPrefixWarps * 32)
void prefix_logits_and_stats(
    const half * input,
    const half * weight,
    half * logits,
    float * prefix_logits,
    float * stats,
    int vocab_size,
    int hidden_size,
    int prefix_dim) {
    extern __shared__ unsigned char shared_raw[];
    half * input_shared = reinterpret_cast<half *>(shared_raw);
    float * warp_values =
        reinterpret_cast<float *>(input_shared + prefix_dim);

    for (int index = threadIdx.x;
         index < prefix_dim;
         index += blockDim.x) {
        input_shared[index] = input[index];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kPrefixWarps + warp;
    float value = 0.0f;
    if (row < vocab_size) {
        const half2 * input2 =
            reinterpret_cast<const half2 *>(input_shared);
        const half2 * weight2 = reinterpret_cast<const half2 *>(
            weight + static_cast<int64_t>(row) * hidden_size);
        const int pairs = prefix_dim / 2;
        for (int pair = lane; pair < pairs; pair += 32) {
            const float2 x = __half22float2(input2[pair]);
            const float2 w = __half22float2(weight2[pair]);
            value = fmaf(x.x, w.x, value);
            value = fmaf(x.y, w.y, value);
        }
        value = warp_sum(value);
        if (lane == 0) {
            value = __half2float(__float2half_rn(value));
            prefix_logits[row] = value;
            logits[row] = __float2half(-CUDART_INF_F);
        }
    }
    if (lane == 0) {
        warp_values[warp] =
            row < vocab_size ? value : -CUDART_INF_F;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_max = -CUDART_INF_F;
        float block_sum = 0.0f;
        float block_square_sum = 0.0f;
#pragma unroll
        for (int index = 0; index < kPrefixWarps; ++index) {
            const int candidate_row =
                blockIdx.x * kPrefixWarps + index;
            if (candidate_row < vocab_size) {
                const float candidate = warp_values[index];
                block_max = fmaxf(block_max, candidate);
                block_sum += candidate;
                block_square_sum =
                    fmaf(candidate, candidate, block_square_sum);
            }
        }
        atomic_max_float(stats, block_max);
        atomicAdd(stats + 1, block_sum);
        atomicAdd(stats + 2, block_square_sum);
    }
}

__global__ __launch_bounds__(kTailWarps * 32)
void compact_local_tail(
    const half * input,
    const half * weight,
    const float * prefix_logits,
    float * stats,
    half * logits,
    int * candidate_count,
    int vocab_size,
    int hidden_size,
    int prefix_dim,
    float scaled_margin,
    bool count_candidates) {
    const int tail_dim = hidden_size - prefix_dim;
    extern __shared__ unsigned char shared_raw[];
    half * tail_shared = reinterpret_cast<half *>(shared_raw);
    int * local_rows =
        reinterpret_cast<int *>(tail_shared + tail_dim);
    __shared__ int local_count;
    __shared__ float cutoff;

    for (int index = threadIdx.x;
         index < tail_dim;
         index += blockDim.x) {
        tail_shared[index] = input[prefix_dim + index];
    }
    if (threadIdx.x == 0) {
        local_count = 0;
        const float inverse_vocab = 1.0f / static_cast<float>(vocab_size);
        const float mean = stats[1] * inverse_vocab;
        const float mean_square = stats[2] * inverse_vocab;
        const float variance =
            fmaxf(mean_square - mean * mean, 0.0f);
        const float standard_deviation = sqrtf(variance);
        cutoff = stats[0] - scaled_margin * standard_deviation;
        if (blockIdx.x == 0) {
            stats[3] = standard_deviation;
            stats[4] = cutoff;
        }
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int first_row = blockIdx.x * kRowsPerBlock;
    for (int local_row = threadIdx.x;
         local_row < kRowsPerBlock;
         local_row += blockDim.x) {
        const int row = first_row + local_row;
        const bool selected =
            row < vocab_size && prefix_logits[row] >= cutoff;
        const unsigned mask =
            __ballot_sync(0xffffffffu, selected);
        const int selected_in_warp = __popc(mask);
        int warp_base = 0;
        if (lane == 0 && selected_in_warp != 0) {
            warp_base = atomicAdd(&local_count, selected_in_warp);
        }
        warp_base =
            __shfl_sync(0xffffffffu, warp_base, 0);
        if (selected) {
            const unsigned lower_lanes =
                lane == 0 ? 0u : ((1u << lane) - 1u);
            const int offset =
                warp_base + __popc(mask & lower_lanes);
            local_rows[offset] = row;
        }
    }
    __syncthreads();

    if (count_candidates && threadIdx.x == 0 && local_count != 0) {
        atomicAdd(candidate_count, local_count);
    }

    const half2 * input2 =
        reinterpret_cast<const half2 *>(tail_shared);
    const int pairs = tail_dim / 2;
    for (int position = warp;
         position < local_count;
         position += kTailWarps) {
        const int row = local_rows[position];
        const half2 * weight2 = reinterpret_cast<const half2 *>(
            weight +
            static_cast<int64_t>(row) * hidden_size +
            prefix_dim);
        float value = 0.0f;
        for (int pair = lane; pair < pairs; pair += 32) {
            const float2 x = __half22float2(input2[pair]);
            const float2 w = __half22float2(weight2[pair]);
            value = fmaf(x.x, w.x, value);
            value = fmaf(x.y, w.y, value);
        }
        value = warp_sum(value);
        if (lane == 0) {
            logits[row] =
                __float2half_rn(prefix_logits[row] + value);
        }
    }
}

__global__ __launch_bounds__(kTailWarps * 32)
void compact_middle_and_stats(
    const half * input,
    const half * weight,
    const float * prefix_logits,
    const float * prefix_stats,
    float * middle_logits,
    float * middle_stats,
    int * candidate_counts,
    int vocab_size,
    int hidden_size,
    int prefix_dim,
    int middle_dim,
    float first_scaled_margin) {
    extern __shared__ unsigned char shared_raw[];
    half * middle_shared = reinterpret_cast<half *>(shared_raw);
    int * local_rows = reinterpret_cast<int *>(middle_shared + middle_dim);
    float * local_values = reinterpret_cast<float *>(
        local_rows + kRowsPerBlock);
    __shared__ int local_count;
    __shared__ float cutoff;

    for (int index = threadIdx.x;
         index < middle_dim;
         index += blockDim.x) {
        middle_shared[index] = input[prefix_dim + index];
    }
    if (threadIdx.x == 0) {
        local_count = 0;
        const float inverse_vocab = 1.0f / static_cast<float>(vocab_size);
        const float mean = prefix_stats[1] * inverse_vocab;
        const float mean_square = prefix_stats[2] * inverse_vocab;
        const float variance = fmaxf(mean_square - mean * mean, 0.0f);
        const float standard_deviation = sqrtf(variance);
        cutoff = prefix_stats[0] - first_scaled_margin * standard_deviation;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int first_row = blockIdx.x * kRowsPerBlock;
    for (int local_row = threadIdx.x;
         local_row < kRowsPerBlock;
         local_row += blockDim.x) {
        const int row = first_row + local_row;
        if (row < vocab_size) {
            middle_logits[row] = -CUDART_INF_F;
        }
        const bool selected =
            row < vocab_size && prefix_logits[row] >= cutoff;
        const unsigned mask = __ballot_sync(0xffffffffu, selected);
        const int selected_in_warp = __popc(mask);
        int warp_base = 0;
        if (lane == 0 && selected_in_warp != 0) {
            warp_base = atomicAdd(&local_count, selected_in_warp);
        }
        warp_base = __shfl_sync(0xffffffffu, warp_base, 0);
        if (selected) {
            const unsigned lower_lanes =
                lane == 0 ? 0u : ((1u << lane) - 1u);
            const int offset = warp_base + __popc(mask & lower_lanes);
            local_rows[offset] = row;
        }
    }
    __syncthreads();

    const half2 * input2 =
        reinterpret_cast<const half2 *>(middle_shared);
    const int pairs = middle_dim / 2;
    for (int position = warp;
         position < local_count;
         position += kTailWarps) {
        const int row = local_rows[position];
        const half2 * weight2 = reinterpret_cast<const half2 *>(
            weight + static_cast<int64_t>(row) * hidden_size + prefix_dim);
        float value = 0.0f;
        for (int pair = lane; pair < pairs; pair += 32) {
            const float2 x = __half22float2(input2[pair]);
            const float2 w = __half22float2(weight2[pair]);
            value = fmaf(x.x, w.x, value);
            value = fmaf(x.y, w.y, value);
        }
        value = warp_sum(value);
        if (lane == 0) {
            value += prefix_logits[row];
            middle_logits[row] = value;
            local_values[position] = value;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && local_count != 0) {
        float block_max = -CUDART_INF_F;
        float block_sum = 0.0f;
        float block_square_sum = 0.0f;
        for (int index = 0; index < local_count; ++index) {
            const float value = local_values[index];
            block_max = fmaxf(block_max, value);
            block_sum += value;
            block_square_sum = fmaf(value, value, block_square_sum);
        }
        atomic_max_float(middle_stats, block_max);
        atomicAdd(middle_stats + 1, block_sum);
        atomicAdd(middle_stats + 2, block_square_sum);
        atomicAdd(candidate_counts, local_count);
    }
}

__global__ __launch_bounds__(kTailWarps * 32)
void compact_final_tail(
    const half * input,
    const half * weight,
    const float * middle_logits,
    float * middle_stats,
    half * logits,
    int * candidate_counts,
    int vocab_size,
    int hidden_size,
    int tail_start,
    float second_scaled_margin) {
    const int tail_dim = hidden_size - tail_start;
    extern __shared__ unsigned char shared_raw[];
    half * tail_shared = reinterpret_cast<half *>(shared_raw);
    int * local_rows = reinterpret_cast<int *>(tail_shared + tail_dim);
    __shared__ int local_count;
    __shared__ float cutoff;

    for (int index = threadIdx.x;
         index < tail_dim;
         index += blockDim.x) {
        tail_shared[index] = input[tail_start + index];
    }
    if (threadIdx.x == 0) {
        local_count = 0;
        const int raw_first_count = candidate_counts[0];
        const int first_count = raw_first_count > 0 ? raw_first_count : 1;
        const float inverse_count = 1.0f / static_cast<float>(first_count);
        const float mean = middle_stats[1] * inverse_count;
        const float mean_square = middle_stats[2] * inverse_count;
        const float variance = fmaxf(mean_square - mean * mean, 0.0f);
        const float standard_deviation = sqrtf(variance);
        cutoff = middle_stats[0] - second_scaled_margin * standard_deviation;
        if (blockIdx.x == 0) {
            middle_stats[3] = standard_deviation;
            middle_stats[4] = cutoff;
        }
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int first_row = blockIdx.x * kRowsPerBlock;
    for (int local_row = threadIdx.x;
         local_row < kRowsPerBlock;
         local_row += blockDim.x) {
        const int row = first_row + local_row;
        const bool selected =
            row < vocab_size && middle_logits[row] >= cutoff;
        const unsigned mask = __ballot_sync(0xffffffffu, selected);
        const int selected_in_warp = __popc(mask);
        int warp_base = 0;
        if (lane == 0 && selected_in_warp != 0) {
            warp_base = atomicAdd(&local_count, selected_in_warp);
        }
        warp_base = __shfl_sync(0xffffffffu, warp_base, 0);
        if (selected) {
            const unsigned lower_lanes =
                lane == 0 ? 0u : ((1u << lane) - 1u);
            const int offset = warp_base + __popc(mask & lower_lanes);
            local_rows[offset] = row;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && local_count != 0) {
        atomicAdd(candidate_counts + 1, local_count);
    }
    const half2 * input2 = reinterpret_cast<const half2 *>(tail_shared);
    const int pairs = tail_dim / 2;
    for (int position = warp;
         position < local_count;
         position += kTailWarps) {
        const int row = local_rows[position];
        const half2 * weight2 = reinterpret_cast<const half2 *>(
            weight + static_cast<int64_t>(row) * hidden_size + tail_start);
        float value = 0.0f;
        for (int pair = lane; pair < pairs; pair += 32) {
            const float2 x = __half22float2(input2[pair]);
            const float2 w = __half22float2(weight2[pair]);
            value = fmaf(x.x, w.x, value);
            value = fmaf(x.y, w.y, value);
        }
        value = warp_sum(value);
        if (lane == 0) {
            logits[row] = __float2half_rn(middle_logits[row] + value);
        }
    }
}

}  // namespace

std::vector<torch::Tensor> early_exit_lmhead_local_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t prefix_dim,
    double scaled_margin,
    int64_t prefix_warps,
    int64_t tail_warps,
    int64_t rows_per_block,
    int64_t prefix_rows_per_warp,
    bool count_candidates,
    bool packed_weight) {
    check_half_cuda(input, "input");
    check_half_cuda(weight, "weight");
    TORCH_CHECK(
        input.numel() == input.size(-1),
        "only batch-1 single-token decode is supported");
    TORCH_CHECK(weight.dim() == 2, "weight must be [vocab, hidden]");
    TORCH_CHECK(
        input.size(-1) == weight.size(1),
        "hidden dimension mismatch");
    TORCH_CHECK(prefix_dim > 0, "prefix_dim must be positive");
    TORCH_CHECK(prefix_dim % 2 == 0, "prefix_dim must be even");
    TORCH_CHECK(
        prefix_warps == kPrefixWarps,
        "prefix_warps must be 16");
    TORCH_CHECK(tail_warps == kTailWarps, "tail_warps must be 16");
    TORCH_CHECK(
        rows_per_block == kRowsPerBlock,
        "rows_per_block must be 256");
    TORCH_CHECK(
        prefix_rows_per_warp == 1,
        "prefix_rows_per_warp must be 1");
    TORCH_CHECK(!packed_weight, "packed LM-head weight is not supported");

    const int vocab_size = static_cast<int>(weight.size(0));
    const int hidden_size = static_cast<int>(weight.size(1));
    const int tail_dim = hidden_size - static_cast<int>(prefix_dim);
    TORCH_CHECK(tail_dim > 0 && tail_dim % 2 == 0, "invalid tail size");

    auto logits = torch::empty(
        {input.size(0), input.size(1), vocab_size},
        input.options());
    auto prefix_logits = torch::empty(
        {vocab_size}, input.options().dtype(torch::kFloat32));
    auto stats = torch::empty(
        {5}, input.options().dtype(torch::kFloat32));
    auto candidate_count = torch::empty(
        {1}, input.options().dtype(torch::kInt32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    initialize_stats<<<1, 1, 0, stream>>>(
        stats.data_ptr<float>(),
        candidate_count.data_ptr<int>());
    const int prefix_blocks =
        (vocab_size + kPrefixWarps - 1) / kPrefixWarps;
    const size_t prefix_shared =
        static_cast<size_t>(prefix_dim) * sizeof(half) +
        kPrefixWarps * sizeof(float);
    prefix_logits_and_stats<<<
        prefix_blocks,
        kPrefixWarps * 32,
        prefix_shared,
        stream>>>(
            reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
            reinterpret_cast<half *>(logits.data_ptr<at::Half>()),
            prefix_logits.data_ptr<float>(),
            stats.data_ptr<float>(),
            vocab_size,
            hidden_size,
            static_cast<int>(prefix_dim));

    const int tail_blocks =
        (vocab_size + kRowsPerBlock - 1) / kRowsPerBlock;
    const size_t tail_shared =
        static_cast<size_t>(tail_dim) * sizeof(half) +
        kRowsPerBlock * sizeof(int);
    compact_local_tail<<<
        tail_blocks,
        kTailWarps * 32,
        tail_shared,
        stream>>>(
            reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
            prefix_logits.data_ptr<float>(),
            stats.data_ptr<float>(),
            reinterpret_cast<half *>(logits.data_ptr<at::Half>()),
            candidate_count.data_ptr<int>(),
            vocab_size,
            hidden_size,
            static_cast<int>(prefix_dim),
            static_cast<float>(scaled_margin),
            count_candidates);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {logits, candidate_count, stats};
}

std::vector<torch::Tensor> early_exit_lmhead_three_group_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t prefix_dim,
    int64_t middle_dim,
    double first_scaled_margin,
    double second_scaled_margin) {
    check_half_cuda(input, "input");
    check_half_cuda(weight, "weight");
    TORCH_CHECK(
        input.numel() == input.size(-1),
        "only batch-1 single-token decode is supported");
    TORCH_CHECK(weight.dim() == 2, "weight must be [vocab, hidden]");
    TORCH_CHECK(
        input.size(-1) == weight.size(1),
        "hidden dimension mismatch");
    TORCH_CHECK(prefix_dim > 0 && prefix_dim % 2 == 0, "invalid prefix size");
    TORCH_CHECK(middle_dim > 0 && middle_dim % 2 == 0, "invalid middle size");

    const int vocab_size = static_cast<int>(weight.size(0));
    const int hidden_size = static_cast<int>(weight.size(1));
    const int tail_start = static_cast<int>(prefix_dim + middle_dim);
    const int tail_dim = hidden_size - tail_start;
    TORCH_CHECK(tail_dim > 0 && tail_dim % 2 == 0, "invalid final tail size");

    auto logits = torch::empty(
        {input.size(0), input.size(1), vocab_size}, input.options());
    auto prefix_logits = torch::empty(
        {vocab_size}, input.options().dtype(torch::kFloat32));
    auto middle_logits = torch::empty(
        {vocab_size}, input.options().dtype(torch::kFloat32));
    auto stats = torch::empty(
        {10}, input.options().dtype(torch::kFloat32));
    auto candidate_counts = torch::empty(
        {2}, input.options().dtype(torch::kInt32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    initialize_three_group_stats<<<1, 1, 0, stream>>>(
        stats.data_ptr<float>(), candidate_counts.data_ptr<int>());

    const int prefix_blocks =
        (vocab_size + kPrefixWarps - 1) / kPrefixWarps;
    const size_t prefix_shared =
        static_cast<size_t>(prefix_dim) * sizeof(half) +
        kPrefixWarps * sizeof(float);
    prefix_logits_and_stats<<<
        prefix_blocks,
        kPrefixWarps * 32,
        prefix_shared,
        stream>>>(
            reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
            reinterpret_cast<half *>(logits.data_ptr<at::Half>()),
            prefix_logits.data_ptr<float>(),
            stats.data_ptr<float>(),
            vocab_size,
            hidden_size,
            static_cast<int>(prefix_dim));

    const int sparse_blocks =
        (vocab_size + kRowsPerBlock - 1) / kRowsPerBlock;
    const size_t middle_shared =
        static_cast<size_t>(middle_dim) * sizeof(half) +
        kRowsPerBlock * (sizeof(int) + sizeof(float));
    compact_middle_and_stats<<<
        sparse_blocks,
        kTailWarps * 32,
        middle_shared,
        stream>>>(
            reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
            prefix_logits.data_ptr<float>(),
            stats.data_ptr<float>(),
            middle_logits.data_ptr<float>(),
            stats.data_ptr<float>() + 5,
            candidate_counts.data_ptr<int>(),
            vocab_size,
            hidden_size,
            static_cast<int>(prefix_dim),
            static_cast<int>(middle_dim),
            static_cast<float>(first_scaled_margin));

    const size_t tail_shared =
        static_cast<size_t>(tail_dim) * sizeof(half) +
        kRowsPerBlock * sizeof(int);
    compact_final_tail<<<
        sparse_blocks,
        kTailWarps * 32,
        tail_shared,
        stream>>>(
            reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
            reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
            middle_logits.data_ptr<float>(),
            stats.data_ptr<float>() + 5,
            reinterpret_cast<half *>(logits.data_ptr<at::Half>()),
            candidate_counts.data_ptr<int>(),
            vocab_size,
            hidden_size,
            tail_start,
            static_cast<float>(second_scaled_margin));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {logits, candidate_counts, stats};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "early_exit_lmhead_local",
        &early_exit_lmhead_local_cuda);
    module.def(
        "early_exit_lmhead_three_group",
        &early_exit_lmhead_three_group_cuda);
}
