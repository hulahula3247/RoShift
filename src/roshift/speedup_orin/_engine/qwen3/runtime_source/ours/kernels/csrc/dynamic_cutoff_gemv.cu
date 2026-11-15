#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

namespace {

constexpr int kModelDim = 4096;
constexpr int kSegment = 128;
constexpr int kSegments = kModelDim / kSegment;

void check_half_cuda(const torch::Tensor & tensor, const char * name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be float16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__device__ __forceinline__ float weighted_score(
    const half * input,
    const half * scale,
    int index) {
    return fabsf(__half2float(input[index])) * __half2float(scale[index]);
}

template <int Policy>
__device__ __forceinline__ half2 load_weight_half2(const half * pointer);

template <>
__device__ __forceinline__ half2 load_weight_half2<0>(
    const half * pointer) {
    return *reinterpret_cast<const half2 *>(pointer);
}

template <>
__device__ __forceinline__ half2 load_weight_half2<1>(
    const half * pointer) {
    const uint32_t packed =
        __ldcg(reinterpret_cast<const uint32_t *>(pointer));
    return *reinterpret_cast<const half2 *>(&packed);
}

template <>
__device__ __forceinline__ half2 load_weight_half2<2>(
    const half * pointer) {
    const uint32_t packed =
        __ldcs(reinterpret_cast<const uint32_t *>(pointer));
    return *reinterpret_cast<const half2 *>(&packed);
}

__global__ __launch_bounds__(kSegment)
void select_segment128_and_zero(
    float * threshold,
    uint16_t * indices,
    int * segment_counts,
    half * output,
    const half * value,
    const half * score_input,
    const half * score_scale,
    int output_dim,
    float raw_rms_ratio) {
    constexpr int warps = kSegment / 32;
    __shared__ int warp_counts[warps];
    __shared__ int warp_offsets[warps];
    __shared__ float cutoff;

    const int local = threadIdx.x;
    const int lane = local & 31;
    const int warp = local >> 5;

    // Estimate the current RMS scaling from the same representative slice
    // used during calibration, then apply the calibrated raw-RMS ratio.
    if (warp == 0) {
        int best_index = lane;
        float best_magnitude = fabsf(__half2float(score_input[lane]));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_magnitude =
                __shfl_down_sync(0xffffffffu, best_magnitude, offset);
            const int other_index =
                __shfl_down_sync(0xffffffffu, best_index, offset);
            if (lane + offset < 32 && other_magnitude > best_magnitude) {
                best_magnitude = other_magnitude;
                best_index = other_index;
            }
        }
        if (lane == 0) {
            const float raw = __half2float(score_input[best_index]);
            const float scaled = __half2float(value[best_index]);
            const float alpha =
                fabsf(scaled / copysignf(fmaxf(fabsf(raw), 1.0e-8f), raw));
            cutoff = raw_rms_ratio / fmaxf(alpha, 1.0e-8f);
            if (blockIdx.x == 0) {
                threshold[0] = cutoff;
            }
        }
    }

    const int global_thread = blockIdx.x * blockDim.x + local;
    const int global_threads = gridDim.x * blockDim.x;
    for (
        int output_index = global_thread;
        output_index < output_dim;
        output_index += global_threads) {
        output[output_index] = __float2half(0.0f);
    }
    __syncthreads();

    const int index = blockIdx.x * kSegment + local;
    const bool keep =
        weighted_score(score_input, score_scale, index) >= cutoff;
    const uint32_t ballot = __ballot_sync(0xffffffffu, keep);
    const uint32_t lower = lane == 0 ? 0u : ((1u << lane) - 1u);
    const int lane_rank = __popc(ballot & lower);
    if (lane == 0) {
        warp_counts[warp] = __popc(ballot);
    }
    __syncthreads();

    if (local == 0) {
        int offset = 0;
#pragma unroll
        for (int item = 0; item < warps; ++item) {
            warp_offsets[item] = offset;
            offset += warp_counts[item];
        }
        segment_counts[blockIdx.x] = offset;
    }
    __syncthreads();

    if (keep) {
        const int slot =
            blockIdx.x * kSegment + warp_offsets[warp] + lane_rank;
        indices[slot] = static_cast<uint16_t>(index);
    }
}

template <int OutputThreads, int LoadPolicy>
__global__ void segmented_input_sparse_gemv(
    half * output,
    const uint16_t * selected_indices,
    const int * segment_counts,
    const half * weight_column_major,
    const half * value,
    int output_dim) {
    const int output_pairs = output_dim / 2;
    const int output_tiles =
        (output_pairs + OutputThreads - 1) / OutputThreads;
    const int segment = blockIdx.x / output_tiles;
    const int output_tile = blockIdx.x - segment * output_tiles;
    const int output_pair =
        output_tile * OutputThreads + threadIdx.x;
    if (segment >= kSegments || output_pair >= output_pairs) {
        return;
    }

    const int first = segment * kSegment;
    const int count = segment_counts[segment];
    const int out = output_pair * 2;
    float2 accum = make_float2(0.0f, 0.0f);
    for (int local = 0; local < count; ++local) {
        const int index =
            static_cast<int>(selected_indices[first + local]);
        const float x = __half2float(value[index]);
        const half2 weights = load_weight_half2<LoadPolicy>(
            weight_column_major + index * output_dim + out);
        const float2 weights_f = __half22float2(weights);
        accum.x = fmaf(x, weights_f.x, accum.x);
        accum.y = fmaf(x, weights_f.y, accum.y);
    }
    atomicAdd(
        reinterpret_cast<half2 *>(output) + output_pair,
        __floats2half2_rn(accum.x, accum.y));
}

template <int OutputThreads>
void launch_sparse_gemv(
    half * output,
    const uint16_t * indices,
    const int * counts,
    const half * weight,
    const half * value,
    int output_dim,
    int load_policy,
    cudaStream_t stream) {
    const int output_pairs = output_dim / 2;
    const int output_tiles =
        (output_pairs + OutputThreads - 1) / OutputThreads;
    const int blocks = kSegments * output_tiles;
#define LAUNCH(Policy)                                                        \
    segmented_input_sparse_gemv<OutputThreads, Policy>                       \
        <<<blocks, OutputThreads, 0, stream>>>(                               \
            output, indices, counts, weight, value, output_dim)
    if (load_policy == 0) {
        LAUNCH(0);
    } else if (load_policy == 1) {
        LAUNCH(1);
    } else if (load_policy == 2) {
        LAUNCH(2);
    } else {
        TORCH_CHECK(false, "load_policy must be 0, 1, or 2");
    }
#undef LAUNCH
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dynamic_input_threshold_compact_gemv_cuda(
    torch::Tensor value,
    torch::Tensor weight,
    torch::Tensor score_input,
    torch::Tensor score_scale,
    int64_t method,
    int64_t sample_size,
    double rms_ratio,
    double max_keep_ratio,
    int64_t selected_chunk,
    int64_t output_threads,
    int64_t load_policy) {
    check_half_cuda(value, "value");
    check_half_cuda(score_input, "score_input");
    check_half_cuda(score_scale, "score_scale");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be float16");
    TORCH_CHECK(weight.stride(1) > 1, "weight must be column major");
    TORCH_CHECK(value.numel() == kModelDim, "value must have D=4096");
    TORCH_CHECK(score_input.numel() == kModelDim, "score_input must have D=4096");
    TORCH_CHECK(score_scale.numel() == kModelDim, "score_scale must have D=4096");
    TORCH_CHECK(weight.size(1) == kModelDim, "weight input dimension mismatch");
    TORCH_CHECK(weight.size(0) % 2 == 0, "output dimension must be even");
    TORCH_CHECK(method == 18, "minimal runtime supports only segment128");
    TORCH_CHECK(sample_size == 256, "sample_size must be 256");
    TORCH_CHECK(max_keep_ratio == 0.5, "minimal runtime supports 50% sparsity");
    TORCH_CHECK(selected_chunk == 128, "selected_chunk must be 128");

    const int output_dim = static_cast<int>(weight.size(0));
    auto threshold = torch::empty(
        {1}, value.options().dtype(torch::kFloat32));
    auto counts = torch::empty(
        {kSegments}, value.options().dtype(torch::kInt32));
    auto indices = torch::empty(
        {kModelDim}, value.options().dtype(torch::kUInt16));
    auto output = torch::empty({1, 1, output_dim}, value.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    select_segment128_and_zero<<<kSegments, kSegment, 0, stream>>>(
        threshold.data_ptr<float>(),
        indices.data_ptr<uint16_t>(),
        counts.data_ptr<int>(),
        reinterpret_cast<half *>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(value.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(score_input.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(score_scale.data_ptr<at::Half>()),
        output_dim,
        static_cast<float>(rms_ratio));

    half * output_ptr =
        reinterpret_cast<half *>(output.data_ptr<at::Half>());
    const half * weight_ptr =
        reinterpret_cast<const half *>(weight.data_ptr<at::Half>());
    const half * value_ptr =
        reinterpret_cast<const half *>(value.data_ptr<at::Half>());
    if (output_threads == 64) {
        launch_sparse_gemv<64>(
            output_ptr, indices.data_ptr<uint16_t>(), counts.data_ptr<int>(),
            weight_ptr, value_ptr, output_dim, static_cast<int>(load_policy),
            stream);
    } else if (output_threads == 128) {
        launch_sparse_gemv<128>(
            output_ptr, indices.data_ptr<uint16_t>(), counts.data_ptr<int>(),
            weight_ptr, value_ptr, output_dim, static_cast<int>(load_policy),
            stream);
    } else {
        TORCH_CHECK(false, "output_threads must be 64 or 128");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, threshold, counts};
}

TORCH_LIBRARY(ours_minimal_dynamic, module) {
    module.def(
        "dynamic_input(Tensor value, Tensor weight, Tensor score_input, "
        "Tensor score_scale, int method, int sample_size, float rms_ratio, "
        "float max_keep_ratio, int selected_chunk, int output_threads, "
        "int load_policy) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ours_minimal_dynamic, CUDA, module) {
    module.impl("dynamic_input", &dynamic_input_threshold_compact_gemv_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
