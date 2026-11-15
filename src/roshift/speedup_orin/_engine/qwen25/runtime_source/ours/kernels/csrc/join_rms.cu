#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <tuple>

namespace {

constexpr int kModelDim = 3584;
constexpr int kThreads = 1024;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ __launch_bounds__(kThreads)
void join_and_normalize(
    half * joined,
    half * normalized,
    const half * transition,
    const half * mlp,
    float eps) {
    constexpr int items_per_thread = (kModelDim + kThreads - 1) / kThreads;
    constexpr int warp_count = kThreads / 32;
    __shared__ float warp_sums[warp_count];
    __shared__ float inverse_rms;
    half local[items_per_thread];
    float sum = 0.0f;

#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        const int index = threadIdx.x + item * kThreads;
        if (index >= kModelDim) continue;
        const float value =
            __half2float(transition[index]) + __half2float(mlp[index]);
        local[item] = __float2half_rn(value);
        joined[index] = local[item];
        sum = fmaf(value, value, sum);
    }

    sum = warp_sum(sum);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        float total = lane < warp_count ? warp_sums[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            inverse_rms =
                rsqrtf(total / static_cast<float>(kModelDim) + eps);
        }
    }
    __syncthreads();

#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        const int index = threadIdx.x + item * kThreads;
        if (index >= kModelDim) continue;
        normalized[index] =
            __float2half_rn(__half2float(local[item]) * inverse_rms);
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> join_rms_cuda(
    torch::Tensor transition,
    torch::Tensor mlp,
    double eps,
    int64_t block_threads) {
    TORCH_CHECK(transition.is_cuda() && mlp.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(
        transition.scalar_type() == at::kHalf &&
            mlp.scalar_type() == at::kHalf,
        "inputs must be float16");
    TORCH_CHECK(
        transition.is_contiguous() && mlp.is_contiguous(),
        "inputs must be contiguous");
    TORCH_CHECK(transition.sizes() == mlp.sizes(), "input shape mismatch");
    TORCH_CHECK(transition.numel() == kModelDim, "inputs must have D=3584");
    TORCH_CHECK(block_threads == kThreads, "block_threads must be 1024");

    auto joined = torch::empty_like(transition);
    auto normalized = torch::empty_like(transition);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    join_and_normalize<<<1, kThreads, 0, stream>>>(
        reinterpret_cast<half *>(joined.data_ptr<at::Half>()),
        reinterpret_cast<half *>(normalized.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(transition.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(mlp.data_ptr<at::Half>()),
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {joined, normalized};
}

TORCH_LIBRARY(ours_minimal_join_rms, module) {
    module.def(
        "join_rms(Tensor transition, Tensor mlp, float eps, "
        "int block_threads) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ours_minimal_join_rms, CUDA, module) {
    module.impl("join_rms", &join_rms_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
