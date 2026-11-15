#include "kernels.cuh"

namespace {
const half* hp(const torch::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr<at::Half>()); }
half* hp(torch::Tensor& t) { return reinterpret_cast<half*>(t.data_ptr<at::Half>()); }

template<int Chunk, int Threads>
void launch(torch::Tensor& out, torch::Tensor& indices, torch::Tensor& selected,
            const torch::Tensor& value, const torch::Tensor& weight, int kind, cudaStream_t stream) {
    const int n=weight.size(0), m=weight.size(1);
    const int tiles=(n/2+Threads-1)/Threads;
    if(kind==0)
        permuted_input_sparse_gemv_kernel<Chunk,Threads><<<(m/Chunk)*tiles,Threads,0,stream>>>(
            hp(out),indices.data_ptr<uint16_t>(),hp(selected),hp(weight),n);
    else if(kind==2)
        permuted_input_sparse_warp_kernel<Chunk,Threads,2><<<(m/Chunk)*tiles,Threads,0,stream>>>(
            hp(out),indices.data_ptr<uint16_t>(),hp(selected),hp(weight),n);
    else if(kind==1)
        input_sparse_o_kernel<Threads><<<(m/128)*tiles,Threads,0,stream>>>(
            hp(out),indices.data_ptr<uint8_t>(),hp(weight),hp(value),n);
    else if(kind==4)
        input_sparse_down_kernel<2,Chunk,Threads><<<((m+Chunk-1)/Chunk)*tiles,Threads,0,stream>>>(
            hp(out),indices.data_ptr<uint8_t>(),hp(weight),hp(value),m/128,n);
}

#define TCASE(C,T) case T: launch<C,T>(out,indices,selected,value,weight,kind,stream);break;
#define CCASE(C) case C: switch(threads) { TCASE(C,32) TCASE(C,64) TCASE(C,128) TCASE(C,256) TCASE(C,512) default: TORCH_CHECK(false,"threads"); } break;
#define UCASE(T) case T: output_sparse_up_kernel<T,2><<<indices.numel(),T,0,stream>>>(hp(out),indices.data_ptr<uint8_t>(),hp(weight),hp(value),m);break;
}

std::tuple<torch::Tensor,torch::Tensor> run_geometry(torch::Tensor value,torch::Tensor weight,
    torch::Tensor score,torch::Tensor scale,torch::Tensor permutation,int64_t kind,int64_t chunk,int64_t threads) {
    const int m=weight.size(1),n=weight.size(0);
    TORCH_CHECK(value.numel()==m && value.scalar_type()==at::kHalf && weight.scalar_type()==at::kHalf,"input");
    TORCH_CHECK(value.is_cuda() && weight.is_cuda() && m%128==0 && n%64==0,"shape/device");
    TORCH_CHECK(kind>=0 && kind<=4,"kind");
    TORCH_CHECK(kind==3 ? weight.is_contiguous() : weight.stride(0)==1,"weight layout");
    if(kind==0 || kind==2) TORCH_CHECK(m==4096,"permuted dimension");
    if(kind==1) TORCH_CHECK(chunk==128,"O input grouping is unchanged");
    auto out=torch::empty({1,1,n},value.options());
    auto indices=kind==4 ? score : torch::empty({(kind==3?n:m)/128,kKeep},value.options().dtype(
        (kind==0 || kind==2)?torch::kUInt16:torch::kUInt8));
    auto selected=torch::empty({(kind==0 || kind==2)?m/128:0,kKeep},value.options());
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    if(kind==0 || kind==2) {
        select_permuted_input_k64_and_zero<<<m/128,128,0,stream>>>(hp(out),indices.data_ptr<uint16_t>(),
            hp(selected),hp(value),hp(score),hp(scale),permutation.data_ptr<int32_t>(),n);
    } else if(kind==1) {
        select_input_k64_and_zero<<<m/128,128,0,stream>>>(hp(out),indices.data_ptr<uint8_t>(),hp(score),hp(scale),n);
    } else if(kind==3) {
        select_output_k64_and_zero<<<n/128,128,0,stream>>>(hp(out),indices.data_ptr<uint8_t>(),hp(score),hp(scale));
    } else {
        TORCH_CHECK(indices.sizes()==torch::IntArrayRef({m/128,kKeep}),"Down support shape");
        C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr(),0,n*sizeof(half),stream));
    }
    if(kind==3) {
        switch(threads) { UCASE(32) UCASE(64) UCASE(128) UCASE(256) UCASE(512) UCASE(1024)
            default: TORCH_CHECK(false,"Up threads"); }
    } else {
        switch(chunk) { CCASE(128) CCASE(256) CCASE(512) CCASE(1024)
            default: TORCH_CHECK(false,"chunk"); }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,indices};
}

#define REGISTER_GEOMETRY(ns) TORCH_LIBRARY(ns,m)
#define IMPLEMENT_GEOMETRY(ns) TORCH_LIBRARY_IMPL(ns,CUDA,m)
REGISTER_GEOMETRY(GEOMETRY_NAMESPACE) {
    m.def("run(Tensor value, Tensor weight, Tensor score, Tensor scale, Tensor permutation, int kind, int chunk, int threads) -> (Tensor, Tensor)");
}
IMPLEMENT_GEOMETRY(GEOMETRY_NAMESPACE) { m.impl("run",&run_geometry); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {}
