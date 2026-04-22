#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// 辅助规约函数：使用模板支持多种精度
template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceMax(scalar_t val) {
    // 使用 float 进行中间规约以保证精度
    static __shared__ float shared[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    float f_val = static_cast<float>(val);

    // 1. Warp 内规约
    for (int offset = 16; offset > 0; offset /= 2)
        f_val = max(f_val, __shfl_down_sync(0xffffffff, f_val, offset));

    if (lane == 0) shared[wid] = f_val;
    __syncthreads();

    // 2. 跨 Warp 规约
    f_val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e20f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            f_val = max(f_val, __shfl_down_sync(0xffffffff, f_val, offset));
    }
    return static_cast<scalar_t>(f_val);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    float f_val = static_cast<float>(val);

    for (int offset = 16; offset > 0; offset /= 2)
        f_val += __shfl_down_sync(0xffffffff, f_val, offset);

    if (lane == 0) shared[wid] = f_val;
    __syncthreads();

    f_val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            f_val += __shfl_down_sync(0xffffffff, f_val, offset);
    }
    return static_cast<scalar_t>(f_val);
}

// 核心 Kernel：模板化实现
template <typename scalar_t>
__global__ void fused_logp_forward_kernel(
    const scalar_t* __restrict__ logits,      // [TotalTokens, VocabSize]
    const int64_t* __restrict__ token_ids,   // [TotalTokens]
    scalar_t* __restrict__ output,           // [TotalTokens]
    int vocab_size) {

    int row = blockIdx.x;
    const scalar_t* row_logits = logits + row * vocab_size;

    // Step 1: Find Max (使用 float 累加防止溢出)
    float local_max = -1e20f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_max = max(local_max, static_cast<float>(row_logits[i]));
    }
    float max_val = blockReduceMax<float>(local_max);
    
    __shared__ float res_max;
    if (threadIdx.x == 0) res_max = max_val;
    __syncthreads();

    // Step 2: Sum Exp
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        // 使用 expf 保证在 GPU 上的执行效率
        local_sum += expf(static_cast<float>(row_logits[i]) - res_max);
    }
    float sum_val = blockReduceSum<float>(local_sum);
    
    __shared__ float res_sum;
    if (threadIdx.x == 0) res_sum = sum_val;
    __syncthreads();

    // Step 3: Final Logprob
    if (threadIdx.x == 0) {
        int64_t target_id = token_ids[row];
        if (target_id >= 0 && target_id < vocab_size) {
            float target_logit = static_cast<float>(row_logits[target_id]);
            // 最终计算结果转回 scalar_t 存储
            output[row] = static_cast<scalar_t>(target_logit - res_max - logf(res_sum));
        } else {
            output[row] = static_cast<scalar_t>(0.0f); 
        }
    }
}

// 宿主函数 (Dispatch 层)
torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids) {
    // 基础检查
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be a CUDA tensor");

    auto logits_contig = logits.contiguous();
    auto token_ids_contig = token_ids.contiguous();
    
    int64_t total_tokens = logits.size(0);
    int64_t vocab_size = logits.size(1);
    auto output = torch::empty({total_tokens}, logits.options());

    const int threads = 256;
    const int blocks = total_tokens;

    // 动态分发：支持 Float, Half (FP16) 和 BFloat16
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
        logits.scalar_type(), "fused_logp_kernel", ([&] {
        fused_logp_forward_kernel<scalar_t><<<blocks, threads>>>(
            logits_contig.data_ptr<scalar_t>(),
            token_ids_contig.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(),
            vocab_size);
    }));

    return output;
}