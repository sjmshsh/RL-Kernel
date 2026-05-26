# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

#include <torch/extension.h>

torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids);

#ifdef __CUDACC__
torch::Tensor fused_logp_sm90_forward(torch::Tensor logits, torch::Tensor labels);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Kernel-Align High-Performance Operator Extension Library";

    m.def(
        "fused_logp",
        &fused_logp_forward,
        "Generic Fused LogP Forward Operator"
    );

#ifdef __CUDACC__
    m.def(
        "fused_logp_sm90",
        &fused_logp_sm90_forward,
        "TMA-accelerated Online Softmax Fused LogP for SM90+ (Warp Specialization)"
    );
#endif
}
