# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

import torch
from rl_engine.kernels.registry import kernel_registry
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger


def test_accuracy():
    device = device_ctx.device
    dtype = device_ctx.get_preferred_dtype()

    logger.info(f"Running Accuracy Test on: {device} | Dtype: {dtype}")

    G, L, V = 16, 128, 4096

    logits = torch.randn(G * L, V, device=device, dtype=dtype)
    token_ids = torch.randint(0, V, (G * L,), device=device, dtype=torch.int32)

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        ref_logp = torch.log_softmax(logits.float(), dim=-1)
        ref_logp = torch.gather(ref_logp, dim=-1, index=token_ids.unsqueeze(-1).long()).squeeze(-1)
        ref_logp = ref_logp.to(dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()

    try:
        logp_operator = kernel_registry.get_op("logp")
        custom_logp = logp_operator(logits, token_ids)
    except Exception as e:
        logger.error(f"Failed to execute FusedLogp: {e}")
        return

    diff = torch.abs(ref_logp.float() - custom_logp.float()).max().item()
    threshold = 1e-3 if dtype == torch.bfloat16 else 1e-5

    print("\n" + "=" * 50)
    print(f"RESULTS FOR {str(device).upper()}")
    print("-" * 50)
    print(f"Dispatched Operator Class: {logp_operator.__class__.__name__}")
    print(f"Max Difference: {diff:.8e}")

    if diff < threshold:
        print("Status: Accuracy Check Passed!")
    else:
        print("Status: Accuracy Check Failed! (Check your CUDA reduction logic)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    test_accuracy()
