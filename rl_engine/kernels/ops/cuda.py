# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

import torch
from rl_engine.utils.logger import logger

try:
    from rl_engine import _C

    _EXT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Core binary extension (_C) unavailable: {e}. Falling back to native code.")
    _EXT_AVAILABLE = False


class FusedLogpSM90Op:
    """TMA-accelerated Fused LogP for SM90+ cards."""

    def __init__(self):
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_logp_sm90"):
            raise RuntimeError(
                "TMA Fused LogP kernel is not compiled or unsupported on this card architecture. "
                "Please rebuild extension using 'pip install -e .'"
            )
        self.op = _C.fused_logp_sm90
        logger.info("Successfully linked to precompiled _C.fused_logp_sm90 kernel.")

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert logits.dtype == torch.bfloat16, "TMA logp currently requires bfloat16 logits"
        assert logits.is_contiguous(), "Logits must be contiguous for TMA block loading"
        labels_fused = labels.to(device=logits.device, dtype=torch.int32).contiguous()
        return self.op(logits, labels_fused)


class FusedLogpGenericOp:
    """Generic custom CUDA/ROCm fallback Fused LogP."""

    def __init__(self):
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_logp"):
            raise RuntimeError("Base custom kernel 'fused_logp' is unavailable.")
        self.op = _C.fused_logp
        logger.info("Successfully linked to precompiled _C.fused_logp fallback kernel.")

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids_fused = token_ids.to(device=logits.device, dtype=torch.int32).contiguous()
        return self.op(logits, token_ids_fused)
