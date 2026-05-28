# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

import torch
import kernel_align_ops


class FusedLogp:
    @staticmethod
    def apply(logits: torch.Tensor, token_ids: torch.Tensor):
        """Return selected-token log probabilities with output dtype matching logits."""

        orig_shape = logits.shape[:-1]
        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1)
        results = kernel_align_ops.fused_logp(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    @staticmethod
    def apply_fp32(logits: torch.Tensor, token_ids: torch.Tensor):
        """Return selected-token log probabilities in float32."""

        orig_shape = logits.shape[:-1]
        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1)
        results = kernel_align_ops.fused_logp_fp32(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    @staticmethod
    def out(logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor):
        """Write selected-token log probabilities into a caller-provided output tensor."""

        orig_shape = logits.shape[:-1]
        if output.shape != orig_shape:
            raise ValueError(
                f"output shape {tuple(output.shape)} must match logits leading shape "
                f"{tuple(orig_shape)}"
            )

        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1)
        output_1d = output.view(-1)
        results = kernel_align_ops.fused_logp_out(logits_2d, token_ids_1d, output_1d)
        return results.view(orig_shape)

    @staticmethod
    def indexed_out(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ):
        """Write selected-token log probabilities only for indexed rows."""

        orig_shape = logits.shape[:-1]
        if output.shape != orig_shape:
            raise ValueError(
                f"output shape {tuple(output.shape)} must match logits leading shape "
                f"{tuple(orig_shape)}"
            )

        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1)
        row_indices_1d = row_indices.view(-1)
        output_1d = output.view(-1)
        results = kernel_align_ops.fused_logp_indexed_out(
            logits_2d,
            token_ids_1d,
            row_indices_1d,
            output_1d,
        )
        return results.view(orig_shape)

    @staticmethod
    def indexed_fp32(logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor):
        """Return dense float32 log probabilities with only indexed rows computed."""

        orig_shape = logits.shape[:-1]
        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1)
        row_indices_1d = row_indices.view(-1)
        results = kernel_align_ops.fused_logp_indexed_fp32(logits_2d, token_ids_1d, row_indices_1d)
        return results.view(orig_shape)

    @staticmethod
    def online_out(logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor):
        """Write selected-token log probabilities using online log-sum-exp."""

        orig_shape = logits.shape[:-1]
        if output.shape != orig_shape:
            raise ValueError(
                f"output shape {tuple(output.shape)} must match logits leading shape "
                f"{tuple(orig_shape)}"
            )

        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1)
        output_1d = output.view(-1)
        results = kernel_align_ops.fused_logp_online_out(logits_2d, token_ids_1d, output_1d)
        return results.view(orig_shape)

    @staticmethod
    def online_fp32(logits: torch.Tensor, token_ids: torch.Tensor):
        """Return online log-sum-exp selected-token log probabilities in float32."""

        orig_shape = logits.shape[:-1]
        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1)
        results = kernel_align_ops.fused_logp_online_fp32(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    @staticmethod
    def online_indexed_out(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ):
        """Write online log-sum-exp results only for indexed rows."""

        orig_shape = logits.shape[:-1]
        if output.shape != orig_shape:
            raise ValueError(
                f"output shape {tuple(output.shape)} must match logits leading shape "
                f"{tuple(orig_shape)}"
            )

        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1)
        row_indices_1d = row_indices.view(-1)
        output_1d = output.view(-1)
        results = kernel_align_ops.fused_logp_online_indexed_out(
            logits_2d,
            token_ids_1d,
            row_indices_1d,
            output_1d,
        )
        return results.view(orig_shape)

    @staticmethod
    def online_indexed_fp32(
        logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ):
        """Return float32 online log-sum-exp results for indexed rows."""

        orig_shape = logits.shape[:-1]
        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1)
        row_indices_1d = row_indices.view(-1)
        results = kernel_align_ops.fused_logp_online_indexed_fp32(
            logits_2d, token_ids_1d, row_indices_1d
        )
        return results.view(orig_shape)
