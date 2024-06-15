# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from contextlib import nullcontext
from typing import List, Optional
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads  # n_heads per gpu
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_eval: bool = False
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, freqs_cis), apply_rotary_emb(xk, freqs_cis)

        if is_eval:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv
        queries = xq.transpose(1, 2)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        enc_out: Optional[torch.Tensor] = None,
        is_eval=False
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask, is_eval=is_eval
        )
        out_mid = self.feed_forward.forward(self.ffn_norm(h))
        out = out_mid + h
        if enc_out is not None:
            out += enc_out[:, :h.size(1), :]
        return out, out_mid


class Encoder(nn.Module):
    def __init__(self, params: ModelArgs, embedding: nn.Embedding):
        super().__init__()
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = embedding
        self.upscaler = nn.Linear(
            params.dim, embedding.embedding_dim, bias=False
        )  # no bias is NEW!!
        self.downscaler = nn.Linear(
            embedding.embedding_dim, params.dim, bias=False
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads, params.max_seq_len * 2
        )

    def forward(
        self,
        tokens: torch.Tensor = None,
        tokens_embeddings: torch.Tensor = None,
        start_pos: int = 0,
        is_eval=False,
    ):

        assert (tokens is not None) ^ (tokens_embeddings is not None), "Either tokens or tokens_embeddings should be provided"
        if tokens is not None:
            h = self.tok_embeddings(tokens)
        else:
            h = tokens_embeddings
        seqlen = h.size(1)
        h = self.downscaler(h)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h, _ = layer(h, start_pos, freqs_cis, mask, is_eval=is_eval)
        h = self.norm(h)  # do we even need the norm?
        return self.upscaler(h)


class EncodersMoK(nn.Module):
    def __init__(self, encoders: List[Encoder | None]):
        super().__init__()
        self.encoders = encoders

        # used to get pytorch to behave properly
        self.encoders_pytorch = nn.ModuleList(filter(None, self.encoders))
        self.num_experts = len(list(filter(None, encoders)))
        generic_encoder = list(filter(None, encoders))[-1]
        self.tok_embeddings = generic_encoder.tok_embeddings
        # using dec hidden dim as here we're routing before passing knowledge
        # to the encoder i.e. right after embedding
        self.dec_hidden_dim = generic_encoder.downscaler.in_features
        self.gate = nn.Parameter(
            torch.empty((self.num_experts, self.dec_hidden_dim))
        )
        with torch.no_grad():
            torch.nn.init.normal_(self.gate, std=math.sqrt(0.1 / self.dec_hidden_dim))

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, is_eval=False):
        batch_size, seqlen = tokens.shape

        h = self.tok_embeddings(tokens)
        # h = h.reshape(batch_size, self.dec_hidden_dim)
        # TODO: maybe full precision for the gate (see SwitchTransformers)?
        gated_h = F.linear(h, self.gate)
        gated_h = F.softmax(gated_h.mean(dim=1), dim=-1)

        top1_idx = torch.argmax(gated_h, dim=-1)
        mask1 = F.one_hot(top1_idx, num_classes=self.num_experts).to(torch.int32)
        logits_except1 = gated_h.masked_fill(mask1.bool(), float("-inf"))
        top2_idx = torch.argmax(logits_except1, dim=-1)

        gate_loss = self.aux_loss(gated_h, torch.stack([top1_idx, top2_idx], dim=-1), self.num_experts) + self.z_loss(gated_h)

        results = []
        enc_idx = 0

        # FIXME: half precision should not be hardcoded to allow swap between fp16 and bf16
        h = h.to(dtype=torch.bfloat16)
        for encoder in self.encoders:
            if encoder is None:
                results.append(None)
                continue
            expanded = torch.zeros_like(h)
            if enc_idx in top1_idx or enc_idx in top2_idx:
                mask = (top1_idx == enc_idx) | (top2_idx == enc_idx)
                h_selected = h[mask]
                # Use h_selected as needed
                result = encoder(tokens_embeddings=h_selected, start_pos=start_pos, is_eval=is_eval)
                expanded[mask] = result
            results.append(expanded)
            enc_idx += 1
        return gate_loss, results

    # Borrowed from ColossalAI
    def aux_loss(self, router_probs: torch.Tensor, expert_indices: torch.Tensor, num_experts: int) -> None:
        """Computes auxiliary load balancing loss as in Switch Transformer.

        See Switch Transformer (https://arxiv.org/abs/2101.03961). This function
        implements the loss function presented in equations (4) - (6). It aims to
        penalize those cases where the routing between experts is unbalanced.

        Args:
            router_probs: Probability assigned to each expert per token. Shape:
                <float32>[num_groups, tokens_per_group, num_experts].
            expert_indices: <int>[num_groups, tokens_per_group, num_selected_experts]
                indices identifying the top num_selected_experts for a given token.
        """
        if router_probs.dim() == expert_indices.dim() == 2:
            router_probs = router_probs.unsqueeze(0)
            expert_indices = expert_indices.unsqueeze(0)
        assert (
            router_probs.dim() == expert_indices.dim() == 3
        ), "router_probs must be 3D tensor and expert_indices must be 4D tensor"

        # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
        expert_mask = F.one_hot(expert_indices, num_experts)
        # For a given token, determine if it was routed to a given expert.
        # Shape: [num_groups, tokens_per_group, num_experts]
        expert_mask = expert_mask.max(dim=-2)[0]

        tokens_per_group_and_expert = torch.mean(expert_mask.float(), dim=-2)
        router_prob_per_group_and_expert = torch.mean(router_probs.float(), dim=-2)
        aux_loss = num_experts**2 * torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert)
        return aux_loss

    # Borrowed from ColossalAI
    def z_loss(self, router_logits: torch.Tensor):
        """Compute router z-loss.

        The router z-loss was introduced in Designing Effective Sparse Expert Models
        (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
        small in an effort to improve stability.

        Args:
            router_logits: <float>[num_groups, tokens_per_group, num_experts] router logits.
        """
        if router_logits.dim() == 2:
            router_logits = router_logits.unsqueeze(0)
        assert router_logits.dim() == 3, "router_logits must be 3D tensor"
        num_groups, tokens_per_group, _ = router_logits.shape
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = torch.sum(log_z**2, dtype=torch.float32) / (num_groups * tokens_per_group)
        return z_loss
