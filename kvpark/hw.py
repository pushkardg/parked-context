"""Model KV-cache geometry, memory tiers/links, and the recompute model.

All model numbers come from the public HuggingFace config.json of each model.
Re-check them against the configs before submission (see README "Verify").
"""
from dataclasses import dataclass, field

BF16 = 2  # bytes per element


@dataclass(frozen=True)
class Model:
    name: str
    attn: str                 # attention family label used in plots
    layers: int
    kv_heads: int             # per-layer KV heads (GQA); ignored for MLA
    head_dim: int
    active_params: float      # parameters touched per token (MoE: active)
    q_heads: int              # query heads (for attention FLOPs)
    mla_dim: int = 0          # MLA: kv_lora_rank + rope dim stored per layer
    window: int = 0           # sliding-window length for local layers (0 = none)
    global_layers: int = -1   # layers with full attention (-1 = all)
    max_ctx: int = 131072
    elem_bytes: int = BF16

    def _full_layers(self) -> int:
        return self.layers if self.global_layers < 0 else self.global_layers

    def per_layer_token_bytes(self) -> int:
        if self.mla_dim:
            return self.mla_dim * self.elem_bytes
        return 2 * self.kv_heads * self.head_dim * self.elem_bytes  # K and V

    def kv_bytes(self, ctx):
        """KV bytes for a context of `ctx` tokens (scalar or numpy array)."""
        import numpy as np
        ctx = np.minimum(np.asarray(ctx, dtype=float), self.max_ctx)
        b = self.per_layer_token_bytes()
        full = self._full_layers()
        local = self.layers - full
        out = full * b * ctx
        if local:
            out = out + local * b * np.minimum(ctx, self.window)
        return out

    def kv_shards(self, gpus: int) -> int:
        """Distinct KV shards across a replica of `gpus` GPUs.
        GQA under tensor parallelism splits KV heads over at most kv_heads GPUs;
        beyond that, heads are replicated. MLA keeps one latent per token, so it
        is served with data-parallel attention: a session's KV lives on one GPU."""
        if self.mla_dim:
            return 1
        return max(1, min(gpus, self.kv_heads))

    def kv_replication(self, gpus: int) -> float:
        """HBM copies of each KV byte across the replica."""
        if self.mla_dim:
            return 1.0
        return gpus / self.kv_shards(gpus)

    def bytes_per_token_full(self) -> float:
        """Asymptotic bytes per extra token (local layers saturated)."""
        return self._full_layers() * self.per_layer_token_bytes()

    def prefill_flops(self, ctx):
        """Approximate FLOPs to recompute a `ctx`-token prefix from scratch.
        2*N per token for weights + causal attention score/value FLOPs."""
        import numpy as np
        ctx = np.minimum(np.asarray(ctx, dtype=float), self.max_ctx)
        lin = 2.0 * self.active_params * ctx
        d = self.q_heads * self.head_dim
        full = self._full_layers()
        local = self.layers - full
        # causal: sum_{i<L} i ~ L^2/2 ; QK^T and AV each 2*d flops per pair
        attn = full * 2.0 * d * ctx ** 2
        if local:
            w = np.minimum(ctx, self.window)
            pairs = ctx * w - w ** 2 / 2.0       # banded causal pairs
            attn = attn + local * 4.0 * d * pairs
        return lin + attn


# --- Models (values from public config.json files; VERIFY before submission) ---
MODELS = {
    "Llama-3.1-8B": Model("Llama-3.1-8B", "GQA", 32, 8, 128, 8.0e9, 32),
    "Llama-3.1-70B": Model("Llama-3.1-70B", "GQA", 80, 8, 128, 70.6e9, 64),
    "Llama-3.1-405B": Model("Llama-3.1-405B", "GQA", 126, 8, 128, 405e9, 128),
    "Qwen3-235B-A22B": Model("Qwen3-235B-A22B", "GQA+MoE", 94, 4, 128, 22e9, 64),
    # MLA: kv_lora_rank 512 + qk_rope_head_dim 64 cached per layer per token.
    # q_heads*head_dim for attention FLOPs: 128 heads x (128 nope + 64 rope) = 192.
    "DeepSeek-V3": Model("DeepSeek-V3", "MLA+MoE", 61, 0, 192, 37e9, 128,
                         mla_dim=576),
    # Gemma 3 27B: 5 local (window 1024) : 1 global; 62 layers -> 10 global.
    "Gemma-3-27B": Model("Gemma-3-27B", "SWA 5:1", 62, 16, 128, 27e9, 32,
                         window=1024, global_layers=10),
    # gpt-oss-120b: alternating full / 128-token banded attention, 36 layers.
    "gpt-oss-120b": Model("gpt-oss-120b", "SWA 1:1+MoE", 36, 8, 64, 5.1e9, 64,
                          window=128, global_layers=18),
}


@dataclass(frozen=True)
class Tier:
    name: str
    bw_gbs: float      # nominal unidirectional bandwidth per GPU, GB/s
    note: str


# Nominal per-direction numbers; EFF converts to sustained.
EFF = 0.8
TIERS = {
    "NVMe Gen5 SSD": Tier("NVMe Gen5 SSD", 14.0, "one Gen5 x4 drive per GPU"),
    "Host DRAM (PCIe 5.0 x16)": Tier("Host DRAM (PCIe 5.0 x16)", 64.0, "PCIe 5.0 x16"),
    "CXL 2.0 x16 expander": Tier("CXL 2.0 x16 expander", 64.0, "same PHY as PCIe 5.0"),
    "Grace LPDDR5X (NVLink-C2C)": Tier("Grace LPDDR5X (NVLink-C2C)", 450.0,
                                       "GH200: 900 GB/s bidirectional"),
}


@dataclass(frozen=True)
class GPU:
    name: str = "H100-SXM"
    hbm_gb: float = 80.0
    bf16_tflops: float = 989.0   # dense
    mfu_prefill: float = 0.5


H100 = GPU()


def reload_seconds(model: Model, ctx, tier: Tier, gpus: int = 1, eff: float = EFF):
    """Time to bring a parked KV back to HBM. The KV is split into
    model.kv_shards(gpus) shards, each loaded over its own GPU's link."""
    return model.kv_bytes(ctx) / (tier.bw_gbs * 1e9 * eff * model.kv_shards(gpus))


def recompute_seconds(model: Model, ctx, gpu: GPU = H100, gpus: int = 1):
    return model.prefill_flops(ctx) / (gpu.bf16_tflops * 1e12 * gpu.mfu_prefill * gpus)


def breakeven_bw_gbs(model: Model, ctx, gpu: GPU = H100, gpus: int = 1):
    """Sustained per-link bandwidth (GB/s) above which reloading beats
    recomputing on a replica of `gpus` GPUs. Equals B*P*u/F when the KV is
    split across all GPUs, and grows by gpus/shards when it is not."""
    return (model.kv_bytes(ctx) / model.kv_shards(gpus)) / recompute_seconds(model, ctx, gpu, gpus) / 1e9
