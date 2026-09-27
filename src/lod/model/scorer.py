"""OptionScoringModel: a prefill-only backbone with a scalar option-scoring head.

No LM head, no generation. The backbone runs once over the state and every question
block; the hidden state at each option's last token is scored by a shared scalar head
(stage 1), an optional comparison layer corrects the top options against each other, and
a question's distribution is a softmax over its own options only -- so an answer is
always one of the options the caller supplied.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from lod.paths import CONFIG_NAME, config_path


def _hf_attn_impl(attn_impl: str) -> str:
    if attn_impl == "flex":
        return "flex_attention"
    if attn_impl in ("sdpa", "nomask"):
        return "sdpa"
    raise ValueError(f"unknown attn_impl {attn_impl!r} "
                     f"(expected 'sdpa', 'flex' or the 'nomask' diagnostic)")

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Backbones with linear/recurrent layers carry state across question blocks and
# silently break independence; the 4-D attention mask cannot stop them.
FORBIDDEN_BACKBONE_HINTS = ("qwen3_next", "qwen3.5", "qwen3_5", "mamba", "rwkv", "deltanet")


def _check_backbone(config) -> None:
    name = (getattr(config, "model_type", "") or "").lower()
    arch = " ".join(getattr(config, "architectures", None) or []).lower()
    for hint in FORBIDDEN_BACKBONE_HINTS:
        if hint in name or hint in arch:
            raise ValueError(
                f"backbone {name or arch!r} has non-softmax attention layers; the block mask "
                "cannot enforce question independence on it"
            )
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and any("attention" not in str(t) for t in layer_types):
        raise ValueError(f"backbone has non-attention layer types {sorted(set(layer_types))}")


class OptionScoringModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        lora_r: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        torch_dtype: torch.dtype = torch.float32,
        max_state_tokens: int = 512,
        max_total_tokens: int = 768,
        temperature: float = 1.0,
        mask_mode: str = "block",
        tokenizer=None,
        backbone=None,
        attach_lora: bool = True,
        attn_impl: str = "sdpa",
        n_heads: int = 1,
        head_seed: int = 0,
        option_mode: str = "sequential",
        mixer_dim: int = 0,
        mixer_heads: int = 4,
        mixer_topk: int = 64,
        temperature_logn: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_state_tokens = max_state_tokens
        self.max_total_tokens = max_total_tokens
        # "block": questions attend to the state and themselves only. "full-causal" is an
        # ablation in which every question also sees the ones before it.
        self.mask_mode = mask_mode
        # Execution detail, not a property of the trained model: "sdpa" takes the dense
        # additive mask, "flex" rebuilds the identical rule as a FlexAttention predicate.
        # Deliberately absent from config_dict() so checkpoints stay interchangeable.
        self.attn_impl = attn_impl
        # "head" (the trained confidence head) or "maxprob" (the top probability, which
        # needs no parameters and is the default until a head is trained).
        self.confidence_mode = "maxprob"
        self.confidence_head: ConfidenceHead | None = None
        # "independent": each option sees the state, its question's head and itself, from
        # a shared start position (lod/model/packing.py). A property of the trained model
        # -- a checkpoint trained one way is wrong packed the other -- so it is in
        # config_dict and every Packer is built from it (`Packer.for_model`).
        if option_mode not in ("sequential", "independent"):
            raise ValueError(f"unknown option_mode {option_mode!r}")
        self.option_mode = option_mode
        # T(N) = temperature * (N/2)^temperature_logn. log N is a feature of the question
        # the calibration map may need: at fixed logits a 200-way softmax is not
        # miscalibrated the way a 2-way one is. 0 is a plain scalar temperature.
        self.temperature_logn = float(temperature_logn)

        if backbone is None:
            backbone = AutoModel.from_pretrained(
                backbone_name, dtype=torch_dtype,
                attn_implementation=_hf_attn_impl(attn_impl),
            )
        _check_backbone(backbone.config)
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        # K pointer heads over the one shared backbone pass. A head is LayerNorm's
        # 2*H affine parameters plus a row of the [H, K] matrix: at K=8 that is a
        # [N, 8] matmul instead of [N, 1], which does not show up in a profile next
        # to a 0.6B prefill. K=1 reproduces the single-head model exactly.
        self.n_heads = int(n_heads)
        self.head_seed = int(head_seed)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, self.n_heads)).float()
        if self.n_heads > 1:
            init_head_rows(self.head[-1], self.head_seed)
        # The comparison layer: one attention layer over a question's option vectors.
        # The head above stays the stage-1 scorer -- independent per option, so it can be
        # sharded and used to shortlist -- and the mixer adds a zero-initialised
        # correction on the top `mixer_topk`, so a mixer bolted onto a trained checkpoint
        # starts out as exactly that checkpoint.
        self.mixer: OptionMixer | None = None
        self.mixer_topk = int(mixer_topk)
        if mixer_dim > 0:
            if self.n_heads > 1:
                raise ValueError("the comparison layer is single-head only")
            self.mixer = OptionMixer(hidden, int(mixer_dim), int(mixer_heads)).float()
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(backbone_name)

        if lora_r > 0 and attach_lora:
            self.attach_lora()

    # ---- construction helpers -------------------------------------------------

    def attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, cfg)

    def enable_gradient_checkpointing(self) -> None:
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if self.lora_r > 0 and hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def backbone_dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    # ---- forward --------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
        position_ids: torch.Tensor,  # [B, T]
        attention_mask: torch.Tensor | None,  # [B, 1, T, T] additive; None under flex
        option_pos: torch.Tensor,  # [N_opt] flattened b*T + pos
        block_ids: torch.Tensor | None = None,  # [B, T], required under flex
        token_valid: torch.Tensor | None = None,  # [B, T], required under flex
        return_option_hidden: bool = False,  # the confidence head trains on these
        option_ids: torch.Tensor | None = None,  # [B, T], independent options only
        option_group: torch.Tensor | None = None,  # [N_opt]; mixer and T(N) need it
        option_slot: torch.Tensor | None = None,  # [N_opt]; mixer needs it
        return_parts: bool = False,  # -> dict(logits, stage1, rows)
        option_grid: tuple[int, int] | None = None,  # (questions, max options), host-side
    ) -> torch.Tensor:
        extra: dict = {}
        if self.attn_impl == "nomask":
            # DIAGNOSTIC ONLY, and semantically wrong: plain causal attention, every
            # question seeing every earlier question. It exists to price the mask --
            # the gap between this and "sdpa" is the whole budget any mask optimisation
            # can win back. Never train or evaluate with it; it breaks independence.
            attention_mask = None
        elif self.attn_impl == "flex":
            if block_ids is None or token_valid is None:
                raise ValueError("attn_impl='flex' needs block_ids and token_valid; "
                                 "the dense mask is not built on this path")
            from lod.model.flexattn import build_flex_mask, kernel_options

            # built here rather than in collate so it lands on-device and never crosses
            # PCIe, and so every caller gets it without threading a BlockMask through
            attention_mask = build_flex_mask(block_ids, token_valid, option_ids=option_ids)
            ko = kernel_options()
            if ko is not None:
                # transformers forwards this from the model kwargs to flex_attention
                extra["kernel_options"] = ko
        elif option_ids is not None and attention_mask is None:
            raise ValueError("independent options need the block mask built with option_ids")
        out = self.backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **extra,
        )
        hidden = out.last_hidden_state  # [B, T, H]
        rows = hidden.reshape(-1, hidden.shape[-1]).index_select(0, option_pos)
        # the head is always fp32, whatever AMP is doing around it
        with torch.autocast(device_type=rows.device.type, enabled=False):
            logits = self.head(rows.float())
        if self.n_heads == 1:
            # [N_opt] for a single head; [N_opt, K] for a K-head ensemble
            logits = logits.squeeze(-1)
        stage1 = logits
        if self.mixer is not None:
            if option_group is None or option_slot is None:
                raise ValueError("the comparison layer needs option_group and option_slot")
            with torch.autocast(device_type=rows.device.type, enabled=False):
                logits = self.mixer.apply_flat(rows.float(), stage1, option_group,
                                               option_slot, self.mixer_topk, option_grid)
        # else [N_opt, K]. The temperature divides every head's logits by the same
        # scalar, which is what `lod.json.temperature` means: one scalar applied to all
        # logits at serve time, fitted against the NLL of the ensemble mean p̄ rather than
        # of any one head (see fit_temperature_ensemble).
        logits = logits / self.temperature_for(option_group, logits)
        if return_parts:
            return {"logits": logits, "stage1": stage1, "rows": rows}
        if return_option_hidden:
            # detached: the confidence head trains against a frozen scoring model
            return logits, rows.detach()
        return logits

    def temperature_for(self, option_group: torch.Tensor | None,
                        logits: torch.Tensor) -> torch.Tensor:
        """The divisor per option: T, or T * (N/2)^slope with N its question's options."""
        if self.temperature_logn == 0.0 or option_group is None:
            return self.temperature
        n = torch.bincount(option_group).to(logits.dtype)[option_group]
        t = self.temperature * torch.exp(self.temperature_logn * (torch.log(n) - math.log(2.0)))
        return t if logits.dim() == 1 else t.unsqueeze(-1)

    # ---- save / load ----------------------------------------------------------

    def config_dict(self) -> dict:
        return {
            "backbone": self.backbone_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "temperature": float(self.temperature.item()),
            "max_state_tokens": self.max_state_tokens,
            "max_total_tokens": self.max_total_tokens,
            "mask_mode": self.mask_mode,
            "confidence_mode": self.confidence_mode,
            "n_heads": self.n_heads,
            "head_seed": self.head_seed,
            "option_mode": self.option_mode,
            "mixer_dim": self.mixer.dim if self.mixer is not None else 0,
            "mixer_heads": self.mixer.n_heads if self.mixer is not None else 0,
            "mixer_topk": self.mixer_topk,
            "temperature_logn": self.temperature_logn,
            "confidence_features": (list(self.confidence_head.features)
                                    if self.confidence_head is not None else []),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / CONFIG_NAME).write_text(json.dumps(self.config_dict(), indent=2) + "\n")
        torch.save(self.head.state_dict(), path / "head.pt")
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), path / "mixer.pt")
        if self.confidence_head is not None:
            torch.save(self.confidence_head.state_dict(), path / "confidence_head.pt")
        if self.lora_r > 0:
            shutil.rmtree(path / "adapter", ignore_errors=True)
            self.backbone.save_pretrained(path / "adapter")
        else:
            shutil.rmtree(path / "backbone", ignore_errors=True)
            self.backbone.save_pretrained(path / "backbone")
        if self.tokenizer is not None:
            # The backbone's tokenizer carries Qwen's chat-and-tool-calling template, and
            # save_pretrained writes it out as chat_template.jinja. Nothing here reads it:
            # packing encodes with add_special_tokens=False and builds its own format. A
            # shipped template advertises a chat interface this model does not have, so
            # drop it rather than let every checkpoint regenerate it.
            self.tokenizer.chat_template = None
            self.tokenizer.save_pretrained(path)

    def attach_confidence_head(self, state: dict | None = None,
                               features: tuple[str, ...] | list[str] = ()) -> "ConfidenceHead":
        """Create (and optionally load) the confidence head sized to this backbone.

        `features` names scalar features of the question's distribution the head reads
        besides the pooled hidden state (log N and the shape of the distribution; see
        DISTRIBUTION_FEATURES). With none it reads the pooled hidden state only.
        """
        head = ConfidenceHead(self.backbone.config.hidden_size
                              if hasattr(self.backbone, "config")
                              else self.head[-1].in_features, features=tuple(features))
        if state is not None:
            head.load_state_dict(state)
        self.confidence_head = head.to(self.device).float()
        self.confidence_mode = "head"
        return self.confidence_head

    @staticmethod
    def load(
        path: str | Path,
        torch_dtype: torch.dtype = torch.float32,
        merge_lora: bool = True,
        attn_impl: str = "sdpa",
    ) -> "OptionScoringModel":
        path = Path(path)
        cfg = json.loads(config_path(path).read_text())
        base = str(path / "backbone") if (path / "backbone").exists() else cfg["backbone"]
        backbone = AutoModel.from_pretrained(base, dtype=torch_dtype,
                                             attn_implementation=_hf_attn_impl(attn_impl))

        lora_r = int(cfg.get("lora_r", 0))
        adapter = path / "adapter"
        if lora_r > 0 and adapter.exists():
            from peft import PeftModel

            backbone = PeftModel.from_pretrained(backbone, str(adapter), is_trainable=not merge_lora)
            if merge_lora:
                backbone = backbone.merge_and_unload()
                lora_r = 0  # merged: a re-save writes backbone/, not adapter/

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(path))
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])

        head_state = torch.load(path / "head.pt", map_location="cpu")
        # A checkpoint whose config has no n_heads has a [1, H] final Linear. Reading K
        # off the tensor rather than trusting the config keeps checkpoints
        # self-describing.
        n_heads = int(cfg.get("n_heads", 0)) or int(head_state["1.weight"].shape[0])

        model = OptionScoringModel(
            backbone_name=cfg["backbone"],
            lora_r=lora_r,
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            max_state_tokens=int(cfg.get("max_state_tokens", 512)),
            max_total_tokens=int(cfg.get("max_total_tokens", 768)),
            temperature=float(cfg.get("temperature", 1.0)),
            mask_mode=str(cfg.get("mask_mode", "block")),
            attn_impl=attn_impl,
            tokenizer=tokenizer,
            backbone=backbone,
            attach_lora=False,
            n_heads=n_heads,
            head_seed=int(cfg.get("head_seed", 0)),
            option_mode=str(cfg.get("option_mode", "sequential")),
            mixer_dim=int(cfg.get("mixer_dim", 0)),
            mixer_heads=int(cfg.get("mixer_heads", 4)) or 4,
            mixer_topk=int(cfg.get("mixer_topk", 64)),
            temperature_logn=float(cfg.get("temperature_logn", 0.0)),
        )
        model.head.load_state_dict(head_state)
        if model.mixer is not None:
            model.mixer.load_state_dict(torch.load(path / "mixer.pt", map_location="cpu"))
        model.confidence_mode = str(cfg.get("confidence_mode", "maxprob"))
        cpath = path / "confidence_head.pt"
        if cpath.exists():
            # load() reconstructs the shipped model, confidence head included, with no
            # flags
            model.attach_confidence_head(torch.load(cpath, map_location="cpu"),
                                         features=cfg.get("confidence_features", []))
        return model

    # ---- one call that returns everything a caller serves ---------------------

    @torch.no_grad()
    def score_batch(self, batch: dict, device, amp_dtype: torch.dtype | None = None) -> dict:
        """Run one collated batch -> grouped logits (temperature applied), probabilities,
        and the confidence the checkpoint ships (head, or max-prob).

        The one scoring path shared by serving, the probes and the confidence trainer, so
        a new forward input cannot be forgotten at one call site.
        """
        with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            parts = forward_batch(self, batch, device, return_parts=True)
        group = batch["option_group"].to(device)
        slot = batch["option_slot"].to(device)
        valid = batch["valid"].to(device)
        grouped = grouped_logits(parts["logits"].float(), group, slot, valid)
        probs = probs_from_grouped(grouped, valid)
        rows, keep = parts["rows"].float(), None
        if self.mixer is not None and parts["stage1"].dim() == 1:
            # confidence reads the shortlist the comparison layer saw, which is also all
            # a sharded pass keeps (lod/model/sharded.py)
            s1 = grouped_logits(parts["stage1"].float(), group, slot, valid)
            k = min(self.mixer_topk, s1.shape[1])
            top = s1.topk(k, dim=-1).indices
            short = torch.zeros_like(valid).scatter_(1, top, True) & valid
            keep = short[group, slot]
        pooled = (pool_option_hidden(rows, group, grouped.shape[0]) if keep is None else
                  pool_option_hidden(rows[keep], group[keep], grouped.shape[0]))
        feats = distribution_features(grouped, valid)
        conf = probs.max(dim=-1).values
        if self.confidence_mode == "head" and self.confidence_head is not None:
            conf = self.confidence_head(pooled, feats)
        return {"grouped": grouped, "valid": valid, "probs": probs, "pooled": pooled,
                "features": feats, "confidence": conf,
                "stage1": grouped_logits(parts["stage1"].float(), group, slot, valid)
                if parts["stage1"].dim() == 1 else None}


def init_head_rows(linear: nn.Linear, seed: int) -> None:
    """Re-initialise each output row of `linear` from its own seeded generator.

    For a K-head ensemble. `nn.Linear` already draws U(-1/sqrt(H), 1/sqrt(H)); this
    draws the same distribution but per head, from `seed + k`, so head k's starting
    point is a function of (head_seed, k) alone and a run is reproducible head by head.
    """
    fan_in = linear.in_features
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        for k in range(linear.out_features):
            g = torch.Generator().manual_seed(int(seed) * 1_000_003 + k)
            linear.weight[k].copy_(
                torch.empty(fan_in).uniform_(-bound, bound, generator=g))
            if linear.bias is not None:
                linear.bias[k].copy_(
                    torch.empty(1).uniform_(-bound, bound, generator=g)[0])


def grouped_head_logits(
    logits: torch.Tensor,  # [N_opt, H]
    option_group: torch.Tensor,  # [N_opt] question index
    option_slot: torch.Tensor,  # [N_opt] option index within the question
    valid: torch.Tensor,  # [Nq, K] bool
) -> torch.Tensor:
    """Scatter per-head option logits into [Nq, K, H], -inf where a slot has no option."""
    nq, k = valid.shape
    h = logits.shape[-1]
    out = torch.full((nq, k, h), float("-inf"), dtype=logits.dtype, device=logits.device)
    return out.index_put((option_group, option_slot), logits)


def log_mean_prob(head_grouped: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[Nq, K, H] per-head logits -> [Nq, K] log p̄, p̄ = mean_k softmax(logits_k).

    The ensemble predicts with the mean of the per-head *probability* distributions,
    not the mean of the logits. Mean-of-logits is a geometric mean of the
    distributions after renormalising: it is dominated by whichever head is most
    confident that an option is wrong (one head saying 1e-6 drags the product down
    however sure the other seven are), so a single mistrained head can veto the
    ensemble. The arithmetic mean cannot be more confident than its most confident
    member, which is the behaviour we want from a model whose cited failure is being
    confidently wrong.

    The result is returned as log p̄ rather than as p̄ so that everything downstream --
    temperature division, softmax, metrics, decision_loss -- keeps working on a
    [Nq, K] "logits" tensor with the usual -inf-on-invalid convention. softmax(log p̄)
    is p̄ exactly, so at T = 1 nothing is changed by the round trip.
    """
    masked = head_grouped.float().masked_fill(~valid.unsqueeze(-1), float("-inf"))
    logp = torch.log_softmax(masked, dim=1)  # over a question's own options
    n = head_grouped.shape[-1]
    mean_logp = torch.logsumexp(logp, dim=-1) - math.log(n)
    return torch.where(valid, mean_logp, torch.full_like(mean_logp, float("-inf")))


def grouped_logits(
    logits: torch.Tensor,  # [N_opt], or [N_opt, H] for a K-head ensemble
    option_group: torch.Tensor,  # [N_opt] question index
    option_slot: torch.Tensor,  # [N_opt] option index within the question
    valid: torch.Tensor,  # [Nq, K] bool
) -> torch.Tensor:
    """Scatter flat option logits into [Nq, K], -inf where a slot has no option.

    Given per-head logits it returns the ensemble's log p̄ instead, so every caller
    (serve, predict, engine, the probes) gets the shipped distribution without
    knowing whether the checkpoint has one head or eight.
    """
    if logits.dim() == 2:
        return log_mean_prob(grouped_head_logits(logits, option_group, option_slot, valid),
                             valid)
    nq, k = valid.shape
    out = torch.full((nq, k), float("-inf"), dtype=logits.dtype, device=logits.device)
    out = out.index_put((option_group, option_slot), logits)
    return out


def shard_select(head_grouped: torch.Tensor, shard: torch.Tensor) -> torch.Tensor:
    """[Nq, K, H] and a per-question head index -> [Nq, K], that head's logits.

    For a K-head ensemble: each example's loss is read off exactly one head, so head
    k's weights only ever see its own ~1/H of the corpus, which decorrelates them. The
    *backbone* still gets gradient from every example -- routed through whichever
    head owns it -- which is the point of paying for one shared pass.
    """
    idx = shard.to(head_grouped.device).view(-1, 1, 1).expand(-1, head_grouped.shape[1], 1)
    return head_grouped.gather(2, idx).squeeze(-1)


class ConfidenceHead(nn.Module):
    """P(the argmax is correct and the right answer is among the options), per question.

    A Linear on the mean of a question's option hidden states, plus optionally a Linear
    on scalar features of its distribution. Max-prob is already available for free, so a
    head that only re-read the probabilities would add little; reading the hidden states
    lets it notice that a question is unlike anything in training even when the
    distribution looks confident.

    The gradient is stopped at the hidden states: the head is trained against a *frozen*
    scoring model, so the backbone must not move.
    """

    def __init__(self, hidden: int, features: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, 1)
        self.features = tuple(features)
        unknown = set(self.features) - set(DISTRIBUTION_FEATURES)
        if unknown:
            raise ValueError(f"unknown confidence features {sorted(unknown)}")
        # log N and the distribution's shape. Zero-initialised, so a head with features
        # starts as the pooled-hidden head it extends.
        self.feat = nn.Linear(len(self.features), 1, bias=False) if self.features else None
        if self.feat is not None:
            nn.init.zeros_(self.feat.weight)

    def forward(self, pooled: torch.Tensor, feats: dict | None = None) -> torch.Tensor:
        z = self.proj(pooled.float()).squeeze(-1)
        if self.feat is not None:
            if feats is None:
                raise ValueError(f"this confidence head reads {self.features}")
            x = torch.stack([feats[k].float() for k in self.features], dim=-1)
            z = z + self.feat(x).squeeze(-1)
        return torch.sigmoid(z)


DISTRIBUTION_FEATURES = ("log_n", "max_p", "entropy_ratio", "margin", "logit_gap")


def distribution_features(grouped: torch.Tensor, valid: torch.Tensor) -> dict:
    """Scalar features of each question's (global) distribution, [Nq] each.

    `log_n` is how many options the question had; `entropy_ratio` is entropy / log N,
    so it is comparable across N; `margin` is p1 - p2; `logit_gap` is the top logit minus
    the runner-up's, which is what a sharded or shortlisted score still has when the tail
    was never materialised.
    """
    g = grouped.float().masked_fill(~valid, float("-inf"))
    p = torch.softmax(g, dim=-1)
    p = torch.where(valid, p, torch.zeros((), dtype=p.dtype, device=p.device))
    n = valid.sum(-1).clamp_min(1).to(p.dtype)
    logp = torch.where(p > 0, p.clamp_min(1e-30).log(), torch.zeros_like(p))
    ent = -(p * logp).sum(-1)
    top = p.topk(min(2, p.shape[-1]), dim=-1).values
    p2 = top[:, 1] if top.shape[-1] > 1 else torch.zeros_like(top[:, 0])
    lt = g.topk(min(2, g.shape[-1]), dim=-1).values
    gap = (lt[:, 0] - lt[:, 1]) if lt.shape[-1] > 1 else torch.zeros_like(lt[:, 0])
    gap = torch.nan_to_num(gap, posinf=50.0, neginf=0.0).clamp(max=50.0)
    return {"log_n": n.log(), "max_p": top[:, 0], "entropy_ratio": ent / n.log().clamp_min(1e-6),
            "margin": top[:, 0] - p2, "logit_gap": gap}


class OptionMixer(nn.Module):
    """The comparison layer: the only place a question's options see each other.

    One pre-norm transformer layer over the per-option vectors, with no positional
    encoding, so it is permutation-equivariant and the question's distribution is
    order-invariant to floating-point error. Its input is each option's hidden state plus
    its stage-1 log-probability (so it can compare scores as well as content); its output
    is an additive correction to the stage-1 logit, zero-initialised.

    Two-stage by construction. Only the top `topk` options by stage-1 score are mixed;
    the rest keep their stage-1 logit, and the mixed ones are shifted so that together
    they keep exactly the stage-1 probability mass of the shortlist:

        final(o) = s(o)                                  o outside the shortlist
        final(o) = s(o) + d(o) - [lse_S(s + d) - lse_S(s)]   o in shortlist S

    so softmax(final) = (1 - tail) * softmax_S(s + d) on S and the stage-1 distribution on
    the tail. That is computable from a sharded pass -- the tail mass is a global
    logsumexp of independent scores -- and it is the same function in training, where N
    is usually below `topk` and the shortlist is every option.
    """

    def __init__(self, hidden: int, dim: int = 256, n_heads: int = 4) -> None:
        super().__init__()
        self.dim, self.n_heads = int(dim), int(n_heads)
        self.norm_in = nn.LayerNorm(hidden)
        self.inp = nn.Linear(hidden, dim)
        self.score_in = nn.Linear(1, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.out = nn.Linear(dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def delta(self, rows: torch.Tensor, logp1: torch.Tensor,
              keep: torch.Tensor) -> torch.Tensor:
        """rows [Nq, K, H], logp1 [Nq, K] stage-1 log-probs, keep [Nq, K] -> d [Nq, K]."""
        x = self.inp(self.norm_in(rows)) + self.score_in(
            torch.where(keep, logp1, torch.zeros_like(logp1)).unsqueeze(-1))
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=~keep, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        d = self.out(x).squeeze(-1)
        return torch.where(keep, d, torch.zeros_like(d))

    def combine(self, rows: torch.Tensor, s: torch.Tensor, valid: torch.Tensor,
                topk: int) -> torch.Tensor:
        """Grouped form: rows [Nq, K, H], stage-1 logits s [Nq, K] -> final [Nq, K]."""
        s = s.float()
        neg = torch.full_like(s, float("-inf"))
        sm = torch.where(valid, s, neg)
        if s.shape[1] > topk:
            idx = sm.topk(topk, dim=-1).indices
            keep = torch.zeros_like(valid).scatter_(1, idx, True) & valid
        else:
            keep = valid
        logp1 = torch.log_softmax(sm, dim=-1)
        d = self.delta(rows, logp1, keep)
        mixed = torch.where(keep, s + d, neg)
        shift = torch.logsumexp(mixed, -1, keepdim=True) - \
            torch.logsumexp(torch.where(keep, s, neg), -1, keepdim=True)
        final = torch.where(keep, s + d - shift, s)
        return torch.where(valid, final, neg)

    def apply_flat(self, rows: torch.Tensor, s: torch.Tensor, option_group: torch.Tensor,
                   option_slot: torch.Tensor, topk: int,
                   grid: tuple[int, int] | None = None) -> torch.Tensor:
        """Flat [N_opt] in and out, as the model's forward carries them.

        `grid` is (questions, max options) when the caller knows it on the host -- the
        collated batch does -- which saves two device->host syncs per forward."""
        if grid is not None:
            nq, k = int(grid[0]), int(grid[1])
        else:
            nq = int(option_group.max().item()) + 1 if option_group.numel() else 0
            k = int(option_slot.max().item()) + 1 if option_slot.numel() else 0
        valid = torch.zeros(nq, k, dtype=torch.bool, device=s.device)
        valid[option_group, option_slot] = True
        g_s = torch.zeros(nq, k, dtype=torch.float32, device=s.device)
        g_s[option_group, option_slot] = s.float()
        g_r = torch.zeros(nq, k, rows.shape[-1], dtype=rows.dtype, device=rows.device)
        g_r[option_group, option_slot] = rows
        final = self.combine(g_r, g_s, valid, topk)
        return final[option_group, option_slot]


def pool_option_hidden(rows: torch.Tensor, option_group: torch.Tensor,
                       n_questions: int) -> torch.Tensor:
    """Mean of each question's option hidden states -> [n_questions, H]."""
    hidden = rows.shape[-1]
    out = torch.zeros(n_questions, hidden, dtype=rows.dtype, device=rows.device)
    counts = torch.zeros(n_questions, 1, dtype=rows.dtype, device=rows.device)
    idx = option_group.to(rows.device).unsqueeze(-1).expand(-1, hidden)
    out.scatter_add_(0, idx, rows)
    counts.scatter_add_(0, option_group.to(rows.device).unsqueeze(-1),
                        torch.ones_like(option_group, dtype=rows.dtype).unsqueeze(-1))
    return out / counts.clamp(min=1.0)


def forward_batch(model, batch: dict, device, non_blocking: bool = False,
                  **kw) -> torch.Tensor:
    """Move one collated batch to `device` and run the model on it.

    One place that knows what the model's forward needs (block ids, option ids, the
    option grid), so no call site can quietly fall back to the dense path by forgetting
    one. `attention_mask` is None under flex.
    """
    am = batch["attention_mask"]
    oi = batch.get("option_ids")
    return model(
        batch["input_ids"].to(device, non_blocking=non_blocking),
        batch["position_ids"].to(device, non_blocking=non_blocking),
        am.to(device, non_blocking=non_blocking) if am is not None else None,
        batch["option_pos"].to(device, non_blocking=non_blocking),
        block_ids=batch["block_ids"].to(device, non_blocking=non_blocking),
        token_valid=batch["token_valid"].to(device, non_blocking=non_blocking),
        option_ids=oi.to(device, non_blocking=non_blocking) if oi is not None else None,
        option_group=batch["option_group"].to(device, non_blocking=non_blocking),
        option_slot=batch["option_slot"].to(device, non_blocking=non_blocking),
        option_grid=tuple(batch["valid"].shape),
        **kw,
    )


def decision_loss(
    grouped: torch.Tensor,  # [Nq, K]
    valid: torch.Tensor,  # [Nq, K] bool
    target: torch.Tensor,  # [Nq, K] probabilities over valid slots
    kind: str = "ce",
    has_target: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Proper scoring rule over each question's own options, mean over questions.

    `ce` is the log score (cross-entropy to the target distribution, which may be
    hard or soft); `brier` is the squared error of the probability vector.
    """
    if has_target is not None:
        grouped, valid, target = grouped[has_target], valid[has_target], target[has_target]
        if weight is not None:
            weight = weight[has_target]
    if grouped.numel() == 0:
        return grouped.sum() * 0.0

    g = grouped.float()
    zero = torch.zeros((), dtype=g.dtype, device=g.device)
    if kind == "ce":
        logp = torch.log_softmax(g, dim=-1)
        logp = torch.where(valid, logp, zero)  # invalid slots are -inf; 0 * -inf is NaN
        per_q = -(target * logp).sum(dim=-1)
    elif kind == "brier":
        p = torch.softmax(g, dim=-1)
        p = torch.where(valid, p, zero)
        per_q = ((p - target) ** 2 * valid).sum(dim=-1)
    else:
        raise ValueError(f"unknown loss kind {kind!r}")
    if weight is None:
        return per_q.mean()
    w = weight.to(per_q.dtype)
    return (per_q * w).sum() / w.sum().clamp_min(1e-8)


def probs_from_grouped(grouped: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Softmax over each question's valid options; exactly 0 elsewhere."""
    p = torch.softmax(grouped.float(), dim=-1)
    return torch.where(valid, p, torch.zeros((), dtype=p.dtype, device=p.device))
