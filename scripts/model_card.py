#!/usr/bin/env python
"""Write MODEL_CARD.md for a released checkpoint, from artefacts only.

Every number is read off disk -- the checkpoint's config and `confidence_summary.json`,
the report directory the release recipe writes (`gates.json`, `probes.json`,
`regression.json`, `by_domain.json`, `per_source.json`), the training run's `args.json`
and `log.jsonl`, and the corpus `meta.json`. Nothing is retyped, so the card cannot drift
from the run it describes, and a missing artefact leaves "not measured" rather than a
guess.

    model_card.py --ckpt runs/<run>-release --report runs/<run>-release/report \\
        --run runs/<run> --corpus data/<corpus> --out runs/<run>-release/MODEL_CARD.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lod.paths import config_path


def load(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def _gates(g: dict) -> list[str]:
    def row(name, measured, bar):
        return f"| {name} | {measured} | {bar} | {fmt((g.get(name) or {}).get('pass'))} |"

    def get(name, key):
        return fmt((g.get(name) or {}).get(key))

    return [
        "| gate | measurement | bar | verdict |", "|---|---|---|---|",
        row("independence", f"max abs Δlogit {get('independence', 'max_abs_delta_logit')}",
            "< 1e-4"),
        row("calibration", f"ECE {get('calibration', 'ece_real')} on real labels "
                           f"({get('calibration', 'ece_all')} on all)",
            "≤ 0.05, ≤ 0.08 per type"),
        row("probability_recovery", f"MAE {get('probability_recovery', 'mae')}, "
                                    f"r {get('probability_recovery', 'r')}", "≤ 0.10, ≥ 0.80"),
        row("temperature_transfer", f"shipped {get('temperature_transfer', 't_shipped')}, "
                                    f"test-fitted {get('temperature_transfer', 't_oracle_testreal')}",
            "≤ 0.25 apart"),
        row("abstention", f"confidence gap {get('abstention', 'gap')}", "> 0.15"),
        row("confidence_beats_maxprob", f"AURC ratio {get('confidence_beats_maxprob', 'ratio')}",
            "≤ 0.95"),
        row("reserved_families", f"{get('reserved_families', 'conf_reserved')} vs "
                                 f"{get('reserved_families', 'conf_trained_families')}", "lower"),
        row("score_coherence", f"max error {get('score_coherence', 'max_abs_error')}", "< 1e-9"),
        row("unseen_entity_abstention", f"gap {get('unseen_entity_abstention', 'gap')}", "> 0.15"),
        row("rule_transfer", f"trained {get('rule_transfer', 'acc_trained_families')}, held out "
                             f"{get('rule_transfer', 'acc_heldout_families')}",
            "drop ≤ 0.10, trained ≥ 0.70"),
        row("sentinel_wordings", f"p(sentinel) {get('sentinel_wordings', 'p_sentinel_heldout')} "
                                 f"on unseen wordings, gap {get('sentinel_wordings', 'gap')}",
            "< 0.30, < 0.15"),
        row("reefer_probe", "see the probes below", "right, or conf < 0.40"),
    ]


def _eval_table(gates: dict, ci: dict | None) -> list[str]:
    comp = gates.get("composition") or {}
    out = ["| questions | n | accuracy | NLL | ECE |", "|---|---|---|---|---|"]
    for name, m in (("all", gates.get("testreal") or {}),
                    ("labels from people or real systems", comp.get("real") or {}),
                    ("labels computed by code", comp.get("generated") or {})):
        if m.get("n"):
            out.append(f"| {name} | {m['n']:,} | {fmt(m.get('accuracy'))} | "
                       f"{fmt(m.get('nll'))} | {fmt(m.get('ece'))} |")
    if ci:
        out += ["", "95 % bootstrap intervals (micro over questions, macro over tasks):", "",
                "| | accuracy | NLL | ECE |", "|---|---|---|---|"]
        for level in ("micro", "macro"):
            cells = [f"{ci[level][k]['value']:.4f} [{ci[level][k]['ci95'][0]:.4f}, "
                     f"{ci[level][k]['ci95'][1]:.4f}]" for k in ("acc", "nll", "ece")]
            out.append(f"| {level} | " + " | ".join(cells) + " |")
    sel = gates.get("selective_prediction") or {}
    if sel:
        pw = sel.get("p_wrong_at_0.9_all") or {}
        wrong = (f"P(wrong | p ≥ 0.9) {fmt(pw.get('p_wrong'))} over {pw['share']:.1%} of "
                 "questions" if pw.get("n") else "no question reached p ≥ 0.9")
        out += ["", f"Selective prediction (max-prob): AURC {fmt(sel.get('aurc_all'))}; "
                    f"{wrong}."]
    return out


def _probe_table(probes: dict) -> list[str]:
    card = probes.get("card_probes") or {}
    bare, crit = card.get("bare") or {}, card.get("with_criteria") or {}
    if not bare and not crit:
        return ["_No `probes.json` in the report._"]
    out = ["| probe | bare options | with criteria |", "|---|---|---|"]
    for qid in sorted(set(bare) | set(crit)):
        def cell(d):
            r = d.get(qid)
            if not r:
                return "not measured"
            return (f"`{r['argmax']}` {max(r['probs']):.3f}, "
                    f"confidence {r['confidence']:.3f}")
        out.append(f"| `{qid}` | {cell(bare)} | {cell(crit)} |")
    return out


def _regression(r: dict | None) -> list[str]:
    if not r:
        return ["_No `regression.json` in the report._"]
    out = []
    kp = r.get("known_posterior") or {}
    if kp.get("n"):
        out.append(f"- **Known posteriors** ({kp['n']} probes with exact Bayes targets): "
                   f"KL(true‖model) {fmt(kp.get('kl'))}, total variation {fmt(kp.get('tv'))}, "
                   f"calibration slope {fmt(kp.get('calib_slope'))} (1 is calibrated), "
                   f"ECE against the posterior {fmt(kp.get('ece_soft'))}.")
    nz = r.get("nonce") or {}
    if nz.get("n_questions"):
        out.append(f"- **Nonce robustness** ({nz['n_questions']} question pairs, random UUIDs "
                   f"inserted where they cannot change the answer): mean |Δ log p| "
                   f"{fmt(nz.get('mean_abs_dlogp'))}, max {fmt(nz.get('max_abs_dlogp'))}, "
                   f"argmax flip rate {fmt(nz.get('flip_rate'))}.")
    pm = r.get("permutation") or {}
    if pm.get("n_questions"):
        out.append(f"- **Option order** ({pm['n_questions']} questions × {pm.get('k')} "
                   f"permutations): mean |Δp| {fmt(pm.get('mean_abs_dp'))}, max |Δp| "
                   f"{fmt(pm.get('max_abs_dp'))}, argmax flip rate {fmt(pm.get('flip_rate'))}.")
    oc = r.get("option_count") or {}
    cells = [f"{b['bucket']}: ECE {b['ece']:.3f} (n {b['n']:,})"
             for b in oc.get("buckets") or [] if b.get("n") and b.get("ece") is not None]
    if cells:
        out.append("- **Calibration by option count**: " + "; ".join(cells) + ".")
    sweep = [s for s in oc.get("sweep") or [] if s.get("fits") and s.get("n")]
    if sweep:
        out.append("- **Synthetic lookup, accuracy by option count**: "
                   + "; ".join(f"N={s['N']}: {s['acc']:.2f}" for s in sweep) + ".")
    return out or ["_`regression.json` is empty._"]


def _by_domain(res: dict | None) -> list[str]:
    if not res or not res.get("domains"):
        return ["_No `by_domain.json` in the report._"]
    out = ["| # | domain | questions | accuracy | NLL | ECE |", "|---|---|---|---|---|---|"]
    for d, m in sorted(res["domains"].items(), key=lambda kv: int(kv[0])):
        out.append(f"| {d} | {m['name']} | {m['n']:,} | {m['acc']:.4f} | {m['nll']:.4f} | "
                   f"{m['ece']:.4f} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="the training run directory")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--name", default=None, help="model name (default: the run's name)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load(config_path(args.ckpt), {})
    gates = load(args.report / "gates.json", {})
    g = gates.get("gates") or {}
    conf = load(args.ckpt / "confidence_summary.json", {})
    run = load(args.run / "args.json", {})
    targs = run.get("args") or {}
    meta = load(args.corpus / "meta.json", {})
    name = args.name or args.run.name

    counts = {}
    for split in ("train", "val", "devreal", "devood", "testreal"):
        p = args.corpus / f"{split}.jsonl"
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                counts[split] = sum(1 for line in fh if line.strip())

    best = None
    if (args.run / "log.jsonl").exists():
        evals = [json.loads(line) for line in (args.run / "log.jsonl").open()
                 if '"kind": "eval"' in line and '"devood"' in line]
        best = min(evals, key=lambda e: e["nll"]) if evals else None

    lora = (f"LoRA r={targs.get('lora_r')} α={targs.get('lora_alpha')}, merged"
            if targs.get("lora_r") else "full fine-tuning")
    t_line = fmt(cfg.get("temperature"))
    if cfg.get("temperature_logn"):
        t_line += f" × (N/2)^{fmt(cfg['temperature_logn'])}"
    lines = [
        f"# {name}",
        "",
        "A prefill-only option-scoring model: one forward pass, no generated tokens, a",
        "calibrated distribution over exactly the options the caller supplied, plus a",
        "separate confidence that the right answer is among them.",
        "",
        "## What it is",
        "",
        "| | |", "|---|---|",
        f"| backbone | `{cfg.get('backbone')}` ({lora}) |",
        f"| options | `{cfg.get('option_mode')}`; comparison layer width "
        f"{cfg.get('mixer_dim')} over the top {cfg.get('mixer_topk')} |",
        f"| budget | {cfg.get('max_state_tokens')} state / {cfg.get('max_total_tokens')} "
        "total tokens; more options are scored in shards |",
        f"| temperature | {t_line} |",
        f"| confidence | `{cfg.get('confidence_mode')}`"
        + (f", features {', '.join(cfg['confidence_features'])}"
           if cfg.get("confidence_features") else "") + " |",
        "",
        "## Evaluation on the test split",
        "",
        "Test tasks are whole tasks never trained on (splits are by task), so these numbers",
        "measure transfer to unseen schemas.",
        "",
        *_eval_table(gates, load(args.report / "per_source.json")),
        "",
        "### By domain",
        "",
        *_by_domain(load(args.report / "by_domain.json")),
        "",
        "## Gates",
        "",
        *_gates(g),
        "",
        ("Blocking gates: independence, calibration, probability_recovery. "
         + ("No gates.json in the report." if not g else
            f"Failed: {', '.join(gates['blocking_failed'])}." if gates.get("blocking_failed")
            else "None failed.")),
        "",
        "## Probes",
        "",
        *_probe_table(load(args.report / "probes.json", {})),
        "",
        "## Regression probes",
        "",
        *_regression(load(args.report / "regression.json")),
        "",
        "## Confidence head",
        "",
        "| | |", "|---|---|",
        f"| mean confidence, options intact | {fmt(conf.get('conf_intact'))} |",
        f"| mean confidence, correct option withheld | {fmt(conf.get('conf_withheld'))} |",
        f"| AURC, head | {fmt(conf.get('head_aurc'))} |",
        f"| AURC, max-prob | {fmt(conf.get('maxprob_aurc'))} |",
        f"| ECE against correctness, head | {fmt(conf.get('head_ece'))} |",
        f"| ECE against correctness, max-prob | {fmt(conf.get('maxprob_ece'))} |",
        "",
        "The head is trained after the scorer, against the frozen model. Its target is *does",
        "the supplied option set contain the right answer, and did the model pick it* -- 0 on",
        "every question whose correct option was withheld, however the distribution fell.",
        "",
        "## Training",
        "",
        "| split | examples |", "|---|---|",
        *[f"| {s} | {n:,} |" for s, n in counts.items()],
        "",
        f"{len(meta.get('tasks') or [])} tasks. {run.get('n_train', 0):,} training examples, "
        f"{run.get('trainable_params', 0):,} trainable parameters, "
        f"{run.get('total_steps', 0):,} steps.",
        "",
        (f"Checkpoint chosen by dev-OOD NLL: {fmt(best['nll'])} at step {best['step']} "
         f"(accuracy {fmt(best['acc'])}, ECE {fmt(best['ece'])})." if best else
         "No dev-OOD evals in the run log."),
        "",
        "```",
        " ".join(f"--{k.replace('_', '-')}" + ("" if v is True else f" {v}")
                 for k, v in sorted(targs.items()) if v not in (None, False, "")),
        "```",
        "",
        "## Limits",
        "",
        "- It cannot answer off-list: the softmax ranges only over the options supplied. If",
        "  the right answer is missing, only `confidence` says so.",
        "- Only the top options by first-stage score are compared with each other; the rest",
        "  are scored on their own.",
        "- Long states are a small part of the training mix.",
        "- English only.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
