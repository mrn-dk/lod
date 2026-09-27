"""Release gates, computed from a report directory and the checkpoint it describes.

Every gate is read off artefacts already on disk -- the per-question eval dumps, the
independence check's output, the checkpoint's config and `confidence_summary.json`, and
`probes.json` (`lod.evaluation.probes`) -- so this measures and never trains. A gate whose
inputs are missing reports `pass: null` and names what was missing, rather than passing.

Blocking: independence, calibration, probability_recovery.

The test split mixes labels from people and real systems with labels computed by code;
calibration is gated on the real half, and every dump-derived number that matters is
reported for both halves.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from lod.evaluation.dumps import (
    aurc,
    ece,
    p_wrong_at,
    read_dump,
    real_map,
    split_real,
    summarise,
)
from lod.paths import config_path

BLOCKING = ("independence", "calibration", "probability_recovery")


def reserved_prefixes() -> tuple[str, ...]:
    """Task prefixes of the families held out of training whole (reserved for test)."""
    from lod.corpus.services.sources.real.lichess import RESERVED_THEMES

    return tuple(f"lichess_theme_{t}" for t in RESERVED_THEMES) + ("chaosnli_",)


def _independence(report: Path) -> dict:
    path = report / "independence.txt"
    delta = None
    if path.exists():
        m = re.search(r"max \|delta logit\| = ([0-9.eE+-]+)", path.read_text())
        delta = float(m.group(1)) if m else None
    return {"max_abs_delta_logit": delta, "bar": 1e-4,
            "pass": delta is not None and delta < 1e-4}


def _calibration(test: list[dict], real: list[dict], gen: list[dict]) -> dict:
    """ECE <= 0.05 on the real half, and <= 0.08 for each question type within it."""
    def kind(r):
        return "noul" if len(r["probs"]) == 2 else "choice"

    by_kind, by_kind_all = defaultdict(list), defaultdict(list)
    for r in real:
        by_kind[kind(r)].append(r)
    for r in test:
        by_kind_all[kind(r)].append(r)
    per_type = {k: ece(v) for k, v in by_kind.items()}
    ece_real = ece(real)
    return {"ece_real": ece_real, "ece_generated": ece(gen), "ece_all": ece(test),
            "ece_per_type": per_type,
            "ece_per_type_all": {k: ece(v) for k, v in by_kind_all.items()},
            "bar": {"real": 0.05, "per_type": 0.08},
            "pass": (ece_real is not None and ece_real <= 0.05
                     and all(v is not None and v <= 0.08 for v in per_type.values()))}


def _pearson(a: list[float], b: list[float]) -> float | None:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else None


def _probability_recovery(test: list[dict]) -> dict:
    """Binary questions whose target is a known probability (`meta.p`): MAE and r."""
    withp = [r for r in test if (r.get("meta") or {}).get("p") is not None
             and len(r["probs"]) == 2]
    mae = r_corr = None
    by_source = {}
    if withp:
        pm = [r["probs"][1] for r in withp]
        pt = [float(r["meta"]["p"]) for r in withp]
        mae = sum(abs(a - b) for a, b in zip(pm, pt)) / len(withp)
        r_corr = _pearson(pm, pt)
        # the sources are different problems (a count read off the state, a forecaster's
        # number, a market price); the gate is pooled, the breakdown says which one fails
        for src in sorted({r["task"].split("_")[0] for r in withp}):
            rs = [r for r in withp if r["task"].split("_")[0] == src]
            a, b = [r["probs"][1] for r in rs], [float(r["meta"]["p"]) for r in rs]
            by_source[src] = {"n": len(rs), "mae": sum(abs(x - y) for x, y in zip(a, b)) / len(rs),
                              "r": _pearson(a, b)}
    return {"n": len(withp), "mae": mae, "r": r_corr, "bar": {"mae": 0.10, "r": 0.80},
            "pass": mae is not None and mae <= 0.10 and r_corr is not None and r_corr >= 0.80,
            "by_source": by_source}


def _score_coherence(test: list[dict]) -> dict:
    """The wire `score` is exactly sum(i * p[i]) and the wire distribution is the one
    given: every test distribution is put through `api.build_answer` as a score question,
    so the formula is not compared against itself."""
    from lod.serving.api import Parsed, build_answer

    worst, checked = 0.0, 0
    for r in test:
        n = len(r["probs"])
        if not 2 <= n <= 10:
            continue
        keys = [str(i) for i in range(n)]
        answer = build_answer(Parsed("score", None, keys, {k: k for k in keys}), r["probs"], None)
        worst = max(worst, abs(answer["score"] - sum(i * p for i, p in enumerate(r["probs"]))),
                    max(abs(answer["probabilities"][k] - r["probs"][i])
                        for i, k in enumerate(keys)))
        checked += 1
    return {"n_checked": checked, "max_abs_error": worst, "bar": 1e-9,
            "pass": checked > 0 and worst < 1e-9}


def compute(report: Path, ckpt: Path, corpus: Path | None = None,
            probes: Path | None = None) -> dict:
    report, ckpt = Path(report), Path(ckpt)
    test = read_dump(report / "eval-testreal.jsonl")
    real, gen = split_real(test, real_map(corpus))
    cfg_file = config_path(ckpt)
    cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
    summary_file = ckpt / "confidence_summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    gates: dict[str, dict] = {}

    gates["independence"] = _independence(report)
    gates["calibration"] = _calibration(test, real, gen)
    gates["probability_recovery"] = _probability_recovery(test)

    # the shipped temperature against the one that would be fitted on the test split itself
    t_oracle = None
    txt = report / "eval-testreal.txt"
    if txt.exists():
        m = re.search(r"ORACLE temperature.*?T_total = ([0-9.]+)", txt.read_text(), re.S)
        t_oracle = float(m.group(1)) if m else None
    t_ship = cfg.get("temperature")
    gates["temperature_transfer"] = {
        "t_shipped": t_ship, "t_oracle_testreal": t_oracle, "bar": 0.25,
        "pass": (None if t_ship is None or t_oracle is None
                 else abs(t_ship - t_oracle) <= 0.25)}

    # from the confidence head's training run
    gap = summary.get("conf_gap")
    gates["abstention"] = {
        "conf_intact": summary.get("conf_intact"), "conf_withheld": summary.get("conf_withheld"),
        "gap": gap, "bar": 0.15, "pass": None if gap is None or gap != gap else gap > 0.15}
    ratio = summary.get("aurc_ratio")
    gates["confidence_beats_maxprob"] = {
        "head_aurc": summary.get("head_aurc"), "maxprob_aurc": summary.get("maxprob_aurc"),
        "ratio": ratio, "bar": 0.95, "pass": None if ratio is None else ratio <= 0.95}

    prefixes = reserved_prefixes()
    reserved = [r for r in test if r["task"].startswith(prefixes)]
    trained = [r for r in test if not r["task"].startswith(prefixes)]
    cr = sum(r["confidence"] for r in reserved) / len(reserved) if reserved else None
    ct = sum(r["confidence"] for r in trained) / len(trained) if trained else None
    gates["reserved_families"] = {
        "n_reserved": len(reserved), "conf_reserved": cr, "conf_trained_families": ct,
        "pass": None if cr is None or ct is None else cr < ct,
        "note": "max-prob confidence on families never trained on vs the rest"}

    gates["score_coherence"] = _score_coherence(test)

    probes = Path(probes) if probes else report / "probes.json"
    got = json.loads(probes.read_text()) if probes.exists() else {}
    for key in ("unseen_entity_abstention", "rule_transfer", "sentinel_wordings",
                "reefer_probe"):
        if key in got:
            g = dict(got[key])
            g.pop("detail", None)
            gates[key] = g
        else:
            gates[key] = {"pass": None, "missing": f"{probes} (scripts/gates.py probes)"}

    for name in BLOCKING:
        gates[name]["blocking"] = True
    return {
        "n_testreal": len(test),
        "testreal": summarise(test),
        "composition": {"real": summarise(real), "generated": summarise(gen),
                        "generated_share": len(gen) / len(test) if test else None,
                        "source": "corpus meta.json" if corpus else "task-name prefixes"},
        "selective_prediction": {"aurc_real": aurc(real), "aurc_generated": aurc(gen),
                                 "aurc_all": aurc(test),
                                 "p_wrong_at_0.9_real": p_wrong_at(real),
                                 "p_wrong_at_0.9_all": p_wrong_at(test)},
        "gates": gates,
        "blocking_failed": [k for k in BLOCKING if not gates[k].get("pass")],
    }
