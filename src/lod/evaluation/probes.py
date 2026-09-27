"""Generated probe sets for the gates an eval dump cannot measure.

Each of these needs a *contrast* a test-split dump does not carry, so each is a small
generated probe set scored directly, with the checkpoint's shipped confidence:

    unseen_entity_abstention  the same unseen entities with the fact present, contradicted
                              or absent: mean confidence on supported minus not-in-context
                              > 0.15
    rule_transfer             accuracy on held-out combinations of rule primitives within
                              0.10 of trained combinations, trained >= 0.70
    sentinel_wordings         with the right answer present, p("none of the above") under
                              wordings never trained on < 0.30, and within 0.15 of trained
                              wordings -- the sentinel is read, not matched as a string
    reefer_probe              a rule that names no field (cargo outside its temperature
                              range): right, or wrong with confidence < 0.40

`scripts/gates.py probes` writes the result; `lod.evaluation.gates` folds it into the gate
report.
"""

from __future__ import annotations

import itertools
import json
import random

import torch

from lod import sentinel
from lod.model.packing import Packer, collate
from lod.schema import Example, Question


@torch.no_grad()
def score(model, packer: Packer, examples, device, batch_size: int = 16) -> list[dict]:
    """-> one record per question: probs, the shipped confidence, whether it was right.

    Questions without a target are kept: they are the not-in-context rows, whose whole
    purpose here is that the confidence should be low on them.
    """
    out: list[dict] = []
    for start in range(0, len(examples), batch_size):
        packed, kept = [], []
        for e in examples[start: start + batch_size]:
            pk = packer.pack(e)
            if pk is None:
                continue
            packed.append(pk)
            # `pack` keeps a prefix of the question list when it has to drop to fit
            kept += [(e.task, q) for q in e.questions[: pk.n_questions]]
        if not packed:
            continue
        res = model.score_batch(collate(packed, packer.pad_id, model.backbone_dtype), device)
        probs, conf = res["probs"].cpu(), res["confidence"].float().cpu()
        for qi, (task, q) in enumerate(kept):
            n = len(q.options)
            p = probs[qi, :n]
            tgt = q.target_probs()
            out.append({
                "task": task, "qid": q.id, "meta": q.meta or {}, "options": list(q.options),
                "probs": [float(x) for x in p], "maxprob": float(p.max()),
                "confidence": float(conf[qi]),
                "argmax": q.options[int(p.argmax())],
                "correct": (None if tgt is None
                            else bool(int(p.argmax()) == max(range(n), key=tgt.__getitem__))),
            })
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ---- unseen entities ------------------------------------------------------------------

def unseen_entity_abstention(model, packer, device, n: int = 600) -> dict:
    from lod.corpus.services.sources.synth import entities

    task = {t.name: t for t in entities.tasks()}["entity_fact_unseen"]
    recs = score(model, packer, list(itertools.islice(task.load(n), n)), device)
    by: dict = {}
    for r in recs:
        by.setdefault(r["meta"].get("case"), []).append(r["confidence"])
    sup, nic = _mean(by.get("supported", [])), _mean(by.get("not_in_context", []))
    gap = None if sup is None or nic is None else sup - nic
    return {"n": len(recs), "conf_supported": sup, "conf_refuted": _mean(by.get("refuted", [])),
            "conf_not_in_context": nic, "gap": gap, "bar": 0.15,
            "pass": gap is not None and gap > 0.15}


# ---- rule transfer --------------------------------------------------------------------

def rule_transfer(model, packer, device, per_family: int = 150) -> dict:
    """Held-out rule families are whole combinations of primitives the model trained on.

    Trained and held-out groups are matched (same depth, same leaf count, the same
    primitives on both sides), so the difference measures composition, not a change of
    shape. A floor on trained accuracy keeps "equally bad at both" from passing, and the
    per-primitive accuracies are the diagnosis when a combination fails.
    """
    from lod.corpus.services.sources.synth import rules

    by_name = {t.name: t for t in rules.probe_tasks()}
    acc: dict[str, float | None] = {}
    by_mode: dict[str, list[int]] = {"direct": [0, 0], "indirect": [0, 0]}
    for fam in rules.FAMILIES:
        task = by_name[f"rule_{fam}"]
        recs = [r for r in score(model, packer,
                                 list(itertools.islice(task.load(per_family), per_family)),
                                 device) if r["correct"] is not None]
        acc[fam] = _mean(r["correct"] for r in recs)
        if fam in rules.TRAINED_FAMILIES:
            # a rule either names its field or only describes what the field records
            for r in recs:
                mode = r["meta"].get("mode")
                if mode in by_mode:
                    by_mode[mode][0] += 1
                    by_mode[mode][1] += bool(r["correct"])
    trained = _mean(acc[f] for f in rules.TRAINED_COMBO_FAMILIES)
    heldout = _mean(acc[f] for f in rules.HELDOUT_COMBO_FAMILIES)
    drop = None if trained is None or heldout is None else trained - heldout
    floor = 0.70
    return {"acc_trained_families": trained, "acc_heldout_families": heldout, "drop": drop,
            "bar": {"drop": 0.10, "acc_trained_floor": floor},
            "pass": (drop is not None and drop <= 0.10 and trained is not None
                     and trained >= floor),
            "acc_per_family": {k: round(v, 4) for k, v in acc.items() if v is not None},
            "acc_per_primitive": {p: round(acc[p], 4) for p in rules.PRIMITIVES
                                  if acc.get(p) is not None},
            "acc_by_reference_mode": {m: {"n": n, "acc": round(k / n, 4)}
                                      for m, (n, k) in by_mode.items() if n},
            "heldout_families": list(rules.HELDOUT_COMBO_FAMILIES),
            # reported, not gated: every family, and the different-shape control
            "acc_trained_all_families": _mean(acc[f] for f in rules.TRAINED_FAMILIES),
            "acc_heldout_all_families": _mean(acc[f] for f in rules.HELDOUT_FAMILIES),
            "acc_shape_control": {f: round(acc[f], 4) for f in rules.HELDOUT_SHAPE_FAMILIES
                                  if acc.get(f) is not None}}


# ---- sentinel wordings ----------------------------------------------------------------

SUPPORT_STATE = (
    "From: dana@northfield.example\nSubject: charged twice for September\n\n"
    "Hi — my card shows two charges of 49.00 for the September subscription, three "
    "days apart. I only have one account. Please refund the duplicate; I do not want "
    "to cancel the plan."
)
SUPPORT_OPTIONS = ["billing", "platform", "design"]
SUPPORT_DESCS = ["Charges, invoices, refunds and payment disputes",
                 "Outages, errors, latency and anything the service itself did wrong",
                 "Layout, wording, colours and other appearance feedback"]
SUPPORT_GOLD = 0
PARAPHRASES = ["Anything else", "Some other team", "A team not listed here",
               "Does not fit the teams above", "Other"]


def _sentinel_probe(model, packer, device, wordings: list[str], key: str = "other"):
    examples = [Example(state=SUPPORT_STATE, task="probe_sentinel", questions=[
        Question("route", "Which team should handle this message?", SUPPORT_OPTIONS + [key],
                 None, None, None, SUPPORT_DESCS + [w])]) for w in wordings]
    return [{"wording": w, "p_sentinel": r["probs"][-1], "p_gold": r["probs"][SUPPORT_GOLD],
             "argmax": r["argmax"], "confidence": r["confidence"]}
            for w, r in zip(wordings, score(model, packer, examples, device))]


def sentinel_wordings(model, packer, device, k: int = 24) -> dict:
    rng = random.Random(0)
    trained = [sentinel.sample(rng, "train")[1] for _ in range(k)]
    heldout = [sentinel.sample(rng, "eval")[1] for _ in range(k)]
    rows = {"trained_wordings": _sentinel_probe(model, packer, device, trained),
            "heldout_wordings": _sentinel_probe(model, packer, device, heldout),
            "paraphrases": _sentinel_probe(model, packer, device, PARAPHRASES)}
    p_tr = _mean(r["p_sentinel"] for r in rows["trained_wordings"])
    p_ho = _mean(r["p_sentinel"] for r in rows["heldout_wordings"])
    gap = None if p_tr is None or p_ho is None else abs(p_tr - p_ho)
    return {"p_sentinel_trained": p_tr, "p_sentinel_heldout": p_ho,
            "p_sentinel_paraphrase": _mean(r["p_sentinel"] for r in rows["paraphrases"]),
            "gap": gap,
            "wrong_answer_rate_heldout": _mean(r["argmax"] != SUPPORT_OPTIONS[SUPPORT_GOLD]
                                               for r in rows["heldout_wordings"]),
            "bar": {"p_heldout": 0.30, "gap": 0.15},
            "pass": p_ho is not None and p_ho < 0.30 and gap is not None and gap < 0.15,
            "detail": rows,
            "note": "the correct answer (billing) is present in every one of these"}


# ---- the two card probes --------------------------------------------------------------

REEFER_STATE = json.dumps({
    "cargo": "refrigerated fish", "reefer_temp_c": 4.8, "setpoint_c": -18.0,
    "hours_above_setpoint": 31, "port_eta_hours": 62})
SEMVER_STATE = ("refactor: rename `getUser` to `fetchUser` across the public SDK, "
                "no deprecation shim")


def card_probes(model, packer, device, with_criteria: bool = True) -> dict:
    """Two hand-written questions, with and without option criteria.

    Both forms are reported because they are different questions: bare keys test what the
    model assumes a key means, criteria test whether it reads the definitions.
    """
    def q(qid, text, options, descs):
        return Question(qid, text, options, None, None, None, descs if with_criteria else None)

    reefer = Example(state=REEFER_STATE, task="probe_reefer", questions=[
        q("cargo_at_risk", "Is the cargo at risk of spoiling?", ["no", "yes"],
          ["The cargo is being held within its required temperature range.",
           "The cargo has been outside its required temperature range long enough to be "
           "at risk."]),
        q("action", "What should the operations team do?",
          ["none", "notify_consignee", "divert_to_nearest_port", "raise_temperature_alarm"],
          ["No action: the shipment is within tolerance.",
           "Tell the consignee the cargo may be compromised on arrival.",
           "Divert so the cargo can be discharged sooner than the current ETA.",
           "Raise an alarm because the reefer is not holding its setpoint."]),
    ])
    semver = Example(state=SEMVER_STATE, task="probe_semver", questions=[
        q("semver", "Which release does this change require?", ["patch", "minor", "major"],
          ["A backwards-compatible bug fix.", "Backwards-compatible new functionality.",
           "A backwards-incompatible change to the public interface."]),
    ])
    return {r["qid"]: {"options": r["options"], "probs": [round(x, 4) for x in r["probs"]],
                       "argmax": r["argmax"], "confidence": round(r["confidence"], 4)}
            for r in score(model, packer, [reefer, semver], device)}


def reefer_probe(card: dict) -> dict:
    """Refrigerated fish at +4.8 °C against a -18 °C setpoint for 31 hours: the cargo is at
    risk, but the criteria never name the field they are about. The bar is not "get it
    right" but "do not be confidently wrong": either `yes`, or `no` with confidence below
    0.40 -- the model reporting that it cannot read the rule."""
    out = {}
    for form in ("bare", "with_criteria"):
        r = (card.get(form) or {}).get("cargo_at_risk")
        if not r:
            continue
        conf = r["confidence"]
        out[form] = {"argmax": r["argmax"], "confidence": conf,
                     "p_yes": r["probs"][r["options"].index("yes")],
                     "correct": r["argmax"] == "yes",
                     "abstaining": r["argmax"] != "yes" and conf < 0.40}
    return {"forms": out, "bar": "correct, or wrong with confidence < 0.40",
            "pass": bool(out) and all(v["correct"] or v["abstaining"] for v in out.values())}


def run_all(model, device, n_entities: int = 600, per_family: int = 150) -> dict:
    packer = Packer.for_model(model)
    card = {"bare": card_probes(model, packer, device, False),
            "with_criteria": card_probes(model, packer, device, True)}
    return {"unseen_entity_abstention": unseen_entity_abstention(model, packer, device,
                                                                 n_entities),
            "rule_transfer": rule_transfer(model, packer, device, per_family),
            "sentinel_wordings": sentinel_wordings(model, packer, device),
            "reefer_probe": reefer_probe(card),
            "card_probes": card}
