"""Regression probes: four behaviours every checkpoint is measured on, with numbers.

`scripts/regression_suite.py` is the CLI; this module is importable so tests and other
scripts can run any one probe. Each probe scores through `Scorer`, which goes through the
public path only -- `DecisionDataset` + `engine.collect_logits` -- so the probes keep
meaning the same thing when the model's internals change; nothing here reads hidden rows.

    a. known posterior   exact Bayesian posteriors (dice, urns, a noisy sensor,
                         conflicting sources), computed with `fractions.Fraction`; KL,
                         TV, calibration slope, ECE against the soft target
    b. nonce robustness  random UUIDs inserted into states where they cannot change the
                         answer; |Δ log p|, TV and argmax flips, by insertion type
    c. option order      k random permutations per question, un-permuted; |Δp| and flips
    d. option count      (i) an eval dump bucketed by option count; (ii) a synthetic
                         lookup whose answer is in the state, at N = 2 .. 256 options

THE POSTERIOR PROBE MUST NEVER BE ADDED TO A CORPUS. It is a measuring instrument: the
moment it is trained on, (a) measures memorisation of these templates. Its surface
templates are deliberately different from the training generator for the same domain
(`lod/corpus/services/sources/synth/posterior.py`); keep them apart in both directions --
do not copy wording from there into here, or from here into there.
Its task names all start with `probe_posterior_`.
"""

from __future__ import annotations

import json
import math
import random
import re
import uuid
from fractions import Fraction
from typing import Callable, Iterable, Sequence

from lod.schema import Example, Question

# ---- scoring ------------------------------------------------------------------------


class Scorer:
    """Examples in, per-question log-probabilities out, through the public forward.

    `score(examples)` -> for each example, for each question, the list of log p over its
    options in the order given, or None where the packer dropped the question (it drops
    trailing questions that do not fit the budget, and whole examples whose first
    question does not). Unlabelled questions get a uniform placeholder target, because
    `collect_logits` keeps only labelled rows; the target is never read here.
    """

    def __init__(self, model, device="cpu", batch_size: int = 4) -> None:
        from lod.model.packing import Packer

        self.model = model
        self.device = device
        self.batch_size = batch_size
        # the checkpoint's own layout: an independent-option model packed sequentially
        # is a different model (it scored 0.00 at N=128 on the lookup sweep that way)
        self.packer = Packer.for_model(model)

    def n_tokens(self, example: Example) -> int | None:
        """Packed length, or None when it does not fit without dropping questions."""
        pk = self.packer.pack(example, drop_questions_to_fit=False)
        return None if pk is None else len(pk)

    def fits(self, example: Example) -> bool:
        """True when every question fits AND the state is not truncated.

        The packer keeps a request whose questions fit by cutting the state to the
        budget left over, so a packed length alone does not say the state survived: a
        256-option lookup packs to exactly 3072 tokens with half its state -- and
        possibly its answer -- cut off.
        """
        from lod.model.packing import STATE_PREFIX

        pk = self.packer.pack(example, drop_questions_to_fit=False)
        if pk is None:
            return False
        tok = self.packer.tok
        full = len(tok(STATE_PREFIX + example.state, add_special_tokens=False)["input_ids"])
        full += 1 if getattr(tok, "bos_token_id", None) is not None else 0
        return pk.block_ids.count(0) >= full

    def score(self, examples: Sequence[Example]) -> list[list[list[float] | None]]:
        import torch

        from lod.training.data import DecisionDataset, make_loader
        from lod.evaluation.engine import collect_logits

        out: list[list[list[float] | None]] = [[None] * len(e.questions) for e in examples]
        if not examples:
            return out
        # similar lengths share a batch: padding is most of the cost on CPU
        order = sorted(range(len(examples)), key=lambda i: _approx_len(examples[i]))
        prepared = [_with_targets(examples[i]) for i in order]
        ds = DecisionDataset(prepared, self.packer, augment=False, drop_questions_to_fit=True)
        loader = make_loader(ds, self.packer.pad_id, self.model.backbone_dtype,
                             batch_size=self.batch_size, num_workers=0)
        logits, valid, _, _, refs = collect_logits(self.model, loader, self.device,
                                                   return_refs=True)
        masked = logits.double().masked_fill(~valid, float("-inf"))
        logp = torch.log_softmax(masked, dim=-1)
        for row, (ei, qi) in enumerate(refs):
            n = int(valid[row].sum())
            out[order[ei]][qi] = [float(x) for x in logp[row, :n]]
        return out


def _approx_len(e: Example) -> int:
    return len(e.state) + sum(len(q.question) + sum(len(o) + 3 for o in q.options)
                              + sum(len(d or "") for d in (q.descriptions or []))
                              for q in e.questions)


def _with_targets(e: Example) -> Example:
    qs = []
    for q in e.questions:
        if q.target is None:
            q = Question(q.id, q.question, list(q.options), [1.0 / len(q.options)] * len(q.options),
                         q.meta, q.instructions, q.descriptions)
        qs.append(q)
    return Example(state=e.state, questions=qs, task=e.task)


# ---- shared metrics ------------------------------------------------------------------

def _mean(xs: Iterable[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def ece_top(conf: Sequence[float], hit: Sequence[float], n_bins: int = 15) -> float | None:
    """Equal-width ECE over the top probability, bins as in `calibration.metrics`.

    `hit` may be soft (the true probability of the predicted option) or 0/1.
    """
    n = len(conf)
    if n == 0:
        return None
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        sel = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == lo)]
        if sel:
            total += (len(sel) / n) * abs(sum(hit[i] for i in sel) / len(sel)
                                          - sum(conf[i] for i in sel) / len(sel))
    return total


def _argmax(xs: Sequence[float]) -> int:
    return max(range(len(xs)), key=xs.__getitem__)


# ---- (a) known posterior -------------------------------------------------------------

COLOURS = ["red", "blue", "green", "amber", "violet", "grey"]


def _frac(x: Fraction, pct: bool) -> str:
    if pct and (x * 100).denominator == 1:
        return f"{int(x * 100)}%"
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)


def _normalise(w: list[Fraction]) -> list[Fraction]:
    s = sum(w)
    return [x / s for x in w]


def _pick(rng: random.Random, probs: Sequence[Fraction]) -> int:
    u, acc = rng.random(), 0.0
    for i, p in enumerate(probs):
        acc += float(p)
        if u < acc:
            return i
    return len(probs) - 1


def dice_posterior(dice: list[list[Fraction]], prior: list[Fraction],
                   rolls: list[int]) -> list[Fraction]:
    """P(die | rolls); `dice[i][f-1]` is P(face f | die i)."""
    return _normalise([prior[i] * math.prod(dice[i][r - 1] for r in rolls)
                       for i in range(len(dice))])


def urn_likelihood(comp: list[int], draws: list[int], replace: bool) -> Fraction:
    """P(this colour sequence | urn), with or without replacement; exact."""
    pool, out = list(comp), Fraction(1)
    for c in draws:
        tot = sum(pool)
        if tot == 0 or pool[c] == 0:
            return Fraction(0)
        out *= Fraction(pool[c], tot)
        if not replace:
            pool[c] -= 1
    return out


def urn_posterior(comps: list[list[int]], prior: list[Fraction], draws: list[int],
                  replace: bool) -> list[Fraction]:
    return _normalise([prior[i] * urn_likelihood(comps[i], draws, replace)
                       for i in range(len(comps))])


def sensor_posterior(base: Fraction, fpr: Fraction, fnr: Fraction,
                     reads: list[bool]) -> Fraction:
    """P(condition | independent readings), positive = True."""
    yes = base * math.prod((1 - fnr) if r else fnr for r in reads)
    no = (1 - base) * math.prod(fpr if r else (1 - fpr) for r in reads)
    return yes / (yes + no)


def sources_posterior(prior: Fraction, rel: list[Fraction], says: list[bool]) -> Fraction:
    """P(claim | reports); source i reports the truth with probability rel[i]."""
    yes = prior * math.prod(r if s else 1 - r for r, s in zip(rel, says))
    no = (1 - prior) * math.prod(1 - r if s else r for r, s in zip(rel, says))
    return yes / (yes + no)


def _dice(rng: random.Random, pct: bool) -> tuple[str, Question, str]:
    k = rng.choice([2, 2, 3])
    names = rng.sample(COLOURS, k)
    dice, lines = [], []
    for i, nm in enumerate(names):
        if i == 0:
            dice.append([Fraction(1, 6)] * 6)
            lines.append(f"the {nm} die is fair")
            continue
        face = rng.randint(1, 6)
        pf = rng.choice([Fraction(1, 2), Fraction(1, 3), Fraction(2, 5), Fraction(1, 4)])
        other = (1 - pf) / 5
        dice.append([pf if f == face else other for f in range(1, 7)])
        lines.append(f"the {nm} die lands on {face} with probability {_frac(pf, pct)} and "
                     f"on each other face with probability {_frac(other, pct)}")
    if rng.random() < 0.5:
        prior = [Fraction(1, k)] * k
        prior_text = "chosen uniformly at random"
    else:
        w = [Fraction(rng.randint(1, 4)) for _ in range(k)]
        prior = _normalise(w)
        prior_text = "chosen with these odds: " + ", ".join(
            f"{nm} {_frac(p, pct)}" for nm, p in zip(names, prior))
    true = _pick(rng, prior)
    rolls = [1 + _pick(rng, dice[true]) for _ in range(rng.randint(1, 5))]
    post = dice_posterior(dice, prior, rolls)
    tpl = rng.randrange(2)
    if tpl == 0:
        state = (f"Probe D. There are {k} dice in a cup: " + "; ".join(lines) + ". "
                 f"One die was {prior_text} and thrown {len(rolls)} time(s). "
                 f"Results in order: {', '.join(map(str, rolls))}.")
    else:
        state = ("Dice inventory:\n" + "\n".join(f"* {ln}" for ln in lines)
                 + f"\nSelection: {prior_text}.\nObserved throws of the selected die: "
                 + " ".join(map(str, rolls)))
    text = rng.choice(["Which die was thrown?", "Which die produced the results?"])
    return state, _question("die", text, names, post, rng), "dice"


def _urns(rng: random.Random, pct: bool) -> tuple[str, Question, str]:
    k = rng.choice([2, 2, 3])
    labels = ["A", "B", "C"][:k]
    cols = rng.sample(["red", "white", "black", "green"], rng.choice([2, 3]))
    comps: list[list[int]] = []
    while len(comps) < k:
        c = [rng.randint(0, 6) for _ in cols]
        if sum(c) >= 4 and c not in comps:
            comps.append(c)
    if rng.random() < 0.5:
        prior = [Fraction(1, k)] * k
        prior_text = "each jar equally likely"
    else:
        prior = _normalise([Fraction(rng.randint(1, 5)) for _ in range(k)])
        prior_text = ", ".join(f"jar {lb} with probability {_frac(p, pct)}"
                               for lb, p in zip(labels, prior))
    replace = rng.random() < 0.5
    true = _pick(rng, prior)
    n_draw = rng.randint(1, min(4, sum(comps[true])))
    pool = list(comps[true])
    draws = []
    for _ in range(n_draw):
        tot = sum(pool)
        c = _pick(rng, [Fraction(x, tot) for x in pool])
        draws.append(c)
        if not replace:
            pool[c] -= 1

    post = urn_posterior(comps, prior, draws, replace)
    contents = "; ".join(f"jar {lb}: " + ", ".join(f"{n} {c}" for n, c in zip(comp, cols))
                         for lb, comp in zip(labels, comps))
    how = "each ball is put back before the next draw" if replace else \
        "balls are not put back"
    state = (f"Probe U. Marbles by jar -- {contents}. A jar is selected ({prior_text}); "
             f"{n_draw} marble(s) are drawn from it and {how}. Drawn, in order: "
             + ", ".join(cols[c] for c in draws) + ".")
    text = rng.choice(["Which jar were the marbles drawn from?", "Which jar was selected?"])
    return state, _question("jar", text, [f"jar {lb}" for lb in labels], post, rng), "urns"


SENSOR_THINGS = [("valve", "stuck", "the pressure test"), ("sample", "contaminated", "the assay"),
                 ("crate", "damaged", "the drop sensor"), ("server", "compromised", "the scanner")]


def _sensor(rng: random.Random, pct: bool) -> tuple[str, Question, str]:
    thing, cond, sensor = rng.choice(SENSOR_THINGS)
    base = Fraction(rng.choice([2, 5, 10, 20, 30, 50]), 100)
    fpr = Fraction(rng.choice([5, 10, 15, 20]), 100)
    fnr = Fraction(rng.choice([5, 10, 20, 30]), 100)
    truth = rng.random() < float(base)
    m = rng.randint(1, 3)
    reads = [(rng.random() >= float(fnr)) if truth else (rng.random() < float(fpr))
             for _ in range(m)]

    p_yes = sensor_posterior(base, fpr, fnr, reads)
    shown = ", ".join("positive" if r else "negative" for r in reads)
    state = (f"Probe S. Base rate: {_frac(base, True)} of units of this kind are {cond}. "
             f"{sensor.capitalize()} flags a {cond} {thing} as positive with probability "
             f"{_frac(1 - fnr, True)}, and flags a sound one as positive with probability "
             f"{_frac(fpr, True)}; repeated runs are independent. This {thing} was run "
             f"{m} time(s): {shown}.")
    q = Question("state", f"Is this {thing} {cond}?", ["no", "yes"],
                 [float(1 - p_yes), float(p_yes)], {"probe": True, "family": "sensor"})
    return state, q, "sensor"


CLAIMS = [("the ferry sailed on schedule", "Did the ferry sail on schedule?"),
          ("the invoice was paid", "Was the invoice paid?"),
          ("the gate was left open", "Was the gate left open?")]
SOURCES = ["the harbour log", "a deckhand", "the payment clerk", "an automated feed",
           "a neighbour", "the night guard"]


def _sources(rng: random.Random, pct: bool) -> tuple[str, Question, str]:
    claim, text = rng.choice(CLAIMS)
    prior = Fraction(rng.choice([20, 30, 50, 50, 70]), 100)
    k = rng.choice([2, 3])
    names = rng.sample(SOURCES, k)
    rel = [Fraction(rng.choice([60, 65, 70, 75, 80, 85, 90, 95]), 100) for _ in range(k)]
    for _ in range(50):             # conflicting sources: resample until they disagree
        truth = rng.random() < float(prior)
        says = [truth if rng.random() < float(r) else (not truth) for r in rel]
        if len(set(says)) > 1:
            break
    else:
        says[-1] = not says[0]

    p_yes = sources_posterior(prior, rel, says)
    lines = [f"{nm} (correct {_frac(r, True)} of the time) says it "
             f"{'is so' if s else 'is not so'}" for nm, r, s in zip(names, rel, says)]
    state = (f"Probe R. Claim: {claim}. Before any report, the claim holds with probability "
             f"{_frac(prior, True)}. Reports, each independent of the others given the "
             f"truth: " + "; ".join(lines) + ".")
    q = Question("claim", text, ["no", "yes"], [float(1 - p_yes), float(p_yes)],
                 {"probe": True, "family": "sources"})
    return state, q, "sources"


def _question(qid: str, text: str, names: list[str], post: list[Fraction],
              rng: random.Random) -> Question:
    order = list(range(len(names)))
    rng.shuffle(order)
    return Question(qid, text, [names[i] for i in order], [float(post[i]) for i in order],
                    {"probe": True})


FAMILIES: dict[str, Callable] = {"dice": _dice, "urns": _urns, "sensor": _sensor,
                                 "sources": _sources}


def posterior_probes(n: int, seed: int = 0) -> list[Example]:
    """`n` probes, round-robin over the four families; deterministic in `seed`.

    Every target is the exact posterior (`Fraction` arithmetic, then one float cast).
    """
    rng = random.Random(f"posterior-probe:{seed}")
    fams = list(FAMILIES)
    out = []
    for i in range(n):
        fam = fams[i % len(fams)]
        state, q, name = FAMILIES[fam](rng, pct=rng.random() < 0.5)
        q.meta = {**(q.meta or {}), "family": name}
        out.append(Example(state=state, questions=[q], task=f"probe_posterior_{name}"))
    return out


def _posterior_metrics(rows: list[tuple[list[float], list[float]]]) -> dict:
    """rows: (true posterior, model p) per question."""
    if not rows:
        return {"n": 0}
    kl, tv, conf, hit, agree, xs, ys = [], [], [], [], [], [], []
    for t, p in rows:
        kl.append(sum(ti * math.log(ti / max(pi, 1e-12)) for ti, pi in zip(t, p) if ti > 0))
        tv.append(0.5 * sum(abs(ti - pi) for ti, pi in zip(t, p)))
        j = _argmax(p)
        conf.append(p[j])
        hit.append(t[j])
        agree.append(float(j == _argmax(t)))
        xs += t
        ys += p
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else None
    return {"n": len(rows), "kl": _mean(kl), "tv": _mean(tv),
            "calib_slope": slope,
            "calib_intercept": (my - slope * mx) if slope is not None else None,
            "ece_soft": ece_top(conf, hit), "argmax_agree": _mean(agree),
            "mean_true_maxp": _mean(max(t) for t, _ in rows),
            "mean_model_maxp": _mean(conf)}


def run_posterior(scorer, n: int = 160, seed: int = 0) -> dict:
    """(a). KL(true‖model), TV, slope of model p on true p (OLS over every option cell,
    1.0 is calibrated, < 1 is underconfident-shrunk), ECE of the top option against its
    true posterior, and the rate at which the model's argmax is the posterior's."""
    examples = posterior_probes(n, seed)
    logps = scorer.score(examples)
    rows: dict[str, list] = {}
    for e, lp in zip(examples, logps):
        q, l = e.questions[0], lp[0]
        if l is None:
            continue
        rows.setdefault(q.meta["family"], []).append((q.target, [math.exp(x) for x in l]))
    allrows = [r for v in rows.values() for r in v]
    out = _posterior_metrics(allrows)
    out["per_family"] = {f: _posterior_metrics(v) for f, v in sorted(rows.items())}
    out["n_generated"] = len(examples)
    return out


# ---- (b) nonce robustness ------------------------------------------------------------

# a question mentioning any of these might be about the structure a nonce changes
# (how many fields, which key, which line); the example is skipped rather than guessed
_RISKY = re.compile(r"how many|number of|\bcount|\bfield|\bkeys?\b|\blines?\b|identifier|"
                    r"\bids?\b|uuid|attribute|length|longest|shortest|\bwords?\b|character|"
                    r"token|format|json|schema|propert|column|duplicate|same record|"
                    r"\bhash|checksum|random|nonce", re.I)
_ID_KEY = re.compile(r"^(id|uuid|guid|.+_id)$", re.I)
NONCE_KEYS = ["trace_id", "request_id", "uuid", "correlation_id", "event_uuid", "nonce_id"]
NONCE_LABELS = ["id", "ref", "request-id", "trace id", "msg-id"]


def _question_text(e: Example) -> str:
    parts = []
    for q in e.questions:
        parts += [q.question, q.instructions or ""] + list(q.options)
        parts += [d or "" for d in (q.descriptions or [])]
    return "\n".join(parts)


def _uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _json_style(state: str):
    """(obj, dumps kwargs) that reproduce `state` byte for byte, or None."""
    s = state.strip()
    if not s.startswith("{"):
        return None
    try:
        obj = json.loads(s)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    for indent in (None, 2, 4, 1, 3, "\t"):
        for ascii_ in (False, True):
            seps = [None] if indent is not None else [None, (",", ":")]
            for sep in seps:
                kw = {"indent": indent, "ensure_ascii": ascii_}
                if sep:
                    kw["separators"] = sep
                if json.dumps(obj, **kw) == s:
                    return obj, kw
    return None


def _wrap(state: str, body: str) -> str:
    """Keep the original's leading/trailing whitespace around a re-rendered body."""
    s = state.strip()
    i = state.index(s[:1]) if s else 0
    return state[:i] + body + state[i + len(s):]


def nonce_variants(e: Example, rng: random.Random) -> dict[str, Example]:
    """-> {insertion type: perturbed copy} for the insertions that are safe here.

    json_key   a new id-like key with a UUID value, at a random position of a JSON object
    text_line  an `id: <uuid>` style line, as its own paragraph, in a text state
    id_replace an existing id-like value (key `id`, `*_id`, `uuid`) replaced by a UUID,
               only when that value occurs exactly once in the state and nowhere in the
               questions, so no cross-reference or option can depend on it

    Skipped entirely when a question's wording touches structure (fields, keys, lines,
    counts, ids): there the nonce might be the answer.
    """
    if _RISKY.search(_question_text(e)):
        return {}
    out: dict[str, Example] = {}
    qtext = _question_text(e)

    def copy(state: str) -> Example:
        return Example(state=state, questions=e.questions, task=e.task)

    style = _json_style(e.state)
    if style is not None:
        obj, kw = style
        free = [k for k in NONCE_KEYS if k not in obj]
        if free:
            items = list(obj.items())
            items.insert(rng.randint(0, len(items)), (rng.choice(free), _uuid(rng)))
            out["json_key"] = copy(_wrap(e.state, json.dumps(dict(items), **kw)))
        cands = []

        def walk(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    if (_ID_KEY.match(str(k)) and isinstance(v, (str, int))
                            and not isinstance(v, bool) and str(v).strip()):
                        cands.append((path, k, v))
                    walk(v, path + [k])
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, path + [i])

        walk(obj, [])
        safe = []
        for path, k, v in cands:
            sv = str(v)
            if len(re.findall(r"(?<![\w-])" + re.escape(sv) + r"(?![\w-])", e.state)) != 1:
                continue
            if sv in qtext:
                continue
            safe.append((path, k))
        if safe:
            path, k = rng.choice(safe)
            new = json.loads(json.dumps(obj))
            node = new
            for p in path:
                node = node[p]
            node[k] = _uuid(rng)
            out["id_replace"] = copy(_wrap(e.state, json.dumps(new, **kw)))
    elif not e.state.lstrip().startswith(("{", "[")):
        paras = e.state.split("\n\n")
        pos = rng.randint(0, len(paras))
        line = f"{rng.choice(NONCE_LABELS)}: {_uuid(rng)}"
        out["text_line"] = copy("\n\n".join(paras[:pos] + [line] + paras[pos:]))
    return out


def _compare(base: list[float], other: list[float]) -> tuple[float, float, float, bool]:
    """-> (mean |Δ log p|, max |Δ log p|, TV, argmax flipped) for one question."""
    d = [abs(a - b) for a, b in zip(base, other)]
    pa, pb = [math.exp(x) for x in base], [math.exp(x) for x in other]
    tv = 0.5 * sum(abs(a - b) for a, b in zip(pa, pb))
    return sum(d) / len(d), max(d), tv, _argmax(pa) != _argmax(pb)


def run_nonce(scorer, examples: Sequence[Example], seed: int = 0,
              base_logps: list | None = None) -> dict:
    """(b). Scores each example with and without each safe nonce insertion."""
    rng = random.Random(f"nonce:{seed}")
    variants: list[tuple[int, str, Example]] = []
    skipped = 0
    for i, e in enumerate(examples):
        vs = nonce_variants(e, rng)
        if not vs:
            skipped += 1
        variants += [(i, kind, v) for kind, v in vs.items()]
    if base_logps is None:
        base_logps = scorer.score(examples)
    scored = scorer.score([v for _, _, v in variants])
    per: dict[str, list] = {}
    for (i, kind, _), lp in zip(variants, scored):
        for b, o in zip(base_logps[i], lp):
            if b is None or o is None:
                continue
            per.setdefault(kind, []).append(_compare(b, o))

    def summ(rows):
        if not rows:
            return {"n_questions": 0}
        return {"n_questions": len(rows),
                "mean_abs_dlogp": _mean(r[0] for r in rows),
                "max_abs_dlogp": max(r[1] for r in rows),
                "mean_tv": _mean(r[2] for r in rows),
                "flip_rate": _mean(float(r[3]) for r in rows)}

    allrows = [r for v in per.values() for r in v]
    out = summ(allrows)
    out.update({"n_examples": len(examples), "n_examples_skipped": skipped,
                "n_variants": len(variants),
                "by_type": {k: summ(v) for k, v in sorted(per.items())}})
    return out


# ---- (c) option order ----------------------------------------------------------------

def permute_question(q: Question, perm: list[int]) -> Question:
    """New option i is old option perm[i]; descriptions and target follow."""
    descs = None
    if q.descriptions:
        d = list(q.descriptions) + [None] * (len(q.options) - len(q.descriptions))
        descs = [d[j] for j in perm]
    tgt = q.target
    if isinstance(tgt, bool):
        tgt = int(tgt)
    if isinstance(tgt, int):
        tgt = perm.index(tgt)
    elif isinstance(tgt, list):
        tgt = [tgt[j] for j in perm]
    return Question(q.id, q.question, [q.options[j] for j in perm], tgt, q.meta,
                    q.instructions, descs)


def _random_perm(n: int, rng: random.Random) -> list[int]:
    ident = list(range(n))
    if n < 2:
        return ident
    while True:
        p = ident[:]
        rng.shuffle(p)
        if p != ident:
            return p


def run_permutation(scorer, examples: Sequence[Example], k: int = 3, seed: int = 0,
                    base_logps: list | None = None) -> dict:
    """(c). Each question under k random non-identity permutations, un-permuted and
    compared with the given order. Every question in an example is permuted
    independently, in the same pass."""
    rng = random.Random(f"perm:{seed}")
    if base_logps is None:
        base_logps = scorer.score(examples)
    jobs, perms = [], []
    for i, e in enumerate(examples):
        for _ in range(k):
            ps = [_random_perm(len(q.options), rng) for q in e.questions]
            jobs.append(Example(state=e.state, task=e.task,
                                questions=[permute_question(q, p)
                                           for q, p in zip(e.questions, ps)]))
            perms.append((i, ps))
    scored = scorer.score(jobs)
    dps, maxes, flips = [], [], []
    per_q_flip: dict[tuple[int, int], bool] = {}
    for (i, ps), lp in zip(perms, scored):
        for qi, (p, l) in enumerate(zip(ps, lp)):
            b = base_logps[i][qi]
            if b is None or l is None:
                continue
            un = [0.0] * len(p)
            for new, old in enumerate(p):
                un[old] = math.exp(l[new])
            pb = [math.exp(x) for x in b]
            d = [abs(x - y) for x, y in zip(pb, un)]
            dps += d
            maxes.append(max(d))
            f = _argmax(pb) != _argmax(un)
            flips.append(float(f))
            per_q_flip[(i, qi)] = per_q_flip.get((i, qi), False) or f
    return {"k": k, "n_questions": len(per_q_flip), "n_permuted_scores": len(flips),
            "mean_abs_dp": _mean(dps), "mean_max_abs_dp": _mean(maxes),
            "max_abs_dp": max(maxes) if maxes else None,
            "flip_rate": _mean(flips),
            "any_flip_rate": _mean(float(v) for v in per_q_flip.values())}


# ---- (d) option count ----------------------------------------------------------------

BUCKETS = [(2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 10**9)]


def bucket_name(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else (f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}")


def option_count_buckets(records: Iterable[dict], n_bins: int = 15) -> list[dict]:
    """(d.i). `records` as in an eval dump (probs, target). ECE is the usual top-label
    one against the hard label argmax(target); NLL is against the (soft) target."""
    by: dict[tuple[int, int], list] = {b: [] for b in BUCKETS}
    for r in records:
        n = len(r["probs"])
        for b in BUCKETS:
            if b[0] <= n <= b[1]:
                by[b].append(r)
                break
    out = []
    for (lo, hi), rs in by.items():
        row = {"bucket": bucket_name(lo, hi), "n": len(rs)}
        if rs:
            conf, hit, nll = [], [], []
            for r in rs:
                p, t = r["probs"], r["target"]
                j = _argmax(p)
                conf.append(p[j])
                hit.append(float(j == _argmax(t)))
                nll.append(-sum(ti * math.log(max(pi, 1e-12)) for pi, ti in zip(p, t)))
            row.update({"acc": _mean(hit), "nll": _mean(nll), "ece": ece_top(conf, hit, n_bins),
                        "mean_maxp": _mean(conf)})
        out.append(row)
    return out


ADJ = ["amber", "brisk", "calm", "dusky", "eager", "faint", "gilded", "hollow", "ivory",
       "jolly", "keen", "lunar", "mossy", "noble", "olive", "pale", "quiet", "rusty",
       "silent", "tidy", "urban", "vivid", "woolly", "young", "zesty", "bold", "coral",
       "dry", "early", "frosty", "grand", "husky"]
NOUN = ["badger", "crane", "dingo", "egret", "ferret", "gecko", "heron", "ibis", "jackal",
        "koala", "lemur", "marten", "newt", "otter", "panda", "quail", "raven", "stoat",
        "tapir", "urchin", "vole", "walrus", "yak", "zebra", "bison", "cobra", "drake",
        "finch", "gibbon", "hyena", "kestrel", "llama"]


def lookup_example(n: int, rng: random.Random) -> Example:
    """One synthetic lookup with `n` options: items and their bins are listed in the
    state, the question names a bin, the options are every item, in another order."""
    names = rng.sample([f"{a}-{b}" for a in ADJ for b in NOUN], n)
    bins = rng.sample(range(1, max(10 * n, 100)), n)
    state = "Storage map (item: bin):\n" + "\n".join(f"{nm}: bin {b}"
                                                     for nm, b in zip(names, bins))
    j = rng.randrange(n)
    opts = names[:]
    rng.shuffle(opts)
    q = Question("lookup", f"Which item is stored in bin {bins[j]}?", opts,
                 opts.index(names[j]), {"probe": True, "n": n})
    return Example(state=state, questions=[q], task=f"probe_lookup_n{n}")


SWEEP_NS = (2, 4, 8, 16, 32, 64, 128, 256)


def run_sweep(scorer, ns: Sequence[int] = SWEEP_NS, reps: int = 10, seed: int = 0) -> dict:
    """(d.ii). Accuracy, ECE and mean p(correct) against N, for every N whose
    examples fit whole (no question dropped, no state truncated)."""
    rng = random.Random(f"sweep:{seed}")
    rows = []
    for n in ns:
        exs = [lookup_example(n, rng) for _ in range(reps)]
        lens = [scorer.n_tokens(e) if scorer.fits(e) else None for e in exs]
        fit = [e for e, ln in zip(exs, lens) if ln is not None]
        row = {"N": n, "chance": 1.0 / n, "n_generated": reps, "n": len(fit),
               "fits": bool(fit),
               "mean_tokens": _mean(ln for ln in lens if ln is not None)}
        if fit:
            lps = scorer.score(fit)
            conf, hit, pc = [], [], []
            for e, lp in zip(fit, lps):
                l = lp[0]
                if l is None:
                    continue
                p = [math.exp(x) for x in l]
                t = e.questions[0].target
                j = _argmax(p)
                conf.append(p[j])
                hit.append(float(j == t))
                pc.append(p[t])
            row.update({"n": len(pc), "acc": _mean(hit), "ece": ece_top(conf, hit),
                        "mean_p_correct": _mean(pc), "mean_maxp": _mean(conf)})
        rows.append(row)
    fitting = [r["N"] for r in rows if r["fits"] and r["n"]]
    return {"reps": reps, "rows": rows, "tested_up_to": max(fitting) if fitting else None}


# ---- sampling real examples ----------------------------------------------------------

def sample_examples(path, n: int, scorer=None, seed: int = 0, max_tokens: int = 1024,
                    per_task: int = 2) -> list[Example]:
    """`n` examples from a data file, at most `per_task` from any one task (so a few big
    tasks cannot be the whole sample), and at most `max_tokens` packed (CPU cost)."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    rng = random.Random(f"sample:{seed}")
    order = list(range(len(lines)))
    rng.shuffle(order)
    taken: dict[str, int] = {}
    out: list[Example] = []
    for i in order:
        if len(out) >= n:
            break
        e = Example.from_dict(json.loads(lines[i]))
        if not e.questions or taken.get(e.task, 0) >= per_task:
            continue
        if len(e.state) > 6 * max_tokens:        # cheap reject before tokenising
            continue
        if scorer is not None:
            ln = scorer.n_tokens(e)
            if ln is None or ln > max_tokens or not scorer.fits(e):
                continue            # a truncated state could lose an inserted nonce
        taken[e.task] = taken.get(e.task, 0) + 1
        out.append(e)
    return out
