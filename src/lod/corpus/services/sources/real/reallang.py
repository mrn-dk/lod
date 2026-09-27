"""Source-table row 50 (release round) -- broad real-language decisions (domain 24).

The Decision Index 0.2 read put the model at or near chance wherever a benchmark asks a
natural-language judgement the corpus had no real labels for: entailment beyond SNLI,
physical and social commonsense, sarcasm and humour, stance, social nuance (politeness,
condescension, intimacy, empathy), dialogue acts, and choosing a tool for a request. This
row adds human-labelled data for each of those judgements -- never a DI member, and every
state checked against the DI 0.2 blocklist before it is used.

Raw rows come from `scripts/fetch_data.py --rows 50` (`rl_<name>` in the raw store); this module
owns what every label means. Every option has a criteria description (the source's own
definition where it publishes one). Descriptions are attached to half of each task's
questions (hash-ranked, as in `descriptions.attach`), except the tool-choice tasks, where
the function descriptions are the only statement of what an option does and are always
attached.

Splits are whole datasets (`family` = the dataset), forced per task, so no dataset's
column tasks straddle two splits:

  devreal   scitail (NLI), copa (causal commonsense), dailydialog acts and emotions
            (dialogue), emobank valence (emotion, ordinal), bragging (social)
  testreal  cosmosqa (reading commonsense), scruples (moral judgement, soft), toolace
            (tool choice), flute (figurative NLI), rumoureval (stance), complaints
  train     everything else

Two placements are forced by the rest of the corpus, measured against an earlier build by
exact and 10-word-shingle overlap: HWU64 trains, because MASSIVE (English in that train,
and its translations) is built from the same utterances, so a held-out HWU64 measured
recall (152 of 1,999 dev states matched that train); DailyDialog is held out, because
BIG-bench TimeDial (in devreal) is built from DailyDialog dialogues and a trained
DailyDialog would have deleted 160 of its examples in the full build's dedup.

Soft targets only where the source publishes more than one judgement per item: Scruples'
community vote counts, and three mean ratings (SICK relatedness over 10 raters, EmoBank
valence, HaHackathon humour rating) spread over the two nearest ordinal levels so the
target's mean is the published mean. The tool-choice labels (Glaive, ToolACE) are what a
data-generating model did, not a person: those questions carry `meta.label_source =
"model"`, every other question `"human"`.

Pool rules, per task: identical (state, question, options) appear once; a state claimed
by an earlier task in `PRIORITY` (eval tasks first) is not reused; a state over
`MAX_STATE_CHARS` is dropped whole; a state (with its question and options) that the DI
blocklist matches is skipped and counted in `DI_DROPS`; from a pool larger than the quota
no class takes more than `MAX_CLASS_SHARE` of the sample.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from lod import sentinel
from lod.schema import Example, Question
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask
from lod.paths import DI_BLOCKLIST

ROW = 50
PREFIX = "rl_"
MAX_STATE_CHARS = 6000
MAX_CLASS_SHARE = 0.5
DESCRIBE_SHARE = 0.5
# This row is sized at 60-90k questions; `task_quotas` would otherwise hand
# 36 tasks 10 % of --target-questions between them (1,111 each at 400k). Still capped by
# --sample-per-task.
MIN_QUOTA = 2000

LIC_SOCKET = "CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: "
URL_SOCKET = "https://huggingface.co/datasets/Blablablab/SOCKET"

NLI3 = ("entailment", "neutral", "contradiction")
NLI3_CRITERIA = (
    "the hypothesis is definitely true given the premise",
    "the hypothesis might or might not be true given the premise",
    "the hypothesis is definitely false given the premise",
)


@dataclass(frozen=True)
class Spec:
    task: str                         # without PREFIX
    key: str                          # raw-store key (without the rl_ prefix)
    split: str
    family: str
    licence: str
    url: str
    question: str                     # a per-row reader may override it
    options: tuple[str, ...] | None   # None: per-example options from the reader
    criteria: tuple[str, ...] | None  # one per fixed option
    read: Callable[[dict], dict | None] = field(default=lambda r: None)
    ordinal: bool = False
    soft: bool = False
    label_source: str = "human"
    always_describe: bool = False
    # (i, n): keep the i-th of n hash slices of the family's rows, so two tasks reading
    # one pool never share a record. `slice_on` names the record (default: row index).
    slice: tuple[int, int] | None = None
    slice_on: Callable[[dict, int], str] | None = None
    notes: str = ""

    @property
    def name(self) -> str:
        return PREFIX + self.task

    @property
    def per_example(self) -> bool:
        # hwu64's 64 fixed options are read from the store, not written here
        return self.options is None and self.task != "hwu64_intent"


# ------------------------------------------------------------------ helpers ----------

def _clean(text) -> str:
    text = str(text or "").replace("﻿", "").replace("�", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _pair(a_name: str, a: str, b_name: str, b: str) -> str:
    return f"{a_name}: {_clean(a)}\n{b_name}: {_clean(b)}"


def interp(value: float, points: tuple[float, ...]) -> list[float]:
    """A published mean rating as a distribution over ordinal levels at `points`: the
    mass is split between the two nearest levels so that its mean is the value."""
    v = min(max(float(value), points[0]), points[-1])
    out = [0.0] * len(points)
    for i in range(len(points) - 1):
        lo, hi = points[i], points[i + 1]
        if lo <= v <= hi:
            w = (v - lo) / (hi - lo) if hi > lo else 0.0
            out[i] += 1 - w
            out[i + 1] += w
            return out
    out[-1] = 1.0
    return out


def _argmax(t) -> int:
    return max(range(len(t)), key=lambda i: t[i]) if isinstance(t, list) else int(t)


def _h(*parts) -> int:
    return int.from_bytes(hashlib.sha1("\x00".join(map(str, parts)).encode()).digest()[:8], "big")


# ------------------------------------------------------------------ readers ----------

def _read_wanli(r):
    if r.get("gold") not in NLI3:
        return None
    return {"state": _pair("Premise", r["premise"], "Hypothesis", r["hypothesis"]),
            "target": NLI3.index(r["gold"])}


def _read_sick_ent(r):
    lab = r.get("label")
    if lab not in (0, 1, 2):
        return None
    return {"state": _pair("Sentence A", r["sentence_A"], "Sentence B", r["sentence_B"]),
            "target": int(lab)}


RELATED = ("unrelated", "slightly related", "somewhat related", "closely related",
           "nearly equivalent")


def _read_sick_rel(r):
    s = r.get("relatedness_score")
    if s is None:
        return None
    t = interp(float(s), (1, 2, 3, 4, 5))
    return {"state": _pair("Sentence A", r["sentence_A"], "Sentence B", r["sentence_B"]),
            "target": t, "cls": _argmax(t)}


def _read_scitail(r):
    lab = {"entailment": 0, "neutral": 1}.get(r.get("gold_label"))
    if lab is None:
        return None
    return {"state": _pair("Premise", r["sentence1"], "Hypothesis", r["sentence2"]),
            "target": lab}


def _choice(state: str, options: list, label: int, question: str | None = None,
            desc: str | None = None) -> dict | None:
    options = [_clean(o) for o in options]
    if not all(options) or len(set(o.lower() for o in options)) != len(options):
        return None
    if not 0 <= label < len(options):
        return None
    out = {"state": state, "options": options, "target": label}
    if question:
        out["question"] = question
    if desc:
        out["descriptions"] = [desc] * len(options)
    return out


def _read_piqa(r):
    return _choice(f"Goal: {_clean(r['goal'])}", [r["sol1"], r["sol2"]], int(r["label"]),
                   desc="a way to carry out the goal; the right one would actually work "
                        "in the physical world")


def _read_siqa(r):
    lab = int(r["label"]) - 1
    return _choice(_clean(r["context"]), [r["answerA"], r["answerB"], r["answerC"]], lab,
                   question=_clean(r["question"]),
                   desc="an answer to the question about the people in the situation; the "
                        "right one is what most people would infer about their motives, "
                        "reactions or next steps")


def _read_cosmos(r):
    opts = [r[f"answer{i}"] for i in range(4)]
    return _choice(_clean(r["context"]), opts, int(r["label"]),
                   question=_clean(r["question"]),
                   desc="an answer to the question; the right one is the most plausible "
                        "reading of what the passage implies but may not state")


def _read_swag(r):
    stem = _clean(r.get("sent2"))
    opts = [f"{stem} {_clean(r[f'ending{i}'])}" for i in range(4)]
    return _choice(f"Context: {_clean(r['sent1'])}", opts, int(r["label"]),
                   desc="a possible next sentence; the right one is what most plausibly "
                        "happens next in the described scene")


def _read_obqa(r):
    ch = r["choices"]
    key = r.get("answerKey")
    if key not in ch["label"]:
        return None
    state = f"Question: {_clean(r['question_stem'])}"
    if r.get("fact1"):
        state += f"\nA relevant science fact: {_clean(r['fact1'])}"
    return _choice(state, list(ch["text"]), ch["label"].index(key),
                   desc="a candidate completion or answer; the right one follows from "
                        "the science fact combined with common knowledge")


def _read_copa(r):
    q = {"cause": "What was the CAUSE of this?",
         "effect": "What happened as a RESULT?"}.get(r.get("question"))
    if q is None:
        return None
    return _choice(f"Premise: {_clean(r['premise'])}", [r["choice1"], r["choice2"]],
                   int(r["label"]), question=q,
                   desc="a candidate cause or effect of the premise; the right one is the "
                        "more plausible alternative")


def _read_ethics_cs(r):
    # the long ETHICS commonsense items are Reddit AITA posts, the same community (and in
    # 282 cases the same posts) as the Scruples anecdotes held out in testreal: only the
    # short, crowd-written scenarios train
    if str(r.get("is_short")) != "True":
        return None
    return {"state": _clean(r["input"]), "target": int(r["label"])}


def _read_ethics_deon(r):
    return {"state": _pair("Request or duty", r["scenario"], "Excuse", r["excuse"]),
            "target": int(r["label"])}


def _read_ethics_just(r):
    return {"state": _clean(r["scenario"]), "target": int(r["label"])}


def _read_moral(r):
    act = r.get("moral_action") or r.get("immoral_action")
    if not act or r.get("label") not in ("0", "1"):
        return None
    return {"state": _pair("Norm", r["norm"], "Action", act), "target": int(r["label"])}


SCRUPLES = ("AUTHOR", "OTHER", "EVERYBODY", "NOBODY", "INFO")
SCRUPLES_MIN_VOTES = 5


def _read_scruples(r):
    sc = r.get("label_scores") or {}
    votes = [int(sc.get(k, 0) or 0) for k in SCRUPLES]
    n = sum(votes)
    if n < SCRUPLES_MIN_VOTES:
        return None
    t = [v / n for v in votes]
    state = f"{_clean(r.get('title'))}\n\n{str(r.get('text') or '').strip()}"
    return {"state": state.strip(), "target": t, "cls": _argmax(t),
            "meta": {"votes": n, "source_label": r.get("label")}}


def _read_flute(r):
    lab = {"Entailment": 0, "Contradiction": 1}.get(r.get("label"))
    if lab is None:
        return None
    return {"state": _pair("Premise", r["premise"], "Hypothesis", r["hypothesis"]),
            "target": lab, "meta_extra": {"figure": r.get("type")}}


RUMOUR = ("support", "deny", "query", "comment")


def _read_rumour(r):
    lab = r.get("label")
    if lab not in RUMOUR:
        return None
    return {"state": _pair("Source tweet", r["source_text"], "Reply", r["reply_text"]),
            "target": RUMOUR.index(lab)}


DD_ACTS = ("inform", "question", "directive", "commissive")
DD_EMOTIONS = ("no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise")
DD_CONTEXT = 4


def _dd_state(dialog: list, i: int) -> str:
    lo = max(0, i - DD_CONTEXT)
    lines = []
    for j in range(lo, i + 1):
        who = "A" if j % 2 == 0 else "B"
        mark = " <-- target utterance" if j == i else ""
        lines.append(f"{who}: {_clean(dialog[j])}{mark}")
    return "\n".join(lines)


def _dd_pick(r, field_: str) -> int | None:
    dialog, labs = r.get("dialog") or [], r.get(field_) or []
    if not dialog or len(dialog) != len(labs):
        return None
    return _h(field_, dialog[0]) % len(dialog)


def _read_dd_act(r):
    i = _dd_pick(r, "act")
    if i is None or not 1 <= int(r["act"][i]) <= 4:
        return None
    return {"state": _dd_state(r["dialog"], i), "target": int(r["act"][i]) - 1}


def _read_dd_emotion(r):
    i = _dd_pick(r, "emotion")
    if i is None or not 0 <= int(r["emotion"][i]) <= 6:
        return None
    return {"state": _dd_state(r["dialog"], i), "target": int(r["emotion"][i])}


WOZ_DOMAINS = ("restaurant", "hotel", "attraction", "train", "taxi", "hospital", "police",
               "general")
WOZ_CRITERIA = (
    "finding or booking a restaurant in Cambridge",
    "finding or booking a hotel or guesthouse",
    "finding an attraction: a museum, college, park, theatre or other place to visit",
    "finding or booking a train",
    "booking a taxi",
    "finding a hospital department",
    "finding the police station",
    "no service: greeting, thanks, goodbye or other conversational turn",
)


def _read_woz(r):
    t = r["turns"]
    speakers, utts, acts = t["speaker"], t["utterance"], t["dialogue_acts"]
    user = [i for i, s in enumerate(speakers) if s == 0]
    if not user:
        return None
    i = user[_h("woz", r["dialogue_id"]) % len(user)]
    types = acts[i]["dialog_act"]["act_type"]
    doms = {a.split("-", 1)[0].lower() for a in types}
    if not doms:
        return None
    real = doms - {"general"}
    if len(real) > 1:
        return None                 # two services in one turn: no single answer
    dom = next(iter(real)) if real else "general"
    if dom not in WOZ_DOMAINS:
        return None
    lo = max(0, i - 2)
    lines = [f"{'User' if speakers[j] == 0 else 'System'}: {_clean(utts[j])}"
             + (" <-- target turn" if j == i else "") for j in range(lo, i + 1)]
    return {"state": "\n".join(lines), "target": WOZ_DOMAINS.index(dom)}


def _text_key(r: dict, j: int) -> str:
    return norm_key(str(r.get("text") or ""))


def _socket_bin(neg: str, pos: str):
    def read(r):
        lab = r.get("label")
        if lab not in (neg, pos):
            return None
        return {"state": _clean(r["text"]), "target": int(lab == pos)}
    return read


def _read_condescension(r):
    lab = r.get("label")
    if lab not in ("not condescension", "condescension"):
        return None
    parts = str(r["text"]).split("[SEP]", 1)
    if len(parts) != 2:
        return None
    return {"state": _pair("Quoted comment", parts[0], "Reply", parts[1]),
            "target": int(lab == "condescension")}


INTIMACY = ("Not-intimate-at-all", "Not-intimate", "Not-very-intimate", "Somewhat-intimate",
            "Intimate", "Very-intimate")


def _read_intimacy(r):
    if r.get("label") not in INTIMACY:
        return None
    return {"state": _clean(r["text"]), "target": INTIMACY.index(r["label"])}


def _rating(points: tuple[float, ...]):
    def read(r):
        try:
            v = float(r["label"])
        except (TypeError, ValueError):
            return None
        t = interp(v, points)
        return {"state": _clean(r["text"]), "target": t, "cls": _argmax(t),
                "meta": {"mean_rating": v}}
    return read


# hwu64: options are the published intent names; the criterion spells out the
# scenario/action pair the name encodes (Liu et al. 2019 annotate exactly that pair)
_HWU_NAMES: list[str] | None = None


def hwu_names() -> list[str]:
    global _HWU_NAMES
    if _HWU_NAMES is None:
        rows = sorted(store.load("rl_hwu64_intents"), key=lambda r: int(r["id"]))
        _HWU_NAMES = [r["name"] for r in rows]
    return _HWU_NAMES


def hwu_criterion(name: str) -> str:
    scen, _, act = name.partition("_")
    return f"scenario: {scen}; action: {act.replace('_', ' ') or scen}"


def _read_hwu(r):
    names = hwu_names()
    lab = int(r["label"])
    if not 0 <= lab < len(names):
        return None
    return {"state": _clean(r["utterance"]), "target": lab}


# ---------------------------------------------------------------- tool choice --------

TOOL_Q = ("Which of the available functions, if any, should be called first to handle "
          "this request?")
_REFUSAL = re.compile(
    r"\b(sorry|cannot|can't|can not|unable|not able|don't have|do not have|"
    r"isn't available|not available|none of the|no function|not supported|"
    r"doesn't support|does not support|outside (?:of )?my|beyond my|not possible)\b", re.I)


def _fn_desc(f: dict) -> str:
    desc = _clean(f.get("description")) or "(no description)"
    props = ((f.get("parameters") or {}).get("properties") or {})
    req = set((f.get("parameters") or {}).get("required") or [])
    if props:
        ps = ", ".join(f"{p}{' (required)' if p in req else ''}" for p in list(props)[:8])
        desc += f" Parameters: {ps}."
    return desc[:500]


def _json_objects(text: str) -> list:
    dec, out, i = json.JSONDecoder(), [], 0
    while True:
        j = min([k for k in (text.find("{", i), text.find("[", i)) if k >= 0], default=-1)
        if j < 0:
            return out
        try:
            obj, end = dec.raw_decode(text, j)
        except ValueError:
            i = j + 1
            continue
        out.append(obj)
        i = end


def _tool_item(task: str, request: str, funcs: list[dict], called: str | None,
               split: str) -> dict | None:
    names = [str(f.get("name") or "").strip() for f in funcs]
    if not funcs or not all(names) or len(set(names)) != len(names):
        return None
    if called is not None and called not in names:
        return None
    # seeded on the function set too: Glaive repeats one request across many function
    # lists, and a request-only seed repeated one sentinel wording 144 times (audit bar 50)
    rng = random.Random(_h(task, request, "\x1f".join(names)))
    s_key, s_desc = sentinel.sample(rng, "train" if split == "train" else "eval")
    if s_key in names:
        return None
    options = names + [s_key]
    descs = [_fn_desc(f) for f in funcs] + [s_desc]
    order = list(range(len(options)))
    rng.shuffle(order)
    target = order.index(names.index(called) if called is not None else len(names))
    return {"state": f"User request: {request}", "options": [options[i] for i in order],
            "descriptions": [descs[i] for i in order], "target": target,
            "cls": int(called is None)}


def _read_glaive(r, split="train"):
    funcs = [o for o in _json_objects(str(r.get("system") or "")) if isinstance(o, dict)
             and "name" in o]
    chat = str(r.get("chat") or "")
    m = re.match(r"\s*USER:\s*(.*?)\n\n\nASSISTANT:\s*(.*?)(?:<\|endoftext\|>|\n\n\n|$)",
                 chat, re.S)
    if not funcs or not m:
        return None
    request, reply = _clean(m.group(1)), m.group(2).strip()
    if reply.startswith("<functioncall>"):
        n = re.search(r'"name"\s*:\s*"([^"]+)"', reply)
        called = n.group(1) if n else None
        if called is None:
            return None
    elif _REFUSAL.search(reply):
        called = None
    else:
        return None      # a direct answer or a clarifying question: whether a function
        #                  fits is not stated by the transcript
    return _tool_item("glaive", request, funcs, called, split)


def _read_toolace(r, split="testreal"):
    sysmsg = str(r.get("system") or "")
    k = sysmsg.find("[")
    funcs = []
    for o in _json_objects(sysmsg[k:] if k >= 0 else ""):
        if isinstance(o, list):
            funcs = [f for f in o if isinstance(f, dict) and "name" in f]
            break
    conv = r.get("conversations") or []
    if len(conv) < 2 or conv[0].get("from") != "user" or conv[1].get("from") != "assistant":
        return None
    request, reply = _clean(conv[0]["value"]), str(conv[1]["value"]).strip()
    names = [str(f.get("name")) for f in funcs]
    if reply.startswith("[") and reply.endswith("]"):
        hit = {n for n in names if re.search(re.escape(n) + r"\s*\(", reply)}
        if len(hit) != 1:
            return None
        called = hit.pop()
    elif _REFUSAL.search(reply) and "?" not in reply:
        called = None
    else:
        return None
    return _tool_item("toolace", request, funcs, called, split)


# ------------------------------------------------------------------ the table --------

B = ("no", "yes")

SPECS: tuple[Spec, ...] = (
    # ---- testreal -------------------------------------------------------------------
    Spec("cosmosqa", "cosmosqa", "testreal", "cosmosqa", "CC-BY-4.0",
         "https://huggingface.co/datasets/allenai/cosmos_qa", "", None, None, _read_cosmos,
         notes="Cosmos QA (Huang et al. 2019) train; question and 4 answers per example"),
    Spec("scruples_anecdotes", "scruples", "testreal", "scruples", "Apache-2.0",
         "https://huggingface.co/datasets/metaeval/scruples",
         "Who is in the wrong in this situation, according to the community?",
         ("author", "other", "everybody", "nobody", "info"),
         ("the author of the post is in the wrong (YTA)",
          "the other party is in the wrong, not the author (NTA)",
          "everyone involved is in the wrong (ESH)",
          "no one is in the wrong (NAH)",
          "more information is needed to judge (INFO)"),
         _read_scruples, soft=True,
         notes="Scruples anecdotes (Lourie et al. 2021): share of community votes per "
               "judgement; posts with >= 5 votes"),
    Spec("toolace_tool", "toolace", "testreal", "toolace", "Apache-2.0",
         "https://huggingface.co/datasets/Team-ACE/ToolACE", TOOL_Q, None, None,
         lambda r: _read_toolace(r, "testreal"), label_source="model", always_describe=True,
         notes="ToolACE (Liu et al. 2024), synthetic dialogues: the function the "
               "assistant's first turn calls, or a sentinel where it declines; "
               "multi-function first turns and clarifying questions dropped"),
    Spec("flute", "flute", "testreal", "flute", "AFL-3.0",
         "https://huggingface.co/datasets/ColumbiaNLP/FLUTE",
         "Does the premise entail or contradict this figurative hypothesis?",
         ("entailment", "contradiction"),
         ("the figurative sentence (sarcasm, simile, metaphor or idiom) means something "
          "the premise supports",
          "the figurative sentence means something the premise rules out"),
         _read_flute,
         notes="FLUTE (Chakrabarty et al. 2022): figurative hypotheses written with model "
               "help and verified and labelled by experts"),
    Spec("rumoureval_stance", "rumoureval", "testreal", "rumoureval", "CC-BY-4.0",
         "https://huggingface.co/datasets/strombergnlp/rumoureval_2019",
         "What stance does the reply take toward the rumour in the source tweet?",
         RUMOUR,
         ("the reply supports the veracity of the rumour",
          "the reply denies the veracity of the rumour",
          "the reply asks for more evidence about the rumour",
          "the reply comments without a clear contribution to its veracity"),
         _read_rumour, notes="RumourEval 2019 subtask A (Gorrell et al. 2019), train+val"),
    Spec("complaints", "sk_complaints", "testreal", "complaints",
         LIC_SOCKET + "Preotiuc-Pietro et al. 2019 complaints", URL_SOCKET,
         "Is this tweet a complaint?", B,
         ("the tweet does not express a complaint",
          "the tweet expresses a mismatch between reality and the author's expectations, "
          "a breach of expectation directed at a company or person"),
         _socket_bin("not complaint", "complaint"),
         notes="Complaints (Preotiuc-Pietro et al. 2019) via SocKET"),
    # ---- devreal --------------------------------------------------------------------
    Spec("scitail", "scitail", "devreal", "scitail", "Apache-2.0",
         "https://huggingface.co/datasets/allenai/scitail",
         "Does the premise entail the hypothesis?", ("entails", "neutral"),
         ("the premise supports the hypothesis",
          "the premise does not support the hypothesis"),
         _read_scitail, notes="SciTail (Khot et al. 2018) train+validation"),
    Spec("copa", "copa", "devreal", "copa", "CC-BY-4.0",
         "https://huggingface.co/datasets/pkavumba/balanced-copa", "", None, None, _read_copa,
         notes="Balanced COPA (Kavumba et al. 2019), train+test, mirrored items included"),
    Spec("hwu64_intent", "hwu64", "train", "hwu64",
         "CC-BY-SA-3.0 (Liu et al. 2019, NLU-Evaluation-Data)",
         "https://huggingface.co/datasets/DeepPavlov/hwu64",
         "Which intent does this request to a home assistant express?", None, None,
         _read_hwu, notes="HWU64 train; 64 scenario_action intents"),
    Spec("emobank_valence", "sk_emobank_valence", "devreal", "emobank",
         LIC_SOCKET + "EmoBank, CC-BY-SA-4.0 (Buechel & Hahn 2017)", URL_SOCKET,
         "How negative or positive is the feeling this sentence expresses?",
         ("very negative", "negative", "neutral", "positive", "very positive"),
         ("valence 1 on the 5-point Self-Assessment Manikin scale",
          "valence 2", "valence 3, neither negative nor positive", "valence 4",
          "valence 5, the most positive"),
         _rating((1, 2, 3, 4, 5)), ordinal=True, soft=True,
         notes="EmoBank valence mean rating, spread over the two nearest levels"),
    Spec("bragging", "sk_brag_achievement", "devreal", "bragging",
         LIC_SOCKET + "Jin et al. 2022 bragging", URL_SOCKET,
         "Does this tweet brag about an achievement?", B,
         ("the tweet is not bragging about an achievement",
          "the author says something positive about an achievement of theirs to impress"),
         _socket_bin("not achievement bragging", "achievement bragging"),
         notes="Bragging (Jin et al. 2022) via SocKET"),
    # ---- train ----------------------------------------------------------------------
    Spec("wanli", "wanli", "train", "wanli", "CC-BY-4.0",
         "https://huggingface.co/datasets/alisawuffles/WANLI",
         "What is the relation between the premise and the hypothesis?", NLI3, NLI3_CRITERIA,
         _read_wanli, notes="WANLI (Liu et al. 2022): model-drafted pairs, labelled and "
                            "revised by crowdworkers; gold label"),
    Spec("sick_entailment", "sick", "train", "sick", "CC-BY-NC-SA-3.0",
         "https://huggingface.co/datasets/sick",
         "What is the relation of sentence A to sentence B?", NLI3,
         ("sentence B is true if sentence A is",
          "the truth of sentence B does not follow from sentence A",
          "sentence B is false if sentence A is true"),
         _read_sick_ent, slice=(0, 2), notes="SICK (Marelli et al. 2014) train, half 0"),
    Spec("sick_relatedness", "sick", "train", "sick", "CC-BY-NC-SA-3.0",
         "https://huggingface.co/datasets/sick",
         "How related in meaning are the two sentences?", RELATED,
         ("relatedness 1 of 5", "relatedness 2 of 5", "relatedness 3 of 5",
          "relatedness 4 of 5", "relatedness 5 of 5: same meaning"),
         _read_sick_rel, ordinal=True, soft=True, slice=(1, 2),
         notes="SICK mean of 10 relatedness ratings, spread over the nearest two levels; half 1"),
    Spec("piqa", "piqa", "train", "piqa", "AFL-3.0",
         "https://huggingface.co/datasets/ybisk/piqa",
         "Which solution is the more sensible way to achieve the goal?", None, None,
         _read_piqa, notes="PIQA (Bisk et al. 2020) train"),
    Spec("siqa", "siqa", "train", "siqa", "CC-BY-4.0",
         "https://huggingface.co/datasets/allenai/social_i_qa", "", None, None, _read_siqa,
         notes="Social IQa (Sap et al. 2019) train"),
    Spec("swag", "swag", "train", "swag", "MIT",
         "https://huggingface.co/datasets/allenai/swag",
         "Which sentence most plausibly comes next?", None, None, _read_swag,
         notes="SWAG (Zellers et al. 2018) regular train; not HellaSwag"),
    Spec("obqa", "obqa", "train", "obqa", "Apache-2.0",
         "https://huggingface.co/datasets/allenai/openbookqa",
         "Which answer is correct?", None, None, _read_obqa,
         notes="OpenBookQA additional train (Mihaylov et al. 2018), with its core fact"),
    Spec("ethics_commonsense", "ethics_commonsense", "train", "ethics", "MIT",
         "https://huggingface.co/datasets/hendrycks/ethics",
         "By ordinary moral standards, did the narrator do something clearly wrong?",
         ("not wrong", "wrong"),
         ("the narrator's action is morally acceptable",
          "the narrator's action is clearly morally wrong"),
         _read_ethics_cs, notes="ETHICS commonsense (Hendrycks et al. 2021) train, short "
                                "crowd-written scenarios only (the long ones are AITA posts)"),
    Spec("ethics_deontology", "ethics_deontology", "train", "ethics", "MIT",
         "https://huggingface.co/datasets/hendrycks/ethics",
         "Is the excuse a reasonable reason not to fulfil the request or duty?",
         ("unreasonable", "reasonable"),
         ("the excuse does not exempt the speaker",
          "the excuse is a valid reason the duty does not apply"),
         _read_ethics_deon, notes="ETHICS deontology train"),
    Spec("ethics_justice", "ethics_justice", "train", "ethics", "MIT",
         "https://huggingface.co/datasets/hendrycks/ethics",
         "Is the claim of what the narrator deserves or how they treat others reasonable?",
         ("unreasonable", "reasonable"),
         ("the justification is unfair: undeserved or treats like cases unequally",
          "the justification is fair and reasonable"),
         _read_ethics_just, notes="ETHICS justice train"),
    Spec("moral_stories", "moral_stories", "train", "moral_stories", "MIT",
         "https://huggingface.co/datasets/demelin/moral_stories",
         "Does the action follow or violate the norm?", ("violates", "follows"),
         ("the action goes against the stated social norm",
          "the action is in line with the stated social norm"),
         _read_moral, notes="Moral Stories (Emelin et al. 2021) action+norm, norm_distance train"),
    Spec("dailydialog_act", "dailydialog", "devreal", "dailydialog", "CC-BY-NC-SA-4.0",
         "https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
         "What dialogue act does the target utterance perform?", DD_ACTS,
         ("the speaker provides information",
          "the speaker asks for information",
          "the speaker requests, instructs, suggests or advises the listener to do something",
          "the speaker commits to, accepts or rejects doing something"),
         _read_dd_act, slice=(0, 2), notes="DailyDialog (Li et al. 2017) acts, dialogues half 0"),
    Spec("dailydialog_emotion", "dailydialog", "devreal", "dailydialog", "CC-BY-NC-SA-4.0",
         "https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
         "Which emotion does the speaker express in the target utterance?", DD_EMOTIONS,
         ("no particular emotion", "anger", "disgust", "fear", "happiness or joy",
          "sadness", "surprise"),
         _read_dd_emotion, slice=(1, 2),
         notes="DailyDialog emotions (Ekman six + none), dialogues half 1"),
    Spec("multiwoz_domain", "multiwoz", "train", "multiwoz", "Apache-2.0",
         "https://huggingface.co/datasets/pfb30/multi_woz_v22",
         "Which service is the user's target turn about?", WOZ_DOMAINS, WOZ_CRITERIA,
         _read_woz, notes="MultiWOZ 2.2 (Zang et al. 2020): domain of the user turn's "
                          "annotated dialogue acts; one turn per dialogue"),
    Spec("glaive_tool", "glaive", "train", "glaive", "Apache-2.0",
         "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2", TOOL_Q,
         None, None, lambda r: _read_glaive(r, "train"), label_source="model",
         always_describe=True,
         notes="Glaive function calling v2, synthetic: the function the first assistant "
               "turn calls, or a sentinel where it declines; the row's own functions only"),
    Spec("sarcasm_reddit", "sk_sarc", "train", "sarc",
         LIC_SOCKET + "SARC (Khodak et al. 2018)", URL_SOCKET,
         "Is this Reddit comment sarcastic?", ("literal", "sarcastic"),
         ("the comment means what it says",
          "the comment is sarcastic: it says the opposite of or mocks what is meant "
          "(its author marked it /s)"),
         _socket_bin("literal", "sarcastic"), notes="SARC via SocKET; author-labelled"),
    Spec("humor", "sk_is_humor", "train", "hahackathon",
         LIC_SOCKET + "HaHackathon, SemEval-2021 Task 7 (Meaney et al. 2021)", URL_SOCKET,
         "Is this text intended to be humorous?", ("not humor", "humor"),
         ("most raters judged the text not intended to be funny",
          "most raters judged the text intended to be funny"),
         _socket_bin("not humor", "humor"), slice=(0, 2), slice_on=_text_key,
         notes="HaHackathon is_humor via SocKET; texts half 0"),
    Spec("humor_rating", "sk_humor_rating", "train", "hahackathon",
         LIC_SOCKET + "HaHackathon, SemEval-2021 Task 7 (Meaney et al. 2021)", URL_SOCKET,
         "How funny do readers find this joke?",
         ("not funny", "slightly funny", "moderately funny", "funny", "very funny"),
         ("mean rating 0 of 5", "mean rating about 1.25 of 5", "mean rating about 2.5 of 5",
          "mean rating about 3.75 of 5", "mean rating 5 of 5"),
         _rating((0, 1.25, 2.5, 3.75, 5)), ordinal=True, soft=True, slice=(1, 2),
         slice_on=_text_key,
         notes="HaHackathon mean humour rating, spread over the nearest two levels; texts "
               "half 1 (the rated texts are a subset of the is_humor texts)"),
    Spec("hyperbole", "sk_hyperbole", "train", "hypo_l",
         LIC_SOCKET + "HYPO-L (Zhang & Wan 2022)", URL_SOCKET,
         "Does this sentence use hyperbole?", ("not hyperbole", "hyperbole"),
         ("the sentence is meant literally",
          "the sentence exaggerates deliberately for effect"),
         _socket_bin("not hyperbole", "hyperbole"), notes="HYPO-L via SocKET"),
    Spec("condescension", "sk_condescension", "train", "talkdown",
         LIC_SOCKET + "TalkDown (Wang & Potts 2019)", URL_SOCKET,
         "Is the reply condescending toward the person it answers?",
         ("not condescending", "condescending"),
         ("the reply engages without talking down",
          "the reply talks down to the other person, implying they are inferior"),
         _read_condescension, notes="TalkDown via SocKET"),
    Spec("politeness", "sk_politeness", "train", "stanford_politeness",
         LIC_SOCKET + "Stanford Politeness Corpus, CC-BY-4.0 (Danescu-Niculescu-Mizil et al. 2013)",
         URL_SOCKET, "Is this request polite or impolite?", ("impolite", "polite"),
         ("raters judged the request impolite", "raters judged the request polite"),
         _socket_bin("impolite", "polite"),
         notes="Stanford Politeness (Wikipedia/StackExchange requests) via SocKET"),
    Spec("question_intimacy", "sk_intimacy", "train", "intimacy",
         LIC_SOCKET + "Question intimacy (Pei & Jurgens 2020)", URL_SOCKET,
         "How intimate is this question?", INTIMACY,
         ("impersonal, about facts or the world", "not intimate", "slightly personal",
          "somewhat personal", "personal: feelings, relationships, private life",
          "very personal and private"),
         _read_intimacy, ordinal=True, notes="Question intimacy via SocKET"),
    Spec("empathy", "sk_empathy", "train", "empathy",
         LIC_SOCKET + "Empathic reactions to news (Buechel et al. 2018)", URL_SOCKET,
         "Does the writer express empathy in this reaction to a news story?",
         ("not empathy", "empathy"),
         ("the writer's self-reported empathy is low",
          "the writer's self-reported empathy (warmth, compassion) is high"),
         _socket_bin("not empathy", "empathy"), notes="Empathy (Buechel 2018) via SocKET"),
    Spec("person_abuse", "sk_person_abuse", "train", "contextual_abuse",
         LIC_SOCKET + "Contextual Abuse Dataset (Vidgen et al. 2021)", URL_SOCKET,
         "Does this Reddit comment abuse a person?",
         ("not person directed abuse", "person directed abuse"),
         ("no abuse directed at an individual",
          "the comment abuses an identifiable person: the interlocutor or someone named"),
         _socket_bin("not person directed abuse", "person directed abuse"),
         notes="CAD person-directed abuse via SocKET"),
    Spec("deception", "sk_deception", "train", "diplomacy",
         LIC_SOCKET + "Diplomacy deception, It Takes Two to Lie (Peskov et al. 2020)",
         URL_SOCKET, "Did the sender of this Diplomacy game message intend it truthfully?",
         ("lie", "truth"),
         ("the sender marked the message as a lie", "the sender marked it as truthful"),
         _socket_bin("lie", "truth"), notes="Diplomacy sender truth via SocKET"),
)
SPEC_BY_NAME = {s.name: s for s in SPECS}
# which task keeps a state found in several pools: eval before train
PRIORITY = [s.name for s in SPECS]


def options_for(spec: Spec) -> tuple[str, ...] | None:
    if spec.task == "hwu64_intent":
        return tuple(hwu_names())
    return spec.options


def criteria_of(spec: Spec) -> tuple[str, ...] | None:
    if spec.task == "hwu64_intent":
        return tuple(hwu_criterion(n) for n in hwu_names())
    return spec.criteria


# ------------------------------------------------------------------ DI filter --------

_BLOCK = None
DI_DROPS: dict[str, int] = defaultdict(int)


def blocklist():
    global _BLOCK
    if _BLOCK is None:
        from lod.corpus.services.decontaminate import Blocklist
        _BLOCK = Blocklist.load(DI_BLOCKLIST) or False
    return _BLOCK or None


def di_text(item: dict, spec: Spec) -> str:
    opts = item.get("options") or options_for(spec) or ()
    return "\n".join([item["state"], item.get("question") or spec.question, *opts])


# ------------------------------------------------------------------ pools ------------

def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


_POOLS: dict[str, list[dict]] = {}
DROPS: dict[str, dict[str, int]] = {}


def raw_pool(spec: Spec) -> tuple[list[dict], dict[str, int]]:
    rows = store.load(f"{PREFIX}{spec.key}")
    drops: dict[str, int] = defaultdict(int)
    items: dict[str, dict] = {}
    for j, r in enumerate(rows):
        if spec.slice is not None:
            i, n = spec.slice
            rec = spec.slice_on(r, j) if spec.slice_on else j
            if _h(spec.family, rec) % n != i:
                continue
        try:
            got = spec.read(r)
        except (KeyError, TypeError, ValueError, IndexError):
            got = None
        if not got or not got.get("state"):
            drops["unreadable"] += 1
            continue
        if len(got["state"]) > MAX_STATE_CHARS:
            drops["over_max_chars"] += 1
            continue
        got.setdefault("cls", _argmax(got["target"]))
        k = norm_key(got["state"] + "\x00" + (got.get("question") or "") + "\x00"
                     + "\x00".join(got.get("options") or ()))
        if k in items:
            drops["duplicate"] += 1
            continue
        items[k] = got
    return list(items.values()), dict(drops)


def pools() -> dict[str, list[dict]]:
    if not _POOLS:
        claimed: set[str] = set()
        for name in PRIORITY:
            spec = SPEC_BY_NAME[name]
            if not store.has(f"{PREFIX}{spec.key}"):
                continue
            items, drops = raw_pool(spec)
            kept = [it for it in items if norm_key(it["state"]) not in claimed]
            drops["claimed_by_other_task"] = len(items) - len(kept)
            claimed.update(norm_key(it["state"]) for it in kept)
            _POOLS[name] = kept
            DROPS[name] = drops
    return _POOLS


def select(items: list[dict], n: int, spec: Spec, cap: float = MAX_CLASS_SHARE) -> list[dict]:
    """n items in hashed order, skipping DI-blocked ones; from a pool larger than n no
    class takes more than `cap` of the sample while another class still has items."""
    order = sorted(items, key=lambda it: _h(spec.name, it["state"], it.get("question", "")))
    bl = blocklist()
    limit = max(1, int(cap * n)) if len(order) > n else n
    taken, rest, count = [], [], defaultdict(int)
    for it in order:
        if len(taken) >= n:
            break
        if bl is not None and bl.blocked(di_text(it, spec)):
            DI_DROPS[spec.name] += 1
            continue
        if count[it["cls"]] < limit:
            taken.append(it)
            count[it["cls"]] += 1
        else:
            rest.append(it)
    if len(taken) < n:               # the pool cannot satisfy the cap: fill naturally
        taken += rest[: n - len(taken)]
    return taken


# ------------------------------------------------------------------ tasks ------------

def examples_for(spec: Spec, n: int) -> Iterator[Example]:
    fixed = options_for(spec)
    crit = criteria_of(spec)
    for it in select(pools().get(spec.name, []), n, spec):
        options = list(it.get("options") or fixed)
        descs = it.get("descriptions") or (list(crit) if crit else None)
        describe = spec.always_describe or (
            _h("describe", spec.name, it["state"]) % 1000 < DESCRIBE_SHARE * 1000)
        meta = {"label_source": spec.label_source, **(it.get("meta") or {}),
                **(it.get("meta_extra") or {})}
        if spec.soft:
            meta.setdefault("source_label", options[it["cls"]])
        t = it["target"]
        yield Example(task=spec.name, state=it["state"], questions=[Question(
            id=spec.task, question=it.get("question") or spec.question,
            options=options, target=list(t) if isinstance(t, list) else int(t),
            descriptions=list(descs) if (describe and descs) else None, meta=meta)])


def criteria_for(task: str, options: list[str]) -> list[str] | None:
    spec = SPEC_BY_NAME.get(task)
    if spec is None or spec.per_example:
        return None
    crit = criteria_of(spec)
    if not crit or list(options) != list(options_for(spec) or ()):
        return None
    return list(crit)


def tasks() -> list[RealTask]:
    """Empty until `scripts/fetch_data.py --rows 50` has stored the rows."""
    out = []
    for spec in SPECS:
        if not store.has(f"{PREFIX}{spec.key}"):
            continue
        if spec.task == "hwu64_intent" and not store.has("rl_hwu64_intents"):
            continue
        out.append(RealTask(
            row=ROW, name=spec.name, licence=spec.licence, url=spec.url,
            load=(lambda s: (lambda n: examples_for(s, n)))(spec),
            ordinal=spec.ordinal, soft=spec.soft, force_split=spec.split,
            per_example_options=spec.per_example, family=spec.family, min_quota=MIN_QUOTA,
            notes=spec.notes + (" [labels from a data-generating model]"
                                if spec.label_source == "model" else "")))
    return out
