"""Row 51 (real half) -- decisions over whole long documents (domain 25).

The generated half (`synth/longctx.py`) controls where the evidence sits. This half is the
check that the skill transfers to text nobody wrote for us: licensed long documents with a
label a person or a court gave, where the decision needs the document rather than a
window of it. Every state is between `MIN_TOKENS` and `MAX_TOKENS` Qwen3 tokens and is
never truncated -- a cut could remove the evidence and leave the label standing.

| task | source | question | split (family) |
|---|---|---|---|
| `longdoc_quality_gutenberg` | QuALITY v1.0.1, Project Gutenberg articles | 4-way reading MC | train |
| `longdoc_quality_slate` | QuALITY, Slate (OANC) articles | 4-way reading MC | devreal |
| `longdoc_quality_misc` | QuALITY, CC BY magazine/book articles | 4-way reading MC | testreal |
| `longdoc_cuad_train` | CUAD v1 contracts, 32 clause types | clause present? | train |
| `longdoc_cuad_dev` | CUAD, 6 held-out clause types | clause present? | devreal |
| `longdoc_cuad_test` | CUAD, 6 held-out clause types | clause present? | testreal |
| `longdoc_ecthr_train` | LexGLUE ECtHR (train), Art. 3/5/6/P1-1 | alleged article violated? | train |
| `longdoc_ecthr_dev` | ECtHR (validation), Art. 2/8 | alleged article violated? | devreal |
| `longdoc_ecthr_test` | ECtHR (test), Art. 9/10/11/14 | alleged article violated? | testreal |

What is held out: QuALITY by article *source* (the writing, not just the article), CUAD by
clause type and by contract, ECtHR by Convention article and by case (the dataset's own
train/validation/test). No document is shared between two tasks, so the exact-state dedup
never has to delete an eval side.

Labels are the dataset's: QuALITY `gold_label` (the majority of the validators), CUAD
"any span annotated for this category", ECtHR the articles the Court found violated
(`ecthr_a`) among those the applicant alleged (`ecthr_b`). A `no` for ECtHR is therefore
an article that *was* alleged and not upheld, never an article nobody raised -- which
would be answerable from the topic alone. CUAD and ECtHR are balanced to 50/50 yes/no
per task by choosing which categories/articles to ask, never by changing a label.
Categories that are present in (nearly) every contract -- Document Name, Parties,
Agreement Date -- are not asked.

Every example goes through the Decision Index v0.2 blocklist here (and again in
`services/generate.py`); `DI_DROPS` records what it removed. ContractNLI (a DI member built on
contracts) is not used.

Fetch once, offline afterwards:

    uv run python scripts/fetch_data.py --rows 51
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lod.schema import Example, Question
from lod.corpus.services.sources.base import CACHE
from lod.corpus.repositories import raw_store as store
from lod.corpus.services.sources.real.base import RealTask
from lod.paths import DI_BLOCKLIST

ROW = 51
PREFIX = "longdoc_"
MIN_TOKENS = 2304          # the old state budget: anything shorter teaches nothing new here
MAX_TOKENS = 29000         # under the 30,000-token state budget with room for the prefix
MAX_Q_PER_EXAMPLE = 8      # question blocks share one prefill; keep the total under 32,768
TOKENIZER = "Qwen/Qwen3-0.6B-Base"

QUALITY_ZIP = "https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/QuALITY.v1.0.1.zip"
CUAD_ZIP = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
LEXGLUE = "coastalcph/lex_glue"

LIC_QUALITY = ("CC-BY-4.0 (QuALITY, Pang et al. 2022); article texts: Project Gutenberg "
               "licence (US public domain), OANC licence (Slate), CC-BY-4.0 (misc)")
LIC_CUAD = "CC-BY-4.0 (CUAD v1, The Atticus Project; contracts from SEC EDGAR filings)"
LIC_ECTHR = ("CC-BY-NC-SA-4.0 (LexGLUE ECtHR, Chalkidis et al. 2019/2022; HUDOC case "
             "facts) -- non-commercial project, attribution owed")

# LexGLUE ECtHR label index -> Convention article
ECTHR_ARTICLES = ("2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1")
ECTHR_TITLES = {
    "2": "right to life", "3": "prohibition of torture and inhuman or degrading treatment",
    "5": "right to liberty and security", "6": "right to a fair trial",
    "8": "right to respect for private and family life",
    "9": "freedom of thought, conscience and religion", "10": "freedom of expression",
    "11": "freedom of assembly and association", "14": "prohibition of discrimination",
    "P1-1": "protection of property",
}
ECTHR_GROUPS = {"train": ("3", "5", "6", "P1-1"), "devreal": ("2", "8"),
                "testreal": ("9", "10", "11", "14")}
ECTHR_SPLIT_FILE = {"train": "train", "devreal": "validation", "testreal": "test"}

CUAD_SKIP = frozenset({"Document Name", "Parties", "Agreement Date"})
CUAD_DEV = frozenset({"Audit Rights", "Insurance", "Non-Compete", "Minimum Commitment",
                      "Renewal Term", "Covenant Not To Sue"})
CUAD_TEST = frozenset({"Cap On Liability", "Exclusivity", "Change Of Control",
                       "Post-Termination Services", "Revenue/Profit Sharing",
                       "Ip Ownership Assignment"})
CUAD_PER_SIDE = 3          # at most this many present and this many absent per contract

QUALITY_SOURCES = {"train": ("Gutenberg",), "devreal": ("Slate",),
                   "testreal": ("misc-longshort", "misc-freesouls", "misc-openaccess")}

DI_DROPS: dict[str, int] = defaultdict(int)


def data_dir() -> Path:
    """`<raw root>/longdocs` when the configured raw root points somewhere that has
    it, else `~/.cache/lod-sources/longdocs`."""
    for base in (store.RAW_ROOT, CACHE):
        p = Path(base).expanduser() / "longdocs"
        if p.is_dir():
            return p
    return CACHE / "longdocs"


# ---- tokens and the DI blocklist, cached per document ---------------------------------

_TOK = None
_BL = None
_SIDE: dict | None = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        errs = []
        for kw in ({},):
            try:
                _TOK = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, **kw)
                break
            except Exception as e:  # noqa: BLE001 - try the next cache location
                errs.append(e)
        if _TOK is None:
            raise RuntimeError(f"no local {TOKENIZER} tokenizer: {errs}")
    return _TOK


def blocklist():
    global _BL
    if _BL is None:
        from lod.corpus.services.decontaminate import Blocklist
        _BL = Blocklist.load(DI_BLOCKLIST) or False
    return _BL or None


def _sidecar() -> dict:
    global _SIDE
    if _SIDE is None:
        p = data_dir() / "_doc_meta.json"
        _SIDE = json.loads(p.read_text()) if p.exists() else {}
    return _SIDE


def _save_sidecar() -> None:
    if _SIDE is None:
        return
    p = data_dir() / "_doc_meta.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(_SIDE))
    tmp.replace(p)


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def doc_meta(texts: list[str]) -> list[dict]:
    """[{ntok, di}] per text: Qwen3 token count, and whether the DI v0.2 blocklist
    matches it. Computed once per document and cached beside the raw files."""
    side = _sidecar()
    bl = blocklist()
    tag = f"di:{DI_BLOCKLIST.name}"
    todo = [t for t in texts if _sha(t) not in side or (bl and tag not in side[_sha(t)])]
    if todo:
        tok = tokenizer()
        for i in range(0, len(todo), 64):
            chunk = todo[i:i + 64]
            ids = tok(chunk, add_special_tokens=False)["input_ids"]
            for t, x in zip(chunk, ids):
                m = side.setdefault(_sha(t), {})
                m["ntok"] = len(x)
                if bl:
                    m[tag] = bool(bl.blocked(t))
        _save_sidecar()
    return [{"ntok": side[_sha(t)]["ntok"], "di": bool(side[_sha(t)].get(tag, False))}
            for t in texts]


def text_blocked(text: str) -> bool:
    bl = blocklist()
    return bool(bl and bl.blocked(text))


def _in_range(n: int) -> bool:
    return MIN_TOKENS <= n <= MAX_TOKENS


def _gen(ntok: int, family: str, source: str) -> dict:
    return {"state_tokens": ntok, "depth_tokens": None, "depth_frac": None,
            "n_records": None, "family": family, "source": source}


def _bucket(key: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:{key}".encode()).digest()[:8], "big") % 100


# ---- QuALITY ----------------------------------------------------------------------------

def _quality_articles() -> dict[str, dict]:
    """article_id -> {text, source, title, questions (deduplicated by text)} over the
    labelled train and dev files (the test file's labels are hidden)."""
    arts: dict[str, dict] = {}
    for sp in ("train", "dev"):
        p = data_dir() / f"QuALITY.v1.0.1.htmlstripped.{sp}"
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                a = arts.setdefault(r["article_id"], {
                    "id": r["article_id"], "text": r["article"].strip(),
                    "source": r["source"], "title": r["title"], "questions": {}})
                for q in r["questions"]:
                    if q.get("gold_label") in (1, 2, 3, 4):
                        a["questions"].setdefault(q["question"].strip(), q)
    return arts


def _quality_examples(split: str, family: str) -> list[Example]:
    arts = [a for a in _quality_articles().values() if a["source"] in QUALITY_SOURCES[split]]
    arts.sort(key=lambda a: a["id"])
    metas = doc_meta([a["text"] for a in arts])
    out: list[Example] = []
    for a, m in zip(arts, metas):
        if not _in_range(m["ntok"]):
            continue
        if m["di"]:
            DI_DROPS[family] += 1
            continue
        qs = []
        for text, q in sorted(a["questions"].items(), key=lambda kv: kv[1]["question_unique_id"]):
            opts = [str(o).strip() for o in q["options"]]
            if len(set(opts)) != 4 or any(not o for o in opts):
                continue
            if text_blocked(text + "\n" + "\n".join(opts)):
                DI_DROPS[family] += 1
                continue
            qs.append(Question(
                id=f"longdoc_quality_{q['question_unique_id']}", question=text,
                options=opts, target=int(q["gold_label"]) - 1,
                meta={"gen": _gen(m["ntok"], family, "quality"), "group": f"quality:{a['id']}",
                      "difficult": int(q.get("difficult", 0)), "article_id": a["id"]}))
        for i in range(0, len(qs), MAX_Q_PER_EXAMPLE):
            out.append(Example(state=a["text"], questions=qs[i:i + MAX_Q_PER_EXAMPLE]))
    return out


# ---- CUAD ---------------------------------------------------------------------------------

_CAT = re.compile(r'related to "(.+?)"')


def _cuad_contracts() -> list[dict]:
    raw = json.loads((data_dir() / "CUADv1.json").read_text())["data"]
    out = []
    for d in raw:
        para = d["paragraphs"][0]
        cats = {}
        for q in para["qas"]:
            m = _CAT.search(q["question"])
            if not m:
                continue
            details = q["question"].split("Details:", 1)[1].strip() if "Details:" in q["question"] else ""
            cats[m.group(1)] = (bool(q["answers"]), details)
        out.append({"title": d["title"], "text": para["context"].strip(), "cats": cats})
    return out


def _cuad_split(title: str) -> str:
    v = _bucket(title, "cuad")
    return "train" if v < 70 else ("devreal" if v < 85 else "testreal")


def _cuad_group(split: str, cats) -> list[str]:
    if split == "devreal":
        return sorted(c for c in cats if c in CUAD_DEV)
    if split == "testreal":
        return sorted(c for c in cats if c in CUAD_TEST)
    return sorted(c for c in cats if c not in CUAD_SKIP | CUAD_DEV | CUAD_TEST)


NEAR_DUP = 0.3
_CUAD_TRAIN_SH: dict | None = None


def _cuad_near_dup(text: str) -> float:
    """Largest share of this contract's 10-word shingles found in ONE training contract.
    CUAD carries amended/re-filed versions of the same agreement under different titles:
    2 dev contracts share 0.84 and 0.55 of their shingles with a training contract, while
    the dev median is 0.008 and the test p90 0.038 (boilerplate)."""
    global _CUAD_TRAIN_SH
    from lod.corpus.services.sources.real.base import shingles
    if _CUAD_TRAIN_SH is None:
        _CUAD_TRAIN_SH = {}
        for i, c in enumerate(x for x in _cuad_contracts() if _cuad_split(x["title"]) == "train"):
            for sh in shingles(c["text"]):
                _CUAD_TRAIN_SH.setdefault(sh, set()).add(i)
    mine = shingles(text)
    hits: dict[int, int] = defaultdict(int)
    for sh in mine:
        for i in _CUAD_TRAIN_SH.get(sh, ()):
            hits[i] += 1
    return max(hits.values()) / len(mine) if hits and mine else 0.0


def _cuad_examples(split: str, family: str) -> list[Example]:
    contracts = sorted((c for c in _cuad_contracts() if _cuad_split(c["title"]) == split),
                       key=lambda c: c["title"])
    if split != "train":
        contracts = [c for c in contracts if _cuad_near_dup(c["text"]) < NEAR_DUP]
    metas = doc_meta([c["text"] for c in contracts])
    out: list[Example] = []
    for c, m in zip(contracts, metas):
        if not _in_range(m["ntok"]):
            continue
        if m["di"]:
            DI_DROPS[family] += 1
            continue
        rng = random.Random(f"cuad:{c['title']}")
        group = _cuad_group(split, c["cats"])
        yes = [k for k in group if c["cats"][k][0]]
        no = [k for k in group if not c["cats"][k][0]]
        k = min(len(yes), len(no), CUAD_PER_SIDE)
        if k == 0:
            continue
        picked = rng.sample(yes, k) + rng.sample(no, k)
        rng.shuffle(picked)
        qs = []
        for cat in picked:
            present, details = c["cats"][cat]
            qs.append(Question(
                id=f"longdoc_cuad_{_sha(c['title'])[:10]}_{re.sub(r'[^a-z0-9]+', '_', cat.lower())}",
                question=f'Does this contract contain a "{cat}" clause?',
                instructions=f"{cat}: {details}" if details else None,
                options=["no", "yes"], target=int(present),
                meta={"gen": _gen(m["ntok"], family, "cuad"), "group": f"cuad:{c['title']}",
                      "category": cat, "contract": c["title"]}))
        out.append(Example(state=c["text"], questions=qs))
    return out


# ---- ECtHR --------------------------------------------------------------------------------

def _ecthr_cases(split: str) -> list[dict]:
    p = data_dir() / f"ecthr_{ECTHR_SPLIT_FILE[split]}.jsonl"
    with p.open() as f:
        return [json.loads(line) for line in f]


def _ecthr_examples(split: str, family: str) -> list[Example]:
    group = ECTHR_GROUPS[split]
    cases = []
    for i, r in enumerate(_ecthr_cases(split)):
        violated, alleged = set(r["violated"]), set(r["alleged"])
        if not violated <= alleged:
            continue                              # the two LexGLUE views disagree: skip
        yes = sorted(a for a in violated if a in group)
        no = sorted(a for a in alleged - violated if a in group)
        if not yes and not no:
            continue
        text = "\n".join(p.strip() for p in r["text"]).strip()
        # cheap pre-cut: Qwen3 reads 3-6 characters per token in this prose
        if not MIN_TOKENS * 3 <= len(text) <= MAX_TOKENS * 6:
            continue
        cases.append({"i": i, "text": text, "yes": yes, "no": no})
    metas = doc_meta([c["text"] for c in cases])
    kept = []
    for c, m in zip(cases, metas):
        if not _in_range(m["ntok"]):
            continue
        if m["di"]:
            DI_DROPS[family] += 1
            continue
        c["ntok"] = m["ntok"]
        kept.append(c)
    # balance yes/no over the task: a case with both sides contributes one of each; the
    # one-sided cases fill in, the minority side first, the majority up to the same count
    rng = random.Random(f"ecthr:{split}")
    picks: dict[int, list[tuple[str, int]]] = {}
    one_yes, one_no = [], []
    for c in kept:
        if c["yes"] and c["no"]:
            picks[c["i"]] = [(rng.choice(c["yes"]), 1), (rng.choice(c["no"]), 0)]
        elif c["yes"]:
            one_yes.append(c)
        else:
            one_no.append(c)
    n = min(len(one_yes), len(one_no))
    for c in rng.sample(one_yes, n):
        picks[c["i"]] = [(rng.choice(c["yes"]), 1)]
    for c in rng.sample(one_no, n):
        picks[c["i"]] = [(rng.choice(c["no"]), 0)]
    out = []
    for c in kept:
        if c["i"] not in picks:
            continue
        qs = []
        for art, t in picks[c["i"]]:
            name = f"Article {art}" if art != "P1-1" else "Article 1 of Protocol No. 1"
            qs.append(Question(
                id=f"longdoc_ecthr_{ECTHR_SPLIT_FILE[split]}_{c['i']}_{art}",
                question=(f"The applicant complained of a violation of {name} of the European "
                          f"Convention on Human Rights ({ECTHR_TITLES[art]}). Did the European "
                          f"Court of Human Rights find a violation of {name}?"),
                options=["no", "yes"], target=t,
                meta={"gen": _gen(c["ntok"], family, "ecthr"),
                      "group": f"ecthr:{ECTHR_SPLIT_FILE[split]}:{c['i']}", "article": art}))
        out.append(Example(state=c["text"], questions=qs))
    return out


# ---- tasks --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Spec:
    name: str
    split: str
    build: Callable[[str, str], list[Example]]
    licence: str
    url: str
    needs: tuple[str, ...]
    notes: str


SPECS = (
    Spec("longdoc_quality_gutenberg", "train", _quality_examples, LIC_QUALITY,
         "https://github.com/nyu-mll/quality", ("QuALITY.v1.0.1.htmlstripped.train",),
         "QuALITY train+dev questions over Project Gutenberg articles; gold_label"),
    Spec("longdoc_quality_slate", "devreal", _quality_examples, LIC_QUALITY,
         "https://github.com/nyu-mll/quality", ("QuALITY.v1.0.1.htmlstripped.train",),
         "QuALITY questions over Slate articles (held-out source)"),
    Spec("longdoc_quality_misc", "testreal", _quality_examples, LIC_QUALITY,
         "https://github.com/nyu-mll/quality", ("QuALITY.v1.0.1.htmlstripped.train",),
         "QuALITY questions over CC-BY magazine/book articles (held-out sources)"),
    Spec("longdoc_cuad_train", "train", _cuad_examples, LIC_CUAD,
         "https://www.atticusprojectai.org/cuad", ("CUADv1.json",),
         "clause-type presence, 32 trained clause types, 70 % of contracts"),
    Spec("longdoc_cuad_dev", "devreal", _cuad_examples, LIC_CUAD,
         "https://www.atticusprojectai.org/cuad", ("CUADv1.json",),
         "6 held-out clause types over 15 % of contracts"),
    Spec("longdoc_cuad_test", "testreal", _cuad_examples, LIC_CUAD,
         "https://www.atticusprojectai.org/cuad", ("CUADv1.json",),
         "6 other held-out clause types over another 15 % of contracts"),
    Spec("longdoc_ecthr_train", "train", _ecthr_examples, LIC_ECTHR,
         "https://huggingface.co/datasets/coastalcph/lex_glue", ("ecthr_train.jsonl",),
         "alleged article upheld? Art. 3/5/6/P1-1, LexGLUE train cases"),
    Spec("longdoc_ecthr_dev", "devreal", _ecthr_examples, LIC_ECTHR,
         "https://huggingface.co/datasets/coastalcph/lex_glue", ("ecthr_validation.jsonl",),
         "alleged article upheld? Art. 2/8, LexGLUE validation cases"),
    Spec("longdoc_ecthr_test", "testreal", _ecthr_examples, LIC_ECTHR,
         "https://huggingface.co/datasets/coastalcph/lex_glue", ("ecthr_test.jsonl",),
         "alleged article upheld? Art. 9/10/11/14, LexGLUE test cases"),
)

_BUILT: dict[str, list[Example]] = {}


def build(spec: Spec) -> list[Example]:
    """Every example of a task, in a fixed order. Memoised: the tokenizer pass is the cost."""
    if spec.name not in _BUILT:
        _BUILT[spec.name] = spec.build(spec.split, spec.name)
    return _BUILT[spec.name]


def _loader(spec: Spec):
    def load(n: int):
        exs = list(build(spec))
        random.Random(f"{spec.name}:order").shuffle(exs)
        out = []
        for e in exs[:n]:
            out.append(Example(state=e.state, task=spec.name, questions=[
                Question(id=q.id, question=q.question, options=list(q.options),
                         target=q.target, meta=json.loads(json.dumps(q.meta)),
                         instructions=q.instructions) for q in e.questions]))
        return out
    return load


def tasks() -> list[RealTask]:
    out = []
    for spec in SPECS:
        if not all((data_dir() / f).exists() for f in spec.needs):
            continue
        # Contracts and judgments are written in boilerplate: every CUAD contract shares
        # ten-word runs ("in witness whereof the parties hereto have ...") with some
        # training contract, and every ECtHR judgment with some training judgment. The
        # shingle half of `dedup_against_train` read that as a leak and deleted all 99
        # CUAD dev/test examples and 134 of 151 ECtHR ones on the first build, though no
        # document is shared (the splits are by contract and by the dataset's own case
        # split). The exact-state test still applies. QuALITY articles stay under it.
        legal = spec.name.startswith(("longdoc_cuad", "longdoc_ecthr"))
        out.append(RealTask(row=ROW, name=spec.name, licence=spec.licence, url=spec.url,
                            load=_loader(spec), force_split=spec.split, family=spec.name,
                            template_state=legal, notes=spec.notes))
    return out


# ---- fetch --------------------------------------------------------------------------------

def fetch() -> None:
    import io
    import urllib.request
    import zipfile

    d = CACHE / "longdocs"
    d.mkdir(parents=True, exist_ok=True)
    for url, want in ((QUALITY_ZIP, "QuALITY.v1.0.1.htmlstripped."), (CUAD_ZIP, "CUADv1.json")):
        blob = urllib.request.urlopen(url, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in z.namelist():
                if name.startswith(want) and not name.startswith("__MACOSX"):
                    (d / Path(name).name).write_bytes(z.read(name))
                    print("wrote", d / Path(name).name)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    for sp in ("train", "validation", "test"):
        a = pq.read_table(hf_hub_download(LEXGLUE, f"ecthr_a/{sp}-00000-of-00001.parquet",
                                          repo_type="dataset")).to_pylist()
        b = pq.read_table(hf_hub_download(LEXGLUE, f"ecthr_b/{sp}-00000-of-00001.parquet",
                                          repo_type="dataset")).to_pylist()
        assert len(a) == len(b) and all(x["text"] == y["text"] for x, y in zip(a, b))
        with (d / f"ecthr_{sp}.jsonl").open("w") as f:
            for x, y in zip(a, b):
                f.write(json.dumps({"text": x["text"],
                                    "violated": [ECTHR_ARTICLES[i] for i in x["labels"]],
                                    "alleged": [ECTHR_ARTICLES[i] for i in y["labels"]]}) + "\n")
        print("wrote", d / f"ecthr_{sp}.jsonl", len(a))


def _report() -> None:
    from collections import Counter
    total = Counter()
    for spec in SPECS:
        exs = build(spec)
        nq = sum(len(e.questions) for e in exs)
        hist = Counter()
        tgt = Counter()
        for e in exs:
            n = e.questions[0].meta["gen"]["state_tokens"]
            b = ("<2304" if n < 2304 else "2304-8k" if n < 8000 else "8-16k" if n < 16000
                 else "16-30k")
            hist[b] += len(e.questions)
            total[b] += len(e.questions)
            tgt.update(q.target for q in e.questions)
        print(f"{spec.name:28} {spec.split:8} ex={len(exs):5} q={nq:5} "
              f"di_drop={DI_DROPS.get(spec.name, 0)} lengths={dict(hist)} targets={dict(tgt)}")
    print("total", dict(total), sum(total.values()))


if __name__ == "__main__":
    _report()
