"""The source table, as `HFSpec` rows.

Each entry is one dataset the adapter in `hf.py` reads. Where the source is a live
system or a bulk archive and a packaged mirror of the same records exists on the Hub,
`mirror_of` records the substitution so the provenance manifest states it plainly. The
definition of a real schema turns on where the *labels* came from — people or a real system —
not on which host served the bytes, so a mirror of arXiv's own categories is as real as
the metadata dump, and costs an hour less.

Rows handled elsewhere: 1 (`hub_sweep.py` sweeps the Hub and needs no table),
5 (`gharchive.py`), 9 (`nvd.py`), 15 (`spdx.py`), 28 (`manifold.py`).
"""

from __future__ import annotations

from lod.corpus.services.sources.real.hf import HFSpec

SPECS: tuple[HFSpec, ...] = (
    # --- row 1: curated additions to the sweep ------------------------------------
    # Wayfair WANDS replaces Amazon ESCI (domain-1 audit): ESCI is scored by the Decision Index,
    # and the sweep read it through `product_text` alone, so "how relevant is this product"
    # was asked without the query it is relevant *to*. WANDS is the same judgement --
    # Exact / Partial / Irrelevant for a (query, product) pair, by Wayfair's annotators --
    # so both halves of the pair are the state. `query_class` is left out: it is Wayfair's
    # own category for the query, and next to `product_class` it is half the answer.
    # Irrelevant < Partial < Exact is a declared order.
    HFSpec(1, "napsternxg/wands", "MIT",
           text_cols=("query", "product_name", "product_class", "product_description"),
           label_cols=("label",),
           ordinal_cols=("label",),
           question="How well does this product match the search query?",
           max_rows=4000,
           mirror_of="WANDS, the Wayfair ANnotation DataSet (github.com/wayfair/WANDS)",
           notes="query-product relevance; replaces ESCI, which the Decision Index scores"),

    # --- row 2: tasksource --------------------------------------------------------
    # No generic spec. bigbench is read by `instructions._bigbench_loader`, its dedicated
    # adapter; through this table the generic adapter found no label column and read
    # the per-option `multiple_choice_scores` list as a multi-label vocabulary, asking
    # "Does this text carry the label '0'?" -- 704 questions that answered `yes` every
    # time, caught by `quality.example_rejection` but never worth making (domain-4 audit).
    # MMLU is gone too: its 100 cached rows are MMLU *test* questions, MMLU is in the
    # Decision Index, and every one is a world fact the state does not contain.

    # --- row 3: Super-NaturalInstructions ------------------------------------------
    HFSpec(3, "Muennighoff/natural-instructions", "Apache-2.0",
           notes="filtered to tasks with a small output set"),

    # --- row 4: Jira ---------------------------------------------------------------
    HFSpec(4, "Suvo696/jira_issues", "unknown",
           ordinal_cols=("priority", "Priority"),
           mirror_of="the Public Jira Dataset (Zenodo)",
           notes="issue type, priority (ordinal), resolution, component"),

    # --- row 6: arXiv ---------------------------------------------------------------
    # These are the paper's *own PDF text*, and an arXiv PDF is stamped down its left
    # margin with the submission's identifier and primary category -- "arXiv:1611.03253v1
    # [cs.DS] 10 Nov 2016". Measured on an earlier build: 88.5 % of states carried a stamp and
    # it equalled the gold label on 55.4 % of them, so more than half of row 6 was reading
    # a stamp. The stamp goes, every other bracketed `cs.XX` / `math.XX` mention goes with
    # it (one non-gold category still narrows eleven options), and `scrub_label` removes
    # the gold's own wording from the body the way it does for LEDGAR.
    HFSpec(6, "ccdv/arxiv-classification", "CC0-1.0",
           scrub_label=True,
           strip_state_res=(r"arxiv:\s*\S+(?:\s*\[[^\]\n]{2,30}\])?",
                            r"\[(?:cs|math|stat|physics|q-bio|q-fin|econ|eess|astro-ph|"
                            r"cond-mat|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|math-ph|nlin|"
                            r"nucl-ex|nucl-th|quant-ph)(?:\.[A-Za-z-]+)?\]"),
           mirror_of="the arXiv metadata snapshot",
           notes="primary category from the authors' own submission; the PDF's own "
                 "arXiv category stamp is stripped"),
    HFSpec(6, "mteb/ArxivClassification", "CC0-1.0",
           scrub_label=True,
           strip_state_res=(r"arxiv:\s*\S+(?:\s*\[[^\]\n]{2,30}\])?",),
           mirror_of="the arXiv metadata snapshot"),

    # --- row 7: PubMed ---------------------------------------------------------------
    HFSpec(7, "ml4pubmed/pubmed-classification-20k", "public domain (NLM)",
           drop_label_re=r"###",
           mirror_of="the PubMed baseline",
           notes="section type assigned by indexers; ###<pmid> document separators "
                 "share the label column and are dropped"),

    # --- row 8: OpenAlex ---------------------------------------------------------------
    # The mirror does not publish what the spec row wants. Its only text is the work's
    # title, its field / subfield / topic columns are bare OpenAlex URLs with no published
    # names, and its year / citation / author counts are quantities the title cannot
    # carry -- all four were already dropped by the answerability gate. That left
    # `oa_status` as the whole of row 8 in an earlier build: 1,000 questions asking, given only
    # a title, whether the work is gold, green, bronze, hybrid, diamond or closed. Open
    # access is a property of the venue and the licence, not of the words in the title,
    # and the corpus says so -- a naive-Bayes classifier on the title scores 0.215 against
    # a 0.790 majority, i.e. *below* answering "closed" every time. It is dropped rather
    # than guessed.
    HFSpec(8, "XIfr/Openalex-2005-2025", "CC0-1.0",
           skip_cols=("oa_status",),
           mirror_of="the OpenAlex works dump",
           notes="titles only; no answerable label column survives -- see the comment"),

    # --- row 10: legislation ----------------------------------------------------------
    # The mirror carries no CRS policy area, whatever the old note here said: its columns
    # are `docClass`, `congress`, `is_environment` and the GPO text. `congress` is a bare
    # numeral and the gate drops it, so row 10 is `docClass` -- which measure this is --
    # and every one of the 572 states in an earlier build opened with
    # "[H.R. 3870 Placed on Calendar Senate (PCS)]" followed by a centred "H. R. 3870".
    # 100 % stated their own answer. `scrub_label` cannot reach it: the class is `hr` and
    # the document writes `H.R.`. The bracketed GPO header block and the centred
    # designator line go; what is left still says "A BILL" or "RESOLUTION" and "IN THE
    # SENATE" or "IN THE HOUSE OF REPRESENTATIVES", which is the document's own form and
    # is the thing the question is actually about.
    HFSpec(10, "finnmok/congressional_bills", "public domain (US Government)",
           text_cols=("text",),
           strip_state_res=(r"^[ \t]*\[[^\]\n]*\][ \t]*$",
                            r"^[ \t]*(?:h|s)\.?[ \t]*(?:j\.?[ \t]*res|con\.?[ \t]*res|res|r)?"
                            r"\.?[ \t]*\d+[ \t]*$"),
           mirror_of="Congress.gov bulk data",
           notes="which measure this is; the GPO header naming it is stripped"),

    # --- row 11: clinical (RESERVED to testreal) ---------------------------------------
       # ClinicalTrials is staged, but its snapshot contains a protocol corpus and a
       # separate QA table. It requires a dedicated adapter; do not turn tier flags into
       # generic decisions.

    # --- row 12: SEC EDGAR -------------------------------------------------------------
    HFSpec(12, "eloukas/edgar-corpus", "public domain (US SEC)",
           mirror_of="SEC EDGAR 10-K filings",
           notes="the canonical packaged EDGAR corpus"),

    # --- row 13: Lichess (RESERVED to testreal) ----------------------------------------
       # Lichess is staged, but themes and puzzle ratings need a dedicated structured-state
       # adapter and explicit score buckets; do not expose raw numeric metadata as choices.

    # --- row 14: Loghub ------------------------------------------------------------------
    HFSpec(14, "bolu61/loghub_2", "MIT",
           ordinal_cols=("level", "Level", "severity"),
           mirror_of="Loghub",
           notes="log level and component across systems"),

    # --- row 16: intent -------------------------------------------------------------------
    # The train split is sorted by intent, 100 rows each, so the old 4,000-row cache held
    # 40 of the 151 intents and no `oos` at all (domain-1 audit). The whole split is 15,250
    # short utterances; read all of it.
    # clinc/clinc_oos removed: CLINC150 is a Decision Index benchmark (see
    # hub_sweep.BENCHMARK_OVERLAP). Intent stays covered by MASSIVE and SNIPS.
    # `label` and `label_text` are the same 42 strings -- the mirror ships the intent
    # twice under two names, and the detector made two tasks of it. The split hash then
    # put `label_text` in train and `label` in devreal, so 1,865 of domain 1's 2,420
    # eval questions with a train-identical option set are this one duplicate. Whole-task
    # splitting exists so that eval measures an unseen *schema*; a column copied under a
    # second name is not a second schema.
    # The default config is every locale concatenated, Afrikaans first: the old 4,000-row
    # cache was 100 % `af`, sorted by intent, 42 of 60 intents (domain-1 audit). English, whole
    # train split (11,514 rows).
    HFSpec(16, "mteb/amazon_massive_intent", "CC-BY-4.0", config="en",
           skip_cols=("label", "lang"), max_rows=12000,
           mirror_of="MASSIVE (qanastek/MASSIVE is script-based)"),
    # Sorted by intent: the old 4,000-row cache held 3 of SNIPS's 7 (domain-1 audit).
    HFSpec(16, "benayas/snips", "CC-BY-4.0", max_rows=14000),

    # --- row 17: LexGLUE -------------------------------------------------------------------
    # 53.9 % of these clauses stated their own class name -- "Governing Laws" heads a
    # governing-law clause -- so more than half the task was a string match until
    # `scrub_label`. SPDX has stripped its titles since it was written; the generic
    # adapter had no equivalent.
    HFSpec(17, "coastalcph/lex_glue", "CC-BY-4.0", config="ledgar",
           scrub_label=True,
           notes="100 contract clause types; the clause's own heading is redacted"),
    HFSpec(17, "coastalcph/lex_glue", "CC-BY-4.0", config="scotus",
           notes="13 issue areas"),
    # Fetched through this spec, read by `unfair_tos.py`. The generic adapter turned the
    # `labels` list into eight yes/no "does this carry label X" tasks whose majority
    # baseline was 0.956-0.994 and whose enriched questions asked the clause's *topic*
    # while the label means "flagged as an unfair term of type X" (domain-10 audit).
    HFSpec(17, "coastalcph/lex_glue", "CC-BY-4.0", config="unfair_tos",
           skip_cols=("labels",),
           notes="read by unfair_tos.py, not by the generic adapter"),

    # --- row 18: newsgroups / Reuters --------------------------------------------------------
    HFSpec(18, "SetFit/20_newsgroups", "public domain"),
    # yangwang825/reuters-21578 removed: Reuters-21578 is one of the datasets SATA-Bench
    # (a Decision Index member) is assembled from; 11 of its 249 Reuters paragraphs were
    # in our stored rows (audit v14_topic).

    # --- row 19: Civil Comments (soft) --------------------------------------------------------
    # Each column is the published share of raters who said the attribute applies (for
    # toxicity: "toxic" or "very toxic"); no threshold is applied. The raw store holds
    # 4,000 rows, split disjointly between the tasks, and three attributes are too rare in
    # it to be a question at all (domain-3 audit, counted on the store): `severe_toxicity`
    # never exceeds 0.26 -- no comment a majority calls severely toxic -- and `threat` and
    # `sexual_explicit` have 3 and 12 such comments in 4,000; their tasks were 96-98 %
    # exact zeros. They are skipped, not guessed; a larger pool would bring them back.
    # The rest are sliced positives-first (`soft_enrich`), and each asks its own question
    # instead of "What is the insult of this text?".
    HFSpec(19, "google/civil_comments", "CC0-1.0",
           soft_cols=("toxicity", "insult", "obscene", "identity_attack"),
           skip_cols=("severe_toxicity", "threat", "sexual_explicit"),
           soft_enrich=True,
           questions={
               "toxicity": "Is this comment toxic -- rude, disrespectful or unreasonable "
                           "enough to make someone leave a discussion?",
               "insult": "Is this comment insulting, inflammatory or negative toward a "
                         "person or group of people?",
               "obscene": "Does this comment contain obscene or profane language?",
               "identity_attack": "Does this comment attack or make a negative statement "
                                  "about a group of people because of their identity?",
           },
           notes="per-item rater fractions from the Civil Comments release; no threshold"),

    # --- row 22: DynaSent (ordinal, hard) -------------------------------------------------------
    # The mirror is `text` + one integer `label` (0 negative, 1 neutral, 2 positive per its
    # feature names; the rows bear that out). It carries neither the per-worker label
    # distribution nor the round, so this row is a *hard* 3-way majority label, not soft
    # -- the old note said "soft" (domain-3 audit). Its 97,873 rows are DynaSent R1+R2 train
    # (93,553, re-split 80/20 into train/val) plus R1+R2 test (4,320).
    HFSpec(22, "HelloWorld2307/dynasent", "Apache-2.0",
           ordinal_cols=("gold_label", "label"),
           question="What is the overall sentiment of this sentence?",
           mirror_of="dynabench/dynasent (script-based)",
           notes="majority sentiment label, negative < neutral < positive; no rater "
                 "distribution or round in the mirror"),

    # --- row 23: Social Bias Frames -------------------------------------------------------------
    # one row per annotator (`WorkerId`), not per post: 4,000 rows are 1,357 distinct
    # posts. Without `soft_aggregate_by` each worker's own 0/0.5/1 judgement posed as the
    # item's rater fraction, and the same post reached the corpus several times with
    # contradictory targets -- 184 of them, measured on an earlier build.
    # A worker also gets one row per implied statement they wrote, so the vote is per
    # `WorkerId`, not per row: 220 of 917 posts with >= 3 rows had < 3 workers (domain-3 audit).
    # The values are SBF's own hedged codes -- offensive and sex 1 / 0.5 ("maybe") / 0,
    # intent 1 / 0.66 / 0.33 / 0 -- so the mean is the raters' average credence.
    HFSpec(23, "momererkoc/social_bias_frames", "CC-BY-4.0",
           label_cols=("offensiveYN", "intentYN", "sexYN"),
           soft_cols=("offensiveYN", "intentYN", "sexYN"),
           soft_aggregate_by="post",
           soft_rater_by="WorkerId",
           soft_enrich=True,
           questions={
               "offensiveYN": "Could this post be considered offensive, disrespectful or "
                              "toxic to anyone?",
               "intentYN": "Was the intent of this post to be offensive?",
               "sexYN": "Does this post contain lewd or sexual references?",
           },
           mirror_of="allenai/social_bias_frames (script-based)"),

    # --- row 24: NLI ------------------------------------------------------------------------------
    # the label is a relation between the two sentences, so both are the state. Without
    # text_cols the generic adapter takes the longest string column, which for MNLI is
    # `premise_parse` -- a constituency tree, and only one side of the pair.
    # `genre` is not asked (domain-9 audit): it is a register classifier over the
    # source text -- premise-only naive Bayes 0.663 against a 0.224 majority -- not a
    # relation between the spans, and as the only train-side MNLI task it put 2 of 1,983
    # testreal MNLI premises into train under another question.
    HFSpec(24, "nyu-mll/multi_nli", "CC-BY-SA-3.0/OANC",
           text_cols=("premise", "hypothesis"),
           label_cols=("label",),
           skip_cols=("premise_parse", "hypothesis_parse",
                      "premise_binary_parse", "hypothesis_binary_parse"),
           question="What is the relationship between the premise and the hypothesis?",
           drop_label_re=r"^-1$",
           notes="gold label only; the Hub release does not carry the five annotator labels"),
    # SNLI's `label` is pinned to train because otherwise nothing in the corpus teaches
    # the entailment relation at all: MultiNLI `label` hashes to testreal and ChaosNLI is
    # reserved there whole, so row 24's only trained task was MultiNLI `genre`. MNLI and
    # ChaosNLI stay where they were, so transfer is still measured on tasks never trained.
    HFSpec(24, "stanfordnlp/snli", "CC-BY-SA-4.0",
           text_cols=("premise", "hypothesis"),
           label_cols=("label",),
           force_split_cols={"label": "train"},
           question="What is the relationship between the premise and the hypothesis?",
           drop_label_re=r"^-1$",
           notes="-1 is SNLI's no-consensus marker, not a class; pinned to train"),

       # Row 26 is handled by `kalshi.py`, which groups settled markets into meaningful
       # families. Do not expose the generic adapter: its numeric market metadata columns
       # are not model-facing decision schemas.

    # --- row 29: GitHub pull requests ---------------------------------------------------------------
    # The mirror does not hold what its name says and what it does hold is not answerable
    # from a state. Measured on the cached rows (534 of them, `~/.cache/lod-sources/
    # v4-raw/manoelalmeida_io_github_pullrequests.jsonl`):
    #
    #   * `is_pull_request` is False on 534 of 534 and `pull_request` is null on all of
    #     them. There is not one pull request in the "github-pullrequests" mirror; they
    #     are issues. An earlier build therefore asked "What is the current status of this pull
    #     request?" over 131 issues -- a question wrong about its own subject, 100 % of
    #     the time.
    #   * `repository_url` is `huggingface/datasets` on 534 of 534. One repo, so the row
    #     contributes no project breadth either, which is the whole of its domain-8 brief.
    #   * the three columns that survived detection are outcome and identity metadata:
    #     `state` (open/closed at snapshot time), `state_reason` (null on 235 of 534 and
    #     96.7 % `completed` over what is left) and `author_association`. The state is the
    #     issue body; it says nothing about who filed it or what later became of it.
    #     `author_association` is the same failure the row-1 rule names -- "a column
    #     recording **who annotated the row** is dropped too" -- and only slipped through
    #     because ANNOTATOR_COL matches `submitter`/`contributor` as whole words and this
    #     column is spelled `author_association`.
    #
    # So the columns were skipped rather than asked -- and with them skipped the spec
    # yielded **0 tasks** (domain-8 audit), while still sitting in the catalogue
    # as if row 29 were covered. The one answerable question left in it is the repo's
    # own labels, 130 single-label issues of huggingface/datasets (67 % `enhancement`),
    # which is exactly one more `gh_label_*` task, under an "unknown" licence, beside
    # row 5's ~1,600. **Row 29 is removed from the catalogue**; domain 8 is row 5. A real
    # PR source (GH Archive PullRequestEvent: merged or not, review outcome) would be a
    # new loader in `gharchive.py`, not a Hub mirror.

    # --- row 30: record fields --------------------------------------------------------------------
    # Removed (domain-12 audit). `storytracer/openlibrary_dump_2024-04-30` cached as
    # 4,000 rows, every one `/type/author`, whose only low-cardinality columns are
    # `revision` / `latest_revision` -- edit counters (1 on 3,669 rows), which the
    # provenance filter in `hf.py` rightly refuses. It produced 0 tasks in an earlier build and 0 from
    # current code, and no enum field in the cache could replace them.

    # --- row 31: TabFact ------------------------------------------------------------------------------
    HFSpec(31, "Raywithyou/TabFact", "CC-BY-4.0",
           mirror_of="wenhu/tab_fact (script-based)"),

    # --- row 32: MBPP ---------------------------------------------------------------------------------
    HFSpec(32, "google-research-datasets/mbpp", "CC-BY-4.0", config="full",
           notes="row 32 wants the tests executed"),
)


def specs_for(rows: set[int] | None = None) -> list[HFSpec]:
    return [s for s in SPECS if rows is None or s.row in rows]


def rows_covered() -> set[int]:
    return {s.row for s in SPECS}
