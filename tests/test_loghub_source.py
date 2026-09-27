"""Tests for the row-14 log source (domain 13) after the 16-system rewrite.

Two halves. The pure-function half runs anywhere: it pins one hand-checked real line per
system format, the drop rule (a line whose level cannot be *read* is dropped, never
guessed), the redaction, and the balancing arithmetic. The cache half needs the logs
The fetch stage caches under `<LOD_RAW_ROOT>/loghub/` and skips when
they are absent -- but when they are present it asserts non-emptiness, because a loader
that quietly returns `[]` is a failure that looks like success and is exactly how
`loghub_component` shipped zero rows for as long as it did.

The holdout test is the one that matters most: the supercomputer family is reserved to
`testreal`, and a reserved system leaking into train, val or devreal would turn a
generalisation measurement back into a memorisation one without anything going red.
"""

from __future__ import annotations


import pytest

from lod.corpus.services.sources.real import loghub
from lod.corpus.repositories import raw_store as store
from lod.paths import RAW_ROOT

CACHE = RAW_ROOT


@pytest.fixture(scope="module")
def cached():
    """Point the raw store at the real cache, or skip. Restores the store afterwards."""
    before = store.RAW_ROOT
    store.configure(CACHE)
    if not loghub.systems_available():
        store.configure(before)
        pytest.skip("no cached loghub logs; run scripts/fetch_data.py --rows 14")
    loghub._IN_PROCESS.clear()
    yield loghub
    store.configure(before)
    loghub._IN_PROCESS.clear()


# ------------------------------------------------------------------- the parser

# One real line per format, copied verbatim out of each `<System>_2k.log`. These are the
# hand-read examples the domain brief asks for: if a format drifts, this is where it shows.
REAL_LINES = [
    ("HDFS", "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 "
             "for block blk_38865049064139660 terminating",
     "INFO", "dfs.DataNode$PacketResponder"),
    ("Hadoop", "2015-10-18 18:01:47,978 INFO [main] "
               "org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Created MRAppMaster",
     "INFO", "org.apache.hadoop.mapreduce.v2.app.MRAppMaster"),
    ("Spark", "17/06/09 20:10:40 INFO executor.CoarseGrainedExecutorBackend: "
              "Registered signal handlers for [TERM, HUP, INT]",
     "INFO", "executor.CoarseGrainedExecutorBackend"),
    ("Zookeeper", "2015-07-29 19:04:29,071 - WARN  [SendWorker:188978561024:"
                  "QuorumCnxManager$SendWorker@688] - Send worker leaving thread",
     "WARN", "QuorumCnxManager$SendWorker"),
    ("OpenStack", "nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:00:00.008 25746 "
                  "INFO nova.osapi_compute.wsgi.server [req-38101a0b] 10.11.10.1 GET",
     "INFO", "nova.osapi_compute.wsgi.server"),
    ("BGL", "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 "
            "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected",
     "INFO", "KERNEL"),
    ("BGL", "- 1120241131 2005.07.01 R37-M1-N4 2005-07-01-11.05.31.120732 R37-M1-N4 "
            "NULL DISCOVERY SEVERE Can not get assembly information for node card",
     None, "DISCOVERY"),             # SEVERE is below ERROR on BG/L's scale: no slot
    ("Windows", "2016-09-28 04:30:30, Info                  CBS    Loaded Servicing "
                "Stack v6.1.7601.23505",
     "INFO", "CBS"),
    ("Android", "03-17 16:13:38.811  1702  2395 D WindowManager: printFreezingDisplayLogs",
     "DEBUG", "WindowManager"),      # a bare letter is this system's whole level field
    ("Apache", "[Sun Dec 04 04:47:44 2005] [notice] workerEnv.init() ok", "NOTICE", None),
    ("HPC", "134681 node-246 unix.hw state_change.unavailable 1077804742 1 Component "
            "State Change", None, "unix.hw"),
    ("Thunderbird", "- 1131566461 2005.11.09 dn228 Nov 9 12:01:01 dn228/dn228 "
                    "crond(pam_unix)[2915]: session closed for user root",
     None, "crond(pam_unix)"),
    ("Linux", "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure",
     None, "sshd(pam_unix)"),
    ("Mac", "Jul  1 09:01:05 calvisitor-10-105-160-95 com.apple.CDScheduler[43]: "
            "Thermal pressure state: 1", None, "com.apple.CDScheduler"),
    ("OpenSSH", "Dec 10 06:55:46 LabSZ sshd[24200]: Invalid user webmaster", None, "sshd"),
    ("HealthApp", "20171223-22:15:29:606|Step_LSC|30002312|onStandStepChanged 3579",
     None, "Step_LSC"),
    ("Proxifier", "[10.30 16:49:06] chrome.exe - proxy.cse.cuhk.edu.hk:5070 open through "
                  "proxy", None, "chrome.exe"),
]


@pytest.mark.parametrize("system,line,level,component", REAL_LINES,
                         ids=[f"{r[0]}-{i}" for i, r in enumerate(REAL_LINES)])
def test_each_system_format_is_read_positionally(system, line, level, component):
    got = loghub.read_line(line, system)
    assert got is not None, f"{system}: pattern did not match its own log line"
    assert got["level"] == level
    assert got["component"] == component


def test_a_message_that_merely_mentions_a_level_word_is_not_given_one():
    """The drop rule, and the reason the first keyword parser had to go: OpenSSH
    publishes no severity field, so a line whose *text* says "failure" (and elsewhere
    "error") must come back with no level at all rather than a guessed ERROR."""
    line = ("Dec 10 06:55:46 LabSZ sshd[24200]: error: Received disconnect from "
            "173.234.31.186: Bye Bye [preauth]")
    got = loghub.read_line(line, "OpenSSH")
    assert got is not None and got["level"] is None


def test_a_line_that_does_not_match_its_system_shape_is_dropped_entirely():
    assert loghub.read_line("this is not a log line at all", "HDFS") is None
    assert loghub.read_line("081109 203615 148 INFO dfs.X: ok", "Spark") is None


def test_an_unknown_level_token_is_not_invented():
    """THE INVARIANT: a target code cannot derive is dropped, not guessed. A Spark-shaped
    line whose level field holds something that is not a severity yields level None."""
    got = loghub.read_line("17/06/09 20:10:40 BANANA executor.Foo: hello", "Spark")
    assert got is not None and got["level"] is None and got["component"] == "executor.Foo"


def test_every_system_in_families_has_a_format_and_vice_versa():
    assert set(loghub._FORMATS) == set(loghub.FAMILY_OF)


def test_fetcher_and_reader_agree_on_the_systems():
    """The holdout is defined by the family map; a fetcher that fetched other systems than
    the reader maps would quietly move a system from the reserved side to the trained side."""
    from lod.corpus.repositories import loghub as fetcher

    assert set(fetcher.SYSTEMS) == set(loghub.FAMILY_OF)


# ------------------------------------------------------------------- redaction

def test_level_scrub_blanks_the_field_and_leaves_the_message():
    line = "17/06/09 20:10:40 WARN spark.Foo: could not connect, will retry"
    got = loghub.read_line(line, "Spark")
    state = loghub._scrub(line, got["level_span"], "[LEVEL]")
    assert "[LEVEL]" in state and "WARN" not in state
    assert "could not connect, will retry" in state


def test_component_scrub_removes_every_copy_of_the_name():
    """A syslog line names its daemon in the message as often as in the program field;
    leaving the second copy measured a 23.6 % string match on the held-out family."""
    line = "Dec 10 06:55:46 LabSZ sshd[24200]: sshd restarted by root"
    got = loghub.read_line(line, "OpenSSH")
    state = loghub._scrub(line, got["component_span"], "[COMPONENT]",
                          everywhere=got["component"])
    assert "sshd" not in state
    assert state.count("[COMPONENT]") == 2


# ------------------------------------------------------------------- balancing

def test_balanced_cap_holds_the_majority_class_under_the_target():
    q = loghub.balanced_cap([1000, 400, 300, 200], 0.40, min_pool=10)
    kept = [min(n, q) for n in (1000, 400, 300, 200)]
    assert max(kept) <= 0.40 * sum(kept)


def test_balanced_cap_relaxes_rather_than_emptying_a_thin_pool():
    """BGL writes INFO, FATAL and two ERROR lines. No cap gets the majority under 0.40
    without leaving ten questions, so the honest answer is a looser bound over a usable
    pool -- and never a cap so tight the task disappears."""
    sizes = [177, 93, 2]
    tight = [min(n, loghub.balanced_cap(sizes, 0.40, min_pool=0)) for n in sizes]
    relaxed = [min(n, loghub.balanced_cap(sizes, 0.40, min_pool=60)) for n in sizes]
    assert sum(tight) < 60 <= sum(relaxed)
    assert max(relaxed) <= 0.55 * sum(relaxed)


def test_thin_spreads_across_the_pool_instead_of_taking_a_prefix():
    got = loghub._thin(list(range(100)), 5)
    assert got == [0, 20, 40, 60, 80]


# ------------------------------------------------------------------- bursts

def test_escalation_label_drops_an_all_identical_burst():
    """The first version's degenerate case: one Apache error_log repeats a line for
    pages, and six copies of it is a free "steady"."""
    same = [{"text": "x", "level": "INFO"} for _ in range(6)]
    assert loghub.escalation_label(same) is None


def test_escalation_label_compares_only_against_the_first_leveled_line():
    rows = [{"text": f"l{i}", "level": lv} for i, lv in
            enumerate(["WARN", None, "INFO", "ERROR", "INFO", "INFO"])]
    assert loghub.escalation_label(rows) == "escalates"
    rows[3]["level"] = "DEBUG"
    assert loghub.escalation_label(rows) == "steady"


def test_bursts_never_straddle_two_systems():
    rows = ([{"system": "A", "text": str(i), "level": "INFO"} for i in range(4)]
            + [{"system": "B", "text": str(i), "level": "INFO"} for i in range(8)])
    groups = list(loghub.iter_bursts(rows, size=4))
    assert all(len({r["system"] for r in g}) == 1 for g in groups)


# ------------------------------------------------------------------- against the cache

def test_every_cached_system_parses_almost_every_line(cached):
    report = cached.parse_report()
    assert len(report) >= 10, "expected the 16-system cache, not the single-Apache one"
    for system, stats in report.items():
        assert stats["parse_rate"] >= 0.90, (system, stats)


def test_level_is_not_a_constant_and_component_fires(cached):
    """The two documented claims that failed on the old cache: an ordered severity *scale*,
    and a component target that actually ships."""
    assert len(cached.level_options()) >= 4
    per_system = cached.components_by_system(cached.component_candidates(False))
    assert len(per_system) >= 4
    assert all(len(v) >= 2 for v in per_system.values())


def test_tasks_are_non_empty_and_cover_both_sides_of_the_holdout(cached):
    names = {t.name for t in cached.tasks()}
    assert "loghub_level" in names and "loghub_component" in names
    assert any(n.endswith(cached.HELD_OUT_SUFFIX) for n in names)
    for task in cached.tasks():
        assert list(task.load(20)), f"{task.name} loaded nothing"


def test_reserved_supercomputer_systems_never_reach_train_val_or_devreal(cached):
    """The structural holdout. `carve_val` takes val out of train, so a system that never
    reaches train never reaches val either; devreal and testreal are assigned per task."""
    from lod.corpus.services.sources.real.base import split_of

    seen_in_trainable = set()
    for task in cached.tasks():
        split = task.force_split or split_of(task.name, task.row)
        systems = {e.questions[0].meta["system"] for e in task.load(4000)}
        if split == "testreal":
            continue
        seen_in_trainable |= systems
    assert not (seen_in_trainable & set(cached.HELD_OUT)), seen_in_trainable


def test_held_out_tasks_carry_only_held_out_systems(cached):
    for task in cached.tasks():
        if not task.name.endswith(cached.HELD_OUT_SUFFIX):
            continue
        assert task.force_split == "testreal"
        for ex in task.load(4000):
            assert ex.questions[0].meta["family"] == cached.HELD_OUT_FAMILY


def test_dev_holdout_is_its_own_family_and_holds_out_no_level(cached):
    """Dev holds out a whole family the way test does, disjoint from test's and from
    training -- and, unlike `mobile` (DEBUG) or `server_app` (NOTICE), one whose holdout
    leaves every severity on the scale trained."""
    assert cached.DEV_FAMILY != cached.HELD_OUT_FAMILY
    dev = [t for t in cached.tasks() if t.force_split == "devreal"]
    assert dev and all(t.name.endswith(cached.DEV_SUFFIX) for t in dev)
    trained_systems = set()
    for task in cached.tasks():
        if task.force_split != "train":
            continue
        for ex in task.load(4000):
            trained_systems.add(ex.questions[0].meta["system"])
    assert not trained_systems & set(cached.DEV_HELD_OUT)
    for task in dev:
        for ex in task.load(4000):
            assert ex.questions[0].meta["family"] == cached.DEV_FAMILY
    levels = {r["level"] for r in cached.parsed_rows() if r["level"]}
    trained_levels = {r["level"] for r in cached.parsed_rows()
                      if r["level"] and r["family"] not in (cached.DEV_FAMILY,
                                                             cached.HELD_OUT_FAMILY)}
    assert levels == trained_levels


def test_format_is_a_variable_not_a_constant(cached):
    """The first version's corpus was one 2005 Apache error_log, so
    `[Sun Jul 03 04:07:55 2005]` was learnable once and the row was over."""
    for name in ("loghub_level", "loghub_component", "loghub_burst_escalate"):
        task = next(t for t in cached.tasks() if t.name == name)
        systems = {e.questions[0].meta["system"] for e in task.load(4000)}
        assert len(systems) >= 4, (name, systems)


def test_the_gold_option_is_not_simply_printed_in_the_state(cached):
    """Row 36 shipped at a 100 % string-match rate and so did the first `loghub_level`.
    The level task keeps message words like "error" on purpose -- they are evidence, not
    the printed answer -- so a small residual is expected and a large one is a failed
    task.

    Matched as a whole token. A component name *inside* a longer word (`/bgl/apps/` for
    `APP`, `/etc/xinetd.d/` for `xinetd`) is left in the message as evidence: redacting
    it as a substring is what made the placeholder count name the answer (0.798 against a
    0.473 marginal on the held-out component task)."""
    import re

    limits = {"loghub_level": 0.25, "loghub_component": 0.0,
              "loghub_burst_escalate": 0.0}
    for task in cached.tasks():
        base = task.name.replace(cached.HELD_OUT_SUFFIX, "").replace(cached.DEV_SUFFIX, "")
        examples = list(task.load(4000))
        leaked = sum(1 for e in examples
                     if re.search(r"(?<![\w.$])"
                                  + re.escape(e.questions[0].options[e.questions[0].target])
                                  + r"(?![\w.$])", e.state, re.IGNORECASE))
        assert leaked / len(examples) <= limits[base], (task.name, leaked, len(examples))


def test_no_state_is_under_the_token_floor(cached):
    """Counted in characters, not tokens: 128 characters is the smallest floor at which no
    line in this cache falls under 40 Qwen tokens (measured with `Qwen/Qwen3-0.6B-Base`),
    and `MIN_STATE_CHARS` sits above that. Keeping the check character-based keeps the
    test, like the loader, free of a tokenizer download."""
    assert cached.MIN_STATE_CHARS >= 128
    for task in cached.tasks():
        for ex in task.load(500):
            assert len(ex.state) >= cached.MIN_STATE_CHARS, (task.name, ex.state)


def test_no_class_dominates_any_task(cached):
    from collections import Counter

    for task in cached.tasks():
        examples = list(task.load(4000))
        counts = Counter(e.questions[0].options[e.questions[0].target] for e in examples)
        majority = counts.most_common(1)[0][1] / len(examples)
        assert majority <= 0.70, (task.name, counts)


def test_no_log_line_backs_two_tasks(cached):
    lines_by_task = {}
    for task in cached.tasks():
        lines: set[str] = set()
        for ex in task.load(4000):
            lines.update(ex.state.split("\n"))
        lines_by_task[task.name] = lines
    names = sorted(lines_by_task)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert lines_by_task[a].isdisjoint(lines_by_task[b]), (a, b)


def test_licence_is_the_one_the_repository_publishes(cached):
    """LogHub is research data, not MIT-licensed software, and this project has an open
    licence audit -- so the string every task carries has to say so."""
    assert "MIT" not in loghub.LICENCE
    assert "research" in loghub.LICENCE.lower()
    for task in cached.tasks():
        assert task.licence == loghub.LICENCE
    assert (CACHE / "loghub" / "LICENSE").exists()


# --------------------------------------------- the domain-13 audit: shortcut baselines

def _prior_accuracy(pairs):
    """Best accuracy reachable by memorising one answer per key -- the lookup-table
    ceiling for a reader who sees only `key` and never the message."""
    from collections import Counter, defaultdict

    lut = defaultdict(Counter)
    for key, label in pairs:
        lut[key][label] += 1
    return sum(v.most_common(1)[0][1] for v in lut.values()) / len(pairs)


def test_balance_within_systems_kills_the_conditional_prior():
    """Pure arithmetic, no cache. Two systems, each a landslide the other way: balancing
    the pooled histogram alone leaves a perfect system -> label lookup, which is what
    an earlier build shipped (0.8158 on `loghub_level` against a 0.400 majority)."""
    items = ([{"s": "A", "l": "INFO"} for _ in range(200)]
             + [{"s": "A", "l": "WARN"} for _ in range(40)]
             + [{"s": "B", "l": "WARN"} for _ in range(200)]
             + [{"s": "B", "l": "INFO"} for _ in range(40)])
    got = loghub._balance_within_systems(items, lambda r: r["s"], lambda r: r["l"],
                                         ["INFO", "WARN"], 0.55)
    assert got
    assert _prior_accuracy([(r["s"], r["l"]) for r in got]) <= 0.60


def test_a_system_that_writes_one_level_is_not_asked_about_its_level():
    """Spark and Windows are 2,000/2,000 INFO. Recognising the format *is* the answer,
    so they contribute components and bursts and nothing to the level task."""
    items = ([{"s": "OneLevel", "l": "INFO"} for _ in range(500)]
             + [{"s": "Mixed", "l": "INFO"} for _ in range(100)]
             + [{"s": "Mixed", "l": "WARN"} for _ in range(100)])
    got = loghub._balance_within_systems(items, lambda r: r["s"], lambda r: r["l"],
                                         ["INFO", "WARN"], 0.55)
    assert {r["s"] for r in got} == {"Mixed"}


def test_the_burst_question_states_the_comparison_the_code_performs():
    """An earlier wording asked whether severity "increase[s] at any point after the first
    line"; `escalation_label` compares every later line against the *first* one. A run
    that dips and recovers satisfies the first reading and not the second, and 9.7 % of
    the trained task did exactly that."""
    dip = [{"text": f"l{i}", "level": lv} for i, lv in
           enumerate(["WARN", "INFO", "WARN", "WARN", "INFO", "WARN"])]
    assert loghub.escalation_label(dip) == "steady"
    assert "at any point" not in loghub.BURST_QUESTION
    assert "first line" in loghub.BURST_QUESTION
    steady, escalates = loghub.BURST_CRITERIA
    assert "more severe than the first line" in steady
    assert "higher severity than the first line" in escalates


def test_every_level_and_burst_option_carries_a_criterion():
    """Criteria as input: the key is an identifier, the description carries the meaning. These two
    tasks shipped `described = 0` in an earlier build."""
    assert set(loghub.LEVEL_CRITERIA) == set(loghub.LEVEL_ORDER)
    assert len(loghub.BURST_CRITERIA) == len(loghub.BURST_OPTIONS)
    for key, text in loghub.LEVEL_CRITERIA.items():
        assert key.lower() not in text.lower(), f"{key} restates its own key"


def test_the_line_format_alone_does_not_answer_the_question(cached):
    """The system is readable off the line shape without reading one message word, so the
    per-system label prior is a lower bound on what a format-only reader scores. It was
    0.8158 on `loghub_level` and 0.7664 on `loghub_burst_escalate` in an earlier build."""
    for task in cached.tasks():
        examples = list(task.load(4000))
        pairs = [(e.questions[0].meta["system"],
                  e.questions[0].options[e.questions[0].target]) for e in examples]
        prior = _prior_accuracy(pairs)
        assert prior <= 0.60, (task.name, prior, len(examples))


def test_no_single_system_task_is_two_thirds_one_answer(cached):
    """`balanced_cap` relaxes its target to keep a thin pool alive; the held-out burst
    task reached 0.650 that way. Under the shippable floor it is allowed to be smaller
    instead of more lopsided."""
    from collections import Counter

    for task in cached.tasks():
        examples = list(task.load(4000))
        assert len(examples) >= cached.MIN_SHIPPABLE_EXAMPLES
        counts = Counter(e.questions[0].options[e.questions[0].target] for e in examples)
        assert counts.most_common(1)[0][1] / len(examples) <= 0.60, (task.name, counts)


def test_the_holdout_moves_the_level_prior_and_the_number_is_published(cached):
    """Not a fix -- a fence. BGL is the corpus's only source of FATAL, so a class that is
    correct in no training example is a large share of the eval labels, and DEBUG is the
    reverse. Anything that changes the holdout changes these two numbers, and they belong
    in the corpus documentation rather than in a surprise at eval time."""
    from collections import Counter

    def hist(name):
        task = next(t for t in cached.tasks() if t.name == name)
        return Counter(e.questions[0].options[e.questions[0].target]
                       for e in task.load(4000))

    trained, held = hist("loghub_level"), hist("loghub_level_supercomputer")
    unseen = sum(v for k, v in held.items() if trained.get(k, 0) == 0)
    assert unseen / sum(held.values()) > 0.20, (
        "the FATAL asymmetry has moved; re-measure it and update the module docstring")
    assert trained.get("DEBUG", 0) and not held.get("DEBUG", 0)


# --------------------------------------------- the domain-13 audit, second pass

def test_log4j_padding_does_not_survive_the_level_redaction():
    """Zookeeper pads its level to five columns, so after blanking `INFO` two spaces were
    left and after blanking `ERROR` one: every one of its 13 ERROR lines was readable off
    the width of the gap (an independent all-lines padding check, 13 -> 0)."""
    lines = ["2015-07-29 19:04:29,071 - INFO  [main:Foo@1] - Send worker leaving thread",
             "2015-07-29 19:04:29,071 - ERROR [main:Foo@1] - Send worker leaving thread"]
    states = []
    for line in lines:
        got = loghub.read_line(line, "Zookeeper")
        states.append(loghub.level_state(dict(got, text=line, system="Zookeeper")))
    assert states[0] == states[1]
    assert "[LEVEL] [main" in states[0]


def test_the_bgl_alert_flag_is_not_shown():
    """LogHub's first BGL/Thunderbird field is the dataset's alert annotation, and every
    flagged BGL line is FATAL. A lookup on it scored 0.541 against a 0.400 prior on the
    held-out level task; the `APP*` flags also name the `APP` component."""
    line = ("KERNDTLB 1118536327 2005.06.11 R30-M0-N9-C:J16-U01 2005-06-11-17.32.07.581048 "
            "R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt")
    got = loghub.read_line(line, "BGL")
    assert got["level"] == "FATAL"
    row = dict(got, text=line, system="BGL")
    for state in (loghub.level_state(row), loghub.component_state(row)):
        assert "KERNDTLB" not in state and state.startswith("1118536327 ")
    assert loghub.display_text(line, "Hadoop") == line


def test_a_burst_printing_an_off_scale_level_is_dropped():
    """BG/L ranks SEVERE below ERROR; the shared scale has no slot for it, so a burst that
    prints one cannot be ranked by the reader and is not asked about."""
    assert loghub.canon_level("SEVERE") is None
    rows = [{"text": f"l{i}", "level": lv, "level_raw": raw} for i, (lv, raw) in
            enumerate([("ERROR", "ERROR"), (None, "SEVERE"), ("INFO", "INFO"),
                       ("INFO", "INFO"), ("ERROR", "ERROR"), ("INFO", "INFO")])]
    assert loghub.escalation_label(rows) is None
    rows[1] = {"text": "l1", "level": None, "level_raw": None}   # no level field at all
    assert loghub.escalation_label(rows) == "steady"


def test_component_redaction_is_whole_token():
    line = ("- 1123914312 2005.08.12 R44-M0-N8-I:J18-U01 2005-08-12-23.25.12.135766 "
            "R44-M0-N8-I:J18-U01 RAS APP FATAL ciod: Error loading /bgl/apps/x.rts: app died")
    got = loghub.read_line(line, "BGL")
    state = loghub.component_state(dict(got, text=line, system="BGL"))
    assert "/bgl/apps/" in state                 # evidence inside a word stays
    assert state.count("[COMPONENT]") == 2       # the field and the whole-word echo


def test_the_component_marker_count_does_not_name_the_answer(cached):
    """0.798 against a 0.473 marginal on `loghub_component_supercomputer` before the
    whole-token redaction."""

    for task in cached.tasks():
        if not task.name.startswith("loghub_component"):
            continue
        exs = list(task.load(4000))
        gold = [e.questions[0].options[e.questions[0].target] for e in exs]
        keys = [tuple(e.questions[0].options) for e in exs]
        marks = [(k, e.state.count("[COMPONENT]")) for k, e in zip(keys, exs)]
        marginal = _prior_accuracy(list(zip(keys, gold)))
        marker = _prior_accuracy(list(zip(marks, gold)))
        assert marker <= marginal + 0.05, (task.name, marker, marginal)


def test_no_bgl_state_carries_the_alert_flag(cached):
    for task in cached.tasks():
        for e in task.load(4000):
            if e.questions[0].meta["system"] not in cached.ALERT_FLAG_SYSTEMS:
                continue
            for line in e.state.split("\n"):
                assert line.split()[0].isdigit(), (task.name, line[:80])


def test_questions_are_registered_and_protected_from_enrichment(cached):
    import json

    from lod.corpus.services.enrich import protected
    from lod.paths import ASSETS

    specs = json.loads((ASSETS / "phrasing_specs" / "d13.json").read_text())
    assert specs["d13.level"]["template"] == loghub.LEVEL_QUESTION
    assert specs["d13.component"]["template"] == loghub.COMPONENT_QUESTION
    assert specs["d13.burst"]["template"] == loghub.BURST_QUESTION
    for task in cached.tasks():
        assert protected(task.name), task.name


def test_a_burst_whose_answer_hinges_on_verbose_below_debug_is_dropped():
    """Android ranks V below D; the shared scale folds both into DEBUG. `V D D D D D`
    rises on Android's scale and not on the shared one, so it has no single answer."""
    rows = [{"text": f"l{i}", "level": "DEBUG", "level_raw": raw}
            for i, raw in enumerate(["V", "D", "D", "V", "D", "D"])]
    assert loghub.escalation_label(rows) is None
    rows[0]["level_raw"] = "D"
    assert loghub.escalation_label(rows) == "steady"


def test_no_state_is_asked_twice_within_a_task(cached):
    for task in cached.tasks():
        states = [e.state for e in task.load(4000)]
        assert len(states) == len(set(states)), task.name
