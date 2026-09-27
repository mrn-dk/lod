"""The corpus's 25 domains: which source-table rows each one is built from.

A domain is the unit the corpus is reported and balanced by; a row is one source (a
dataset, a generator). Selection quotas, the domain table and per-domain eval reads all
key off this one mapping.
"""

from __future__ import annotations

DOMAINS: dict[int, tuple[set[int], str]] = {
    1:  ({1, 3, 6, 8, 10, 16, 18, 45}, "Text classification and taxonomy"),  # 45: topic
    2:  ({33, 47}, "Rule application"),     # 47: compositional rule grammar
    3:  ({19, 20, 21, 22, 23, 44}, "Affect and safety"),   # 44: sentiment/stance
    4:  ({2, 13, 43}, "Reasoning and puzzles"),
    5:  ({26, 27, 28}, "Forecasting and probability recovery"),
    6:  ({34, 35}, "Grounding and abstention"),
    7:  ({9}, "Security advisory triage"),
    8:  ({5}, "Software project signals"),
    9:  ({24, 25}, "Entailment and NLI"),
    10: ({15, 17}, "Legal and licence text"),
    11: ({31}, "Tabular entailment"),
    12: ({7}, "Document structure"),     # row 30 removed: 0 tasks
    13: ({14}, "Logs and events"),
    14: ({36}, "Browser and interface"),
    15: ({37}, "Control and sensor"),
    16: ({38}, "Tool-call and action gating"),
    17: ({39}, "Entity resolution"),
    18: ({40}, "Code and diffs"),
    19: ({41}, "Scheduling and allocation"),
    20: ({42}, "Multi-hop and ranking"),
    21: ({46}, "Multilingual classification"),
    22: ({48}, "Exact posteriors"),
    23: ({49}, "Many-option matching and linking"),
    24: ({50}, "Broad real-language decisions"),
    25: ({51}, "Long-context reading"),
}
SPLITS = ("train", "val", "devreal", "testreal")

