"""A phrasing may vary the wording of a question. It may never vary the question.

`PhrasingBank` draws a stored rewording of a template per record, deterministically, and
falls back to the template itself whenever a stored phrasing could not stand in for it.
"""
from __future__ import annotations

from lod.phrasings import PhrasingBank, placeholders

GEFS = "Will this location see at least {mm} mm of precipitation {lead} days after the forecast was issued?"


def test_bank_skips_phrasings_whose_slots_do_not_match():
    b = PhrasingBank({"g": ["Will {a} happen in {b}?", "Only {a} here"]})
    t = "Will {a} happen in {b}?"
    assert all(b.pick("g", t, f"k{i}") == t for i in range(20))


def test_bank_is_deterministic_and_spreads():
    b = PhrasingBank({"g": [f"Ask {{a}} way {i}?" for i in range(5)]})
    t = "Ask {a} way 0?"
    picks = [b.pick("g", t, f"rec{i}") for i in range(200)]
    assert len(set(picks)) >= 4                 # spreads over the bank
    assert picks == [b.pick("g", t, f"rec{i}") for i in range(200)]   # reproducible


def test_placeholders_helper():
    assert placeholders(GEFS) == {"mm", "lead"}


def test_a_reworded_template_never_draws_its_old_phrasings():
    bank = PhrasingBank({"t": ["Does it break?", "Will it break callers?"]},
                        {"t": "Does it break?"})
    assert bank.pick("t", "Is every caller preserved?", "k") == "Is every caller preserved?"
    assert bank.pick("t", "Does it break?", "k") in {"Does it break?", "Will it break callers?"}
