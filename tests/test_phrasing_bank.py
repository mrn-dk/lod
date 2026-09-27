"""The frozen phrasing bank that ships in `lod/assets` matches the templates it rewords."""
import json

from lod.paths import ASSETS

BANK = ASSETS / "phrasings.json"


def test_every_spec_template_is_in_the_bank_verbatim():
    specs = {}
    for f in sorted((ASSETS / "phrasing_specs").glob("*.json")):
        specs.update(json.loads(f.read_text()))
    bank = json.loads(BANK.read_text())
    stale = [t for t, s in specs.items() if t in bank and bank[t]["template"] != s["template"]]
    assert not stale, stale


def test_every_stored_phrasing_keeps_its_template_slots():
    """A phrasing with other placeholders than its template is never drawn; the bank
    should not carry dead entries."""
    from lod.phrasings import placeholders

    bank = json.loads(BANK.read_text())
    bad = [(tid, p) for tid, v in bank.items() for p in v["phrasings"]
           if placeholders(p) != placeholders(v["template"])]
    assert not bad, bad[:5]
