"""ClinicalTrials protocol adapter (row 11) — deliberately contributes nothing.

Row 11 is one of the three whole families reserved to `testreal`, so it is tempting to
make it produce *something*. It should not, and this module records why rather than
leaving the gap to be rediscovered.

The staged corpus (`Parexel/clinical-trials-protocols`, 4,000 protocols) carries exactly
one label-shaped thing beside the text: four flags, `tier_a_member` .. `tier_d_member`.
Measured over all 4,000 rows they are strictly nested and sized by powers of ten:

    tier_d only                        3,000
    tier_c + tier_d                      900
    tier_b + tier_c + tier_d              90
    tier_a + ... + tier_d                 10

That is a 10 / 100 / 1,000 / 4,000 benchmark sampling hierarchy, not a property of the
protocol. Nothing in the protocol text predicts which nested subset a trial was sampled
into, so a model can only learn the base rate -- and the base rate is what an
over-confident classifier already defaults to, which is the failure this corpus exists to
avoid.

An earlier build shipped it anyway, as four boolean tasks asking "What is the tier c member
of this text?", because this adapter read a parquet path that does not exist here,
returned `[]`, and the generic row-1 sweep claimed the dataset instead. The sweep no
longer may (`hub_sweep.DEDICATED_ADAPTERS`).

The corpus rule is that a source whose criteria its publisher does not publish does not
enter the corpus. The publisher documents no meaning for these tiers, so it does not enter.
The `reserved_families` gate is measured on the other two reserved families, rows 13
(Lichess) and 25 (ChaosNLI).
"""

from __future__ import annotations

from lod.corpus.services.sources.real.base import RealTask

PATH = "Parexel/clinical-trials-protocols"
URL = f"https://huggingface.co/datasets/{PATH}"
# nested subset sizes, measured over all 4,000 staged rows
TIER_SIZES = {"tier_a_member": 10, "tier_b_member": 100,
              "tier_c_member": 1000, "tier_d_member": 4000}


def tasks() -> list[RealTask]:
    """No tasks. See the module docstring: the only labels are a sampling hierarchy."""
    return []
