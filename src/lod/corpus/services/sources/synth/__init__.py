"""Code-labelled generators: the label is computed, not collected.

These are the only sources in the corpus whose labels are not a record of something a
person or a system decided, and they are marked `real=False` so the build reports their
share separately. They exist because the shipped model is a label-recovery model and not
a reasoner: it reads `gratitude` at 0.925 and CVSS `NETWORK` at 0.998, and gets the semver
of a breaking rename wrong. Nothing in the real corpus asks it to apply a written rule to
a written fact, so nothing in the real corpus teaches it to.

`MODULES()` names every generator, including ones whose `tasks()` is still empty. A
generator that contributes nothing yet is visible in the registry rather than absent from
it, which is what stops a half-built domain from being silently skipped -- the failure
mode row 36 hit, where `browser.py` existed for two days and reached no corpus because it
was never named here.
"""


# `synth/<module>.py` -> the domain it builds (`lod/corpus/domains.py`).
DOMAINS = {
    "rules": 2,       # row 33, rule application
    "entities": 6,    # row 35, grounding and abstention (the synthetic half)
    "browser": 14,    # row 36, browser and interface
    "sensors": 15,    # row 37, control and sensor
    "toolgate": 16,   # row 38, tool-call and action gating
    "entityres": 17,  # row 39, entity resolution
    "diffs": 18,      # row 40, code and diffs
    "scheduling": 19, # row 41, scheduling and allocation
    "relational": 20, # row 42, multi-hop and ranking
    "chessfacts": 4,  # row 43, chess positions as facts -- joins domain 4
    "compose": 2,     # row 47, compositional rule grammar, held out by composition and depth
    "posterior": 22,  # row 48, exact posteriors (urns, dice, noisy sensors, sources)
    "catalog": 23,    # row 49, many-option matching and linking, heavy-tailed N
    "longctx": 25,    # row 51, long-context reading (generated long states + controlled depth)
}


def MODULES():
    """Imported lazily, like `sources.real.REGISTRY`: importing a package must not
    build a corpus, and `--help` must not pay for it."""
    from lod.corpus.services.sources.synth import (
        browser,
        catalog,
        chessfacts,
        compose,
        diffs,
        entities,
        entityres,
        longctx,
        posterior,
        relational,
        rules,
        scheduling,
        sensors,
        toolgate,
    )
    return (rules, entities, browser, sensors, toolgate, entityres, diffs,
            scheduling, relational, chessfacts, compose, posterior, catalog, longctx)


__all__ = ["MODULES", "DOMAINS"]
