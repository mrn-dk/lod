"""Schema sources. Each module exposes INFO and generate(n, seed) -> SourceResult."""

from lod.corpus.services.sources import api_schema, issue_labels, synth_schema, tabular_cloze
from lod.corpus.services.sources.base import (
    SourceInfo,
    SourceResult,
    coverage_report,
    schema_key,
    write_manifest,
)

# in priority order.
# synth_schema is the answer to "not enough schemas": it invents them, so
# it also registers a prose and a tabular variant to cover those diversity axes.
REGISTRY = {
    "synth_schema": synth_schema,
    "synth_schema_prose": synth_schema.synth_schema_prose,
    "synth_schema_ood": synth_schema.synth_schema_ood,
    "synth_schema_fmtshift": synth_schema.synth_schema_fmtshift,
    "tabular_cloze": tabular_cloze,
    "api_schema": api_schema,
    "issue_labels": issue_labels,
}
# not yet implemented: wikidata_cloze, taxonomy

__all__ = ["REGISTRY", "SourceInfo", "SourceResult", "coverage_report", "schema_key",
           "write_manifest"]
