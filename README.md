# Lod

Lod models answer typed decisions about a piece of text. Given a **state** (a document,
a record, a log, a conversation) and any number of **questions**, each with its own
options, one forward pass returns a calibrated probability for every option of every
question. There is no generation and no sampling: the backbone reads the state once,
every option is scored against it independently, and a small comparison layer lets the
options of a question see each other.

| model | backbone | adapter | context | test accuracy | test NLL | test ECE |
|---|---|---|---|---|---|---|
| lod-lille-0.6B | Qwen3-0.6B-Base | LoRA r32 | 32k | 0.709 | 0.781 | 0.026 |
| lod-stor-4B | Qwen3-4B-Base | LoRA r64 | 32k | 0.789 | 0.665 | 0.036 |

Test = 139,543 questions over 447 held-out tasks. Both models ship at temperature 1.

## Setup

```bash
uv sync                      # model, training and evaluation
uv sync --extra serve        # + the HTTP server
uv sync --extra data         # + what building the corpus needs
```

Locations come from environment variables (`src/lod/paths.py`):

| variable | default | holds |
|---|---|---|
| `LOD_ROOT` | `.` | `data/` (corpora) and `runs/` (checkpoints) |
| `LOD_RAW_ROOT` | `~/.cache/lod-sources` | raw rows downloaded from the sources |
| `LOD_DI_SUITE` | `decision-index/suite-0.2` | a rebuilt Decision Index suite, for decontamination |

## Use a model

```bash
uv run python scripts/predict.py --ckpt runs/lod-lille-release \
  --state "Order #4411 arrived with a cracked screen; the customer wants a replacement." \
  --q "Which team handles this? | billing | shipping | technical support"

uv run python scripts/serve.py --ckpt runs/lod-lille-release --port 8000   # POST /v1/score
```

## Build the corpus

The corpus is built in two stages: fetch the raw sources once, then build from them.
Building is local and deterministic.

```bash
uv run python scripts/fetch_data.py                  # every source, into LOD_RAW_ROOT
uv run python scripts/build_corpus.py blocklist      # hash the Decision Index suite
uv run python scripts/build_corpus.py all --out data/lod-corpus
```

`all` runs generate → enrich → decontaminate → select → probes → audit. Each stage is
also its own subcommand (`build_corpus.py --help`).

- **Enrichment** rewrites question wording and option criteria through an LLM. It starts
  from the answers the release corpus used, so a rebuild needs an `OPENROUTER_API_KEY`
  only for new schemas.
- **The blocklist** removes any state that carries an item of the Decision Index
  benchmark. It is built from a rebuilt copy of that suite
  ([decision-index](https://github.com/apolinario/decision-index)).

## Train

```bash
bash scripts/recipes/lod_stor.sh                     # lod-stor-4B on data/lod-corpus
uv run python scripts/build_corpus.py distill \
  --teacher runs/lod-stor-release --src data/lod-corpus --dst data/lod-corpus-distilled
bash scripts/recipes/lod_lille.sh                    # lod-lille-0.6B, distilled from lod-stor
```

Each recipe trains, attaches the confidence head, and writes `runs/<model>-release/`. That
directory holds a `report/` (evaluations, release gates, probes) and a `MODEL_CARD.md`.

- **Settings:** the recipes use the exact settings of the released models.
- **Hardware:** lod-stor needs a 96 GB GPU at its batch cap; lod-lille needs about 24 GB.
- **Reproducibility:** the released lod-lille was distilled from an earlier lod-stor
  checkpoint, so the recipe reproduces the method rather than the exact weights.

Evaluate any checkpoint on any split:

```bash
uv run python scripts/evaluate.py --ckpt runs/lod-lille-release \
  --data data/lod-corpus/testreal.jsonl --out runs/lod-lille-release/testreal.jsonl
```

## Layout

```
src/lod/
  schema.py …      questions, examples and their serialisation
  model/           the scorer, packing, attention masks, sharded scoring, calibration
  training/        datasets, loaders and criteria augmentation
  evaluation/      metrics, bootstrap, release gates, probes and the regression suite
  serving/         the API wire format and the budget policy
  corpus/
    controllers/   one function per pipeline stage (fetch, generate, enrich, …)
    services/      the transformations: 51 source generators, enrichment, splits, audits
    repositories/  I/O: Hub and web downloads, the raw-row store, corpus directories
  assets/          frozen phrasing banks, enrichment answers and side tables
scripts/           command-line entry points, and recipes/ for the two release runs
tests/
```

## License

The code is Apache-2.0. The corpus is built from third-party sources, each under its own
licence; `src/lod/corpus/DATA_LICENSES.md` lists every source with its licence.
