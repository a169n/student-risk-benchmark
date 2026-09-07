# Student-Risk Benchmark

63 student cohorts from five universities, five learning-management platforms and
four countries, harmonised to one seven-feature schema — 35,529 distinct students.
This repository is the artifact behind the paper in [`paper/`](paper/): the
adapters that build the benchmark, the experiment code, the frozen results and a
script that recomputes every number the paper states.

The paper's own finding is why the benchmark exists: within-cohort ROC-AUC spans
0.384 to 0.953 across these cohorts, so a result measured on one dataset is a
statement about that dataset.

## What is here

    services/ml/src/benchmarks/     one adapter per institution, plus the shared
                                    Moodle log builder
    services/ml/src/experiments/    cohort definitions, the transfer ladder and
                                    every experiment in the paper
    services/ml/scripts/            figure and table generation, and
                                    verify_paper_claims.py
    services/ml/configs/            experiment configurations
    data/artifacts/experiments/     frozen result files
    datasets/PROVENANCE.md          where to download each dataset
    paper/                          main.tex, the built PDF and the figures

The `services/ml/` nesting is inherited from the project this was extracted
from. It is load-bearing: config paths and `verify_paper_claims.py` resolve the
repository root by counting directories up from themselves.

## Reproducing the paper

**1. Get the data.** No student data is stored here. All five releases are
published openly by the institutions that produced them, four under a DOI.
[`datasets/PROVENANCE.md`](datasets/PROVENANCE.md) gives the source, the licence
and the expected local path for each. Budget an afternoon: two archives are
hundreds of megabytes and Oviedo needs an extraction step.

**2. Run the experiments.** The runners live in `services/ml/src/experiments/`.
Each writes into `data/artifacts/experiments/<experiment>/`.

**3. Check the numbers.**

    cd services/ml && uv run python scripts/verify_paper_claims.py

It recomputes each claim from the frozen result files and reports whether the
manuscript states that value. `ABSENT` means the value is not written down, which
is often deliberate; a value stated differently in the text is the error this
catches.

### Status of the frozen results

Regeneration is in progress. Present: `exp_021_trivial_baseline` in full;
`exp_014`, `exp_023` and `exp_024` carry run metadata only. Absent:
`exp_015` through `exp_019` and `exp_022`. Until those are restored,
`verify_paper_claims.py` cannot complete.

## Building the paper

    cd paper && latexmk -pdf main.tex

Uses the standard IEEE conference class, which ships with TeX Live, MiKTeX and
Overleaf, so no template file needs downloading. `tectonic -X compile main.tex`
also works.

The figure file names and the figure numbers disagree: `fig2_cohort_spread.pdf`
is Figure 1 because it was added later and lands earlier in the text. LaTeX
numbers by position, so the labels `fig:spread` and `fig:instability` are the
reliable handles.

## Licence

Code is MIT, see [LICENSE](LICENSE). The datasets are not redistributed here and
each carries its own licence, recorded in `datasets/PROVENANCE.md`.
