# Student-Risk Benchmark

63 student cohorts from five universities, three learning-management platforms and
five countries, harmonised to one seven-feature schema — 35,529 distinct students.
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

**2. Build the caches.** The runners read weekly frames from parquet caches
that are not in git. Rebuild them from the downloads (about 15 minutes):

    cd services/ml && uv run python -m src.benchmarks.build_caches

**3. Run the experiments.** The runners live in `services/ml/src/experiments/`.
Each writes into `data/artifacts/experiments/<experiment>/`. Then compute the
intervals on the explanation-agreement numbers, which the figures and
`verify_paper_claims.py` read:

    uv run python scripts/estimator_intervals.py

**4. Check the numbers.**

    cd services/ml && uv run python scripts/verify_paper_claims.py

It recomputes each claim from the frozen result files and reports whether the
manuscript states that value. `ABSENT` means the value is not written down, which
is often deliberate; a value stated differently in the text is the error this
catches.

The frozen results for every experiment the paper cites, `exp_014` through
`exp_024`, are committed, so step 4 runs without steps 1–3. Against the
camera-ready `paper/main.tex` it reports every recomputed value the paper states
as `PASS`.

## Building the paper

    cd paper && latexmk -pdf main.tex

Uses the standard IEEE conference class, which ships with TeX Live, MiKTeX and
Overleaf, so no template file needs downloading. `tectonic -X compile main.tex`
also works.

The figure file names and the figure numbers disagree because figures were
added later and land earlier in the text: `fig0_pipeline.pdf` is Fig. 1,
`fig2_cohort_spread.pdf` is Fig. 2 and `fig1_explanation_instability.pdf` is
Fig. 3. LaTeX numbers by position, so the labels `fig:pipeline`, `fig:spread`
and `fig:instability` are the reliable handles. The scripts that draw them,
`plot_pipeline.py`, `plot_cohort_spread.py` and `plot_explanation_instability.py`,
write into `paper/figures/`.

## Licence

Code is MIT, see [LICENSE](LICENSE). The datasets are not redistributed here and
each carries its own licence, recorded in `datasets/PROVENANCE.md`.
