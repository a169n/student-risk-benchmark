# Getting the datasets

None of the five student datasets is stored in this repository. All five are
published openly by the institutions that produced them, so the adapters read a
copy you download yourself. This file says where each one comes from and where to
put it.

Nothing here is a fast path: two of the archives are hundreds of megabytes and
Oviedo needs an extraction step. Budget an afternoon the first time.

## The five sources

### Open University (OULAD) — 19 cohorts

- Download: https://analyse.kmi.open.ac.uk/open_dataset
- Cite: Kuzilek, Hlosta & Zdrahal (2017), *Sci. Data* 4:170171
- Put the seven CSVs (`courses.csv`, `studentInfo.csv`, `studentVle.csv`, …)
  directly in `datasets/oulad/`, which is the `raw_dir` the experiment configs
  under `services/ml/configs/experiments/` expect.
- Licence: CC BY 4.0 according to the source page. Nothing in this repository
  records it independently, so confirm it there before you rely on it.

### KU Leuven — 6 cohorts

- Download: Zenodo `10.5281/zenodo.17087849` (78 409 206 bytes zipped)
- Licence: CC BY 4.0
- Unzip so the spreadsheets land in `datasets/ku_leuven/dataset/`. The adapter
  wants twelve files, named `<year>_course_content.xlsx`,
  `<year>_course_participation.xlsx`, `<year>_df_contribution.xlsx` and
  `<year>_df_consumption.xlsx` for years 1819, 1920 and 2021.
- Verify you have the same bytes the paper's numbers came from:

      shasum -a 256 dataset_full.zip
      # 6e58631a2e988a735fe79b28b8d6257dd3a30a705721deb49d5675717bd99d5c

### University of KwaZulu-Natal — 16 cohorts

- Download: Zenodo `10.5281/zenodo.14810607` (about 316 MB)
- Licence: CC BY 4.0
- Extract under `datasets/ukzn/raw/`.

### Universidad de Oviedo — 20 cohorts

- Download: the PostgreSQL dump published beside the code of Riestra-González,
  Paule-Ruiz & Ortin (2021), *Comput. Educ.* 163:104108. **There is no DOI** —
  see the warning below.
- Licence: Apache-2.0
- 781 MB zipped, about 7 GB expanded. Do not restore it into a database: put the
  archive in `datasets/_candidates/oviedo/` and run

      python -m src.benchmarks.oviedo_extract

  which streams the dump once and writes only the four tables that matter into
  `datasets/_candidates/oviedo/tables/`.
- Use the raw `mdl_log` clicks, not the feature CSVs published in that
  repository: those mix grade-derived predictors into a grade-derived label,
  which is the circularity this project avoids.

### University of Zambia — 2 cohorts

- Download: Zenodo `10.5281/zenodo.21292883`
- Licence: CC BY 4.0
- Put the pipe-delimited files under `datasets/zambia/`.
- The published log and examination files carry a `StudentName` column of
  realistic personal names beside the hashed `StudentID`. The adapter never reads
  it. Neither should anything you write: join on `StudentID`.

## Oviedo is the fragile one

Four of the five have a DOI, so they have a landing page, a version history and a
persistence commitment from Zenodo. Oviedo has none: the dump lives beside the
authors' paper code on GitHub and can be deleted or rewritten without a trace.
Twenty of the benchmark's 63 cohorts depend on it.

Its Apache-2.0 licence does permit redistribution with attribution, so archiving
a copy under our own DOI is an option if the original ever disappears. Until
then, record the checksum of what you downloaded, the way the KU Leuven entry
above does.
