# Data and analysis code for *Degree-conditioned anneal offsets reshape certified-optimum sampling for minimum vertex cover*

This repository contains the analysis code, manuscript figures, and supporting metadata for the accompanying preprint.

## Download the data

The complete dataset is available as `dcosqa_mvc_data_v1.0.0.zip` in the [v1.0.0 release](../../releases/tag/v1.0.0). The archive includes the recorded QPU and simulated-annealing candidate tables, analysis-ready and derived tables, metadata, scripts, and manuscript figures.

Large raw tables are kept in the release archive rather than in the Git history. The files in the repository are sufficient to inspect the analysis and its outputs; download the archive to run the complete workflow.

## Contents

```text
code/                    Analysis and figure-generation scripts
data/metadata/           Graph, protocol, and embedding metadata
figures/                 Manuscript figures
DATA_DICTIONARY.md       Column and field definitions
DATA_PROVENANCE.md       Dataset origins and figure crosswalk
FULL_PACKAGE_CONTENTS.md Full archive layout and reproduction notes
requirements.txt         Python dependencies
requirements-lock.txt    Reference validation environment
```

## Reproduce the figures

For complete reproduction, download and extract `dcosqa_mvc_data_v1.0.0.zip`, then run from the extracted package directory:

```bash
python3 -m pip install -r requirements.txt
python3 code/reproduce_all_figures.py
```

Some read-budget and closed-system calculations take longer and can be resumed. See `FULL_PACKAGE_CONTENTS.md` for the commands and `DATA_PROVENANCE.md` for the input used for each figure.

## Citation

Please cite the preprint and the v1.0.0 data release. Citation metadata are provided in `CITATION.cff`.

## License

Data, metadata, documentation, and figures are released under CC BY 4.0. Python analysis code is released under the MIT License. See `LICENSE` for details.
