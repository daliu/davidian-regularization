# Stability Bonus Davidian Regularization: Publication Package

## Overview

This directory contains the complete publication-ready package for the research paper:

**"Stability Bonus Davidian Regularization: A Reward-Based Alternative to Minority Class Rebalancing"**

## Directory Structure

```
publication/
├── README.md                    # This file
├── paper.tex                   # Main LaTeX paper
├── references.bib              # Bibliography
├── src/                        # Core implementation
│   ├── __init__.py
│   ├── davidian_regularization.py
│   ├── data_loaders.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
├── experiments/                # Experimental scripts
│   ├── synthetic_experiments.py
│   ├── real_dataset_experiments.py
│   ├── mechanism_analysis.py
│   └── statistical_validation.py
├── scripts/                   # Utility and pipeline scripts
│   ├── run_full_pipeline.py
│   ├── generate_figures.py
│   └── validate_reproducibility.py
├── data/                      # Experimental data
│   ├── synthetic_results/
│   ├── real_dataset_results/
│   └── processed/
├── figures/                   # Publication figures
│   ├── method_comparison.pdf
│   ├── stability_bonus_analysis.pdf
│   ├── real_dataset_validation.pdf
│   └── mechanism_explanation.pdf
└── docs/                      # Supporting documentation
    ├── methodology.md
    ├── results_summary.md
    └── implementation_guide.md
```

## Quick Start

### Generate All Results
```bash
cd publication
python scripts/run_full_pipeline.py
```

### Compile Paper
```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Key Findings

- **Stability Bonus**: +15-20% improvement over baseline
- **Original Davidian**: -1% to -4% degradation
- **Statistical Significance**: 100% for Stability Bonus
- **Real Dataset Validation**: Consistent across all tested datasets

## Citation

```bibtex
@article{davidian2024stability,
  title={Stability Bonus Davidian Regularization: A Reward-Based Alternative to Minority Class Rebalancing},
  author={[Author Names]},
  journal={arXiv preprint},
  year={2024}
}
```
