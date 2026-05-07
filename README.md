# Double Descent Analysis

> [!NOTE]  
> This repository contains the final project for the _Statistical Methods For Machine Learning_ course at Università degli Studi di Milano (2025/2026).

## Overview

This project provides an empirical analysis of the double descent phenomenon in machine learning, exploring how model performance changes across different model complexities and dataset sizes. The double descent is a phenomenon where test error can decrease again after initially increasing past the _interpolation threshold_, challenging traditional bias-variance tradeoff theory.

## Project Content

The repository contains an implementation and analysis of the double descent phenomenon using different regression models and approaches. It implements Ordinary Least Squares (OLS) regression, Ridge regression, and minimum norm interpolators to empirically study how training and test errors evolve as the model dimension increases. The project is organized to separate data handling, model implementations, experiments, metrics computation, and visualization code, making it easy to modify and extend the analysis.

## Repo Structure

```
.
├── src/                          # Main source code modules
│   ├── data.py                   # Synthetic data generation
│   ├── models.py                 # Model implementations (OLS, Ridge, Min Norm)
│   ├── experiments.py            # Experiment runners
│   ├── metrics.py                # Error metrics computation
│   └── plotting.py               # Visualization utilities
├── notebooks/
│   └── double_descent_analysis.ipynb  # Main analysis (all experiments)
├── assets/                       # Generated plots and figures
├── report/                       # Final project report (Typst & LaTeX formats)
├── requirements.txt              # Python dependencies
├── LICENSE                       # Project license
└── README.md                     # This file
```

## Setup and Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the complete analysis in the Jupyter notebook:

```bash
jupyter lab notebooks/double_descent_analysis.ipynb
```

This notebook generates all experiments and plots shown in the report.

## Report Formats

To accommodate different formatting preferences and strict submission requirements, the final project report is provided in both `Typst` and `LaTeX` formats. Both versions are functionally identical in content and can be found inside the `report/` directory.

