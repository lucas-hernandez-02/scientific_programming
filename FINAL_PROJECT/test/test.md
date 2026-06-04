# test/

This folder contains the minimal testing environment used to verify that the project's modules run correctly as an integrated pipeline. It is designed to validate imports, directory structure, preprocessing routines, visualizations and the complete analysis workflow.

The folder contains the following elements:

---

## 1. `__init__.py`

Marks the directory as a Python package, enabling test modules to import components from the `src/` folder using package-relative paths.

This file does not contain logic itself, but it is essential for ensuring:

- correct module discovery
- proper Python package initialization
- compatibility with automated testing tools

---

## 2. `test.py`

This script serves as the **main testing file** of the project. It loads raw data, runs preprocessing, generates all visualizations, builds the feature table, evaluates clustering, runs ANOVA/Fisher tests, evaluates classifiers with cross-validation, analyzes instrument separability, and finishes with the comparison between optimized and non-optimized algorithms.

### The script performs:

1. Creation of output directories (`data/processed/`, `results/figures/`, `results/tables/`)
2. Loading the raw MIR dataset
3. Spectral preprocessing (SNV)
4. Visualization of raw and processed spectra
5. Building the feature table (Global + Clay + Organic)
6. Computing per-domain distributions
7. Computing domain indexes and correlation matrix
8. K-Means evaluation for k = 2..5 with three quality metrics
9. ANOVA F-test and Fisher Ratio per feature
10. Classification metrics with cross-validation
11. Instrument separability analysis
12. **Optimized vs non-optimized comparison** (Lab 3 grading criterion - 20%)
13. Saving all results into `results/figures/` and `results/tables/`

This file essentially **executes the whole project pipeline**, acting as the validation point that ensures everything inside `src/` works together correctly.

---

## 3. Execution

From the project root:

```bash
python -m test.test
```

The script prints progress to the console and finishes with the phrase `PIPELINE FINISHED`, indicating that all stages completed successfully.

---

## Summary

The `test/` directory provides a lightweight but complete environment for validating the project's functionality. It ensures that preprocessing, visualization, analytical modules and result-saving mechanisms operate properly as a unified workflow.
