# src/

This folder contains the full source code of the project. It is organized into modular components that handle preprocessing, analysis and visualization of the MIR soil spectral dataset. Together, these modules form the project's processing pipeline —from raw data to statistical conclusions.

The folder is organized into three main subpackages:

---

## 1. `preprocessing/`

This directory contains the tools responsible for cleaning and transforming raw data before any analytical procedure is applied. These steps ensure the reliability, comparability and consistency of the spectral measurements.

### The folder contains:

- **spectral.py**
  Module dedicated to preparing MIR spectra.
  It implements the `SpectralProcessor` class, responsible for:
  - Separating identifier columns from spectral columns
  - Applying SNV normalization (*Standard Normal Variate*) per spectrum
  - Optional moving-average smoothing
  - Conversion to NumPy for downstream analysis

  The processed spectral matrix is ready for feature extraction and multivariate analysis.

---

## 2. `analysis/`

This directory contains all mathematical and statistical algorithms used to extract patterns and evaluate relationships across spectral domains. It forms the core analytical engine of the project.

### The folder contains:

- **analysis.py**
  Implements the `AnalysisProcessor` class, which provides a complete analytical pipeline:
  - Global feature extraction (energy, mean power, entropy, slope)
  - Band-specific descriptors (clay 1000–1200 cm⁻¹, organic matter 1350–1450 cm⁻¹)
  - Z-score standardization and per-domain distribution analysis
  - Domain summary indexes (without linear regression between them, addressing the Lab 2 feedback)
  - K-Means evaluation with three metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
  - ANOVA F-test and Fisher Ratio per feature
  - Classifier evaluation (Baseline, Logistic Regression, kNN) with cross-validation
  - Instrument separability analysis

  This module reproduces and formalizes the analytical logic developed in the notebooks of previous labs.

---

## 3. `visualization/`

This directory includes all plotting utilities used to generate the project's figures. Each visualizer produces outputs saved into `results/figures/`.

### The folder contains:

- **spectralviz.py**
  Generates visualizations of raw and processed spectral data:
  - Overlaid raw spectra
  - SNV-normalized spectra
  - Spectra grouped by instrument (subplots)
  - Mean spectrum per instrument

- **analysisviz.py**
  Generates all statistical analysis visualizations:
  - Violin + boxplot of per-domain distributions
  - Correlation heatmap between domain indexes
  - Fisher Ratio bars per feature
  - Clustering metrics vs k
  - Cluster separability scatter (top-2 features)
  - ROC curve
  - Confusion matrix for instrument classification
  - Optimized vs non-optimized timing comparison

---

## Summary

The `src/` folder contains the complete implementation of the project's computational pipeline, covering:

- Data cleaning
- Normalization
- Feature extraction
- Clustering
- Statistical testing
- Classification
- Visualization

Each subfolder contributes a specific layer of functionality, and together they produce a coherent, fully automated analysis framework via `test/test.py`.
