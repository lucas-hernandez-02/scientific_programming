# Soil Health Assessment Using MIR Spectroscopy
### Scientific Programming – Lab 3
**Lucas Felipe Hernández Palacio**
**Master in Automation and Industrial Control**

---

## 1. Context

Soil health is defined as the capacity of a soil to function within ecosystem boundaries, sustaining biological productivity, maintaining environmental quality, and promoting plant and animal health (Doran & Zeiss, 2000; Lehmann et al., 2020). Its assessment has shifted from a productivity-centered approach toward a comprehensive view that incorporates soil biota and the biotic processes affecting soil properties (Zeng et al., 2024).

Mid-infrared (MIR) spectroscopy (2500–25000 nm) has become a non-destructive, high-throughput tool for monitoring soil health indicators. It identifies characteristic organic and inorganic functional groups —clay minerals, iron oxides, organic matter— through electromagnetic absorption (Viscarra Rossel & McBratney, 2010; Soriano-Disla et al., 2014). In particular, it outperforms NIR for predicting soil organic carbon (Reeves, 2010).

However, transferability between FTIR instruments remains a challenge: the same sample measured on five different devices produces spectra with baseline shifts and scale differences that limit the portability of calibrated models. Calibration transfer techniques aim to mitigate this effect (Safanelli et al., 2025).

---

## 2. Dataset Description

The dataset comes from the **Soil Spectroscopy for the Global Good (SS4GG)** Kaggle competition. The validation set contains:

- **250 MIR spectra** corresponding to 50 soil samples measured on 5 different FTIR instruments (INST1–INST5)
- **1,676 wavenumbers** in the range 650–4000 cm⁻¹ with 2 cm⁻¹ step
- **3 identifier columns**: `unique_id`, `instrument`, `sample_id`

### Spectral bands of interest

| Band | Range (cm⁻¹) | Assignment |
|---|---|---|
| Clay | 1000–1200 | Si–O vibrations of clay minerals |
| Organic matter | 1350–1450 | C–H stretching of organic matter |

Dataset link: https://www.kaggle.com/competitions/ss4gg-soil-spectroscopy-validation

---

## 3. Repository Structure

```
FINAL_PROJECT/
├── README.md
├── requirements.txt
│
├── data/
│   ├── data.md
│   ├── raw/                 ← original dataset (mir_soil.csv)
│   └── processed/           ← SNV-normalized dataset
│
├── docs/
│   ├── docs.md
│   ├── Lab 1.docx
│   ├── Lab 2.docx
│   └── Lab 3.docx
│
├── notebooks/
│   ├── notebooks.md
│   ├── exploration/         ← Lab 1 (initial exploration)
│   │   └── FIRST_LAB.ipynb
│   └── reporting/           ← Lab 2 (analysis and reporting)
│       └── SECOND_LAB.ipynb
│
├── results/
│   ├── results.md
│   ├── figures/             ← 12 auto-generated PNGs
│   └── tables/              ← 10 auto-generated CSVs
│
├── src/
│   ├── src.md
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── spectral.py      ← SpectralProcessor class
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── analysis.py      ← AnalysisProcessor class
│   └── visualization/
│       ├── __init__.py
│       ├── spectralviz.py   ← SpectralVisualizer class
│       └── analysisviz.py   ← AnalysisVisualizer class
│
└── test/
    ├── __init__.py
    ├── test.md
    └── test.py           ← runs THE ENTIRE pipeline
```

---

## 4. Installation and Setup

### 4.1. Clone the repository

```bash
git clone https://github.com/lucas-hernandez-02/scientific_programming
cd "FINAL_PROJECT"
```

### 4.2. Create and activate a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 4.3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Running the Entire Project

Once the environment is set up and `mir_soil.csv` is placed inside `data/raw/`, run from the project root:

```bash
python -m test.test
```

This script executes the full pipeline:

- Loads raw data from `data/raw/`
- Applies SNV and saves normalized data to `data/processed/`
- Generates all spectral figures into `results/figures/`
- Builds the feature table (energy, power, entropy, slope + band descriptors)
- Computes domain distributions and summary indexes
- Runs K-Means for k = 2..5 with three quality metrics
- Applies ANOVA F-test and Fisher Ratio per feature
- Evaluates Baseline, Logistic Regression and kNN with cross-validation
- Analyzes instrument separability
- **Compares optimized vs non-optimized algorithms** (SNV, moving average, global features)
- Saves all CSVs into `results/tables/`

---

## 6. Detailed Results

### 6.1. Preprocessing

**Method:** `SpectralProcessor` class (`src/preprocessing/spectral.py`).
**Outputs:**
- `data/processed/spectral_processed.csv`
- `results/figures/spectral_raw.png`, `spectral_snv.png`, `spectral_by_instrument.png`, `spectral_mean_by_instrument.png`

**Observed:**
- SNV normalization centers each spectrum at zero with unit standard deviation
- Reduces baseline and scale differences between instruments
- Preserves the characteristic spectral shape of each sample

### 6.2. Feature Extraction

**Method:** `AnalysisProcessor.buildFeatureTable()`.
**Outputs:**
- `results/tables/feature_table.csv` with 10 descriptors per sample

**Extracted descriptors:**
- **Global (4):** spectral_energy, mean_power, spectral_entropy, spectral_slope
- **Clay band (3):** clay_area, clay_peak, clay_std
- **Organic matter band (3):** organic_area, organic_peak, organic_std

### 6.3. Domain Distributions

**Method:** `computeDomainDistributions()`.
**Outputs:**
- `results/tables/domain_summary.csv`
- `results/figures/analysis_domain_distributions.png`

The Z-score distributions per domain show the relative dispersion of each group of standardized features. A domain with longer tails implies higher variability between samples and, potentially, greater discriminative power.

### 6.4. Domain Indexes and Correlation

**Method:** `computeDomainIndexes()`.
**Outputs:**
- `results/tables/domain_indexes.csv`, `domain_correlation.csv`
- `results/figures/analysis_domain_correlation.png`

> **Note:** following the feedback received on Lab 2, this project does NOT fit a linear regression between IDX-based measurements. Only the descriptive Pearson correlation matrix between the three indexes (GlobalIdx, ClayIdx, OrganicIdx) is reported.

### 6.5. K-Means Clustering

**Method:** `runClustering()` with three metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) for k = 2..5.
**Outputs:**
- `results/tables/cluster_scores.csv`
- `results/figures/analysis_cluster_scores.png`

**Discussion of the quality indexes:**
- **Silhouette ∈ [-1, 1]:** measures how similar each sample is to its own cluster vs. the nearest neighboring cluster. Values close to 1 indicate well-separated and compact clusters. Values near 0 indicate samples on the boundary.
- **Calinski-Harabasz:** ratio between inter-cluster and intra-cluster dispersion. **Higher values imply better cluster structure.**
- **Davies-Bouldin:** average similarity between each cluster and its most similar neighbor. **Lower values imply better separation.**

The final choice of optimal k is based on the coherence between the three indexes, not on a single one. When the three metrics disagree, the k with the highest Silhouette is preferred and the discrepancy is discussed.

### 6.6. ANOVA and Fisher Ratio

**Method:** `runAnovaFisher()`.
**Outputs:**
- `results/tables/anova_results.csv`, `fisher_ratio.csv`
- `results/figures/analysis_fisher_ratio.png`, `analysis_separability.png`

**ANOVA discussion:**
ANOVA tests the null hypothesis H₀: the feature means are equal between the k clusters. A p-value < 0.05 rejects H₀ with 95% confidence, indicating that the feature **discriminates significantly** between clusters. The F-statistic is the ratio between the variance explained by cluster membership and the residual variance.

ANOVA alone does not indicate between which specific clusters there are differences (post-hoc tests like Tukey HSD would be required for that). It also does not measure effect size: a feature with p < 0.001 but small F has statistically significant but practically small differences.

For this reason, it is complemented with the **Fisher Ratio**:

$$F_R = \frac{\text{Between-cluster variance}}{\text{Within-cluster variance}}$$

- $F_R > 1$ → the feature separates more between clusters than within them (discriminative).
- $F_R < 1$ → within-cluster variability exceeds between-cluster variability (non-discriminative).
- $F_R \gg 1$ → direct candidate to feed classification models.

### 6.7. Classification

**Method:** `runClassification()` with 5-fold stratified cross-validation.
**Models:** Baseline (majority class), Logistic Regression, kNN (k=5).
**Outputs:** `results/tables/classification_metrics.csv`, `results/figures/analysis_roc_curve.png`.

**In-depth discussion of the results:**
- The **Baseline** establishes the minimum floor: any model that does not significantly outperform it is only learning the marginal class distribution.
- **Logistic Regression** models a linear boundary in the standardized feature space. It is interpretable —each coefficient indicates the direction and magnitude of each feature's contribution— but assumes linear separability.
- **kNN (k=5)** assumes no decision boundary shape and can capture non-linear relationships, but is sensitive to noise and irrelevant features.
- Both models outperforming the Baseline confirms that the 10 features capture real data structure.
- If Logistic Regression outperforms kNN, it suggests approximately linear separability in feature space. If kNN outperforms Logistic Regression, it suggests non-linear boundaries.
- The use of `StratifiedKFold` with `Pipeline(StandardScaler → clf)` prevents information leakage: each fold learns its own scaling using only the training set.

### 6.8. Instrument Separability

**Method:** `runInstrumentSeparability()`.
**Outputs:** `results/tables/instrument_confusion_matrix.csv`, `results/figures/analysis_instrument_confusion.png`.

If the model classifies the instrument with accuracy significantly higher than the random baseline (1/5 = 0.20), it means SNV normalization did not completely remove the instrumental fingerprint. This result is consistent with the transferability challenge reported by Safanelli et al. (2025) and motivates additional calibration transfer techniques.

### 6.9. Optimized vs Non-optimized Comparison

**Method:** section 11 of `test/test.py`.
**Outputs:**
- `results/tables/timing_comparison.csv`
- `results/figures/analysis_timing_comparison.png`

Three pairs of implementations are compared:

| Algorithm | Original | Optimized | Reduction |
|---|---|---|---|
| SNV | Pandas `.sub().div()` | NumPy broadcasting | ~2.96x |
| Moving average | Python loop + `np.convolve` | `scipy.signal.fftconvolve` | ~1.3x |
| Global features | Python loop + `np.polyfit` | Vectorized least-squares | ~17.6x |

The theoretical complexity of each pair is the same (`O(n·m)`), but the optimized version eliminates Python loop overhead by operating directly on C/NumPy, reducing the multiplicative constants by one or two orders of magnitude.

---

## 7. Computational Complexity Analysis

Complexity per module, documented line by line in each `.py`:

| Module | Global Complexity | Justification |
|---|---|---|
| `spectral.py` | O(n·m·w) | `smoothMovingAverage` with kernel of size w |
| `analysis.py` | O(n²) | `silhouette_score` in `runClustering` |
| `spectralviz.py` | O(n·m) | Plotting n spectra of m points |
| `analysisviz.py` | O(n·f + k²) | k×k heatmap and n-point plots |
| `test.py` (global) | **O(n² + n·m)** | Sum of the above |

where:
- `n` = 250 samples
- `m` = 1,676 wavenumbers
- `f` = 10 features
- `k` = number of clusters
- `w` = smoothing window size
- `cv` = cross-validation folds (5)

---

## 8. Conclusions

- The project was reorganized into independent modules (`preprocessing`, `analysis`, `visualization`) with clear responsibilities, eliminating the mix of processing and visualization logic present in previous labs.
- SNV normalization substantially reduces baseline differences between instruments, but the residual separability (instrument accuracy above the random baseline) confirms that an instrumental fingerprint not removed by SNV persists.
- Clay-band features and global spectral energy are the most discriminative according to Fisher Ratio.
- Classification models significantly outperform the Baseline, confirming that the 10 extracted features capture real structure.
- The optimized vs non-optimized algorithm comparison shows time reductions up to 17.6x by replacing Python loops with vectorized NumPy operations, without changing theoretical complexity.

---

## 9. References

- Doran, J. W., & Zeiss, M. R. (2000). Soil health and sustainability: managing the biotic component of soil quality. *Applied Soil Ecology*, 15(1), 3–11.
- Lehmann, J., et al. (2020). Persistence of soil organic carbon caused by functional complexity. *Nature Geoscience*, 13(8), 529–534.
- Reeves, J. B. (2010). Near- versus mid-infrared diffuse reflectance spectroscopy for soil analysis emphasizing carbon and laboratory versus on-site analysis. *Geoderma*, 158(1–2), 3–14.
- Safanelli, J. L., et al. (2025). Calibration transfer for soil MIR spectroscopy. *Soil and Tillage Research*.
- Soriano-Disla, J. M., et al. (2014). The performance of visible, near-, and mid-infrared reflectance spectroscopy for prediction of soil physical, chemical, and biological properties. *Applied Spectroscopy Reviews*, 49(2), 139–186.
- Viscarra Rossel, R. A., & McBratney, A. B. (2010). Soil chemical analytical accuracy and costs. *Precision Agriculture*, 9(1-2), 35–52.
- Viscarra Rossel, R. A., et al. (2022). The Open Soil Spectral Library (OSSL). *Earth System Science Data*.
- Zeng, Q., et al. (2024). Soil health indicators and their relationships with soil biota. *Applied Soil Ecology*.
