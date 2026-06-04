"""
test.py
==========
Main testing script. Runs the full analytical pipeline for Lab 3:

1. Load raw MIR spectral data from data/raw/
2. Preprocess: remove identifiers, apply SNV normalization
3. Save processed data to data/processed/
4. Generate raw and SNV figures into results/figures/
5. Build feature table (global + clay + organic descriptors)
6. Compute domain distributions and indexes
7. Run K-Means clustering (k = 2..5) with three quality metrics
8. Run ANOVA F-test and Fisher Ratio per feature
9. Evaluate classifiers (Baseline, Logistic Regression, kNN) with CV
10. Run instrument separability analysis
11. Compare optimized vs non-optimized algorithm versions (timing)
12. Save all tables to results/tables/

Author : Lucas Felipe Hernandez Palacio
Course : Scientific Programming - Lab 3
"""

import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from src.preprocessing.spectral import SpectralProcessor
from src.visualization.spectralviz import SpectralVisualizer
from src.analysis.analysis import AnalysisProcessor
from src.visualization.analysisviz import AnalysisVisualizer


# ============================================================
# OUTPUT FOLDERS (O(1))
# ============================================================
Path("data/processed").mkdir(parents=True, exist_ok=True)       # O(1)
Path("results/figures").mkdir(parents=True, exist_ok=True)      # O(1)
Path("results/tables").mkdir(parents=True, exist_ok=True)       # O(1)


# ============================================================
# 1. LOAD RAW DATA (O(n * m))
# ============================================================
print("=" * 60)
print("1. Loading raw MIR spectral data")
print("=" * 60)

# Try local file first, otherwise download from Google Drive
RAW_PATH = "data/raw/mir_soil.csv"
if not Path(RAW_PATH).exists():
    FILE_ID = "1gdKgdHOxGfeaVUj0OHHFrOneaFbt5o2c"
    RAW_PATH = f"https://drive.google.com/uc?id={FILE_ID}&export=download"

df_raw = pd.read_csv(RAW_PATH)                                  # O(n * m)
print(f"   Loaded shape: {df_raw.shape}")
print(f"   Samples per instrument:")
print(df_raw["instrument"].value_counts().to_string())          # O(n)


# ============================================================
# 2. PREPROCESSING PIPELINE (O(n * m))
# ============================================================
print("\n" + "=" * 60)
print("2. Preprocessing: SNV normalization")
print("=" * 60)

specProcessor = SpectralProcessor(df_raw)                       # O(n * m)
specProcessor.removeIdentifiers()                               # O(n * m)
specProcessor.applySNV()                                        # O(n * m)

spectra_snv = specProcessor.toNumpy()                           # O(n * m)
wavenumbers = specProcessor.wavenumbers                         # O(1)
identifiers = specProcessor.identifiers                         # O(1)

# Also keep the raw spectral matrix (before SNV) for figures
spectra_raw = df_raw[specProcessor.spectralCols].to_numpy()     # O(n * m)

# Save processed data (O(n * m))
processed = pd.concat([                                         # O(n * m)
    identifiers.reset_index(drop=True),
    pd.DataFrame(spectra_snv, columns=specProcessor.spectralCols),
], axis=1)
processed.to_csv("data/processed/spectral_processed.csv",       # O(n * m)
                 index=False)
print(f"   Saved: data/processed/spectral_processed.csv")
print(f"   SNV mean per spectrum (~0): "
      f"{spectra_snv.mean(axis=1).mean():.6f}")
print(f"   SNV std per spectrum (~1):  "
      f"{spectra_snv.std(axis=1).mean():.6f}")


# ============================================================
# 3. SPECTRAL VISUALIZATION (O(n * m))
# ============================================================
print("\n" + "=" * 60)
print("3. Spectral visualization")
print("=" * 60)

specViz = SpectralVisualizer(wavenumbers)                       # O(m)

specViz.plotRaw(spectra_raw,                                    # O(n * m)
                "results/figures/spectral_raw.png")
specViz.plotSNV(spectra_snv,                                    # O(n * m)
                "results/figures/spectral_snv.png")
specViz.plotByInstrument(spectra_snv,                           # O(n * m)
                         identifiers["instrument"].values,
                         "results/figures/spectral_by_instrument.png")
specViz.plotMeanByInstrument(spectra_snv,                       # O(n * m)
                              identifiers["instrument"].values,
                              "results/figures/spectral_mean_by_instrument.png")
print("   Saved 4 spectral figures into results/figures/")


# ============================================================
# 4. ANALYSIS - FEATURE EXTRACTION (O(n * m))
# ============================================================
print("\n" + "=" * 60)
print("4. Feature extraction")
print("=" * 60)

analysis = AnalysisProcessor()                                  # O(1)
analysisViz = AnalysisVisualizer()                              # O(1)

featureTable = analysis.buildFeatureTable(                      # O(n * m)
    spectra_snv, wavenumbers,
    identifiers[["instrument", "sample_id"]],
)
featureTable.to_csv("results/tables/feature_table.csv",         # O(n * f)
                    index=False)

globalCols = ["spectral_energy", "mean_power",
              "spectral_entropy", "spectral_slope"]
clayCols = ["clay_area", "clay_peak", "clay_std"]
organicCols = ["organic_area", "organic_peak", "organic_std"]

print(f"   Feature table: {featureTable.shape}")
print(f"   Domains: Global ({len(globalCols)}) + "
      f"Clay ({len(clayCols)}) + Organic ({len(organicCols)})")


# ============================================================
# 5. DOMAIN DISTRIBUTIONS (O(n * f))
# ============================================================
print("\n" + "=" * 60)
print("5. Domain distributions")
print("=" * 60)

Z, dfLong, domainSummary = analysis.computeDomainDistributions( # O(n * f)
    featureTable, globalCols, clayCols, organicCols,
)
domainSummary.to_csv("results/tables/domain_summary.csv")       # O(1)
analysisViz.plotDomainDistributions(                            # O(n * f)
    dfLong, "results/figures/analysis_domain_distributions.png"
)
print(domainSummary)


# ============================================================
# 6. DOMAIN INDEXES (correlation only, no regression) (O(n * f))
# ============================================================
print("\n" + "=" * 60)
print("6. Domain indexes correlation")
print("=" * 60)

idxDf, corrMatrix = analysis.computeDomainIndexes(              # O(n * f)
    featureTable, globalCols, clayCols, organicCols,
)
idxDf.to_csv("results/tables/domain_indexes.csv", index=False)  # O(n)
corrMatrix.to_csv("results/tables/domain_correlation.csv")      # O(1)

analysisViz.plotIndexCorrelation(                               # O(k^2)
    corrMatrix,
    "results/figures/analysis_domain_correlation.png",
)
print(corrMatrix)


# ============================================================
# 7. CLUSTERING (O(n^2))
# ============================================================
print("\n" + "=" * 60)
print("7. K-Means clustering (k = 2..5)")
print("=" * 60)

scoresDf, clusterLabels, bestK = analysis.runClustering(        # O(n^2)
    featureTable, globalCols, clayCols, organicCols,
    kMin=2, kMax=5,
)
scoresDf.to_csv("results/tables/cluster_scores.csv", index=False)
analysisViz.plotClusterScores(                                  # O(K)
    scoresDf, "results/figures/analysis_cluster_scores.png",
)
print(scoresDf)
print(f"   Best k (by silhouette): {bestK}")


# ============================================================
# 8. ANOVA + FISHER (O(k * n * f))
# ============================================================
print("\n" + "=" * 60)
print("8. ANOVA F-test and Fisher Ratio")
print("=" * 60)

anovaDf, fisherDf = analysis.runAnovaFisher(                    # O(k * n * f)
    featureTable, globalCols, clayCols, organicCols, clusterLabels,
)
anovaDf.to_csv("results/tables/anova_results.csv", index=False)
fisherDf.to_csv("results/tables/fisher_ratio.csv", index=False)

analysisViz.plotFisherRatio(                                    # O(f)
    fisherDf, "results/figures/analysis_fisher_ratio.png",
)

topTwo = fisherDf["feature"].iloc[:2].tolist()                  # O(1)
analysisViz.plotSeparability(                                   # O(n)
    featureTable, clusterLabels, topTwo,
    "results/figures/analysis_separability.png",
)
print("   ANOVA (top significant features):")
print(anovaDf.head().to_string(index=False))
print("\n   Fisher Ratio:")
print(fisherDf.to_string(index=False))


# ============================================================
# 9. CLASSIFICATION (O(cv * n * f))
# ============================================================
print("\n" + "=" * 60)
print("9. Classification (cross-validation)")
print("=" * 60)

classMetrics = analysis.runClassification(                      # O(cv * n * f)
    featureTable, globalCols, clayCols, organicCols, clusterLabels,
)
classMetrics.to_csv("results/tables/classification_metrics.csv",
                     index=False)
print(classMetrics.to_string(index=False))

# ROC curve for the best model (if AUC available)
if classMetrics["ROC_AUC"].notna().any():
    bestRow = classMetrics.loc[classMetrics["ROC_AUC"].idxmax()] # O(1)
    bestName = bestRow["Model"]
    print(f"\n   Best classifier by ROC-AUC: "
          f"{bestName} (AUC = {bestRow['ROC_AUC']:.3f})")

    # Recompute probabilities for the best model to draw ROC
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.dummy import DummyClassifier

    bestPipelines = {
        "Baseline": make_pipeline(
            StandardScaler(),
            DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="liblinear", random_state=42)),
        "kNN (k=5)": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = featureTable[globalCols + clayCols + organicCols]
    y = np.array(clusterLabels)
    try:
        probaAll = cross_val_predict(bestPipelines[bestName],
                                      X, y, cv=cv,
                                      method="predict_proba")
        analysisViz.plotRocCurve(                                # O(t)
            y, probaAll[:, 1], bestRow["ROC_AUC"], bestName,
            "results/figures/analysis_roc_curve.png",
        )
    except Exception as e:
        print(f"   (ROC plot skipped: {e})")


# ============================================================
# 10. INSTRUMENT SEPARABILITY (O(cv * n * f))
# ============================================================
print("\n" + "=" * 60)
print("10. Instrument separability")
print("=" * 60)

instAcc, confMat, instClasses = analysis.runInstrumentSeparability( # O(cv * n * f)
    featureTable, globalCols, clayCols, organicCols,
)
pd.DataFrame(                                                   # O(k^2)
    confMat, index=instClasses, columns=instClasses,
).to_csv("results/tables/instrument_confusion_matrix.csv")

analysisViz.plotConfusionMatrix(                                # O(k^2)
    confMat, instClasses,
    "results/figures/analysis_instrument_confusion.png",
)
print(f"   Accuracy: {instAcc:.3f}  "
      f"(random baseline for 5 classes: 0.200)")


# ============================================================
# 11. OPTIMIZED VS NON-OPTIMIZED ALGORITHMS
#     (Lab 3 grading criterion - 20%)
# ============================================================
print("\n" + "=" * 60)
print("11. Optimized vs non-optimized algorithms")
print("=" * 60)

N_REPS = 5
timesDict = {}

# --- 11.1 SNV: Pandas (original) vs NumPy (optimized) ---
def snvPandas(df):
    """Original SNV using Pandas. Complexity O(n * m)."""
    mean = df.mean(axis=1)                                      # O(n * m)
    std = df.std(axis=1).replace(0, 1)                          # O(n * m)
    return df.sub(mean, axis=0).div(std, axis=0)                # O(n * m)


def snvNumpy(arr):
    """Optimized SNV using NumPy broadcasting. Complexity O(n * m)."""
    mean = arr.mean(axis=1, keepdims=True)                      # O(n * m)
    std = arr.std(axis=1, keepdims=True)                        # O(n * m)
    std[std == 0] = 1                                           # O(n)
    return (arr - mean) / std                                   # O(n * m)


df_spec_only = df_raw[specProcessor.spectralCols]
arr_spec_only = df_spec_only.to_numpy()

t_snv_orig = timeit.timeit(
    lambda: snvPandas(df_spec_only), number=N_REPS) / N_REPS
t_snv_opt = timeit.timeit(
    lambda: snvNumpy(arr_spec_only), number=N_REPS) / N_REPS

print(f"   SNV (Pandas vs NumPy):")
print(f"      Original  (Pandas):  {t_snv_orig:.4f}s")
print(f"      Optimized (NumPy):   {t_snv_opt:.4f}s")
print(f"      Speedup:             {t_snv_orig / t_snv_opt:.1f}x")

timesDict["SNV"] = {"original": t_snv_orig, "optimized": t_snv_opt}


# --- 11.2 Moving average: Python loop vs FFT ---
def maLoop(spectra, window=20):
    """Original: Python loop with np.convolve. Complexity O(n * m * w)."""
    kernel = np.ones(window) / window                           # O(w)
    return np.vstack([                                          # O(n * m * w)
        np.convolve(row, kernel, mode="same") for row in spectra
    ])


def maFFT(spectra, window=20):
    """Optimized: vectorized FFT convolution. Complexity O(n * m * log m)."""
    kernel = np.ones((1, window)) / window                      # O(w)
    return fftconvolve(spectra, kernel, mode="same", axes=1)    # O(n * m * log m)


t_ma_orig = timeit.timeit(
    lambda: maLoop(arr_spec_only), number=N_REPS) / N_REPS
t_ma_opt = timeit.timeit(
    lambda: maFFT(arr_spec_only), number=N_REPS) / N_REPS

print(f"\n   Moving average (loop vs FFT):")
print(f"      Original  (loop):    {t_ma_orig:.4f}s")
print(f"      Optimized (FFT):     {t_ma_opt:.4f}s")
print(f"      Speedup:             {t_ma_orig / t_ma_opt:.1f}x")

timesDict["Moving Average"] = {"original": t_ma_orig,
                                "optimized": t_ma_opt}


# --- 11.3 Global features: Python loop vs vectorized NumPy ---
def globalFeaturesLoop(spectra, wavenumbers):
    """Original: Python loops for entropy and polyfit. O(n * m)."""
    from scipy.stats import entropy as _entropy                 # O(1)
    sq = np.square(spectra)                                     # O(n * m)
    total = sq.sum(axis=1)                                      # O(n * m)
    energyDist = (sq.T / total).T                               # O(n * m)
    spec_entropy = np.array([                                   # O(n * m)
        _entropy(row + 1e-12) for row in energyDist
    ])
    slopes = np.array([                                         # O(n * m)
        np.polyfit(wavenumbers, row, 1)[0] for row in spectra
    ])
    return total, spec_entropy, slopes


def globalFeaturesVect(spectra, wavenumbers):
    """Optimized: full vectorized NumPy. O(n * m) with lower constants."""
    sq = np.square(spectra)                                     # O(n * m)
    total = sq.sum(axis=1)                                      # O(n * m)
    energyDist = (sq.T / total).T + 1e-12                       # O(n * m)
    # Vectorized Shannon entropy
    spec_entropy = -(energyDist * np.log(energyDist)).sum(axis=1)  # O(n * m)
    # Vectorized slope via analytical least-squares
    wl_c = wavenumbers - wavenumbers.mean()                     # O(m)
    wl_ss = (wl_c ** 2).sum()                                   # O(m)
    spec_c = spectra - spectra.mean(axis=1, keepdims=True)      # O(n * m)
    slopes = (spec_c * wl_c).sum(axis=1) / wl_ss                # O(n * m)
    return total, spec_entropy, slopes


t_gf_orig = timeit.timeit(
    lambda: globalFeaturesLoop(spectra_snv, wavenumbers),
    number=N_REPS) / N_REPS
t_gf_opt = timeit.timeit(
    lambda: globalFeaturesVect(spectra_snv, wavenumbers),
    number=N_REPS) / N_REPS

print(f"\n   Global features (loop vs vectorized):")
print(f"      Original  (loop):    {t_gf_orig:.4f}s")
print(f"      Optimized (vect):    {t_gf_opt:.4f}s")
print(f"      Speedup:             {t_gf_orig / t_gf_opt:.1f}x")

timesDict["Global Features"] = {"original": t_gf_orig,
                                 "optimized": t_gf_opt}


# Save timing table and figure
timingTable = pd.DataFrame([
    {"algorithm": k,
     "original_s": v["original"],
     "optimized_s": v["optimized"],
     "speedup_x": round(v["original"] / v["optimized"], 2)}
    for k, v in timesDict.items()
])
timingTable.to_csv("results/tables/timing_comparison.csv", index=False)

analysisViz.plotTimingComparison(                               # O(p)
    timesDict,
    "results/figures/analysis_timing_comparison.png",
    title="Optimized vs non-optimized algorithms",
)
print("\n   Timing table saved to results/tables/timing_comparison.csv")


# ============================================================
# FINAL STATUS
# ============================================================
print("\n" + "=" * 60)
print("PIPELINE FINISHED")
print("=" * 60)
print("All figures saved to results/figures/")
print("All tables saved to results/tables/")
print("Processed data saved to data/processed/")


# ============================================================
#  COMPUTATIONAL COMPLEXITY SUMMARY
# ============================================================
# Summing dominant blocks:
# DATA LOAD               : O(n * m)
# PREPROCESSING (SNV)     : O(n * m)
# SPECTRAL VISUALIZATION  : O(n * m)
# FEATURE EXTRACTION      : O(n * m)
# DOMAIN DISTRIBUTIONS    : O(n * f)
# DOMAIN INDEXES          : O(n * f)
# CLUSTERING              : O(n^2)      <- dominant
# ANOVA + FISHER          : O(k * n * f)
# CLASSIFICATION          : O(cv * n * f)
# INSTRUMENT SEPARABILITY : O(cv * n * f)
# TIMING BENCHMARKS       : O(n * m)
# ============================================================
# FINAL RESULT:
# The global computational complexity of the full pipeline is
#
#                    O(n^2 + n * m)
#
# dominated by silhouette_score in K-Means evaluation and by the
# data loading / feature extraction over the full spectral matrix.
# ============================================================
