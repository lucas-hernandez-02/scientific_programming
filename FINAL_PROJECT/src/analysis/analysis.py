import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import entropy

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


class AnalysisProcessor:
    """
    Perform feature extraction, statistical analysis, clustering and
    classification on MIR spectral data of soil samples.

    The class implements the full analytical pipeline:
    - Global spectral feature extraction (energy, power, entropy, slope)
    - Band-specific descriptors (clay band 1000-1200 cm^-1,
      organic matter band 1350-1450 cm^-1)
    - Standardization and domain distribution analysis
    - Domain-level summary indexes (without correlation regression,
      per professor's feedback on Lab 2)
    - K-Means clustering with three quality metrics
    - ANOVA F-test and Fisher Ratio per feature
    - Classification with cross-validation (Baseline, Logistic, kNN)
    - Instrument separability analysis
    """

    def __init__(self):
        """Initialize processor (no persistent state). O(1)"""
        pass

    def extractGlobalFeatures(self, spectra, wavenumbers):
        """
        Compute global spectral descriptors per sample.

        Descriptors:
        - spectral_energy : sum of squared absorbances (total signal).
        - mean_power      : energy divided by number of wavenumbers.
        - spectral_entropy: Shannon entropy of normalized energy
          distribution (uniform = high, peaky = low).
        - spectral_slope  : slope of linear fit over the full spectrum.

        Parameters
        ----------
        spectra : numpy.ndarray
            SNV-normalized matrix of shape (n_samples, n_wavenumbers).
        wavenumbers : numpy.ndarray
            Wavenumber axis in cm^-1.

        Returns
        -------
        df : pandas.DataFrame
            Columns: spectral_energy, mean_power, spectral_entropy,
            spectral_slope.
        """
        df = pd.DataFrame()                                     # O(1)

        sq = np.square(spectra)                                 # O(n * m)
        totalEnergy = sq.sum(axis=1)                            # O(n * m)
        df["spectral_energy"] = totalEnergy                     # O(n)
        df["mean_power"] = totalEnergy / spectra.shape[1]       # O(n)

        # Energy distribution per spectrum, then Shannon entropy
        energyDist = (sq.T / totalEnergy).T                     # O(n * m)
        df["spectral_entropy"] = np.array([                     # O(n * m)
            entropy(row + 1e-12) for row in energyDist
        ])

        # Linear-fit slope per spectrum
        df["spectral_slope"] = np.array([                       # O(n * m)
            np.polyfit(wavenumbers, row, 1)[0] for row in spectra
        ])

        return df                                               # O(1)

    def extractBandDescriptors(self, spectra, wavenumbers,
                               wnMin, wnMax, label):
        """
        Compute three descriptors over a spectral band of interest.

        Descriptors:
        - {label}_area : area under the curve (trapezoidal rule).
        - {label}_peak : maximum absorbance inside the band.
        - {label}_std  : standard deviation inside the band.

        Parameters
        ----------
        spectra : numpy.ndarray
            SNV-normalized spectral matrix.
        wavenumbers : numpy.ndarray
            Wavenumber axis in cm^-1.
        wnMin : float
            Lower bound of the band.
        wnMax : float
            Upper bound of the band.
        label : str
            Prefix for the resulting column names.

        Returns
        -------
        df : pandas.DataFrame
            Columns: {label}_area, {label}_peak, {label}_std.
        """
        bandMask = (wavenumbers >= wnMin) & (wavenumbers <= wnMax)  # O(m)
        bandData = spectra[:, bandMask]                             # O(n * b)
        bandAxis = wavenumbers[bandMask]                            # O(b)

        df = pd.DataFrame()                                         # O(1)
        df[f"{label}_area"] = np.trapezoid(                         # O(n * b)
            bandData, x=bandAxis, axis=1
        )
        df[f"{label}_peak"] = bandData.max(axis=1)                  # O(n * b)
        df[f"{label}_std"] = bandData.std(axis=1)                   # O(n * b)
        return df                                                   # O(1)

    def buildFeatureTable(self, spectra, wavenumbers, identifiers):
        """
        Build full feature table by combining global features and
        band-specific descriptors for clay and organic-matter regions.

        Parameters
        ----------
        spectra : numpy.ndarray
            SNV-normalized spectral matrix.
        wavenumbers : numpy.ndarray
            Wavenumber axis in cm^-1.
        identifiers : pandas.DataFrame
            DataFrame with columns: unique_id, instrument, sample_id.

        Returns
        -------
        featureTable : pandas.DataFrame
            Identifiers + 10 spectral descriptors.
        """
        glob = self.extractGlobalFeatures(spectra, wavenumbers)   # O(n * m)
        clay = self.extractBandDescriptors(                       # O(n * b)
            spectra, wavenumbers, 1000, 1200, label="clay"
        )
        org = self.extractBandDescriptors(                        # O(n * b)
            spectra, wavenumbers, 1350, 1450, label="organic"
        )

        featureTable = pd.concat([                                # O(n * f)
            identifiers.reset_index(drop=True),
            glob.reset_index(drop=True),
            clay.reset_index(drop=True),
            org.reset_index(drop=True),
        ], axis=1)
        return featureTable                                       # O(1)

    def computeDomainDistributions(self, featureTable,
                                    globalCols, clayCols, organicCols):
        """
        Standardize features (z-score) and compute long-format
        distribution per domain.

        Parameters
        ----------
        featureTable : pandas.DataFrame
            Feature matrix including identifier columns.
        globalCols : list of str
            Names of global descriptors.
        clayCols : list of str
            Names of clay-band descriptors.
        organicCols : list of str
            Names of organic-band descriptors.

        Returns
        -------
        Z : pandas.DataFrame
            Standardized feature matrix.
        dfLong : pandas.DataFrame
            Long format with columns: feature, z, domain.
        summary : pandas.DataFrame
            Mean, std, median, min, max per domain.
        """
        descCols = globalCols + clayCols + organicCols              # O(f)
        X = featureTable[descCols].copy()                           # O(n * f)

        Z = (X - X.mean()) / X.std(ddof=0)                          # O(n * f)
        dfLong = Z.melt(var_name="feature", value_name="z")         # O(n * f)

        # Assign domain label per feature (no regression on indexes)
        domainMap = {}                                              # O(1)
        for c in globalCols:                                        # O(f)
            domainMap[c] = "Global"
        for c in clayCols:
            domainMap[c] = "Clay"
        for c in organicCols:
            domainMap[c] = "Organic"
        dfLong["domain"] = dfLong["feature"].map(domainMap)         # O(n * f)

        summary = dfLong.groupby("domain")["z"].agg(                # O(n)
            ["mean", "std", "median", "min", "max"]
        ).round(3)
        return Z, dfLong, summary                                   # O(1)

    def computeDomainIndexes(self, featureTable,
                              globalCols, clayCols, organicCols):
        """
        Compute one summary index per domain (mean z-score across its
        features). Returns only numeric indexes per sample, without
        linear regression between them (per professor's feedback).

        Parameters
        ----------
        featureTable : pandas.DataFrame
            Feature matrix including identifier columns.
        globalCols, clayCols, organicCols : list of str
            Feature names per domain.

        Returns
        -------
        idxDf : pandas.DataFrame
            Columns: GlobalIdx, ClayIdx, OrganicIdx.
        corrMatrix : pandas.DataFrame
            Pearson correlation matrix between the three indexes
            (descriptive only; no regression line is fitted).
        """
        descCols = globalCols + clayCols + organicCols              # O(f)
        X = featureTable[descCols].copy()                           # O(n * f)
        Z = (X - X.mean()) / X.std(ddof=0)                          # O(n * f)

        idxDf = pd.DataFrame({                                      # O(n)
            "GlobalIdx":  Z[globalCols].mean(axis=1),
            "ClayIdx":    Z[clayCols].mean(axis=1),
            "OrganicIdx": Z[organicCols].mean(axis=1),
        })
        corrMatrix = idxDf.corr().round(3)                          # O(n)
        return idxDf, corrMatrix                                    # O(1)

    def runClustering(self, featureTable,
                      globalCols, clayCols, organicCols,
                      kMin=2, kMax=6):
        """
        Evaluate K-Means for k in [kMin, kMax] using three quality
        metrics: silhouette, Calinski-Harabasz and Davies-Bouldin.

        Parameters
        ----------
        featureTable : pandas.DataFrame
        globalCols, clayCols, organicCols : list of str
        kMin : int
        kMax : int

        Returns
        -------
        scoresDf : pandas.DataFrame
            Metrics per k.
        bestLabels : numpy.ndarray
            Cluster labels for best k by silhouette.
        bestK : int
            Selected k value.
        """
        descCols = globalCols + clayCols + organicCols              # O(f)
        X = featureTable[descCols].copy()                           # O(n * f)
        Z = (X - X.mean()) / X.std(ddof=0)                          # O(n * f)

        scores = []                                                 # O(1)
        labelsDict = {}                                             # O(1)

        for k in range(kMin, kMax + 1):                             # O(K)
            km = KMeans(n_clusters=k, n_init=20, random_state=42)   # O(1)
            labels = km.fit_predict(Z)                              # O(n * f * i)
            labelsDict[k] = labels                                  # O(1)

            sil = silhouette_score(Z, labels)                       # O(n^2)
            ch = calinski_harabasz_score(Z, labels)                 # O(n * f)
            db = davies_bouldin_score(Z, labels)                    # O(n * f)

            scores.append({                                         # O(1)
                "k": k,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
            })

        scoresDf = pd.DataFrame(scores).round(3)                    # O(K)

        bestK = int(scoresDf.sort_values(                           # O(K log K)
            "silhouette", ascending=False
        ).iloc[0]["k"])
        bestLabels = labelsDict[bestK]                              # O(1)
        return scoresDf, bestLabels, bestK                          # O(1)

    def runAnovaFisher(self, featureTable,
                       globalCols, clayCols, organicCols, labels):
        """
        Compute ANOVA F-test and Fisher ratio per feature against
        cluster labels.

        Parameters
        ----------
        featureTable : pandas.DataFrame
        globalCols, clayCols, organicCols : list of str
        labels : array-like
            Cluster labels for each sample.

        Returns
        -------
        anovaDf : pandas.DataFrame
            Columns: feature, F, p_value, significant.
        fisherDf : pandas.DataFrame
            Columns: feature, FisherRatio, discriminative.
        """
        descCols = globalCols + clayCols + organicCols              # O(f)
        X = featureTable[descCols].copy()                           # O(n * f)
        y = np.array(labels)                                        # O(n)
        classes = np.unique(y)                                      # O(n)

        def fisherRatio(values, groups):
            """Compute Fisher ratio for one feature. O(k * n)."""
            cls = np.unique(groups)                                 # O(n)
            means = [np.mean(values[groups == c]) for c in cls]     # O(k * n)
            vars_ = [np.var(values[groups == c], ddof=1)
                     for c in cls]                                  # O(k * n)
            if np.mean(vars_) == 0:                                 # O(1)
                return np.nan
            return np.var(means, ddof=1) / np.mean(vars_)           # O(1)

        anovaRows = []                                              # O(1)
        fisherRows = []                                             # O(1)

        for col in descCols:                                        # O(f)
            groupsVals = [X.loc[y == c, col].values                 # O(k * n)
                          for c in classes]
            F, p = stats.f_oneway(*groupsVals)                      # O(k * n)
            anovaRows.append({                                      # O(1)
                "feature": col,
                "F": round(F, 4),
                "p_value": round(p, 4),
                "significant": p < 0.05,
            })

            fr = fisherRatio(X[col].values, y)                      # O(k * n)
            fisherRows.append({                                     # O(1)
                "feature": col,
                "FisherRatio": round(fr, 4) if not np.isnan(fr) else fr,
                "discriminative": (fr > 1) if not np.isnan(fr) else False,
            })

        anovaDf = pd.DataFrame(anovaRows).sort_values(              # O(f log f)
            "p_value"
        ).reset_index(drop=True)
        fisherDf = pd.DataFrame(fisherRows).sort_values(            # O(f log f)
            "FisherRatio", ascending=False
        ).reset_index(drop=True)
        return anovaDf, fisherDf                                    # O(1)

    def runClassification(self, featureTable,
                          globalCols, clayCols, organicCols,
                          labels, cvSplits=5):
        """
        Evaluate three classifiers (Baseline, Logistic Regression, kNN)
        against cluster labels using stratified cross-validation.

        Parameters
        ----------
        featureTable : pandas.DataFrame
        globalCols, clayCols, organicCols : list of str
        labels : array-like
            Cluster labels as class targets.
        cvSplits : int
            Number of folds.

        Returns
        -------
        resultsDf : pandas.DataFrame
            Metrics per model (Accuracy, Precision, Recall, F1, ROC-AUC).
        """
        descCols = globalCols + clayCols + organicCols              # O(f)
        X = featureTable[descCols].copy()                           # O(n * f)
        y = np.array(labels)                                        # O(n)

        cv = StratifiedKFold(                                       # O(1)
            n_splits=cvSplits, shuffle=True, random_state=42
        )

        models = {                                                  # O(1)
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

        rows = []                                                   # O(1)

        for name, clf in models.items():                            # O(3)
            yPred = cross_val_predict(clf, X, y, cv=cv)             # O(cv * n * f)

            try:
                probaAll = cross_val_predict(                       # O(cv * n * f)
                    clf, X, y, cv=cv, method="predict_proba"
                )
                if probaAll.shape[1] >= 2 and len(np.unique(y)) == 2:
                    auc = roc_auc_score(y, probaAll[:, 1])          # O(n log n)
                else:
                    auc = np.nan
            except Exception:
                auc = np.nan

            avg = "binary" if len(np.unique(y)) == 2 else "macro"   # O(1)

            rows.append({                                           # O(1)
                "Model": name,
                "Accuracy":  round(accuracy_score(y, yPred), 3),
                "Precision": round(precision_score(y, yPred,
                                                   average=avg,
                                                   zero_division=0), 3),
                "Recall":    round(recall_score(y, yPred,
                                                average=avg,
                                                zero_division=0), 3),
                "F1":        round(f1_score(y, yPred,
                                            average=avg,
                                            zero_division=0), 3),
                "ROC_AUC":   round(auc, 3) if not np.isnan(auc) else np.nan,
            })

        resultsDf = pd.DataFrame(rows)                              # O(1)
        return resultsDf                                            # O(1)

    def runInstrumentSeparability(self, featureTable,
                                   globalCols, clayCols, organicCols,
                                   cvSplits=5):
        """
        Test whether the extracted features still encode the instrument
        signature after SNV normalization, using Logistic Regression
        with cross-validation. Returns accuracy and confusion matrix.

        Parameters
        ----------
        featureTable : pandas.DataFrame
        globalCols, clayCols, organicCols : list of str
        cvSplits : int

        Returns
        -------
        accuracy : float
        confMat : numpy.ndarray
        labelClasses : numpy.ndarray
        """
        descCols = globalCols + clayCols + organicCols              # O(f)
        X = featureTable[descCols].copy()                           # O(n * f)

        le = LabelEncoder()                                         # O(1)
        yInst = le.fit_transform(featureTable["instrument"])        # O(n)

        cv = StratifiedKFold(                                       # O(1)
            n_splits=cvSplits, shuffle=True, random_state=42
        )
        clf = make_pipeline(                                        # O(1)
            StandardScaler(),
            LogisticRegression(solver="lbfgs", max_iter=1000,
                               random_state=42)
        )
        yPred = cross_val_predict(clf, X, yInst, cv=cv)             # O(cv * n * f)

        accuracy = accuracy_score(yInst, yPred)                     # O(n)
        confMat = confusion_matrix(yInst, yPred)                    # O(n)
        return accuracy, confMat, le.classes_                       # O(1)


# extractGlobalFeatures      : O(n * m)
# extractBandDescriptors     : O(n * b)
# buildFeatureTable          : O(n * m)
# computeDomainDistributions : O(n * f)
# computeDomainIndexes       : O(n * f)
# runClustering              : O(n^2)
# runAnovaFisher             : O(k * n * f)
# runClassification          : O(cv * n * f)
# runInstrumentSeparability  : O(cv * n * f)
# This code has a computational time complexity of O(n^2)
