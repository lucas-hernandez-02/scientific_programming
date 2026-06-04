import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve


class AnalysisVisualizer:
    """
    Visualization tools for the analytical pipeline results.

    Generates:
    - Domain distribution violin + box plots
    - Domain index correlation heatmap (no regression line, per
      professor's feedback)
    - Fisher Ratio bar plot per feature
    - Clustering metrics vs k
    - Cluster separability scatter (top-2 features)
    - ROC curve
    - Confusion matrix for instrument classification
    - Optimized vs non-optimized timing bar plot
    """

    DOMAIN_PALETTE = {
        "Global":  "steelblue",
        "Clay":    "darkorange",
        "Organic": "seagreen",
    }

    def __init__(self):
        """Initialize visualizer. O(1)."""
        pass

    def _ensureDir(self, path):
        """Create directory if it does not exist. O(1)."""
        if path and not os.path.exists(path):                       # O(1)
            os.makedirs(path, exist_ok=True)                        # O(1)

    def plotDomainDistributions(self, dfLong, savepath):
        """
        Violin + box plot of standardized features grouped by domain.

        Parameters
        ----------
        dfLong : pandas.DataFrame
            Columns: feature, z, domain.
        savepath : str
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        plt.figure(figsize=(8, 5))                                  # O(1)
        sns.violinplot(                                             # O(n * f)
            data=dfLong, x="domain", y="z",
            hue="domain", inner=None, legend=False,
            palette=self.DOMAIN_PALETTE,
        )
        sns.boxplot(                                                # O(n * f)
            data=dfLong, x="domain", y="z",
            width=0.2, showcaps=True,
            boxprops={"zorder": 2},
            palette={k: "white" for k in self.DOMAIN_PALETTE},
            hue="domain", legend=False,
        )
        plt.title("Standardized feature distribution by domain")    # O(1)
        plt.ylabel("Z-score")                                       # O(1)
        plt.xlabel("")                                              # O(1)
        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300)                              # O(1)
        plt.close()                                                 # O(1)

    def plotIndexCorrelation(self, corrMatrix, savepath):
        """
        Heatmap of correlation between domain indexes.

        Note: per professor's feedback on Lab 2, no linear regression
        is fitted between IDX-based measurements. Only the correlation
        matrix is reported descriptively.

        Parameters
        ----------
        corrMatrix : pandas.DataFrame
            3x3 correlation matrix (GlobalIdx, ClayIdx, OrganicIdx).
        savepath : str
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        plt.figure(figsize=(5, 4))                                  # O(1)
        sns.heatmap(                                                # O(k^2)
            corrMatrix, annot=True, fmt=".2f",
            cmap="coolwarm", vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": 0.8},
        )
        plt.title("Correlation between domain indexes")             # O(1)
        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300)                              # O(1)
        plt.close()                                                 # O(1)

    def plotFisherRatio(self, fisherDf, savepath):
        """
        Horizontal bar plot of Fisher Ratio per feature, ordered by
        discriminative power and colored by domain.

        Parameters
        ----------
        fisherDf : pandas.DataFrame
            Columns: feature, FisherRatio, discriminative.
        savepath : str
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        # Map each feature to its domain
        domainMap = {}                                              # O(1)
        for col in ["spectral_energy", "mean_power",
                    "spectral_entropy", "spectral_slope"]:          # O(f)
            domainMap[col] = "Global"
        for col in ["clay_area", "clay_peak", "clay_std"]:
            domainMap[col] = "Clay"
        for col in ["organic_area", "organic_peak", "organic_std"]:
            domainMap[col] = "Organic"

        df = fisherDf.copy()                                        # O(f)
        df["domain"] = df["feature"].map(domainMap)                 # O(f)
        barColors = df["domain"].map(self.DOMAIN_PALETTE).tolist()  # O(f)

        plt.figure(figsize=(8, 5))                                  # O(1)
        plt.barh(df["feature"], df["FisherRatio"],                  # O(f)
                 color=barColors, edgecolor="white", height=0.6)
        plt.axvline(x=1, color="gray", linestyle="--",              # O(1)
                    linewidth=1, label="Fisher Ratio = 1")
        plt.xlabel("Fisher Ratio")                                  # O(1)
        plt.title("Fisher Ratio per feature\n"                      # O(1)
                  "(ordered by discriminative power)")
        plt.grid(axis="x", alpha=0.3)                               # O(1)

        legend = [                                                  # O(1)
            Patch(color=v, label=k)
            for k, v in self.DOMAIN_PALETTE.items()
        ]
        plt.legend(handles=legend, loc="lower right")               # O(1)
        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300)                              # O(1)
        plt.close()                                                 # O(1)

    def plotClusterScores(self, scoresDf, savepath):
        """
        Plot silhouette, Calinski-Harabasz and Davies-Bouldin vs k.

        Parameters
        ----------
        scoresDf : pandas.DataFrame
            Columns: k, silhouette, calinski_harabasz, davies_bouldin.
        savepath : str
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))              # O(1)

        axs[0].plot(scoresDf["k"], scoresDf["silhouette"],          # O(K)
                    marker="o", color="steelblue")
        axs[0].set_title("Silhouette (higher is better)")           # O(1)
        axs[0].set_xlabel("k")                                      # O(1)
        axs[0].grid(True, alpha=0.3)                                # O(1)

        axs[1].plot(scoresDf["k"], scoresDf["calinski_harabasz"],   # O(K)
                    marker="o", color="darkorange")
        axs[1].set_title("Calinski-Harabasz (higher is better)")    # O(1)
        axs[1].set_xlabel("k")                                      # O(1)
        axs[1].grid(True, alpha=0.3)                                # O(1)

        axs[2].plot(scoresDf["k"], scoresDf["davies_bouldin"],      # O(K)
                    marker="o", color="seagreen")
        axs[2].set_title("Davies-Bouldin (lower is better)")        # O(1)
        axs[2].set_xlabel("k")                                      # O(1)
        axs[2].grid(True, alpha=0.3)                                # O(1)

        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300)                              # O(1)
        plt.close()                                                 # O(1)

    def plotSeparability(self, featureTable, labels, topTwo, savepath):
        """
        Scatter of the two most discriminative features colored by
        cluster label.

        Parameters
        ----------
        featureTable : pandas.DataFrame
            Feature matrix with the two columns to plot.
        labels : array-like
            Cluster labels per row.
        topTwo : list of str
            Names of the two features to plot.
        savepath : str
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        scatterColors = ["steelblue", "darkorange",                 # O(1)
                          "seagreen", "tomato"]
        labels = np.array(labels)                                   # O(n)

        plt.figure(figsize=(7, 5))                                  # O(1)
        for g in np.unique(labels):                                 # O(k)
            sel = labels == g                                       # O(n)
            plt.scatter(                                            # O(n_g)
                featureTable.loc[sel, topTwo[0]],
                featureTable.loc[sel, topTwo[1]],
                label=f"Cluster {g}", alpha=0.7,
                color=scatterColors[g % len(scatterColors)], s=50,
            )

        plt.xlabel(topTwo[0])                                       # O(1)
        plt.ylabel(topTwo[1])                                       # O(1)
        plt.title("Cluster separability - top 2 Fisher features")   # O(1)
        plt.legend()                                                # O(1)
        plt.grid(True, alpha=0.3)                                   # O(1)
        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300)                              # O(1)
        plt.close()                                                 # O(1)

    def plotRocCurve(self, y, yProba, aucValue, modelName, savepath):
        """
        ROC curve for the best classification model.

        Parameters
        ----------
        y : array-like
            True labels.
        yProba : array-like
            Predicted probabilities for the positive class.
        aucValue : float
            ROC-AUC score.
        modelName : str
            Label for the legend.
        savepath : str
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        fpr, tpr, _ = roc_curve(y, yProba)                          # O(n log n)
        plt.figure(figsize=(6, 5))                                  # O(1)
        plt.plot(fpr, tpr, color="steelblue", linewidth=2,          # O(t)
                 label=f"{modelName}  (AUC = {aucValue:.3f})")
        plt.plot([0, 1], [0, 1], color="gray",                      # O(1)
                 linestyle="--", linewidth=1,
                 label="Random classifier")
        plt.xlabel("False positive rate")                           # O(1)
        plt.ylabel("True positive rate")                            # O(1)
        plt.title("ROC curve - best classifier")                    # O(1)
        plt.legend()                                                # O(1)
        plt.grid(True, alpha=0.3)                                   # O(1)
        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300)                              # O(1)
        plt.close()                                                 # O(1)

    def plotConfusionMatrix(self, confMat, labelClasses, savepath):
        """
        Confusion matrix for instrument classification.

        Parameters
        ----------
        confMat : numpy.ndarray
            Confusion matrix (k x k).
        labelClasses : numpy.ndarray
            Class names.
        savepath : str
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        disp = ConfusionMatrixDisplay(                              # O(1)
            confusion_matrix=confMat,
            display_labels=labelClasses,
        )
        fig, ax = plt.subplots(figsize=(7, 6))                      # O(1)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)              # O(k^2)
        ax.set_title("Confusion matrix - instrument classification")
        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300)                              # O(1)
        plt.close()                                                 # O(1)

    def plotTimingComparison(self, timesDict, savepath,
                              title="Optimized vs non-optimized timing"):
        """
        Bar plot comparing original and optimized execution times.

        Parameters
        ----------
        timesDict : dict
            Structure:
            {'algorithm_name': {'original': float, 'optimized': float}, ...}
        savepath : str
        title : str
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        labels = list(timesDict.keys())                             # O(p)
        originals = [v["original"] for v in timesDict.values()]     # O(p)
        optimized = [v["optimized"] for v in timesDict.values()]    # O(p)

        x = np.arange(len(labels))                                  # O(p)
        width = 0.35                                                # O(1)

        fig, ax = plt.subplots(figsize=(10, 5))                     # O(1)
        bars1 = ax.bar(x - width / 2, originals, width,             # O(p)
                       label="Original", color="steelblue", alpha=0.85)
        bars2 = ax.bar(x + width / 2, optimized, width,             # O(p)
                       label="Optimized", color="seagreen", alpha=0.85)

        ax.set_xlabel("Algorithm")                                  # O(1)
        ax.set_ylabel("Time (seconds)")                             # O(1)
        ax.set_title(title)                                         # O(1)
        ax.set_xticks(x)                                            # O(1)
        ax.set_xticklabels(labels, rotation=15, ha="right")         # O(1)
        ax.legend()                                                 # O(1)

        for bar in list(bars1) + list(bars2):                       # O(p)
            height = bar.get_height()                               # O(1)
            ax.annotate(                                            # O(1)
                f"{height:.4f}s",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300)                              # O(1)
        plt.close()                                                 # O(1)


# plotDomainDistributions : O(n * f)
# plotIndexCorrelation    : O(k^2)
# plotFisherRatio         : O(f)
# plotClusterScores       : O(K)
# plotSeparability        : O(n)
# plotRocCurve            : O(t)
# plotConfusionMatrix     : O(k^2)
# plotTimingComparison    : O(p)
# This code has a computational time complexity of O(n * f + k^2)
