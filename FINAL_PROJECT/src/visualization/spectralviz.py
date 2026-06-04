import os
import numpy as np
import matplotlib.pyplot as plt


class SpectralVisualizer:
    """
    Visualization tools for MIR spectral datasets of soil samples.

    Generates:
    - Raw spectral plots (all samples overlaid)
    - SNV-normalized spectral plots (all samples overlaid)
    - Spectra grouped by instrument (subplot per instrument)
    - Mean spectrum per instrument

    Expected dataset layout:
    - identifiers : DataFrame with columns unique_id, instrument, sample_id
    - spectra     : numpy ndarray of shape (n_samples, n_wavenumbers)
    - wavenumbers : numpy ndarray (n_wavenumbers,)
    """

    INSTRUMENT_COLORS = [
        "magenta", "cyan", "salmon", "olivedrab", "gold"
    ]

    def __init__(self, wavenumbers):
        """
        Store the wavenumber axis used in every plot.

        Parameters
        ----------
        wavenumbers : numpy.ndarray
            Wavenumber axis (n_wavenumbers,).

        Returns
        -------
        None
        """
        self.wavenumbers = wavenumbers                              # O(m)

    def _ensureDir(self, path):
        """Create directory if it does not exist. O(1)."""
        if path and not os.path.exists(path):                       # O(1)
            os.makedirs(path, exist_ok=True)                        # O(1)

    def plotRaw(self, spectra, savepath):
        """
        Plot all raw spectra overlaid in a single axis.

        Parameters
        ----------
        spectra : numpy.ndarray
            Raw spectral matrix of shape (n, m).
        savepath : str
            Output file path.

        Returns
        -------
        None
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        plt.figure(figsize=(12, 5))                                 # O(1)
        plt.plot(self.wavenumbers, spectra.T,                       # O(n * m)
                 alpha=0.3, linewidth=0.5, color="steelblue")
        plt.xlabel("Wavenumber (cm^-1)")                            # O(1)
        plt.ylabel("Absorbance")                                    # O(1)
        plt.title("Raw MIR spectra (all samples)")                  # O(1)
        plt.grid(True, alpha=0.3)                                   # O(1)
        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300)                              # O(1)
        plt.close()                                                 # O(1)

    def plotSNV(self, spectra, savepath):
        """
        Plot all SNV-normalized spectra overlaid in a single axis.

        Parameters
        ----------
        spectra : numpy.ndarray
            SNV-normalized matrix of shape (n, m).
        savepath : str
            Output file path.

        Returns
        -------
        None
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        plt.figure(figsize=(12, 5))                                 # O(1)
        plt.plot(self.wavenumbers, spectra.T,                       # O(n * m)
                 alpha=0.3, linewidth=0.5, color="darkorange")
        plt.xlabel("Wavenumber (cm^-1)")                            # O(1)
        plt.ylabel("SNV absorbance")                                # O(1)
        plt.title("SNV-normalized MIR spectra (all samples)")       # O(1)
        plt.grid(True, alpha=0.3)                                   # O(1)
        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300)                              # O(1)
        plt.close()                                                 # O(1)

    def plotByInstrument(self, spectra, instruments, savepath):
        """
        Plot spectra grouped by instrument in side-by-side subplots.

        Parameters
        ----------
        spectra : numpy.ndarray
            Spectral matrix of shape (n, m).
        instruments : array-like
            Instrument label for each row of 'spectra'.
        savepath : str
            Output file path.

        Returns
        -------
        None
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        uniqueInsts = sorted(np.unique(instruments))                # O(n log n)
        fig, axs = plt.subplots(1, len(uniqueInsts),                # O(1)
                                figsize=(20, 4), sharey=True)

        for i, inst in enumerate(uniqueInsts):                      # O(5)
            mask = np.array(instruments) == inst                    # O(n)
            color = self.INSTRUMENT_COLORS[                         # O(1)
                i % len(self.INSTRUMENT_COLORS)
            ]
            axs[i].plot(self.wavenumbers, spectra[mask].T,          # O(n_i * m)
                        alpha=0.4, linewidth=0.5, color=color)
            axs[i].set_title(inst)                                  # O(1)
            axs[i].set_xlabel("Wavenumber (cm^-1)")                 # O(1)
            axs[i].grid(True, alpha=0.3)                            # O(1)

        axs[0].set_ylabel("Absorbance")                             # O(1)
        plt.suptitle("Spectra by instrument", y=1.02)               # O(1)
        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")         # O(1)
        plt.close()                                                 # O(1)

    def plotMeanByInstrument(self, spectra, instruments, savepath):
        """
        Plot the mean spectrum per instrument in a single axis.

        Parameters
        ----------
        spectra : numpy.ndarray
        instruments : array-like
        savepath : str

        Returns
        -------
        None
        """
        self._ensureDir(os.path.dirname(savepath))                  # O(1)

        uniqueInsts = sorted(np.unique(instruments))                # O(n log n)
        plt.figure(figsize=(12, 5))                                 # O(1)

        for i, inst in enumerate(uniqueInsts):                      # O(5)
            mask = np.array(instruments) == inst                    # O(n)
            meanSpec = spectra[mask].mean(axis=0)                   # O(n_i * m)
            color = self.INSTRUMENT_COLORS[                         # O(1)
                i % len(self.INSTRUMENT_COLORS)
            ]
            plt.plot(self.wavenumbers, meanSpec,                    # O(m)
                     label=inst, color=color, linewidth=1.5)

        plt.xlabel("Wavenumber (cm^-1)")                            # O(1)
        plt.ylabel("Mean absorbance")                               # O(1)
        plt.title("Mean spectrum per instrument")                   # O(1)
        plt.legend()                                                # O(1)
        plt.grid(True, alpha=0.3)                                   # O(1)
        plt.tight_layout()                                          # O(1)
        plt.savefig(savepath, dpi=300)                              # O(1)
        plt.close()                                                 # O(1)


# plotRaw              : O(n * m)
# plotSNV              : O(n * m)
# plotByInstrument     : O(n * m)
# plotMeanByInstrument : O(n * m)
# This code has a computational time complexity of O(n * m)
