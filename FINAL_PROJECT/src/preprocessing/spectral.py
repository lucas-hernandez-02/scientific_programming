import pandas as pd
import numpy as np


class SpectralProcessor:
    """
    Perform preprocessing operations on MIR spectral data.

    This class applies:
    - Removal of non-spectral identifier columns
    - Standard Normal Variate (SNV) normalization
    - Optional smoothing by moving average
    - Conversion to NumPy format for downstream analysis

    Expected dataset structure:
    - Columns 0..2 : identifier columns (unique_id, instrument, sample_id)
    - Columns 3..m : MIR absorbance values, column names are wavenumbers (cm^-1)
    """

    def __init__(self, data):
        """
        Store the input DataFrame and detect spectral columns.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw MIR spectral dataset including identifier and spectral columns.

        Returns
        -------
        None
        """
        self.data = data.copy()                              # O(n * m)
        self.idCols = ["unique_id", "instrument", "sample_id"]  # O(1)
        self.spectralCols = [                                  # O(m)
            c for c in self.data.columns if c not in self.idCols
        ]
        self.wavenumbers = np.array(self.spectralCols).astype(float)  # O(m)

    def removeIdentifiers(self):
        """
        Separate identifier columns from spectral columns.

        Keeps only spectral columns in self.data and stores identifiers
        in self.identifiers for later concatenation.

        Returns
        -------
        SpectralProcessor
            Enables method chaining.
        """
        self.identifiers = self.data[self.idCols].copy()       # O(n)
        self.data = self.data[self.spectralCols].copy()        # O(n * m)
        return self                                            # O(1)

    def applySNV(self):
        """
        Apply Standard Normal Variate (SNV) normalization row-wise.

        SNV transforms each spectrum as:
            (spectrum - row_mean) / row_std

        This removes baseline shifts and scale differences between
        instruments, preserving the spectral shape of each sample.

        Returns
        -------
        SpectralProcessor
            Enables method chaining.
        """
        rowMeans = self.data.mean(axis=1)                      # O(n * m)
        rowStds = self.data.std(axis=1)                        # O(n * m)

        # Avoid division by zero in flat spectra
        rowStds = rowStds.replace(0, 1)                        # O(n)

        self.data = (self.data.sub(rowMeans, axis=0)).div(     # O(n * m)
            rowStds, axis=0
        )
        return self                                            # O(1)

    def smoothMovingAverage(self, window=20):
        """
        Apply moving-average smoothing per spectrum.

        For each spectrum, convolves the absorbance vector with a uniform
        kernel of length 'window'. Reduces high-frequency noise while
        preserving major spectral features.

        Parameters
        ----------
        window : int
            Smoothing window size (number of wavenumbers).

        Returns
        -------
        SpectralProcessor
            Enables method chaining.
        """
        kernel = np.ones(window) / window                      # O(w)
        arr = self.data.to_numpy()                             # O(n * m)
        smoothed = np.vstack([                                 # O(n * m * w)
            np.convolve(row, kernel, mode="same") for row in arr
        ])
        self.data = pd.DataFrame(                              # O(n * m)
            smoothed,
            columns=self.spectralCols,
            index=self.data.index,
        )
        return self                                            # O(1)

    def toNumpy(self):
        """
        Convert processed DataFrame to NumPy array.

        Returns
        -------
        numpy.ndarray
            Spectral matrix of shape (n_samples, n_wavenumbers).
        """
        return self.data.to_numpy()                            # O(n * m)


# __init__        : O(n * m)
# removeIdentifiers: O(n * m)
# applySNV        : O(n * m)
# smoothMovingAverage: O(n * m * w)
# toNumpy         : O(n * m)
# This code has a computational time complexity of O(n * m * w)
