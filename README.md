# Implementation of "One-bit mmWave MIMO Channel Estimation Using Deep Generative Networks"

Submitted to IEEE Wireless Comm. Letters. Preprint posted to arXiv at: https://arxiv.org/pdf/2211.08635.pdf

Code, results and data is structured with reference to the paper as given below:

/code: WGAN_FFT_GP_1bitQ.ipynb implements the QGCE (Section III) algorithm for WGAN-GP, CWGAN and GCE algorithm for WGAN-GP, along with the capacity computations as detailed in Section V-D.

/data: Contains the trained generative models from WGAN-GP and CWGAN training procedures as detailed in Section IV

/results: nmse_1bit.mat is BG-GAMP NMSE, capacity_GAMP is BG-GAMP capacity, capacity_max is SVD based capacity assuming perfect CSI.
