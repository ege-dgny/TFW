import numpy as np

SEED = 11111
np.random.seed(SEED)


def haar2d(matrix):

    M, N = matrix.shape
    sqrt2 = np.sqrt(2)

    row_avg = (matrix[:, 0::2] + matrix[:, 1::2]) / sqrt2
    row_diff = (matrix[:, 0::2] - matrix[:, 1::2]) / sqrt2

    LL = (row_avg[0::2, :] + row_avg[1::2, :]) / sqrt2
    LH = (row_avg[0::2, :] - row_avg[1::2, :]) / sqrt2
    HL = (row_diff[0::2, :] + row_diff[1::2, :]) / sqrt2
    HH = (row_diff[0::2, :] - row_diff[1::2, :]) / sqrt2

    return LL, LH, HL, HH


def ihaar2d(LL, LH, HL, HH):
    sqrt2 = np.sqrt(2)
    M2, N2 = LL.shape

    row_avg = np.zeros((2 * M2, N2))
    row_diff = np.zeros((2 * M2, N2))

    row_avg[0::2, :] = (LL + LH) / sqrt2
    row_avg[1::2, :] = (LL - LH) / sqrt2
    row_diff[0::2, :] = (HL + HH) / sqrt2
    row_diff[1::2, :] = (HL - HH) / sqrt2

    M = 2 * M2
    N = 2 * N2
    reconstructed = np.zeros((M, N))
    reconstructed[:, 0::2] = (row_avg + row_diff) / sqrt2
    reconstructed[:, 1::2] = (row_avg - row_diff) / sqrt2

    return reconstructed
