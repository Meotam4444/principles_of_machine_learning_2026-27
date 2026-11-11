"""
Assignment 01 - Problem 1.1: beware of numerical instabilities

We solve the ordinary least squares (OLS) problem

    w* = argmin_w || X^T w - y ||^2

in different ways and compare how robust each way is on a computer.

Notation follows the lecture:
    matX  = X     (shape 2 x 3, the data matrix, columns = data points)
    matX.T = X^T  (shape 3 x 2, the "tall" design matrix)
    vecY  = y     (shape 3,     the target vector)
"""

import numpy as np
import numpy.linalg as la


# ---------------------------------------------------------------------------
# Task 1.1.1 (given on the sheet) - kept here only for comparison
# ---------------------------------------------------------------------------
def solve_lsq_normal_equations(matX, vecY):
    """
    Pen-and-paper solution  w = (X X^T)^-1 X y.

    Problem: it explicitly forms X X^T. This squares the condition number
    of the problem, so rounding errors get amplified much more.
    """
    return la.inv(matX @ matX.T) @ matX @ vecY


# ---------------------------------------------------------------------------
# Task 1.1.2 - solution via the (thin) QR decomposition
# ---------------------------------------------------------------------------
def solve_lsq_qr(matX, vecY):
    """
    Solve the OLS problem via the thin QR decomposition of the design matrix.

    Idea (eq. (8) on the sheet):
        X^T = Q1 R1                 Q1: 3x2 with orthonormal columns
                                    R1: 2x2 upper triangular
        w*  = R1^-1 Q1^T y

    Why is this more robust?
      - We never form X X^T, so the condition number is NOT squared.
      - Q1 is orthogonal, so multiplying by Q1^T does not amplify errors
        (orthogonal matrices preserve lengths: ||Q1^T v|| <= ||v||).
      - R1 is upper triangular; the inverse of a (non-singular) triangular
        matrix can be computed quickly and robustly.

    Parameters
    ----------
    matX : ndarray, shape (m, n)   data matrix X (we decompose its transpose)
    vecY : ndarray, shape (n,)     targets y

    Returns
    -------
    vecW : ndarray, shape (m,)     least squares solution w*
    """
    # np.linalg.qr uses mode='reduced' by default -> returns Q1 and R1 (thin QR)
    matQ1, matR1 = la.qr(matX.T)

    # literally implement eq. (8) from the sheet:  w* = R1^-1 Q1^T y
    vecW = la.inv(matR1) @ matQ1.T @ vecY

    return vecW


# ---------------------------------------------------------------------------
# Task 1.1.3 - solution via numpy's special purpose function lstsq()
# ---------------------------------------------------------------------------
def solve_lsq_lstsq(matX, vecY):
    """
    Solve the OLS problem  argmin_w || X^T w - y ||^2  with la.lstsq().

    lstsq(A, b) solves  argmin_x || A x - b ||^2, so we pass A = X^T, b = y.
    rcond=None uses machine precision times max(A.shape) as cutoff for
    small singular values (the current numpy default).

    lstsq returns (solution, residuals, rank, singular values); we only
    need the solution.
    """
    vecW, _, _, _ = la.lstsq(matX.T, vecY, rcond=None)

    return vecW


if __name__ == '__main__':
    # ingredients from eq. (2) on the sheet
    matX = np.array([[ 1.00000, 0.00000, 0.00000],
                     [-1.00000, 0.00001, 0.00000]])
    vecY = np.array( [ 0.00000, 0.00001, 0.00000] )

    np.set_printoptions(precision=17)

    vecW_ne = solve_lsq_normal_equations(matX, vecY)
    vecW_qr = solve_lsq_qr(matX, vecY)
    vecW_ls = solve_lsq_lstsq(matX, vecY)

    print('pen-and-paper solution :', vecW_ne)
    print('QR solution            :', vecW_qr)
    print('lstsq solution         :', vecW_ls)
    print('exact solution         : [1. 1.]')
    print()
    print('residual pen-and-paper :', la.norm(matX.T @ vecW_ne - vecY))
    print('residual QR            :', la.norm(matX.T @ vecW_qr - vecY))
    print('residual lstsq         :', la.norm(matX.T @ vecW_ls - vecY))
