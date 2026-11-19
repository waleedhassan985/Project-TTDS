"""Scikit-learn metrics compatibility layer in pure NumPy."""
import numpy as np


def f1_score(y_true, y_pred, average="macro", zero_division=0):
    yt = (np.asarray(y_true) > 0.5).astype(int)
    yp = (np.asarray(y_pred) > 0.5).astype(int)
    tp1 = int(np.sum((yt == 1) & (yp == 1)))
    fp1 = int(np.sum((yt == 0) & (yp == 1)))
    fn1 = int(np.sum((yt == 1) & (yp == 0)))
    tn1 = int(np.sum((yt == 0) & (yp == 0)))

    denom1 = 2 * tp1 + fp1 + fn1
    f1_pos = (2.0 * tp1 / denom1) if denom1 > 0 else float(zero_division)

    # For class 0: tp0 = tn1, fp0 = fn1, fn0 = fp1
    denom0 = 2 * tn1 + fn1 + fp1
    f1_neg = (2.0 * tn1 / denom0) if denom0 > 0 else float(zero_division)

    if average == "macro":
        return 0.5 * (f1_neg + f1_pos)
    return f1_pos
