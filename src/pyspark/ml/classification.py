"""PySpark MLlib LogisticRegression using exact Newton-Raphson (IRLS) on the L2-regularized average log-loss."""
import numpy as np
from ..sql import DataFrame


class _LogisticRegressionModel:
    def __init__(self, w, b, featuresCol, predictionCol="prediction"):
        self.w = w
        self.b = b
        self.featuresCol = featuresCol
        self.predictionCol = predictionCol

    def transform(self, dataset):
        pdf = dataset._pdf.copy()
        X = np.vstack(pdf[self.featuresCol].to_numpy()).astype(float)
        logits = X @ self.w + self.b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))
        pdf[self.predictionCol] = (probs >= 0.5).astype(float)
        return DataFrame(pdf, dataset._ops + ["LogisticRegressionModel"])


class LogisticRegression:
    def __init__(self, featuresCol="features", labelCol="label", regParam=0.01, maxIter=100):
        self.featuresCol = featuresCol
        self.labelCol = labelCol
        self.regParam = regParam
        self.maxIter = maxIter

    def fit(self, dataset):
        pdf = dataset._pdf
        X = np.vstack(pdf[self.featuresCol].to_numpy()).astype(float)
        y = (pdf[self.labelCol].to_numpy() > 0.5).astype(float)
        n, d = X.shape
        # Augment X with intercept column (intercept is unpenalized in Spark MLlib)
        X_aug = np.hstack([X, np.ones((n, 1), dtype=float)])
        theta = np.zeros(d + 1, dtype=float)
        p_bar = np.clip(np.mean(y), 1e-5, 1.0 - 1e-5)
        theta[-1] = np.log(p_bar / (1.0 - p_bar))

        reg_diag = np.full(d + 1, max(1e-6, self.regParam), dtype=float)
        reg_diag[-1] = 1e-9  # do not regularize intercept

        for _ in range(min(self.maxIter, 25)):
            z = np.clip(X_aug @ theta, -25.0, 25.0)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = (X_aug.T @ (p - y)) / n + reg_diag * theta
            w_vec = np.maximum(p * (1.0 - p), 1e-6) / n
            H = (X_aug.T * w_vec) @ X_aug + np.diag(reg_diag)
            step = np.linalg.solve(H, grad)
            theta -= step
            if np.max(np.abs(step)) < 1e-7:
                break

        return _LogisticRegressionModel(theta[:-1], float(theta[-1]), self.featuresCol)
