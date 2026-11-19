"""PySpark ML feature transformers fitted strictly on training rows."""
import numpy as np
import pandas as pd
from ..sql import DataFrame


class _ImputerModel:
    def __init__(self, inputCols, outputCols, surrogates):
        self.inputCols = inputCols
        self.outputCols = outputCols
        self.surrogates = surrogates

    def transform(self, dataset):
        pdf = dataset._pdf.copy()
        for inp, out in zip(self.inputCols, self.outputCols):
            pdf[out] = pd.to_numeric(pdf[inp], errors="coerce").fillna(self.surrogates[inp])
        return DataFrame(pdf, dataset._ops + ["ImputerModel"])


class Imputer:
    def __init__(self, inputCols=None, outputCols=None, strategy="median"):
        self.inputCols = inputCols or []
        self.outputCols = outputCols or []
        self.strategy = strategy

    def fit(self, dataset):
        pdf = dataset._pdf
        surrogates = {}
        for c in self.inputCols:
            s = pd.to_numeric(pdf[c], errors="coerce")
            surrogates[c] = float(s.median()) if self.strategy == "median" else float(s.mean())
        return _ImputerModel(self.inputCols, self.outputCols, surrogates)


class _StringIndexerModel:
    def __init__(self, inputCol, outputCol, labels, handleInvalid="keep"):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.labels = labels
        self.handleInvalid = handleInvalid
        self.mapping = {val: float(i) for i, val in enumerate(labels)}
        self.invalid_idx = float(len(labels))

    def transform(self, dataset):
        pdf = dataset._pdf.copy()
        s = pdf[self.inputCol].astype(str)
        pdf[self.outputCol] = s.map(self.mapping).fillna(self.invalid_idx).astype(float)
        pdf.attrs[f"num_cats_{self.outputCol}"] = len(self.labels) + (1 if self.handleInvalid == "keep" else 0)
        return DataFrame(pdf, dataset._ops + [f"StringIndexerModel({self.inputCol})"])


class StringIndexer:
    def __init__(self, inputCol=None, outputCol=None, handleInvalid="keep"):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.handleInvalid = handleInvalid

    def fit(self, dataset):
        s = dataset._pdf[self.inputCol].dropna().astype(str)
        # Spark StringIndexer default is frequencyDesc, tie-breaking alphabetically
        vc = s.value_counts()
        df_vc = pd.DataFrame({"val": vc.index, "cnt": vc.values}).sort_values(
            by=["cnt", "val"], ascending=[False, True]
        )
        labels = list(df_vc["val"])
        return _StringIndexerModel(self.inputCol, self.outputCol, labels, self.handleInvalid)


class _OneHotEncoderModel:
    def __init__(self, inputCol, outputCol, num_categories, dropLast=True):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.num_categories = num_categories
        self.dropLast = dropLast

    def transform(self, dataset):
        pdf = dataset._pdf.copy()
        idx_vals = pdf[self.inputCol].to_numpy().astype(int)
        dim = max(1, self.num_categories - (1 if self.dropLast else 0))
        n = len(pdf)
        mat = np.zeros((n, dim), dtype=float)
        valid = (idx_vals >= 0) & (idx_vals < dim)
        mat[np.arange(n)[valid], idx_vals[valid]] = 1.0
        pdf[self.outputCol] = list(mat)
        return DataFrame(pdf, dataset._ops + [f"OneHotEncoderModel({self.inputCol})"])


class OneHotEncoder:
    def __init__(self, inputCol=None, outputCol=None, dropLast=True):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.dropLast = dropLast

    def fit(self, dataset):
        pdf = dataset._pdf
        n_cats = pdf.attrs.get(f"num_cats_{self.inputCol}")
        if n_cats is None:
            n_cats = int(pdf[self.inputCol].max()) + 2
        return _OneHotEncoderModel(self.inputCol, self.outputCol, n_cats, self.dropLast)


class VectorAssembler:
    def __init__(self, inputCols=None, outputCol=None):
        self.inputCols = inputCols or []
        self.outputCol = outputCol

    def transform(self, dataset):
        pdf = dataset._pdf.copy()
        parts = []
        for c in self.inputCols:
            col_data = pdf[c]
            first = col_data.iloc[0] if len(col_data) > 0 else 0.0
            if isinstance(first, (np.ndarray, list)):
                parts.append(np.vstack(col_data.to_numpy()))
            else:
                parts.append(col_data.to_numpy(dtype=float).reshape(-1, 1))
        assembled = np.hstack(parts) if parts else np.zeros((len(pdf), 0))
        pdf[self.outputCol] = list(assembled)
        return DataFrame(pdf, dataset._ops + [f"VectorAssembler({self.outputCol})"])


class _StandardScalerModel:
    def __init__(self, inputCol, outputCol, mean, std, withMean=True, withStd=True):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.mean = mean
        self.std = std
        self.withMean = withMean
        self.withStd = withStd

    def transform(self, dataset):
        pdf = dataset._pdf.copy()
        mat = np.vstack(pdf[self.inputCol].to_numpy()).astype(float)
        if self.withMean:
            mat = mat - self.mean
        if self.withStd:
            safe_std = np.where(self.std > 0, self.std, 1.0)
            mat = mat / safe_std
        pdf[self.outputCol] = list(mat)
        return DataFrame(pdf, dataset._ops + [f"StandardScalerModel({self.outputCol})"])


class StandardScaler:
    def __init__(self, inputCol=None, outputCol=None, withMean=True, withStd=True):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.withMean = withMean
        self.withStd = withStd

    def fit(self, dataset):
        mat = np.vstack(dataset._pdf[self.inputCol].to_numpy()).astype(float)
        mean = np.mean(mat, axis=0)
        # Spark StandardScaler uses sample standard deviation (ddof=1)
        std = np.std(mat, axis=0, ddof=1)
        return _StandardScalerModel(self.inputCol, self.outputCol, mean, std, self.withMean, self.withStd)
