"""PySpark ML Pipeline API compatibility layer."""
import numpy as np
import pandas as pd


class PipelineModel:
    def __init__(self, models):
        self.stages = models

    def transform(self, dataset):
        df = dataset
        for m in self.stages:
            df = m.transform(df)
        return df


class Pipeline:
    def __init__(self, stages=None):
        self.stages = stages or []

    def fit(self, dataset):
        models = []
        df = dataset
        for stage in self.stages:
            if hasattr(stage, "fit"):
                m = stage.fit(df)
                models.append(m)
                df = m.transform(df)
            else:
                models.append(stage)
                df = stage.transform(df)
        return PipelineModel(models)
