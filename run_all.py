"""
TTDS End Semester Project — Bank Marketing Pipeline on Apache Spark.

    python run_all.py --demo     build a small synthetic dataset and run on it
    python run_all.py            run on data/bank_marketing.csv named in project.json

Writes, every time, from scratch:
    results/train_ids.csv, results/test_ids.csv   the 75/25 split
    results/spark_plan.txt                        Spark's physical plan for the prepared test rows
    results/metrics.csv                           one row per model, on f1_macro with 95% bootstrap CIs
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# On Windows, Spark must be told which Python to start for its workers.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import pyspark  # noqa: F401
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(HERE, "src"))

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import vector_to_array
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

SEED = 808                      # every random choice below uses this fixed seed
TEST_SHARE = 0.25               # 25% test share (between 10% and 40% required by G3)
BOOTSTRAP = 1000                # 1,000 bootstrap resamples for 95% CIs

os.chdir(HERE)
CFG = json.load(open("project.json", encoding="utf-8"))
METRIC = CFG["primary_metric"]
TARGET = CFG["target"]
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------- data
def make_demo(path, n=5500):
    """Fallback synthetic demo table if invoked with --demo."""
    rng = np.random.default_rng(SEED)
    d = pd.DataFrame({
        "age": rng.integers(18, 85, n).astype(float),
        "job": rng.choice(["admin.", "blue-collar", "technician", "services", "management", "retired"], n),
        "marital": rng.choice(["married", "single", "divorced"], n),
        "education": rng.choice(["university.degree", "high.school", "basic.9y", "professional.course"], n),
        "default": rng.choice(["no", "unknown"], n, p=[0.8, 0.2]),
        "housing": rng.choice(["yes", "no", "unknown"], n, p=[0.52, 0.45, 0.03]),
        "loan": rng.choice(["no", "yes", "unknown"], n, p=[0.82, 0.15, 0.03]),
        "contact": rng.choice(["cellular", "telephone"], n, p=[0.64, 0.36]),
        "month": rng.choice(["may", "jul", "aug", "jun", "nov", "apr"], n),
        "day_of_week": rng.choice(["mon", "thu", "wed", "tue", "fri"], n),
        "duration": rng.gamma(2.5, 110, n).round(),
        "campaign": rng.integers(1, 15, n).astype(float),
        "pdays": rng.choice([999.0, 6.0, 3.0], n, p=[0.96, 0.02, 0.02]),
        "previous": rng.choice([0.0, 1.0, 2.0], n, p=[0.86, 0.11, 0.03]),
        "poutcome": rng.choice(["nonexistent", "failure", "success"], n, p=[0.86, 0.11, 0.03]),
        "emp_var_rate": rng.choice([1.1, 1.4, -1.8, -1.1], n),
        "cons_price_idx": rng.normal(93.58, 0.58, n).round(3),
        "cons_conf_idx": rng.normal(-40.5, 4.6, n).round(1),
        "euribor3m": rng.choice([4.857, 4.961, 1.405, 1.028], n),
        "nr_employed": rng.choice([5191.0, 5228.1, 5099.1, 4963.6], n),
    })
    z = -2.2 + 0.6 * (d.contact == "cellular") + 1.8 * (d.poutcome == "success") - 0.35 * (d.euribor3m - 3.0)
    d[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    d.insert(0, "row_id", np.arange(1, n + 1))
    d.to_csv(path, index=False)


DATA = CFG["dataset"]["file"]
if "--demo" in sys.argv or not os.path.exists(DATA):
    if "--demo" not in sys.argv:
        sys.exit(f"{DATA} not found. Put your data there, or run with --demo.")
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    make_demo(DATA)

# Columns by kind. Note on T4 (Data Leakage):
# `duration` (last call duration in seconds) is present in the raw UCI dataset
# but is strictly excluded from NUMERIC features below because call duration is
# not known before placing a telemarketing call and leaks the target outcome.
ID = CFG.get("id_column", "row_id")
LEAKAGE_COLS = ["duration"]
NUMERIC = [
    "age", "campaign", "pdays", "previous",
    "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"
]
CATEGORICAL = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "day_of_week", "poutcome"
]

# ---------------------------------------------------------------- split
# Deterministic split from row_id and fixed SEED so train/test sets are identical
# across operating systems and CPU core counts.
ids = pd.read_csv(DATA, usecols=[ID])[ID].to_numpy()
rng = np.random.default_rng(SEED)
test_mask = rng.random(len(ids)) < TEST_SHARE
pd.Series(ids[~test_mask], name=ID).to_csv("results/train_ids.csv", index=False)
pd.Series(ids[test_mask], name=ID).to_csv("results/test_ids.csv", index=False)

# ---------------------------------------------------------------- Spark
spark = (SparkSession.builder.master("local[*]").appName("ttds-bank-marketing")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(DATA, header=True, inferSchema=True)

# Remove target-leakage column (`duration`) before modeling (T4)
df = df.drop(*LEAKAGE_COLS)

test_ids = spark.createDataFrame(pd.DataFrame({ID: ids[test_mask].tolist()}))
df = df.join(test_ids.withColumn("_test", F.lit(True)), on=ID, how="left") \
       .withColumn("_test", F.coalesce(F.col("_test"), F.lit(False)))
for c in NUMERIC:
    df = df.withColumn(c, F.col(c).cast("double"))
df = df.withColumn(
    "label",
    F.when(F.col(TARGET).cast("string").isin("yes", "1", "true"), 1.0).otherwise(0.0).cast("double")
)

train = df.filter(~F.col("_test"))
test = df.filter(F.col("_test"))

# Technical Check T3: Every stage that learns parameters from data (Imputer medians,
# StringIndexer category vocabularies, OneHotEncoder dimensions, StandardScaler
# means and standard deviations) is inside this Pipeline and fitted on `train` ONLY.
imputed = [c + "_imp" for c in NUMERIC]
stages = [Imputer(inputCols=NUMERIC, outputCols=imputed, strategy="median")]
for c in CATEGORICAL:
    stages += [
        StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep"),
        OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_oh")
    ]
stages += [
    VectorAssembler(inputCols=imputed + [c + "_oh" for c in CATEGORICAL], outputCol="raw_features"),
    StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
]
prep = Pipeline(stages=stages).fit(train)
train_p = prep.transform(train)
test_p = prep.transform(test)

with open("results/spark_plan.txt", "w", encoding="utf-8") as fh:
    fh.write(test_p._jdf.queryExecution().toString())


# ---------------------------------------------------------------- models
def to_numpy(sdf):
    p = sdf.select(ID, vector_to_array("features").alias("x"), "label").toPandas()
    p = p.sort_values(ID)
    return p[ID].to_numpy(), np.vstack(p["x"].to_numpy()), p["label"].to_numpy()


te_id, Xte, yte = to_numpy(test_p)
tr_id, Xtr, ytr = to_numpy(train_p)

predictions = {}

# Model 1: Spark MLlib Logistic Regression (Linear family, engine = spark)
lr = LogisticRegression(featuresCol="features", labelCol="label", regParam=0.01, maxIter=100).fit(train_p)
p = lr.transform(test_p).select(ID, "prediction").toPandas().sort_values(ID)
predictions["spark_logistic"] = ("linear", "spark", p["prediction"].to_numpy())

# Model 2: Scikit-Learn Histogram-Based Gradient Boosting (Tree ensemble family, engine = sklearn)
gb = HistGradientBoostingClassifier(
    max_iter=80, learning_rate=0.15, max_leaf_nodes=31, min_samples_leaf=20, random_state=SEED
).fit(Xtr, ytr)
predictions["sklearn_boosting"] = ("tree ensemble", "sklearn", gb.predict(Xte))

# Baseline: Majority class predictor on training rows (family = baseline, engine = other)
majority = float(pd.Series(ytr).mode()[0])
predictions["majority_class"] = ("baseline", "other", np.full(len(yte), majority))


# ---------------------------------------------------------------- metric & bootstrap
def score(y, yhat):
    return f1_score(y, yhat, average="macro", zero_division=0)


def bootstrap(y, yhat, b=BOOTSTRAP):
    r = np.random.default_rng(SEED)
    n = len(y)
    vals = [score(y[idx], yhat[idx]) for idx in (r.integers(0, n, n) for _ in range(b))]
    return np.percentile(vals, [2.5, 97.5])


rows = []
for name, (family, engine, yhat) in predictions.items():
    lo, hi = bootstrap(yte, yhat)
    rows.append({
        "model": name,
        "family": family,
        "engine": engine,
        "metric": METRIC,
        "value": round(score(yte, yhat), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4)
    })
met = pd.DataFrame(rows)
met.to_csv("results/metrics.csv", index=False)
print(met.to_string(index=False))
spark.stop()
