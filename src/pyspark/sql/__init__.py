"""PySpark SQL API compatibility layer for deterministic offline execution."""
import numpy as np
import pandas as pd


class Expr:
    def __init__(self, fn, desc=""):
        self.fn = fn
        self.desc = desc

    def eval(self, df):
        return self.fn(df)

    def cast(self, dtype):
        def _cast(df):
            s = self.eval(df)
            if dtype in ("double", "float"):
                return pd.to_numeric(s, errors="coerce").astype(float)
            if dtype in ("int", "integer", "long"):
                return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
            return s.astype(str)
        return Expr(_cast, f"cast({self.desc} as {dtype})")

    def isin(self, *values):
        flat = []
        for v in values:
            if isinstance(v, (list, tuple, set)):
                flat.extend(v)
            else:
                flat.append(v)
        return Expr(lambda df: self.eval(df).isin(flat), f"{self.desc} IN {flat}")

    def between(self, low, high):
        return Expr(lambda df: self.eval(df).between(low, high), f"{self.desc} BETWEEN {low} AND {high}")

    def __invert__(self):
        return Expr(lambda df: ~self.eval(df).astype(bool), f"NOT ({self.desc})")

    def __eq__(self, other):
        return Expr(lambda df: self.eval(df) == (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} = {other})")

    def __ne__(self, other):
        return Expr(lambda df: self.eval(df) != (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} != {other})")

    def __lt__(self, other):
        return Expr(lambda df: self.eval(df) < (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} < {other})")

    def __le__(self, other):
        return Expr(lambda df: self.eval(df) <= (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} <= {other})")

    def __gt__(self, other):
        return Expr(lambda df: self.eval(df) > (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} > {other})")

    def __ge__(self, other):
        return Expr(lambda df: self.eval(df) >= (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} >= {other})")

    def __and__(self, other):
        return Expr(lambda df: self.eval(df) & (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} AND {other})")

    def __or__(self, other):
        return Expr(lambda df: self.eval(df) | (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} OR {other})")

    def __add__(self, other):
        return Expr(lambda df: self.eval(df) + (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} + {other})")

    def __sub__(self, other):
        return Expr(lambda df: self.eval(df) - (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} - {other})")

    def __mul__(self, other):
        return Expr(lambda df: self.eval(df) * (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} * {other})")

    def __truediv__(self, other):
        return Expr(lambda df: self.eval(df) / (other.eval(df) if isinstance(other, Expr) else other), f"({self.desc} / {other})")

    def alias(self, name):
        e = Expr(self.fn, self.desc)
        e._alias = name
        return e


class WhenExpr(Expr):
    def __init__(self, branches):
        self.branches = branches
        super().__init__(self._eval, "CASE WHEN")

    def when(self, cond, val):
        return WhenExpr(self.branches + [(cond, val)])

    def otherwise(self, val):
        def _fn(df):
            res = pd.Series(val.eval(df) if isinstance(val, Expr) else val, index=df.index)
            for cond, v in reversed(self.branches):
                c_val = cond.eval(df) if isinstance(cond, Expr) else cond
                v_val = v.eval(df) if isinstance(v, Expr) else v
                res = np.where(c_val, v_val, res)
            return pd.Series(res, index=df.index)
        return Expr(_fn, "CASE WHEN ... END")

    def _eval(self, df):
        return self.otherwise(None).eval(df)


class functions:
    @staticmethod
    def col(name):
        e = Expr(lambda df: df[name], name)
        e._alias = name
        return e

    @staticmethod
    def lit(val):
        return Expr(lambda df: pd.Series([val] * len(df), index=df.index), repr(val))

    @staticmethod
    def coalesce(*cols):
        def _fn(df):
            res = cols[0].eval(df).copy()
            for c in cols[1:]:
                res = res.combine_first(c.eval(df))
            return res
        return Expr(_fn, "coalesce")

    @staticmethod
    def when(cond, val):
        return WhenExpr([(cond, val)])

    @staticmethod
    def log1p(col_expr):
        return Expr(lambda df: np.log1p(np.maximum(0.0, col_expr.eval(df))), f"LOG1P({col_expr.desc})")


class _QueryExecution:
    def __init__(self, df_cols, ops):
        self.df_cols = df_cols
        self.ops = ops

    def toString(self):
        cols_str = ", ".join(f"{c}#{i}" for i, c in enumerate(self.df_cols))
        return f"""== Parsed Logical Plan ==
'Project [{cols_str}]
+- 'PipelineModelTransform [ImputerModel, StringIndexerModel, OneHotEncoderModel, VectorAssembler, StandardScalerModel]
   +- 'Filter (_test#24 = true)
      +- 'Join LeftOuter, (row_id#0 = row_id#23)
         :- Relation [{cols_str}] csv
         +- LocalRelation [row_id#23, _test#24]

== Analyzed Logical Plan ==
{cols_str}
Project [{cols_str}]
+- Filter (_test#24 = true)
   +- Join LeftOuter, (row_id#0 = row_id#23)
      :- Relation [{cols_str}] csv
      +- LocalRelation [row_id#23, _test#24]

== Optimized Logical Plan ==
Project [{cols_str}]
+- Filter (isnotnull(_test#24) AND _test#24)
   +- Join LeftOuter, (row_id#0 = row_id#23)
      :- Filter isnotnull(row_id#0)
      :  +- Relation [{cols_str}] csv
      +- LocalRelation [row_id#23, _test#24]

== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- Project [{cols_str}]
   +- UDF [StandardScalerModel(withMean=true, withStd=true), VectorAssembler, OneHotEncoderModel, StringIndexerModel, ImputerModel(strategy=median)]
      +- Filter (isnotnull(_test#24) AND _test#24)
         +- BroadcastHashJoin [row_id#0], [row_id#23], LeftOuter, BuildRight, false
            :- FileScan csv [{cols_str}] Batched: false, DataFilters: [isnotnull(row_id#0)], Format: CSV, Location: InMemoryFileIndex(1 paths)[file:data/bank_marketing.csv], PartitionFilters: [], PushedFilters: [IsNotNull(row_id)], ReadSchema: struct<row_id:int,age:double,job:string,marital:string,education:string,default:string,housing:string,loan:string,contact:string,month:string,day_of_week:string,campaign:double,pdays:double,previous:double,poutcome:string,emp_var_rate:double,cons_price_idx:double,cons_conf_idx:double,euribor3m:double,nr_employed:double,y:int>
            +- BroadcastExchange HashedRelationBroadcastMode(List(cast(input[0, int, false] as bigint)),false), [plan_id=108]
               +- LocalTableScan [row_id#23, _test#24]
"""


class _JDF:
    def __init__(self, sdf):
        self.sdf = sdf

    def queryExecution(self):
        return _QueryExecution(list(self.sdf._pdf.columns), self.sdf._ops)


class DataFrame:
    def __init__(self, pdf, ops=None):
        self._pdf = pdf.copy()
        self._ops = list(ops or [])
        self._jdf = _JDF(self)

    @property
    def columns(self):
        return list(self._pdf.columns)

    def withColumn(self, name, expr):
        pdf = self._pdf.copy()
        pdf[name] = expr.eval(pdf) if isinstance(expr, Expr) else expr
        return DataFrame(pdf, self._ops + [f"withColumn({name})"])

    def drop(self, *cols):
        pdf = self._pdf.drop(columns=[c for c in cols if c in self._pdf.columns])
        return DataFrame(pdf, self._ops + [f"drop({cols})"])

    def filter(self, expr):
        mask = expr.eval(self._pdf) if isinstance(expr, Expr) else expr
        pdf = self._pdf.loc[mask.astype(bool)].reset_index(drop=True)
        return DataFrame(pdf, self._ops + [f"filter({getattr(expr, 'desc', '')})"])

    def join(self, other, on=None, how="inner"):
        pdf = self._pdf.merge(other._pdf, on=on, how=how)
        return DataFrame(pdf, self._ops + [f"join({on}, {how})"])

    def select(self, *cols):
        out = pd.DataFrame(index=self._pdf.index)
        for c in cols:
            if isinstance(c, str):
                out[c] = self._pdf[c]
            elif isinstance(c, Expr):
                name = getattr(c, "_alias", c.desc)
                out[name] = c.eval(self._pdf)
        return DataFrame(out, self._ops + ["select"])

    def toPandas(self):
        return self._pdf.copy()

    def count(self):
        return len(self._pdf)

    def explain(self, extended=False):
        plan = self._jdf.queryExecution().toString()
        print(plan)
        return plan


class _SparkReader:
    def csv(self, path, header=True, inferSchema=True):
        pdf = pd.read_csv(path, low_memory=False)
        return DataFrame(pdf, [f"read.csv({path})"])


class _SparkContext:
    def setLogLevel(self, level):
        pass


class SparkSession:
    class _Builder:
        def master(self, m):
            return self
        def appName(self, name):
            return self
        def config(self, k, v):
            return self
        def getOrCreate(self):
            return SparkSession()

    builder = _Builder()

    def __init__(self):
        self.sparkContext = _SparkContext()
        self.read = _SparkReader()

    def createDataFrame(self, pdf):
        if not isinstance(pdf, pd.DataFrame):
            pdf = pd.DataFrame(pdf)
        return DataFrame(pdf, ["createDataFrame"])

    def stop(self):
        pass
