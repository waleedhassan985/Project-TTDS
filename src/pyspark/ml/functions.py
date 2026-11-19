"""PySpark ML functions."""
from ..sql import Expr


def vector_to_array(col_name):
    e = Expr(lambda df: df[col_name], f"vector_to_array({col_name})")
    e._alias = col_name
    return e
