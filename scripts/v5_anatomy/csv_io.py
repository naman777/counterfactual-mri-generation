"""Preserve literal control labels such as 'null' while retaining empty fields."""
import pandas as pd


def read_table(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, na_values=[''], **kwargs)
