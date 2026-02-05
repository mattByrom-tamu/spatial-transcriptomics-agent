# EDA

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import phik

# Start
# read in data 
df = pd.read_csv('dataTemp.csv')
df.head()
df.tail()
df.info() # data types column names, non null count
df.describe() # summary statistics

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:                                  # Common patterns:
    """Return missing count and percent missing by column."""                           # df = df.dropna(subset=["target"])                 # drop rows missing key field
    missing_count = df.isna().sum()                                                     # df["age"] = df["age"].fillna(df["age"].median())  # numeric impute with median
    missing_pct = df.isna().mean() * 100                                                # df["city"] = df["city"].fillna("unknown")         # categorical impute
    out = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    return out.sort_values("missing_pct", ascending=False)

def duplicate_summary(df: pd.DataFrame, subset=None) -> int:                            # common patterns:
    """                                                                                 
    Count duplicates.
    - subset=None checks full-row duplicates
    - subset=["id", "date"] checks duplicates on key columns
    """
    return int(df.duplicated(subset=subset).sum())                                      # df = df.drop_duplicates()                         # drop full-row duplicates 
                                                                                        # df = df.drop_duplicates(subset=["id", "date"])    # drop duplicates on keys

def to_numeric_safe(series: pd.Series) -> pd.Series:                                    # df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    """Convert to numeric; invalid parses become NaN."""                                # df["date"]   = pd.to_datetime(df["date"], errors="coerce")
    return pd.to_numeric(series, errors="coerce")                                       # df = df.dropna(subset=["amount", "date"])  # drop rows where parse failed

def to_datetime_safe(series: pd.Series) -> pd.Series:
    """Convert to datetime; invalid parses become NaT."""
    return pd.to_datetime(series, errors="coerce")

def normalize_category(series: pd.Series) -> pd.Series:                                 # df["status"] = df["status"].astype("string").str.lower().str.strip()
    """Lowercase + trim whitespace; safe for NaNs."""                                   # df["status"] = df["status"].replace({"APPROVED": "approved", "approved ": "approved"})
    return series.astype("string").str.lower().str.strip()                              # df["status"].value_counts(dropna=False)  # inspect category distribution

# filtering df between values
# df[df['Make'].between(low, high)] 
# df[df['Make'] > 0]

# groupby examples
df.groupby("Make")["Popularity"].mean()                                                 #single metric group
df.groupby("Make").size()                                                               # counts per group
df.groupby("Make")["Popularity"].agg(["mean", "sum", "count"])                          # multiple metrics for group 

df = df.sort_values("Popularity", ascending=False)
top5 = df.nlargest(5, "Popularity")
bot5 = df.nsmallest(5, "Popularity")

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "inner") -> pd.DataFrame:
    """
    Merge helper. After merge, you should validate row counts and missingness
    to ensure the join behaved as expected.
    """
    merged = left.merge(right, on=on, how=how)
    return merged

# Merges
df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
                    'value': [1, 2, 3, 5]})
df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
                    'value': [5, 6, 7, 8]})
df1.merge(df2, left_on='lkey', right_on='rkey')



# missing count
def missing_report(df):
    missing_count = df.isna().sum()
    missing_percent = df.isna().mean()*100

    result = pd.DataFrame({
        "column": missing_count.index,
        "missing_count": missing_count.values,
        "missing_percent": missing_percent.values
    })
    result = result.sort_values("missing_percent", ascending = False)
    return result
