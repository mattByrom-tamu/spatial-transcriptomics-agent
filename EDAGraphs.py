import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# =============================================================================
# 1) Histogram (Distribution Shape)
# Shows skew, spikes, gaps, rough outliers
plt.figure()
df["age"].dropna().hist()
plt.title("Histogram of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# =============================================================================
# 2) Box Plot (Outliers + Spread)
# Shows median, IQR, extreme outliers
plt.figure()
df[["income"]].dropna().boxplot()
plt.title("Box Plot of Income")
plt.ylabel("Income")
plt.show()

# =============================================================================
# 3) Scatter Plot (Relationships + Anomalies)
# Shows correlation trends, nonlinear patterns, clusters
plt.figure()
plt.scatter(df["age"], df["income"])
plt.title("Income vs Age")
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()

# =============================================================================
# 4) Correlation Matrix (Numeric Relationships)
# Correlation between all numeric columns
corr = df.corr(numeric_only=True)

print("\nCorrelation Matrix:\n")
print(corr)

# Heatmap visualization
plt.figure()
plt.imshow(corr)
plt.title("Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.show()

# =============================================================================
# 5) Correlation Between Two Columns
corr_val = df[["age", "income"]].corr().iloc[0, 1]
print(f"\nCorrelation(age, income) = {corr_val:.3f}")

# =============================================================================
# 6) QQ Plot (Normality Check)
# Points close to line => roughly normal distribution
plt.figure()
stats.probplot(df["income"].dropna(), dist="norm", plot=plt)
plt.title("QQ Plot of Income")
plt.show()

# =============================================================================
# 7) Bar Plot (Categorical Counts)
# Useful for category imbalance + inconsistent labels
counts = df["status"].value_counts(dropna=False)

plt.figure()
counts.plot(kind="bar")
plt.title("Status Counts")
plt.xlabel("Status")
plt.ylabel("Count")
plt.show()