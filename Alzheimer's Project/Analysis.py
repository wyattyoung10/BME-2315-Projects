import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import linregress
import numpy as np
from scipy.stats import f_oneway
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind

print("----------------------------------------------------------------")
# Load the two CSV files
luminex = pd.read_csv("UpdatedLuminex.csv")
metadata = pd.read_csv("UpdatedMetaData.csv")

# Merge on Donor ID
merged = pd.merge(luminex, metadata, on="Donor ID")

#Adds a new columns of ABeta40 divided by Abeta42
merged["ABeta_ratio"] = merged["ABeta40 pg/ug"] / merged["ABeta42 pg/ug"]


# --- Define groups ---
# Group 1: APOE ε4 carriers (any genotype containing "4")
group1 = merged[merged["APOE Genotype"].str.contains("4")]["ABeta_ratio"]

# Group 2: Non-carriers (no "4" in genotype)
group2 = merged[~merged["APOE Genotype"].str.contains("4")]["ABeta_ratio"]

# --- Descriptive statistics ---
print("Mean ratio (APOE ε4 carriers):", group1.mean())
print("Mean ratio (Non-carriers):", group2.mean())

# --- Bar plot (2 bars) ---
means = [group1.mean(), group2.mean()]
labels = ["APOE ε4 carriers", "Non-carriers"]

plt.bar(labels, means, color=["skyblue", "lightgreen"])
plt.ylabel("Aβ40/42 Ratio")
plt.title("Comparison of Aβ40/42 ratio by APOE group")
plt.show()

# --- T-test ---
t_stat, p_val = ttest_ind(group1, group2, nan_policy="omit")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# --- Interpretation ---
if p_val < 0.05:
    print("Result: Statistically significant difference (p < 0.05)")
else:
    print("Result: No significant difference (p >= 0.05)")

print("----------------------------------------------------------------")
# --- Scatterplot with regression line and R^2 ------------------------
# Pick variables: pTau vs MMSE
x = merged["pTAU pg/ug"]
y = merged["Last MMSE Score"]

# Remove missing values
mask = ~x.isna() & ~y.isna()
x = x[mask]
y = y[mask]

# --- Linear regression using scipy ---
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# --- Scatterplot ---
plt.scatter(x, y, alpha=0.7, label="Donors")

# Regression line
plt.plot(x, slope*x + intercept, color="red", label=f"y={slope:.2f}x+{intercept:.2f}")

plt.xlabel("pTau (pg/ug)")
plt.ylabel("Last MMSE Score")
plt.title("Scatterplot: pTau vs MMSE with Regression Line")
plt.legend()
plt.show()

# --- Stats ---
print("Slope:", slope)
print("Intercept:", intercept)
print("R-squared:", r_value**2)
print("Correlation coefficient (r):", r_value)
print("p-value:", p_value)


# New bar graph comparing Homozygous carriers to heterozygous carriers to Non-carriers
# === APOE4 three-group analysis (robust parsing of both alleles) ===
# This section supersedes any earlier classification that simply counted "4".
# It parses both alleles explicitly (first number + month tag).

# Map month label to allele number
month_to_num = {"Feb": 2, "Mar": 3, "Apr": 4}

def parse_apoe_label(label):
    """Return (allele1, allele2) as ints from labels like '4-Mar', '3-Feb', '4-Apr'."""
    if pd.isna(label):
        return (pd.NA, pd.NA)
    try:
        first, month = str(label).split("-")
        a1 = int(first)
        a2 = month_to_num.get(month, pd.NA)
        return (a1, a2)
    except Exception:
        # Fallback: extract any digits present
        s = str(label)
        digits = [int(ch) for ch in s if ch.isdigit()]
        if len(digits) == 2:
            return (digits[0], digits[1])
        elif len(digits) == 1:
            return (digits[0], pd.NA)
        return (pd.NA, pd.NA)

# Extract alleles and count ε4 copies
alleles = merged["APOE Genotype"].apply(parse_apoe_label)
merged[["Allele1", "Allele2"]] = pd.DataFrame(alleles.tolist(), index=merged.index)

merged["APOE4_copies"] = merged[["Allele1", "Allele2"]].apply(lambda row: (row == 4).sum(), axis=1)

def classify_from_copies(c):
    if pd.isna(c):
        return pd.NA
    if c == 0:
        return "Non-carrier"
    elif c == 1:
        return "Heterozygous"
    elif c == 2:
        return "Homozygous"
    return pd.NA

merged["APOE4_status"] = merged["APOE4_copies"].apply(classify_from_copies)

# Show counts
print("\nAPOE genotype counts:")
print(merged["APOE Genotype"].value_counts())
print("\nAPOE4_status counts:")
print(merged["APOE4_status"].value_counts())

# Build groups
non_carriers  = merged.loc[merged["APOE4_status"] == "Non-carrier", "ABeta_ratio"].dropna()
heterozygotes = merged.loc[merged["APOE4_status"] == "Heterozygous", "ABeta_ratio"].dropna()
homozygotes   = merged.loc[merged["APOE4_status"] == "Homozygous", "ABeta_ratio"].dropna()

groups = [non_carriers, heterozygotes, homozygotes]
labels = ["Non-carrier", "Het ε4", "Hom ε4"]
non_empty = [(lab, g) for lab, g in zip(labels, groups) if len(g) > 0]

print("\nGroup sizes:", {lab: len(g) for lab, g in non_empty})

# --- ANOVA p-value only ---
anova_p = np.nan
if len(non_empty) >= 3:
    _, anova_p = f_oneway(*(g for _, g in non_empty))
    print(f"ANOVA p-value: {anova_p:.6g}")
else:
    print("ANOVA p-value: N/A (need ≥3 non-empty groups)")

# Plot only the groups that exist
plt.figure()
plt.boxplot([g for _, g in non_empty], labels=[lab for lab, _ in non_empty])
plt.ylabel("Aβ40/42 Ratio")
title = "Comparison of Aβ40/42 ratio by APOE genotype"
if np.isfinite(anova_p):
    title += f" (ANOVA p={anova_p:.3g})"
plt.title(title)
plt.show()

# Print anova statistics
f_stat, p_val = f_oneway(*(g for _, g in non_empty))
print("\nANOVA (≥3 groups)")
print("F-statistic:", f_stat)
print("p-value:", p_val)

# --- Tukey HSD post-hoc ---
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Flatten data for Tukey
data = np.concatenate([np.asarray(g) for _, g in non_empty])
group_labels = np.concatenate([[lab]*len(g) for lab, g in non_empty])

tukey = pairwise_tukeyhsd(data, group_labels, alpha=0.05)

print("\nTukey HSD Post-hoc Results (α = 0.05):")
print(tukey.summary())

