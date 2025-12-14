import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data" / "processed"

mortality_path = data_processed / "mortality_2000_2020_age_adjusted.csv"

print(f"Lade Mortality-Daten von: {mortality_path}")
mort = pd.read_csv(mortality_path)

mort = mort.rename(columns={
    "Age Adjusted Rate": "AgeAdj_Rate"
})

mort["Year"] = pd.to_numeric(mort["Year"], errors="coerce")
mort["State"] = mort["State"].astype(str)

mort = mort[(mort["Year"] >= 2000) & (mort["Year"] <= 2020)]

mort = mort[["State", "Year", "AgeAdj_Rate"]]

print("Mortality-Daten nach Filterung:", mort.shape)
print(mort.head())

restrictive_states = [
    "Hawaii", "Rhode Island", "New York", "New Jersey", "California",
    "Delaware", "Vermont", "Maryland", "Oregon"
]

permissive_states = [
    "Arkansas", "Georgia", "Idaho", "Mississippi", "Montana",
    "North Dakota", "Oklahoma"
]

def assign_group(state):
    if state in restrictive_states:
        return "always_restrictive"
    elif state in permissive_states:
        return "always_permissive"
    else:
        return "excluded"

mort["LegGroup"] = mort["State"].apply(assign_group)

df_compare = mort[mort["LegGroup"].isin(["always_restrictive", "always_permissive"])].copy()

df_compare = df_compare.dropna(subset=["AgeAdj_Rate", "Year"])

df_compare["Restrictive"] = (df_compare["LegGroup"] == "always_restrictive").astype(int)

print("\nDaten für Regime-Modell:", df_compare.shape)
print(df_compare.head())

formula = "AgeAdj_Rate ~ Restrictive + C(Year)"

print("\nLaufe Policy-Regime-Regression mit AgeAdj_Rate ...")
model = smf.ols(formula=formula, data=df_compare).fit(cov_type="HC3")

coef = model.params.get("Restrictive")
pval = model.pvalues.get("Restrictive")
ci_low, ci_high = model.conf_int().loc["Restrictive"]
r2 = model.rsquared
nobs = int(model.nobs)

print("\n=== Policy Regime Model (Outcome: AgeAdj_Rate) ===")
print(f"Restrictive coefficient: {coef:.3f}")
print(f"p-value: {pval:.3f}")
print(f"95% confidence interval: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"R-squared: {r2:.3f}")
print(f"Number of observations: {nobs}")

results_path = data_processed / "policy_groups_model_results_age_adjusted_only.txt"
with open(results_path, "w") as f:
    f.write(model.summary().as_text())

print(f"\nPolicy Regime Model results saved to: {results_path}")

group_year_means = (
    df_compare
    .groupby(["Year", "LegGroup"])["AgeAdj_Rate"]
    .mean()
    .unstack("LegGroup")
    .sort_index()
)

means_path = data_processed / "policy_groups_means_by_year_age_adjusted.csv"
group_year_means.to_csv(means_path)

print(f"Yearly group means saved to: {means_path}")
