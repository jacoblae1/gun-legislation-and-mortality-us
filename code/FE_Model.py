import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data" / "processed"

mortality_path = data_processed / "mortality_2000_2020_age_adjusted.csv"
law_path = data_processed / "RAND_Law_Changes_Counts_2000_2020.xlsx"

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

print(f"\nLade Law-Index-Daten von: {law_path}")
law = pd.read_excel(law_path)

law["Year"] = pd.to_numeric(law["Year"], errors="coerce")
law["State"] = law["State"].astype(str)

law = law[["State", "Year", "Restrictive_Count", "Permissive_Count"]]

print("Law-Daten (Counts):", law.shape)
print(law.head())

df = mort.merge(law, on=["State", "Year"], how="inner")

df = df.dropna(subset=["AgeAdj_Rate", "Restrictive_Count", "Permissive_Count"])

print("\nGemergtes DataFrame:", df.shape)
print(df.head())

formula = "AgeAdj_Rate ~ Restrictive_Count + Permissive_Count + C(State) + C(Year)"

print("\nLaufe Fixed-Effects-Regression mit AgeAdj_Rate ...")
model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")

beta_r = model.params["Restrictive_Count"]
se_r = model.bse["Restrictive_Count"]
pval_r = model.pvalues["Restrictive_Count"]

beta_p = model.params["Permissive_Count"]
se_p = model.bse["Permissive_Count"]
pval_p = model.pvalues["Permissive_Count"]

r2 = model.rsquared
nobs = int(model.nobs)

print("\n=== Fixed-effects regression (Outcome: AgeAdj_Rate) ===")
print(f"Number of observations: {nobs}")
print(f"R-squared: {r2:.3f}")
print("Restrictive_Count coefficient:")
print(f"  beta = {beta_r:.3f}, SE = {se_r:.3f}, p-value = {pval_r:.3f}")
print("Permissive_Count coefficient:")
print(f"  beta = {beta_p:.3f}, SE = {se_p:.3f}, p-value = {pval_p:.3f}")

results_path = data_processed / "fe_model_results.txt"
with open(results_path, "w") as f:
    f.write(model.summary().as_text())

print(f"\nFixed-effects regression results saved to: {results_path}")
