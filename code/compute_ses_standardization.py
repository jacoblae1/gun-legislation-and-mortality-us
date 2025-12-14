import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data" / "processed"

ses_path = data_processed / "ses_standardized_2000_2020.csv"

ses = pd.read_csv(ses_path)

z_cols = ["z_Income", "z_Unemployment", "z_Population_Density", "z_AgeAdj_Rate"]

df = ses[z_cols].dropna()

corr_matrix = df.corr()

print("\n=== SES ↔ Firearm Mortality (Full Correlation Matrix) ===")
print(corr_matrix)

corr_with_mortality = corr_matrix["z_AgeAdj_Rate"].drop("z_AgeAdj_Rate")

print("\n=== SES → Firearm Mortality (Correlations with z_AgeAdj_Rate) ===")
print(corr_with_mortality)

print("\n=== Interpretation ===")
for var, r in corr_with_mortality.items():
    print(f"{var} vs z_AgeAdj_Rate: r = {r:.3f}")
