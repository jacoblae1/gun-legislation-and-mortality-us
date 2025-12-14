import pandas as pd
from scipy.stats import zscore
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data" / "processed"

mortality_path = data_processed / "mortality_2000_2020_age_adjusted.csv"
income_path = data_processed / "median_income_2000_2020.xlsx"
density_path = data_processed / "population_density_2000_2020.csv"
unemp_path = data_processed / "Unemployment_Rates_USA_2000_2020.xlsx"

mortality = pd.read_csv(mortality_path)
mortality = mortality.rename(columns={"Age Adjusted Rate": "AgeAdj_Rate"})
mortality["Year"] = pd.to_numeric(mortality["Year"], errors="coerce")

income = pd.read_excel(income_path)
income = income.rename(columns={income.columns[0]: "State"})
year_cols_inc = [c for c in income.columns if str(c).isdigit()]
income_long = income.melt(
    id_vars="State",
    value_vars=year_cols_inc,
    var_name="Year",
    value_name="Income"
)
income_long["Year"] = income_long["Year"].astype(int)

unemp = pd.read_excel(unemp_path)
unemp = unemp.rename(columns={unemp.columns[0]: "State"})
year_cols_unemp = [c for c in unemp.columns if str(c).isdigit()]
unemp_long = unemp.melt(
    id_vars="State",
    value_vars=year_cols_unemp,
    var_name="Year",
    value_name="Unemployment"
)
unemp_long["Year"] = unemp_long["Year"].astype(int)

density = pd.read_csv(density_path)
density_cols = [c for c in density.columns if c.startswith("Density_")]

density_long = density.melt(
    id_vars=["State"],
    value_vars=density_cols,
    var_name="Year_raw",
    value_name="Density"
)

density_long["Year"] = density_long["Year_raw"].str.replace("Density_", "", regex=False).astype(int)
density_base = density_long.pivot(index="State", columns="Year", values="Density")

rows = []
for state, row in density_base.iterrows():
    d2000 = row.get(2000)
    d2010 = row.get(2010)
    d2020 = row.get(2020)
    for year in range(2000, 2010):
        rows.append([state, year, d2000])
    for year in range(2010, 2020):
        rows.append([state, year, d2010])
    rows.append([state, 2020, d2020])

density_full = pd.DataFrame(rows, columns=["State", "Year", "Population_Density"])

df = mortality.merge(income_long, on=["State", "Year"], how="inner")
df = df.merge(unemp_long, on=["State", "Year"], how="inner")
df = df.merge(density_full, on=["State", "Year"], how="inner")

for col in ["Income", "Unemployment", "Population_Density", "AgeAdj_Rate"]:
    df[f"z_{col}"] = zscore(df[col].astype(float))

cols = [
    "State", "Year",
    "Income", "Unemployment", "Population_Density", "AgeAdj_Rate",
    "z_Income", "z_Unemployment", "z_Population_Density", "z_AgeAdj_Rate"
]

df_out = df[cols].sort_values(["State", "Year"])

out_path = data_processed / "ses_standardized_2000_2020.csv"
df_out.to_csv(out_path, index=False)

print(f"SES-standardized dataset saved to: {out_path}")
print(df_out.head(20))
