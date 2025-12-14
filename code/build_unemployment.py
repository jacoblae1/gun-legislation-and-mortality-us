import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / "data" / "raw" / "unemployment"
data_processed = project_root / "data" / "processed"
data_processed.mkdir(parents=True, exist_ok=True)

old_path = data_raw / "Unemployment_Rates_USA_1990_2018.xlsx"
new_path = data_raw / "BLS_Unemployment_Rates_2019_2020.xlsx"
output_path = data_processed / "Unemployment_Rates_USA_2000_2020_full.xlsx"

df_old = pd.read_excel(old_path)
df_new = pd.read_excel(new_path)

df_old.columns = df_old.columns.astype(str).str.strip()
df_new.columns = df_new.columns.astype(str).str.strip()

df_old.rename(columns={"Area": "State"}, inplace=True)
df_old.drop(columns=["Fips"], inplace=True)

years = [str(y) for y in range(2000, 2019)]
df_old = df_old[["State"] + years]

df_new["State"] = df_new["State"].astype(str).str.strip()

df_combined = df_old.merge(df_new, on="State", how="left")

df_combined.sort_values("State", inplace=True)

df_combined.to_excel(output_path, index=False)

print(f"Unemployment dataset saved to: {output_path}")
