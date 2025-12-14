import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / "data" / "raw" / "population"
data_processed = project_root / "data" / "processed"
data_processed.mkdir(parents=True, exist_ok=True)

area_path = data_raw / "2025_Gaz_state_national.txt"
pop_path = data_raw / "Population 2000-2020.xlsx"

area = pd.read_csv(area_path, sep="|", engine="python")
area = area[["NAME", "ALAND_SQMI"]].copy()
area["Area_km2"] = area["ALAND_SQMI"] * 2.58999
area["NAME_norm"] = area["NAME"].astype(str).str.strip().str.upper()

pop = pd.read_excel(pop_path)
pop.columns = pop.columns.astype(str).str.strip()

if "State" not in pop.columns:
    first_col = pop.columns[0]
    pop = pop.rename(columns={first_col: "State"})

pop["State_norm"] = pop["State"].astype(str).str.strip().str.upper()

merged = pd.merge(
    pop,
    area[["NAME_norm", "Area_km2"]],
    left_on="State_norm",
    right_on="NAME_norm",
    how="left"
)

years_available = [
    c for c in pop.columns
    if c.isdigit() and 2000 <= int(c) <= 2020
]

for year in years_available:
    merged[f"Density_{year}"] = merged[year].astype(float) / merged["Area_km2"]

cols = ["State", "Area_km2"] + [f"Density_{y}" for y in years_available]

out_path = data_processed / "population_density_2000_2020.csv"
merged[cols].to_csv(out_path, index=False)

print(f"Population density dataset saved to: {out_path}")
