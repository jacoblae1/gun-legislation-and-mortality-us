import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / "data" / "raw" / "law"
data_processed = project_root / "data" / "processed"
data_processed.mkdir(parents=True, exist_ok=True)

input_path = data_raw / "law_data.xlsx"
output_path = data_processed / "RAND_Law_Changes_Counts_2000_2020.xlsx"

df = pd.read_excel(input_path, sheet_name="Database")

df = df[["State", "Effective Date Year", "Effect"]]

df = df.rename(columns={
    "Effective Date Year": "Year"
})

df = df[(df["Year"] >= 2000) & (df["Year"] <= 2020)]

df = df[df["Effect"].isin(["Restrictive", "Permissive"])]

agg = df.groupby(["State", "Year", "Effect"]).size().unstack(fill_value=0)

agg = agg.rename(columns={
    "Restrictive": "Restrictive_Count",
    "Permissive": "Permissive_Count"
})


states = sorted(df["State"].unique())
years = list(range(2000, 2021))
full = pd.MultiIndex.from_product([states, years], names=["State", "Year"]).to_frame(index=False)

out = full.merge(agg.reset_index(), on=["State", "Year"], how="left").fillna(0)
out["Restrictive_Count"] = out["Restrictive_Count"].astype(int)
out["Permissive_Count"] = out["Permissive_Count"].astype(int)

out.to_excel(output_path, index=False)

print(f"Law change counts saved to: {output_path}")
