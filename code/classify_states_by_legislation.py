import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data" / "processed"
law_path = data_processed / "RAND_Law_Changes_Counts_2000_2020.xlsx"

df_law = pd.read_excel(law_path)

effects = df_law.groupby("State")[["Restrictive_Count", "Permissive_Count"]].sum()


has_change = effects[(effects["Restrictive_Count"] + effects["Permissive_Count"]) > 0]

mixed_states = has_change[
    (has_change["Restrictive_Count"] > 0) &
    (has_change["Permissive_Count"] > 0)
].index.tolist()

only_restrictive_states = has_change[
    (has_change["Restrictive_Count"] > 0) &
    (has_change["Permissive_Count"] == 0)
].index.tolist()


only_permissive_states = has_change[
    (has_change["Permissive_Count"] > 0) &
    (has_change["Restrictive_Count"] == 0)
].index.tolist()


non_mixed_states = sorted(set(only_restrictive_states + only_permissive_states))

mixed_states = sorted(mixed_states)
only_restrictive_states = sorted(only_restrictive_states)
only_permissive_states = sorted(only_permissive_states)
non_mixed_states = sorted(non_mixed_states)


print("=== Mixed states (restrictive AND permissive reforms) ===")
print(mixed_states)

print("\n=== Non-mixed states (only restrictive OR only permissive) ===")
print(non_mixed_states)

print("\n=== Only restrictive states (0 permissive reforms) ===")
print(only_restrictive_states)

print("\n=== Only permissive states (0 restrictive reforms) ===")
print(only_permissive_states)
