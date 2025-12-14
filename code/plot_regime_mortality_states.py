import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data" / "processed"
fig_dir = project_root / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

mortality_path = data_processed / "mortality_2000_2020_age_adjusted.csv"

mort = pd.read_csv(mortality_path)
mort = mort.rename(columns={"Age Adjusted Rate": "AgeAdj_Rate"})
mort["Year"] = pd.to_numeric(mort["Year"], errors="coerce")
mort = mort[(mort["Year"] >= 2000) & (mort["Year"] <= 2020)]

always_restrictive_states = [
    "California", "Delaware", "Hawaii", "Maryland",
    "New Jersey", "New York", "Oregon", "Rhode Island", "Vermont"
]

always_permissive_states = [
    "Arkansas", "Georgia", "Idaho", "Mississippi",
    "Montana", "North Dakota", "Oklahoma"
]

states_of_interest = always_restrictive_states + always_permissive_states
mort_subset = mort[mort["State"].isin(states_of_interest)]

mean_ageadj_by_state = (
    mort_subset
    .groupby("State")["AgeAdj_Rate"]
    .mean()
    .sort_values()
)

def classify_state(state: str) -> str:
    if state in always_restrictive_states:
        return "always_restrictive_+1"
    if state in always_permissive_states:
        return "always_permissive_-1"
    return "other"

summary = mean_ageadj_by_state.to_frame(name="mean_ageadj_rate")
summary["group"] = summary.index.map(classify_state)

mean_restrictive = summary.loc[
    summary["group"] == "always_restrictive_+1", "mean_ageadj_rate"
].mean()

mean_permissive = summary.loc[
    summary["group"] == "always_permissive_-1", "mean_ageadj_rate"
].mean()

print("\nMean age-adjusted firearm mortality (2000–2020) by group:")
print(f"Restrictive states: {mean_restrictive:.2f} deaths per 100,000")
print(f"Permissive states: {mean_permissive:.2f} deaths per 100,000")

group_means = summary.groupby("group")["mean_ageadj_rate"].mean()
group_means = group_means.loc[["always_restrictive_+1", "always_permissive_-1"]]
group_means.index = ["Restrictive states", "Permissive states"]

fig, ax = plt.subplots()
group_means.plot(kind="barh", ax=ax)

ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()

output_path = fig_dir / "mortality_states_groups_age_adjusted.pdf"
fig.savefig(output_path)
plt.close(fig)

state_abbrev = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY"
}

map_df = summary.reset_index()[["State", "group"]]
map_df["state_code"] = map_df["State"].map(state_abbrev)
map_df = map_df.dropna(subset=["state_code"])

color_map = {
    "always_restrictive_+1": "blue",
    "always_permissive_-1": "orange"
}

fig_map = px.choropleth(
    map_df,
    locations="state_code",
    locationmode="USA-states",
    color="group",
    scope="usa",
    color_discrete_map=color_map
)

fig_map.update_layout(
    title_text="",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.15,
        xanchor="center",
        x=0.5
    )
)

map_output_path = fig_dir / "mortality_states_groups_map.html"
fig_map.write_html(str(map_output_path))

pdf_output_path = fig_dir / "mortality_states_groups_map.pdf"
fig_map.write_image(str(pdf_output_path))
