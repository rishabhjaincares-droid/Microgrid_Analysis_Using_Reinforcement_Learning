from __future__ import annotations

import pandas as pd
import streamlit as st

from config import TEST_MONTHS, TRAIN_MONTHS, VALIDATION_MONTHS
from demo_scenarios import SCENARIOS
from evaluate import evaluate_scenario


st.set_page_config(page_title="Smart Microgrid RL Demo", layout="wide")


@st.cache_data(show_spinner=False)
def run_demo_view(scenario_key: str, algorithm: str):
    return evaluate_scenario(scenario_key, algorithm=algorithm)


def metric_row(comparison: pd.DataFrame) -> None:
    rl_row = comparison.loc[comparison["Controller"] == "RL"].iloc[0]
    baseline_row = comparison.loc[comparison["Controller"] == "Rule-Based"].iloc[0]

    savings = baseline_row["Bill Cost (INR)"] - rl_row["Bill Cost (INR)"]
    peak_grid_cut = baseline_row["Peak-Hour Grid Import (kWh)"] - rl_row["Peak-Hour Grid Import (kWh)"]
    import_cut = baseline_row["Grid Import (kWh)"] - rl_row["Grid Import (kWh)"]
    cycle_delta = rl_row["Battery Cycles"] - baseline_row["Battery Cycles"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("RL Bill", f"Rs {rl_row['Bill Cost (INR)']:.2f}")
    col2.metric("Baseline Bill", f"Rs {baseline_row['Bill Cost (INR)']:.2f}")
    col3.metric("Bill Savings", f"Rs {savings:.2f}")
    col4.metric("Peak-Hour Import Cut", f"{peak_grid_cut:.2f} kWh")
    col5.metric("Cycle Delta", f"{cycle_delta:+.2f}", delta_color="off")

    st.caption(
        f"RL also reduced total grid import by {import_cut:.2f} kWh in this frozen test scenario."
    )


def build_overview_chart(logs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rl = logs["RL"]
    baseline = logs["Rule-Based"]
    return pd.DataFrame(
        {
            "Load (kWh)": rl["load_kwh"],
            "Solar (kWh)": rl["solar_kwh"],
            "Tariff (Rs/kWh)": rl["import_rate"],
            "RL SOC": rl["soc_after"],
            "Baseline SOC": baseline["soc_after"],
        }
    )


def build_cumulative_bill_chart(logs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RL": logs["RL"]["cumulative_bill_inr"],
            "Rule-Based": logs["Rule-Based"]["cumulative_bill_inr"],
        }
    )


def build_peak_import_chart(logs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    peak_mask = logs["RL"].index.hour.isin(range(18, 22))
    return pd.DataFrame(
        {
            "RL Peak Import": logs["RL"].loc[peak_mask, "grid_import_kwh"],
            "Baseline Peak Import": logs["Rule-Based"].loc[peak_mask, "grid_import_kwh"],
        }
    )


def build_action_table(logs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    action_counts = []
    for name, log in logs.items():
        counts = log["action_name"].value_counts().sort_index()
        for action_name, count in counts.items():
            action_counts.append(
                {
                    "Controller": name,
                    "Action": action_name,
                    "Hours": int(count),
                }
            )
    return pd.DataFrame(action_counts)


st.title("Smart Microgrid RL Demo")
st.write(
    "Compare a trained RL controller against a tariff-aware rule-based baseline on held-out "
    "November-December 2024 scenarios using the same solar, load, battery, and tariff setup."
)

scenario_options = {scenario.label: scenario.key for scenario in SCENARIOS}

with st.sidebar:
    st.header("Demo Controls")
    selected_label = st.selectbox("Scenario", list(scenario_options.keys()), index=0)
    algorithm = st.selectbox("RL Model", ["DQN", "PPO"], index=0)
    st.markdown("**Split**")
    st.write(f"Train: months {min(TRAIN_MONTHS)}-{max(TRAIN_MONTHS)}")
    st.write(f"Validation: month {VALIDATION_MONTHS[0]}")
    st.write(f"Test: months {TEST_MONTHS[0]}-{TEST_MONTHS[-1]}")
    st.markdown("**Fairness**")
    st.write("Same battery, tariff, initial SOC, and held-out data for both controllers.")

scenario_key = scenario_options[selected_label]

try:
    comparison, logs, focus = run_demo_view(scenario_key, algorithm)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

metric_row(comparison)

st.info(f"Scenario focus: {focus}")

left, right = st.columns([1.3, 1.0])

with left:
    st.subheader("Cumulative Electricity Bill")
    st.line_chart(build_cumulative_bill_chart(logs), use_container_width=True)

    st.subheader("System Conditions and Battery Response")
    st.line_chart(build_overview_chart(logs), use_container_width=True)

with right:
    st.subheader("Comparison Table")
    st.dataframe(comparison.set_index("Controller"), use_container_width=True)

    st.subheader("Action Mix")
    st.dataframe(build_action_table(logs), use_container_width=True, hide_index=True)

st.subheader("Peak-Hour Grid Import")
st.bar_chart(build_peak_import_chart(logs), use_container_width=True)

st.subheader("How To Explain This Screen")
st.markdown(
    """
    1. Start with bill savings, because cost is the primary metric.
    2. Use the cumulative-bill chart to show when the policies diverge.
    3. Point to the SOC traces during peak tariff hours to explain timing behavior.
    4. Use the action table to show that RL is not simply cycling more aggressively.
    """
)

st.caption(
    "Demo tariff: Rs 5.60/kWh solar hours, Rs 7.00/kWh normal, Rs 8.40/kWh peak, "
    "Rs 3.00/kWh export credit. Keep this tariff fixed across all live comparisons."
)
