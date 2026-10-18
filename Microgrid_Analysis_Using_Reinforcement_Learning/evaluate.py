from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import BATTERY, DQN_MODEL_PATH, PPO_MODEL_PATH, RESULTS_DIR, TARIFF
from controllers import RuleBasedController, StableBaselinesController
from data_loader import build_dataset, month_slice, split_dataset
from demo_scenarios import get_scenario, scenario_keys, slice_scenario
from microgrid_env import SmartMicrogridEnv


def _resolve_model_path(model_path: Path | str | None, algorithm: str) -> Path:
    """Pick the correct saved-model path when the caller didn't specify one."""
    if model_path is not None:
        return Path(model_path)
    return PPO_MODEL_PATH if algorithm.upper() == "PPO" else DQN_MODEL_PATH


def run_controller(data: pd.DataFrame, controller) -> tuple[pd.DataFrame, dict[str, float]]:
    env = SmartMicrogridEnv(data)
    observation, _ = env.reset()
    records = []
    done = False

    while not done:
        row = env.current_row()
        action = controller.act(row, env.battery_soc, observation)
        observation, reward, done, truncated, info = env.step(action)
        records.append(info)
        done = done or truncated

    log = pd.DataFrame(records)
    if log.empty:
        raise ValueError("Controller run produced no records.")

    log["timestamp"] = pd.to_datetime(log["timestamp"])
    log = log.set_index("timestamp").sort_index()
    log["controller"] = getattr(controller, "name", controller.__class__.__name__)
    log["cumulative_bill_inr"] = log["bill_cost"].cumsum()
    log["cumulative_grid_import_kwh"] = log["grid_import_kwh"].cumsum()
    log["cumulative_export_kwh"] = log["export_kwh"].cumsum()

    metrics = summarize_log(log)
    return log, metrics


def summarize_log(log: pd.DataFrame) -> dict[str, float]:
    peak_mask = log.index.hour.isin(TARIFF.peak_hours)
    total_bill = float(log["bill_cost"].sum())
    cycles = float(log["battery_throughput_kwh"].sum() / (2.0 * BATTERY.capacity_kwh))
    return {
        "bill_cost": total_bill,
        "battery_cycles": cycles,
        "grid_import_kwh": float(log["grid_import_kwh"].sum()),
        "export_kwh": float(log["export_kwh"].sum()),
        "peak_grid_import_kwh": float(log.loc[peak_mask, "grid_import_kwh"].sum()),
        "grid_charge_kwh": float(log["grid_to_battery_kwh"].sum()),
    }


def evaluate_period(
    selected: pd.DataFrame,
    model_path: Path | str | None = None,
    algorithm: str = "DQN",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    model_path = _resolve_model_path(model_path, algorithm)
    controllers = {
        "RL": StableBaselinesController(model_path=model_path, algorithm=algorithm),
        "Rule-Based": RuleBasedController(),
    }

    logs: dict[str, pd.DataFrame] = {}
    rows = []
    for name, controller in controllers.items():
        log, metrics = run_controller(selected, controller)
        logs[name] = log
        rows.append(
            {
                "Controller": name,
                "Bill Cost (INR)": round(metrics["bill_cost"], 2),
                "Battery Cycles": round(metrics["battery_cycles"], 2),
                "Grid Import (kWh)": round(metrics["grid_import_kwh"], 2),
                "Exported Energy (kWh)": round(metrics["export_kwh"], 2),
                "Peak-Hour Grid Import (kWh)": round(metrics["peak_grid_import_kwh"], 2),
                "Grid Charging (kWh)": round(metrics["grid_charge_kwh"], 2),
            }
        )

    comparison = pd.DataFrame(rows)
    return comparison, logs


def evaluate_month(
    month: int,
    model_path: Path | str | None = None,
    algorithm: str = "DQN",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    data = build_dataset()
    test_data = split_dataset(data)["test"]
    selected = month_slice(test_data, month)
    return evaluate_period(selected, model_path=model_path, algorithm=algorithm)


def evaluate_scenario(
    scenario_key: str,
    model_path: Path | str | None = None,
    algorithm: str = "DQN",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], str]:
    data = build_dataset()
    test_data = split_dataset(data)["test"]
    scenario, selected = slice_scenario(test_data, scenario_key)
    comparison, logs = evaluate_period(selected, model_path=model_path, algorithm=algorithm)
    return comparison, logs, scenario.focus


def evaluate_test_split(model_path: Path | str | None = None, algorithm: str = "DQN") -> pd.DataFrame:
    data = build_dataset()
    test_data = split_dataset(data)["test"]
    months = sorted(test_data.index.month.unique())
    frames = []
    for month in months:
        comparison, _ = evaluate_period(month_slice(test_data, int(month)), model_path=model_path, algorithm=algorithm)
        comparison.insert(0, "Month", int(month))
        frames.append(comparison)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL vs rule-based controller.")
    parser.add_argument("--month", type=int, default=None, help="Test month number, e.g. 11 or 12.")
    parser.add_argument("--scenario", choices=scenario_keys(), default=None, help="Frozen demo scenario key.")
    parser.add_argument("--algorithm", choices=["DQN", "PPO"], default="DQN")
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()

    if args.month is not None and args.scenario is not None:
        raise ValueError("Use either --month or --scenario, not both.")

    RESULTS_DIR.mkdir(exist_ok=True)
    if args.scenario is not None:
        comparison, _, focus = evaluate_scenario(args.scenario, model_path=args.model_path, algorithm=args.algorithm)
        scenario = get_scenario(args.scenario)
        output = RESULTS_DIR / f"{args.scenario}_comparison.csv"
        print(f"Scenario: {scenario.label}")
        print(f"Focus: {focus}")
    elif args.month is None:
        comparison = evaluate_test_split(model_path=args.model_path, algorithm=args.algorithm)
        output = RESULTS_DIR / "test_split_comparison.csv"
    else:
        comparison, _ = evaluate_month(args.month, model_path=args.model_path, algorithm=args.algorithm)
        output = RESULTS_DIR / f"month_{args.month:02d}_comparison.csv"

    comparison.to_csv(output, index=False)
    print(comparison.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
