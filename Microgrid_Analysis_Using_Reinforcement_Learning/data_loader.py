from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import NASA_CSV, RANDOM_SEED, SOLAR, TARIFF, TEST_MONTHS, TRAIN_MONTHS, VALIDATION_MONTHS
from load_simulator import build_household_load


def _find_header_row(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.startswith("YEAR,MO,DY,HR"):
                return index
    raise ValueError(f"Could not find NASA POWER data header in {csv_path}")


def load_nasa_power(csv_path: Path = NASA_CSV) -> pd.DataFrame:
    """Load hourly NASA POWER data and convert irradiance into PV output."""
    csv_path = Path(csv_path)
    header_row = _find_header_row(csv_path)
    raw = pd.read_csv(csv_path, skiprows=header_row)
    raw = raw.replace(-999, np.nan).ffill().fillna(0)

    timestamp = pd.to_datetime(
        raw[["YEAR", "MO", "DY", "HR"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"}
        )
    )

    irradiance = raw["ALLSKY_SFC_SW_DWN"].astype(float).clip(lower=0)
    solar_kwh = (
        irradiance / 1000.0 * SOLAR.pv_capacity_kw * SOLAR.performance_ratio
    ).clip(upper=SOLAR.inverter_limit_kwh)

    data = pd.DataFrame(
        {
            "timestamp": timestamp,
            "temp_c": raw["T2M"].astype(float),
            "irradiance_wh_m2": irradiance,
            "solar_kwh": solar_kwh,
        }
    ).set_index("timestamp")

    data["hour"] = data.index.hour
    data["month"] = data.index.month
    data["dayofweek"] = data.index.dayofweek
    data["import_rate"] = [TARIFF.import_rate(hour) for hour in data["hour"]]
    data["export_rate"] = TARIFF.export_rate
    return data


def build_dataset(seed: int = RANDOM_SEED, csv_path: Path = NASA_CSV) -> pd.DataFrame:
    """Return the final hourly simulation dataset used by training and demo."""
    data = load_nasa_power(csv_path)
    load = build_household_load(data.index, data["temp_c"], seed=seed)
    final = data.join(load)
    final["net_without_battery_kwh"] = final["load_kwh"] - final["solar_kwh"]
    return final


def split_dataset(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": data[data.index.month.isin(TRAIN_MONTHS)].copy(),
        "validation": data[data.index.month.isin(VALIDATION_MONTHS)].copy(),
        "test": data[data.index.month.isin(TEST_MONTHS)].copy(),
    }


def test_months(data: pd.DataFrame) -> list[int]:
    test = split_dataset(data)["test"]
    return sorted(test.index.month.unique().tolist())


def month_slice(data: pd.DataFrame, month: int) -> pd.DataFrame:
    selected = data[data.index.month == month].copy()
    if selected.empty:
        raise ValueError(f"No data available for month {month}")
    return selected
