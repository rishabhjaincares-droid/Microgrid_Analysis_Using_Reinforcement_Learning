from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_event_mask(index: pd.DatetimeIndex, probability: float, rng: np.random.Generator) -> dict[pd.Timestamp, bool]:
    days = pd.Index(index.normalize().unique())
    return {day: bool(rng.random() < probability) for day in days}


def build_household_load(
    index: pd.DatetimeIndex,
    temperature_c: pd.Series,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a deterministic, appliance-style home load profile in kWh per hour."""
    rng = np.random.default_rng(seed)
    hours = index.hour.to_numpy()
    dayofweek = index.dayofweek.to_numpy()
    weekend = dayofweek >= 5
    temp = temperature_c.reindex(index).to_numpy()

    base = np.full(len(index), 0.16)
    fridge = 0.06 + 0.025 * (np.sin((hours - 2) * np.pi / 6) > 0)
    fans = np.clip((temp - 24.0) * 0.035, 0.0, 0.35)
    fans += np.where((hours >= 21) | (hours <= 1), 0.08, 0.0)
    lights = np.where((hours >= 18) & (hours <= 23), 0.24, 0.02)
    electronics = np.where((hours >= 19) & (hours <= 23), 0.18, 0.05)

    cooking = np.zeros(len(index))
    cooking += np.where((hours >= 6) & (hours <= 8), 0.26, 0.0)
    cooking += np.where((hours >= 18) & (hours <= 20), 0.34, 0.0)

    pump = np.zeros(len(index))
    pump_days = _daily_event_mask(index, probability=0.55, rng=rng)
    for i, ts in enumerate(index):
        if pump_days[ts.normalize()] and ts.hour in (6, 7):
            pump[i] = 0.55 if ts.hour == 6 else 0.15

    washing = np.zeros(len(index))
    washing_days = _daily_event_mask(index, probability=0.28, rng=rng)
    for i, ts in enumerate(index):
        if washing_days[ts.normalize()] and (weekend[i] or rng.random() < 0.25) and ts.hour in (10, 11):
            washing[i] = 0.55 if ts.hour == 10 else 0.30

    random_small = rng.normal(0.0, 0.035, len(index))
    load = base + fridge + fans + lights + electronics + cooking + pump + washing + random_small
    load = np.clip(load, 0.18, 3.8)

    return pd.DataFrame(
        {
            "load_kwh": load,
            "base_kwh": base + fridge,
            "comfort_kwh": fans,
            "lights_kwh": lights,
            "appliances_kwh": cooking + pump + washing + electronics,
        },
        index=index,
    )
