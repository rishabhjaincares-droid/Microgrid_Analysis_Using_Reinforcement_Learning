from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from config import BATTERY, TARIFF, BatteryConfig, TariffConfig

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # The lightweight Q-learning demo does not require Gymnasium.
    gym = None
    spaces = None


ACTION_FRACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)
ACTION_NAMES = {
    0: "discharge_high",
    1: "discharge_low",
    2: "idle",
    3: "charge_low",
    4: "charge_high",
}


def state_key_from_values(
    soc: float,
    solar_kwh: float,
    load_kwh: float,
    import_rate: float,
    hour: int,
    tariff: TariffConfig = TARIFF,
) -> str:
    soc_bin = int(np.digitize(soc, [0.25, 0.40, 0.55, 0.70, 0.85]))
    solar_bin = int(np.digitize(solar_kwh, [0.15, 0.80, 1.80, 2.80]))
    load_bin = int(np.digitize(load_kwh, [0.35, 0.75, 1.20, 1.80]))
    if import_rate >= tariff.peak_import_rate - 1e-9:
        price_bin = 2
    elif import_rate <= tariff.solar_hour_import_rate + 1e-9:
        price_bin = 0
    else:
        price_bin = 1
    return f"{soc_bin}:{solar_bin}:{load_bin}:{price_bin}:{int(hour)}"


BaseEnv = gym.Env if gym is not None else object


class SmartMicrogridEnv(BaseEnv):
    """Hourly home microgrid simulator with solar export and battery constraints."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        battery: BatteryConfig = BATTERY,
        tariff: TariffConfig = TARIFF,
        start_soc: float | None = None,
        solar_data: np.ndarray | None = None,
        load_data: np.ndarray | None = None,
        price_data: np.ndarray | None = None,
        temp_data: np.ndarray | None = None,
        forecast_data: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if gym is not None:
            super().__init__()
        self.battery = battery
        self.tariff = tariff
        self.start_soc = battery.start_soc if start_soc is None else start_soc
        self.data = self._coerce_data(data, solar_data, load_data, price_data, temp_data)
        self.current_step = 0
        self.battery_soc = self.start_soc
        self.last_info: dict[str, Any] = {}

        if spaces is not None:
            self.action_space = spaces.Discrete(len(ACTION_FRACTIONS))
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    @staticmethod
    def _coerce_data(
        data: pd.DataFrame | None,
        solar_data: np.ndarray | None,
        load_data: np.ndarray | None,
        price_data: np.ndarray | None,
        temp_data: np.ndarray | None,
    ) -> pd.DataFrame:
        if data is not None:
            required = {"solar_kwh", "load_kwh", "import_rate", "export_rate", "hour"}
            missing = required.difference(data.columns)
            if missing:
                raise ValueError(f"Simulation data is missing columns: {sorted(missing)}")
            prepared = data.copy()
            if "timestamp" not in prepared.columns:
                prepared = prepared.reset_index().rename(columns={"index": "timestamp"})
            else:
                prepared = prepared.reset_index(drop=True)
            return prepared.copy()

        if solar_data is None or load_data is None or price_data is None:
            raise ValueError("Pass a prepared DataFrame, or pass solar_data/load_data/price_data arrays.")

        n = len(solar_data)
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        price = 5.60 + np.asarray(price_data, dtype=float) * (8.40 - 5.60)
        return pd.DataFrame(
            {
                "timestamp": idx,
                "solar_kwh": np.asarray(solar_data, dtype=float) * 3.5,
                "load_kwh": np.asarray(load_data, dtype=float) * 2.0,
                "import_rate": price,
                "export_rate": TARIFF.export_rate,
                "hour": idx.hour,
                "temp_c": np.asarray(temp_data, dtype=float) * 35 if temp_data is not None else 25.0,
            }
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if gym is not None:
            super().reset(seed=seed)
        self.current_step = 0
        self.battery_soc = self.start_soc
        self.last_info = {}
        return self._observation(), {}

    def current_row(self) -> pd.Series:
        return self.data.iloc[min(self.current_step, len(self.data) - 1)]

    def state_key(self) -> str:
        row = self.current_row()
        return state_key_from_values(
            self.battery_soc,
            row["solar_kwh"],
            row["load_kwh"],
            row["import_rate"],
            row["hour"],
            self.tariff,
        )

    def _observation(self) -> np.ndarray:
        row = self.current_row()
        hour_angle = 2.0 * np.pi * float(row["hour"]) / 24.0
        obs = np.array(
            [
                self.battery_soc,
                min(float(row["solar_kwh"]) / 3.5, 1.0),
                min(float(row["load_kwh"]) / 3.8, 1.0),
                min(float(row["import_rate"]) / self.tariff.peak_import_rate, 1.0),
                np.sin(hour_angle),
                np.cos(hour_angle),
                1.0 if row["hour"] in self.tariff.peak_hours else 0.0,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int):
        if self.current_step >= len(self.data):
            raise RuntimeError("Episode finished; call reset() before stepping again.")

        row = self.current_row()
        soc_before = self.battery_soc
        action = int(np.clip(action, 0, len(ACTION_FRACTIONS) - 1))
        fraction = ACTION_FRACTIONS[action]

        solar = float(row["solar_kwh"])
        load = float(row["load_kwh"])
        import_rate = float(row["import_rate"])
        export_rate = float(row["export_rate"])

        solar_to_load = min(solar, load)
        net_load = load - solar_to_load
        solar_surplus = solar - solar_to_load
        grid_to_battery = 0.0
        battery_to_load = 0.0
        battery_delta_kwh = 0.0

        if fraction < 0:
            requested_output = abs(fraction) * self.battery.max_discharge_kwh
            available_output = max(0.0, (self.battery_soc - self.battery.min_soc) * self.battery.capacity_kwh)
            available_output *= self.battery.discharge_efficiency
            battery_to_load = min(requested_output, available_output, net_load)
            battery_delta_kwh = -battery_to_load / self.battery.discharge_efficiency
            net_load -= battery_to_load

        elif fraction > 0:
            requested_input = fraction * self.battery.max_charge_kwh
            room_input = max(0.0, (self.battery.max_soc - self.battery_soc) * self.battery.capacity_kwh)
            room_input /= self.battery.charge_efficiency
            charge_input = min(requested_input, room_input)
            solar_to_battery = min(solar_surplus, charge_input)
            solar_surplus -= solar_to_battery
            remaining_charge_input = charge_input - solar_to_battery

            # Grid charging is allowed only during cheap solar-hour ToD periods, and
            # only when there is no PV surplus available for charging.
            if solar_to_battery == 0 and import_rate <= self.tariff.solar_hour_import_rate + 1e-9:
                grid_to_battery = remaining_charge_input

            actual_charge_input = solar_to_battery + grid_to_battery
            battery_delta_kwh = actual_charge_input * self.battery.charge_efficiency

        self.battery_soc = float(
            np.clip(
                self.battery_soc + battery_delta_kwh / self.battery.capacity_kwh,
                self.battery.min_soc,
                self.battery.max_soc,
            )
        )

        grid_import = max(0.0, net_load + grid_to_battery)
        export_kwh = min(max(0.0, solar_surplus), self.tariff.export_limit_kwh)
        bill_cost = grid_import * import_rate - export_kwh * export_rate
        throughput = abs(self.battery_soc - soc_before) * self.battery.capacity_kwh
        reward = -bill_cost - throughput * self.battery.degradation_cost_per_kwh

        self.current_step += 1
        terminated = self.current_step >= len(self.data)

        self.last_info = {
            "timestamp": row["timestamp"],
            "action": action,
            "action_name": ACTION_NAMES[action],
            "soc_before": soc_before,
            "soc_after": self.battery_soc,
            "solar_kwh": solar,
            "load_kwh": load,
            "grid_import_kwh": grid_import,
            "export_kwh": export_kwh,
            "battery_to_load_kwh": battery_to_load,
            "grid_to_battery_kwh": grid_to_battery,
            "bill_cost": bill_cost,
            "battery_throughput_kwh": throughput,
            "import_rate": import_rate,
            "export_rate": export_rate,
            "battery": asdict(self.battery),
        }
        return self._observation(), float(reward), terminated, False, self.last_info


def create_demo_environment(num_days: int = 30) -> SmartMicrogridEnv:
    index = pd.date_range("2024-11-01", periods=num_days * 24, freq="h")
    hour = index.hour.to_numpy()
    solar = np.maximum(0, np.sin((hour - 6) * np.pi / 12)) * 3.0
    load = 0.45 + 0.55 * ((hour >= 18) & (hour <= 22)) + 0.25 * ((hour >= 6) & (hour <= 8))
    data = pd.DataFrame(
        {
            "timestamp": index,
            "solar_kwh": solar,
            "load_kwh": load,
            "hour": hour,
            "import_rate": [TARIFF.import_rate(int(h)) for h in hour],
            "export_rate": TARIFF.export_rate,
        },
        index=index,
    )
    return SmartMicrogridEnv(data)
