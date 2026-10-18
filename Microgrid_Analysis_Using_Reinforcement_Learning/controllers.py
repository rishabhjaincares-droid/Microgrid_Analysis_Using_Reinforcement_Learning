from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import BATTERY, DQN_MODEL_PATH, TARIFF, BatteryConfig, TariffConfig


class RuleBasedController:
    """Fair non-MPC baseline with tariff awareness and evening reserve logic."""

    name = "Rule-Based"

    def __init__(
        self,
        battery: BatteryConfig = BATTERY,
        tariff: TariffConfig = TARIFF,
        reserve_soc: float = 0.35,
        peak_reserve_soc: float = 0.45,
        cheap_charge_target_soc: float = 0.70,
    ) -> None:
        self.battery = battery
        self.tariff = tariff
        self.reserve_soc = reserve_soc
        self.peak_reserve_soc = peak_reserve_soc
        self.cheap_charge_target_soc = cheap_charge_target_soc

    def act(self, row: pd.Series, soc: float, observation: np.ndarray | None = None) -> int:
        solar = float(row["solar_kwh"])
        load = float(row["load_kwh"])
        hour = int(row["hour"])
        import_rate = float(row["import_rate"])
        net_load = load - solar
        solar_surplus = solar - load

        cheap_hour = import_rate <= self.tariff.solar_hour_import_rate + 1e-9
        peak_hour = hour in self.tariff.peak_hours
        late_solar_hours = set(self.tariff.solar_hours[-2:])

        # Always absorb real PV surplus first when the battery has room.
        if solar_surplus > 0.10 and soc < self.battery.max_soc - 0.02:
            if solar_surplus > 0.75 or soc < 0.45:
                return 4
            return 3

        # Save charge for the most expensive evening window and respond harder when
        # the home would otherwise import a lot from the grid.
        if peak_hour and net_load > 0.05 and soc > self.peak_reserve_soc:
            if net_load > 0.90 or soc > 0.65:
                return 0
            return 1

        # Use late cheap hours to build reserve before the evening tariff spike.
        if cheap_hour and hour in late_solar_hours and soc < self.cheap_charge_target_soc and solar_surplus <= 0.05:
            if soc < 0.50:
                return 4
            return 3

        # Recover a minimum reserve on cheaper hours, but do not overcharge early in
        # the day when later PV may still be available.
        if cheap_hour and soc < self.reserve_soc and solar_surplus <= 0.05:
            return 3

        # Outside peak periods, discharge only when imports are meaningfully high and
        # the battery is already well above reserve.
        if not peak_hour and net_load > 1.10 and soc > 0.75 and import_rate >= self.tariff.normal_import_rate:
            return 1

        return 2


class StableBaselinesController:
    """Stable-Baselines3 DQN/PPO controller used as the actual RL controller."""

    name = "RL"

    def __init__(self, model_path: Path | str = DQN_MODEL_PATH, algorithm: str = "DQN") -> None:
        self.algorithm = algorithm.upper()
        if self.algorithm == "DQN":
            from stable_baselines3 import DQN

            self.model = DQN.load(str(model_path))
        elif self.algorithm == "PPO":
            from stable_baselines3 import PPO

            self.model = PPO.load(str(model_path))
        else:
            raise ValueError("algorithm must be DQN or PPO")
        self.expected_observation_shape = tuple(int(dim) for dim in self.model.observation_space.shape)

    def act(self, row: pd.Series, soc: float, observation: np.ndarray | None = None) -> int:
        if observation is None:
            raise ValueError("StableBaselinesController requires an observation.")
        observation_shape = tuple(int(dim) for dim in observation.shape)
        if observation_shape != self.expected_observation_shape:
            raise ValueError(
                "Loaded RL model is incompatible with the current environment. "
                f"Model expects observation shape {self.expected_observation_shape}, "
                f"but the environment produced {observation_shape}. "
                "This usually means you are loading an older model trained on a previous pipeline. "
                "Retrain the RL model with train.py before running evaluate.py or dashboard.py."
            )
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
