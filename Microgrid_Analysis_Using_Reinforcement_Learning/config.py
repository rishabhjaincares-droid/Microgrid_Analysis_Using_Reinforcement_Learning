from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
NASA_CSV = ROOT_DIR / "POWER_Point_Hourly_20240101_20241231_012d74N_078d14E_LST.csv"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
DQN_MODEL_PATH = ROOT_DIR / "dqn_model"
PPO_MODEL_PATH = ROOT_DIR / "ppo_model"

TRAIN_MONTHS = tuple(range(1, 10))
VALIDATION_MONTHS = (10,)
TEST_MONTHS = (11, 12)


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 5.0
    min_soc: float = 0.20
    max_soc: float = 0.95
    start_soc: float = 0.50
    max_charge_kwh: float = 2.0
    max_discharge_kwh: float = 2.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    degradation_cost_per_kwh: float = 0.03


@dataclass(frozen=True)
class SolarConfig:
    pv_capacity_kw: float = 4.0
    performance_ratio: float = 0.78
    inverter_limit_kwh: float = 3.5


@dataclass(frozen=True)
class TariffConfig:
    # Ballpark ToD tariff based on Indian ToD patterns: 20% rebate and 20% surcharge.
    # Base rate is a simple domestic demo assumption, not a billed utility claim.
    normal_import_rate: float = 7.00
    solar_hour_import_rate: float = 5.60
    peak_import_rate: float = 8.40
    export_rate: float = 3.00
    export_limit_kwh: float = 2.0
    solar_hours: tuple[int, ...] = tuple(range(9, 17))
    peak_hours: tuple[int, ...] = tuple(range(18, 22))

    def import_rate(self, hour: int) -> float:
        if hour in self.peak_hours:
            return self.peak_import_rate
        if hour in self.solar_hours:
            return self.solar_hour_import_rate
        return self.normal_import_rate


BATTERY = BatteryConfig()
SOLAR = SolarConfig()
TARIFF = TariffConfig()
RANDOM_SEED = 42
