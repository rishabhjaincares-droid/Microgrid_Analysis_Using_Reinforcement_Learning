from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DemoScenario:
    key: str
    label: str
    start: str
    end: str
    focus: str


SCENARIOS: tuple[DemoScenario, ...] = (
    DemoScenario(
        key="november_full",
        label="November 2024",
        start="2024-11-01 00:00:00",
        end="2024-11-30 23:00:00",
        focus="Full held-out month with regular weekday and weekend patterns.",
    ),
    DemoScenario(
        key="december_full",
        label="December 2024",
        start="2024-12-01 00:00:00",
        end="2024-12-31 23:00:00",
        focus="Full held-out month with different seasonal solar and evening peaks.",
    ),
    DemoScenario(
        key="mid_november_peak_week",
        label="Mid-November Peak Week",
        start="2024-11-11 00:00:00",
        end="2024-11-17 23:00:00",
        focus="Compact week that highlights battery reserve decisions before evening peaks.",
    ),
    DemoScenario(
        key="late_november_mixed_solar",
        label="Late-November Mixed-Solar Week",
        start="2024-11-22 00:00:00",
        end="2024-11-28 23:00:00",
        focus="Mixed-solar conditions that stress when to save charge versus export surplus.",
    ),
    DemoScenario(
        key="mid_december_peak_week",
        label="Mid-December Peak Week",
        start="2024-12-09 00:00:00",
        end="2024-12-15 23:00:00",
        focus="Useful live demo window with repeated evening demand spikes.",
    ),
)


SCENARIO_BY_KEY = {scenario.key: scenario for scenario in SCENARIOS}


def scenario_keys() -> list[str]:
    return [scenario.key for scenario in SCENARIOS]


def get_scenario(key: str) -> DemoScenario:
    try:
        return SCENARIO_BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown demo scenario: {key}") from exc


def slice_scenario(data: pd.DataFrame, key: str) -> tuple[DemoScenario, pd.DataFrame]:
    scenario = get_scenario(key)
    selected = data.loc[scenario.start : scenario.end].copy()
    if selected.empty:
        raise ValueError(f"Scenario {key} produced an empty slice.")
    return scenario, selected
