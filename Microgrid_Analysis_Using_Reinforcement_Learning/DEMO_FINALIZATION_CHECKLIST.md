# Demo Finalization Checklist

## Canonical Story

Use only this pipeline in the final demo:

1. `data_loader.py`
2. `load_simulator.py`
3. `microgrid_env.py`
4. `train.py`
5. `evaluate.py`
6. `dashboard.py`

Do not present the older parallel pipeline files as part of the final architecture.

## Priority Order

### 1. Freeze the comparison setup

- Keep the current train/validation/test split:
  - Train: January to September
  - Validation: October
  - Test: November and December
- Freeze one final RL model after validating on October only.
- Freeze one final rule-based baseline after tuning its parameters on October only.
- Keep the tariff fixed across every comparison.

### 2. Validate the demo scenarios

- Run all frozen scenarios from `demo_scenarios.py`.
- Confirm RL beats the baseline on:
  - Bill cost
  - Peak-hour grid import
- Check that battery cycles do not increase in an embarrassing way.
- If a scenario is weak or inconsistent, remove it from the live demo list.

### 3. Make the load model defensible

- Describe the load as appliance-derived, not random.
- Keep these categories in the explanation:
  - Base load
  - Cooling and comfort load
  - Lighting
  - Time-window appliances
- Document the assumptions for cooking, pump, washing machine, and evening electronics use.

### 4. Make the dashboard presentation-proof

- Open on the dashboard, not on notebooks or scripts.
- Default to one strong frozen scenario.
- Keep the first screen focused on:
  - Total bill
  - Bill savings
  - Peak-hour grid import
  - SOC traces
- Do not lead with reward curves.

### 5. Clean the repo narrative

- Keep older experimental files only as archive material.
- Do not mention `custom_env.py` or `phase2_*` files in the final walkthrough.
- Update any README or report text so the architecture matches the current code path.

## Suggested Final Dry Run

1. Train the final chosen RL model.
2. Run `evaluate.py --scenario november_full`.
3. Run `evaluate.py --scenario mid_december_peak_week`.
4. Open Streamlit and rehearse the full talk once without touching code.
5. Record the exact scenario you will open first in the real demo.

## What To Say

- "The controller is evaluated on held-out months, not on the training window."
- "The baseline is tariff-aware and battery-aware, but it is still heuristic rather than predictive optimization."
- "RL performs best when timing decisions matter, especially around evening peak tariffs and limited storage."
- "MPC is an important reference method in literature, but it was not included in the live comparison because it depends on a different optimization and forecast stack."

## What Not To Do

- Do not change tariff parameters in front of evaluators.
- Do not cherry-pick a single random day without also showing aggregate results.
- Do not compare RL against a deliberately weak SOC-threshold baseline in the final presentation.
- Do not claim realism for any appliance behavior you cannot explain clearly.
- Do not rely on a script that retrains or recomputes large results live.
