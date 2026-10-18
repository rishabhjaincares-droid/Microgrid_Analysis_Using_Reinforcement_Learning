from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from stable_baselines3 import DQN, PPO

from config import DQN_MODEL_PATH, PPO_MODEL_PATH, RANDOM_SEED
from data_loader import build_dataset, split_dataset
from microgrid_env import SmartMicrogridEnv


def _progress_bar_available() -> bool:
    return importlib.util.find_spec("tqdm") is not None and importlib.util.find_spec("rich") is not None


def train_dqn(total_timesteps: int, model_path: Path = DQN_MODEL_PATH, progress_bar: bool = False):
    train_data = split_dataset(build_dataset(seed=RANDOM_SEED))["train"]
    env = SmartMicrogridEnv(train_data)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50_000,
        batch_size=32,
        exploration_fraction=0.15,
        target_update_interval=5_000,
        seed=RANDOM_SEED,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar and _progress_bar_available())
    model.save(str(model_path))
    return model


def train_ppo(total_timesteps: int, model_path: Path = PPO_MODEL_PATH, progress_bar: bool = False):
    train_data = split_dataset(build_dataset(seed=RANDOM_SEED))["train"]
    env = SmartMicrogridEnv(train_data)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2_048,
        batch_size=64,
        seed=RANDOM_SEED,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar and _progress_bar_available())
    model.save(str(model_path))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stable-Baselines3 DQN/PPO controllers.")
    parser.add_argument("--algorithm", choices=["dqn", "ppo", "both"], default="dqn")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--progress-bar", action="store_true", help="Show SB3 progress bar when tqdm and rich are installed.")
    args = parser.parse_args()

    if args.algorithm in ("dqn", "both"):
        train_dqn(args.timesteps, progress_bar=args.progress_bar)
    if args.algorithm in ("ppo", "both"):
        train_ppo(args.timesteps, progress_bar=args.progress_bar)


if __name__ == "__main__":
    main()
