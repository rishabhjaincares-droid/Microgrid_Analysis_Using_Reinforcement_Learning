"""
Phase-II: REAL NASA POWER DATA TRAINING
========================================
Uses manually downloaded NASA POWER data for Hosur, Tamil Nadu.
This is the REAL research validation!

Trains DQN with vs without LSTM forecast on actual solar observations.
This is what gets published! 📜
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Tuple, Optional
import glob
import gymnasium as gym

# Ensure you have your custom environment file in the same directory
from custom_env import SmartMicrogridEnv

try:
    from stable_baselines3 import DQN, PPO
except ImportError:
    print("ERROR: stable-baselines3 not installed")
    exit(1)


# ==========================================
# COMPATIBILITY PATCH: Gym -> Gymnasium
# ==========================================
class GymToGymnasiumWrapper(gym.Env):
    """
    Intercepts old Gym outputs from pymgrid and converts them to 
    the new Gymnasium standard required by Stable-Baselines3.
    """
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
            return obs
        return obs, {}
        
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        return result
# ==========================================


def load_nasa_power_csv(csv_path: Optional[str] = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load and clean NASA POWER data using dynamic header detection."""
    
    if csv_path is None:
        csv_files = glob.glob('POWER_Point_Hourly_*.csv')
        if not csv_files:
            print("✗ No POWER_Point_Hourly_*.csv found in current directory")
            return None
        csv_path = csv_files[0]
        print(f"Found: {csv_path}")
    
    try:
        print(f"Loading raw NASA POWER data from: {csv_path}")
        
        # --- THE FIX: Dynamically find the header row ---
        header_row_index = 0
        with open(csv_path, 'r') as f:
            for i, line in enumerate(f):
                if 'ALLSKY_SFC_SW_DWN' in line or 'T2M' in line:
                    header_row_index = i
                    break
        
        # Read the CSV starting exactly at the correct header row
        df = pd.read_csv(csv_path, skiprows=header_row_index)
        # ------------------------------------------------
        
        # Clean NASA's -999.0 missing data markers
        df = df.replace(-999.0, pd.NA).ffill().fillna(0)
        
        if 'ALLSKY_SFC_SW_DWN' not in df.columns or 'T2M' not in df.columns:
            print(f"✗ Could not find raw NASA columns in {df.columns.tolist()}")
            return None
        
        solar = df['ALLSKY_SFC_SW_DWN'].values.astype(np.float32)
        temp = df['T2M'].values.astype(np.float32)
        
        print(f"\n✓ Loaded {len(solar)} hours of real solar data")
        
        solar = np.clip(solar / 1000, 0, 1).astype(np.float32)
        temp = np.clip((temp - 10) / 30, 0, 1).astype(np.float32)
        
        num_hours = len(solar)
        hour_of_day = np.arange(num_hours) % 24
        load_baseline = 150
        load_peak_morning = 100 * np.maximum(0, np.sin((hour_of_day - 7) * np.pi / 6))
        load_peak_evening = 150 * np.maximum(0, np.sin((hour_of_day - 18) * np.pi / 6))
        load = np.clip(
            load_baseline + load_peak_morning + load_peak_evening + np.random.normal(0, 20, num_hours),
            50, 500
        ) / 500
        load = load.astype(np.float32)
        
        base_price = 8
        price = (base_price + 
                4 * np.maximum(0, np.sin((hour_of_day - 9) * np.pi / 12)) +
                3 * np.maximum(0, np.sin((hour_of_day - 19) * np.pi / 8)))
        price = np.clip(price, 6, 14)
        price = ((price - 6) / 8).astype(np.float32)
        
        return solar, load, price, temp
    
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return None


class Phase2RealDataTrainer:
    """Phase-II trainer with REAL NASA POWER data."""
    
    def __init__(self, output_dir: str = './phase2_real_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Phase-II Real Data Trainer initialized")
    
    def train_agent(self, env, agent_name: str, algorithm: str = 'DQN',
                   total_timesteps: int = 100000, learning_rate: float = 1e-4) -> object:
        print(f"\n{'='*70}")
        print(f"Training {algorithm} on REAL NASA POWER DATA - {agent_name}")
        print(f"{'='*70}")
        
        if algorithm == 'DQN':
            model = DQN("MlpPolicy", env, learning_rate=learning_rate, buffer_size=50000, batch_size=32, exploration_fraction=0.15, verbose=1)
        else:
            model = PPO("MlpPolicy", env, learning_rate=learning_rate, batch_size=64, n_steps=2048, verbose=1)
            
        print(f"\nTraining started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        model_path = os.path.join(self.output_dir, f'{agent_name}_model')
        model.save(model_path)
        return model

    def evaluate_agent(self, model, env, num_episodes: int = 5) -> dict:
        episode_rewards = []
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                # The wrapper guarantees exactly 5 variables are returned
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated
                episode_reward += reward
                steps += 1
            
            episode_rewards.append(episode_reward)
            print(f"  Episode {episode+1}/{num_episodes}: Reward={episode_reward:.2f}")
            
        return {'mean_reward': np.mean(episode_rewards), 'std_reward': np.std(episode_rewards)}

    def run_real_data_phase2(self, csv_path: Optional[str] = None):
            print("\n" + "="*70)
            print("PHASE-II: REAL NASA POWER DATA TRAINING")
            print("="*70)
            
            data = load_nasa_power_csv(csv_path)
            if data is None:
                return None
            
            # These are the raw numpy arrays
            solar, load, price, temp = data
            
            print("\n[2/4] Creating environment...")
            
            # THE FIX: Pass the raw arrays directly. No DataFrames!
            raw_env = SmartMicrogridEnv(
                solar_data=solar, 
                load_data=load, 
                price_data=price, 
                temp_data=temp 
            )
            env = GymToGymnasiumWrapper(raw_env)
            print("     ✓ Ready")
            
            print("\n[3/4] Training DQN agent...")
            dqn_model = self.train_agent(env, 'real_nasa_agent', 'DQN', 100000)
            
            print("\n[4/4] Evaluating on real data...")
            results = self.evaluate_agent(dqn_model, env, 5)
            
            print("\n" + "="*70)
            print("FINAL RESULTS ON HOSUR REAL DATA")
            print("="*70)
            print(f"Agent Performance : Mean Reward = {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print("="*70)
            
            return results

def main():
    print("\n" + "="*70)
    print("🚀 PHASE-II: REAL NASA POWER DATA VALIDATION 🚀")
    print("="*70)
    trainer = Phase2RealDataTrainer(output_dir='./phase2_real_results')
    trainer.run_real_data_phase2(csv_path=None)

if __name__ == "__main__":
    main()
