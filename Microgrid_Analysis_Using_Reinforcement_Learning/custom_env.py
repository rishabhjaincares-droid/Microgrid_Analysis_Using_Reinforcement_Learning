"""
Smart Microgrid RL Environment - Simplified Version
====================================================
Custom Gym environment WITHOUT PyMGrid module complexity.
Direct implementation of battery, solar, load, and grid logic.

This avoids all PyMGrid API compatibility issues.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict


class SmartMicrogridEnv(gym.Env):
    """
    Simplified microgrid environment for RL training.
    
    No PyMGrid dependencies - pure Python implementation.
    
    State: [SOC, P_solar, P_load, price, temp, forecast]
    Action: {0: charge, 1: discharge, 2: direct_supply, 3: shed_load, 4: import_grid}
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        solar_data: np.ndarray,
        load_data: np.ndarray,
        price_data: np.ndarray,
        temp_data: np.ndarray,
        forecast_data: np.ndarray = None,
        config: Dict = None
    ):
        """
        Initialize the microgrid environment.
        
        Args:
            solar_data: Hourly solar power (W) - normalized to [0,1]
            load_data: Hourly load demand (W) - normalized to [0,1]
            price_data: Hourly electricity price (₹/kWh) - normalized to [0,1]
            temp_data: Hourly ambient temperature (°C) - normalized to [0,1]
            forecast_data: 1-hour ahead solar forecast - normalized to [0,1]
            config: Configuration dictionary
        """
        super().__init__()
        
        # Default configuration
        self.config = config or self._default_config()
        
        # Data validation
        assert len(solar_data) == len(load_data) == len(price_data) == len(temp_data), \
            "All data arrays must have same length"
        self.timesteps = len(solar_data)
        
        # Store data (assume already normalized to [0,1])
        self.solar_data = np.clip(solar_data.astype(np.float32), 0, 1)
        self.load_data = np.clip(load_data.astype(np.float32), 0, 1)
        self.price_data = np.clip(price_data.astype(np.float32), 0, 1)
        self.temp_data = np.clip(temp_data.astype(np.float32), 0, 1)
        self.forecast_data = np.clip(forecast_data.astype(np.float32), 0, 1) \
            if forecast_data is not None else np.zeros_like(solar_data)
        
        # Simulation state
        self.current_step = 0
        self.done = False
        
        # Battery state
        self.battery_soc = 0.5  # Start at 50%
        self.battery_capacity = self.config['battery_capacity']  # Wh
        self.battery_min_soc = self.config['battery_min_soc']
        self.battery_max_soc = self.config['battery_max_soc']
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Observation space: 6 continuous variables [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Tracking
        self.episode_rewards = []
        self.last_grid_import = 0
        self.last_unmet_load = 0
        
    def _default_config(self) -> Dict:
        """Default system configuration."""
        return {
            'battery_capacity': 84,           # Wh (12V x 7Ah)
            'battery_min_soc': 0.2,           # 20% minimum
            'battery_max_soc': 0.95,          # 95% maximum
            'max_charge_power': 50,           # W
            'max_discharge_power': 50,        # W
            'battery_efficiency': 0.95,       # Round-trip efficiency
            'max_solar_power': 20,            # W peak
            'max_load_power': 500,            # W peak
            'max_price': 20,                  # ₹/kWh max
            'loss_load_penalty': 100,         # ₹ per kWh unmet
            'battery_degradation_cost': 0.01, # ₹ per cycle
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.array([
            self.battery_soc,                              # Battery SOC [0,1]
            self.solar_data[self.current_step],            # Solar power [0,1]
            self.load_data[self.current_step],             # Load [0,1]
            self.price_data[self.current_step],            # Price [0,1]
            self.temp_data[self.current_step],             # Temperature [0,1]
            self.forecast_data[self.current_step]          # Solar forecast [0,1]
        ], dtype=np.float32)
        
        return np.clip(obs, 0.0, 1.0)
    
    def _compute_reward(self, action: int, soc_before: float) -> float:
        """
        Compute reward for the action taken.
        
        Reward = α × energy_saved - β × grid_cost - γ × battery_wear - δ × unmet_load
        """
        # Reward weights
        alpha = 1.0     # Energy savings
        beta = 2.0      # Grid cost (expensive)
        gamma = 0.5     # Battery degradation
        delta = 10.0    # Unmet load (critical)
        
        # Denormalize values
        solar_w = self.solar_data[self.current_step] * self.config['max_solar_power']
        load_w = self.load_data[self.current_step] * self.config['max_load_power']
        price = self.price_data[self.current_step] * self.config['max_price']
        
        # Simulate energy balance for this action
        soc_after = self.battery_soc
        grid_import = 0
        unmet_load = 0
        
        if action == 0:  # Charge battery
            charge_amount = min(
                self.config['max_charge_power'],
                solar_w,
                self.battery_capacity * (self.battery_max_soc - self.battery_soc)
            )
            soc_after = min(self.battery_max_soc, self.battery_soc + charge_amount / self.battery_capacity)
            
            if solar_w < load_w:
                grid_import = (load_w - solar_w) / 1000  # Convert to kWh
                unmet_load = 0
            
        elif action == 1:  # Discharge battery
            discharge_amount = min(
                self.config['max_discharge_power'],
                self.battery_capacity * (self.battery_soc - self.battery_min_soc)
            )
            soc_after = max(self.battery_min_soc, self.battery_soc - discharge_amount / self.battery_capacity)
            
            battery_supply = min(discharge_amount, load_w)
            remaining_load = max(0, load_w - battery_supply)
            
            if solar_w < remaining_load:
                grid_import = (remaining_load - solar_w) / 1000
                unmet_load = 0
            else:
                grid_import = 0
                unmet_load = 0
        
        elif action == 2:  # Direct solar to load
            if solar_w < load_w:
                remaining = load_w - solar_w
                if self.battery_soc > self.battery_min_soc:
                    battery_can_supply = self.battery_capacity * (self.battery_soc - self.battery_min_soc)
                    battery_supply = min(battery_can_supply, remaining)
                    soc_after = self.battery_soc - battery_supply / self.battery_capacity
                    grid_import = max(0, (remaining - battery_supply) / 1000)
                else:
                    grid_import = remaining / 1000
                    unmet_load = remaining / 1000
            else:
                grid_import = 0
                soc_after = self.battery_soc
        
        elif action == 3:  # Shed load
            shed_amount = max(0, load_w - 100)  # Keep 100W baseline
            actual_load = load_w - shed_amount
            
            if solar_w < actual_load:
                grid_import = (actual_load - solar_w) / 1000
            
            soc_after = self.battery_soc
        
        else:  # action == 4: Import from grid
            grid_import = load_w / 1000  # All from grid
            soc_after = self.battery_soc
        
        # Calculate components
        energy_saved = max(0, -grid_import) * 1000  # Wh
        grid_cost = max(0, grid_import) * price
        battery_wear = abs(soc_after - soc_before) * self.battery_capacity * self.config['battery_degradation_cost'] / 100
        
        # Compute reward
        reward = (
            alpha * energy_saved
            - beta * grid_cost
            - gamma * battery_wear
            - delta * unmet_load
        )
        
        return float(reward)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        if self.done:
            raise RuntimeError("Episode finished, call reset()")
        
        soc_before = self.battery_soc
        
        # Execute action
        self._execute_action(action)
        
        # Advance time
        self.current_step += 1
        self.done = self.current_step >= self.timesteps - 1
        
        # Compute reward
        reward = self._compute_reward(action, soc_before)
        self.episode_rewards.append(reward)
        
        # Get observation
        obs = self._get_observation()
        
        # Info
        info = {
            'step': self.current_step,
            'soc': soc_before,
            'action': action,
            'reward': reward,
        }
        
        return obs, reward, self.done, False, info
    
    def _execute_action(self, action: int):
        """Execute action and update battery state."""
        solar_w = self.solar_data[self.current_step] * self.config['max_solar_power']
        load_w = self.load_data[self.current_step] * self.config['max_load_power']
        
        if action == 0:  # Charge
            charge = min(
                self.config['max_charge_power'],
                solar_w,
                self.battery_capacity * (self.battery_max_soc - self.battery_soc)
            )
            self.battery_soc = min(self.battery_max_soc, self.battery_soc + charge / self.battery_capacity)
        
        elif action == 1:  # Discharge
            discharge = min(
                self.config['max_discharge_power'],
                self.battery_capacity * (self.battery_soc - self.battery_min_soc)
            )
            self.battery_soc = max(self.battery_min_soc, self.battery_soc - discharge / self.battery_capacity)
        
        elif action == 2:  # Direct supply
            if solar_w < load_w:
                deficit = load_w - solar_w
                battery_can_supply = self.battery_capacity * (self.battery_soc - self.battery_min_soc)
                battery_supply = min(battery_can_supply, deficit)
                self.battery_soc = max(self.battery_min_soc, self.battery_soc - battery_supply / self.battery_capacity)
        
        elif action == 3:  # Shed
            # Battery stays same
            pass
        
        else:  # Import from grid
            # Battery stays same
            pass
        
        # Keep SOC within bounds
        self.battery_soc = np.clip(self.battery_soc, self.battery_min_soc, self.battery_max_soc)
    
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.done = False
        self.battery_soc = 0.5
        self.episode_rewards = []
        
        obs = self._get_observation()
        return obs, {}
    
    def render(self):
        """Render (optional)."""
        pass
    
    def close(self):
        """Cleanup."""
        pass


def create_demo_environment(num_days: int = 30) -> SmartMicrogridEnv:
    """
    Create environment with synthetic data.
    
    Args:
        num_days: Number of days to simulate
        
    Returns:
        Initialized environment
    """
    num_hours = num_days * 24
    
    # Synthetic solar data
    hour_of_day = np.arange(num_hours) % 24
    solar_data = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12) * 20) / 20  # Normalized
    
    # Synthetic load data
    load_baseline = 150  # W
    load_peak_morning = 100 * np.sin((hour_of_day - 7) * np.pi / 6)
    load_peak_evening = 150 * np.sin((hour_of_day - 18) * np.pi / 6)
    load_data = np.clip(
        load_baseline + np.maximum(load_peak_morning, load_peak_evening) + 
        np.random.normal(0, 20, num_hours),
        50, 500
    ) / 500  # Normalized to [0,1]
    
    # Synthetic price data
    base_price = 8  # ₹/kWh
    price_data = (base_price + 4 * np.sin((hour_of_day - 9) * np.pi / 12)) / 20  # Normalized
    price_data = np.clip(price_data, 0.3, 0.6)
    
    # Synthetic temperature
    temp_data = (25 + 10 * np.sin((hour_of_day - 12) * np.pi / 12)) / 50  # Normalized
    
    # Forecast (simple persistence)
    forecast_data = np.roll(solar_data, 1)
    
    env = SmartMicrogridEnv(
        solar_data=solar_data,
        load_data=load_data,
        price_data=price_data,
        temp_data=temp_data,
        forecast_data=forecast_data
    )
    
    return env


if __name__ == "__main__":
    # Quick test
    print("Testing simplified microgrid environment...")
    
    env = create_demo_environment(num_days=7)
    print(f"Environment: {env}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode finished after {step+1} steps")
            break
        
        if step % 20 == 0:
            print(f"Step {step}: SOC={info['soc']:.2%}, Reward={reward:.2f}")
    
    print(f"Total reward: {total_reward:.2f}")
    print("✓ Simplified environment test PASSED!")
    env.close()