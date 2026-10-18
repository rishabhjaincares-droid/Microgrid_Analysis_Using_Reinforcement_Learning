"""
Smart Microgrid RL Agent Training
==================================
Trains DQN and PPO agents on the custom microgrid environment.

This script:
1. Loads or generates synthetic data
2. Creates the environment
3. Trains DQN and PPO agents
4. Compares them against baselines
5. Generates performance plots

Requires: 
- microgrid_env.py
- stable-baselines3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import os
from datetime import datetime

# Import our custom environment
from microgrid_env import SmartMicrogridEnv, create_demo_environment

# Import RL algorithms from Stable-Baselines3
try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
except ImportError:
    print("ERROR: stable-baselines3 not installed")
    print("Install: pip install stable-baselines3[extra]")
    exit(1)


class MicrogridRLTrainer:
    """Train RL agents on the microgrid environment."""
    
    def __init__(self, env: SmartMicrogridEnv, output_dir: str = './results'):
        """
        Initialize trainer.
        
        Args:
            env: Custom microgrid environment
            output_dir: Directory to save models and results
        """
        self.env = env
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Trainer initialized")
        print(f"Output directory: {output_dir}")
    
    def train_dqn(self,
                  total_timesteps: int = 100000,
                  learning_rate: float = 1e-4,
                  batch_size: int = 32,
                  buffer_size: int = 50000,
                  exploration_fraction: float = 0.15) -> DQN:
        """
        Train DQN agent.
        
        Args:
            total_timesteps: Total training steps
            learning_rate: Learning rate
            batch_size: Batch size for training
            buffer_size: Experience replay buffer size
            exploration_fraction: Fraction of total steps for exploration
            
        Returns:
            Trained DQN model
        """
        print("\n" + "=" * 60)
        print("Training DQN Agent")
        print("=" * 60)
        print(f"Hyperparameters:")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Buffer size: {buffer_size:,}")
        print(f"  Exploration fraction: {exploration_fraction}")
        
        model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            exploration_fraction=exploration_fraction,
            target_update_interval=5000,
            verbose=1
        )
        
        # Train
        print(f"\nTraining started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save
        model_path = os.path.join(self.output_dir, 'dqn_model')
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        return model
    
    def train_ppo(self,
                  total_timesteps: int = 100000,
                  learning_rate: float = 1e-4,
                  batch_size: int = 64,
                  n_steps: int = 2048) -> PPO:
        """
        Train PPO agent.
        
        Args:
            total_timesteps: Total training steps
            learning_rate: Learning rate
            batch_size: Batch size
            n_steps: Number of steps to collect before update
            
        Returns:
            Trained PPO model
        """
        print("\n" + "=" * 60)
        print("Training PPO Agent")
        print("=" * 60)
        print(f"Hyperparameters:")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  n_steps: {n_steps}")
        
        model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            verbose=1
        )
        
        # Train
        print(f"\nTraining started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save
        model_path = os.path.join(self.output_dir, 'ppo_model')
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        return model
    
    def evaluate_agent(self, model, num_episodes: int = 5) -> Dict:
        """
        Evaluate trained agent.
        
        Args:
            model: Trained RL model
            num_episodes: Number of episodes to run
            
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            print(f"  Episode {episode+1}/{num_episodes}: Reward={episode_reward:.2f}, Steps={steps}")
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards
        }
    
    def baseline_rule_based(self, num_episodes: int = 5) -> Dict:
        """
        Evaluate rule-based baseline controller.
        
        Simple rule: Charge if SOC < 40%, Discharge if SOC > 80%, else do nothing
        
        Args:
            num_episodes: Number of episodes
            
        Returns:
            Baseline metrics
        """
        print("\n" + "=" * 60)
        print("Evaluating Rule-Based Baseline")
        print("=" * 60)
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Simple rule-based policy
                soc = obs[0]  # SOC is first element of observation
                
                if soc < 0.4:
                    action = 0  # Charge
                elif soc > 0.8:
                    action = 1  # Discharge
                else:
                    action = 2  # Direct supply
                
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            print(f"  Episode {episode+1}/{num_episodes}: Reward={episode_reward:.2f}")
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episode_rewards': episode_rewards
        }
    
    def baseline_greedy(self, num_episodes: int = 5) -> Dict:
        """
        Evaluate greedy baseline: always maximize immediate reward.
        
        Greedy policy: Always discharge if load is high, charge if solar is high
        
        Args:
            num_episodes: Number of episodes
            
        Returns:
            Baseline metrics
        """
        print("\n" + "=" * 60)
        print("Evaluating Greedy Baseline")
        print("=" * 60)
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Greedy policy: choose action that seems best right now
                solar = obs[1]
                load = obs[2]
                soc = obs[0]
                
                # If high load and high SOC, discharge
                if load > 0.5 and soc > 0.5:
                    action = 1  # Discharge
                # If high solar, charge
                elif solar > 0.5 and soc < 0.8:
                    action = 0  # Charge
                # Otherwise direct supply
                else:
                    action = 2  # Direct
                
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            print(f"  Episode {episode+1}/{num_episodes}: Reward={episode_reward:.2f}")
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episode_rewards': episode_rewards
        }
    
    def compare_controllers(self, dqn_model, ppo_model) -> pd.DataFrame:
        """
        Compare all controllers: Rule-based, Greedy, DQN, PPO
        
        Args:
            dqn_model: Trained DQN model
            ppo_model: Trained PPO model
            
        Returns:
            Comparison DataFrame
        """
        print("\n" + "=" * 60)
        print("Comparing All Controllers")
        print("=" * 60)
        
        num_eval_episodes = 5
        
        # Evaluate all
        results = {
            'Rule-Based': self.baseline_rule_based(num_eval_episodes),
            'Greedy': self.baseline_greedy(num_eval_episodes),
            'DQN': self.evaluate_agent(dqn_model, num_eval_episodes),
            'PPO': self.evaluate_agent(ppo_model, num_eval_episodes),
        }
        
        # Create comparison table
        comparison = pd.DataFrame({
            'Controller': list(results.keys()),
            'Mean Reward': [results[c]['mean_reward'] for c in results.keys()],
            'Std Reward': [results[c]['std_reward'] for c in results.keys()],
            'Min Reward': [results[c]['min_reward'] for c in results.keys()],
            'Max Reward': [results[c]['max_reward'] for c in results.keys()],
        })
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(comparison.to_string(index=False))
        
        # Save
        comparison_path = os.path.join(self.output_dir, 'controller_comparison.csv')
        comparison.to_csv(comparison_path, index=False)
        print(f"\n✓ Comparison saved to {comparison_path}")
        
        return comparison
    
    def plot_comparison(self, comparison: pd.DataFrame):
        """Plot comparison results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Mean reward
        controllers = comparison['Controller']
        mean_rewards = comparison['Mean Reward']
        std_rewards = comparison['Std Reward']
        
        axes[0].bar(controllers, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
        axes[0].set_ylabel('Mean Episode Reward')
        axes[0].set_title('Mean Reward Comparison')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Improvement over baseline
        baseline_reward = mean_rewards[0]  # Rule-based
        improvement = ((mean_rewards - baseline_reward) / abs(baseline_reward) * 100)
        
        colors = ['gray', 'orange', 'blue', 'green']
        axes[1].bar(controllers, improvement, color=colors, alpha=0.7)
        axes[1].set_ylabel('Improvement over Rule-Based (%)')
        axes[1].set_title('Performance Improvement')
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'controller_comparison.png')
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Comparison plot saved to {plot_path}")
        plt.close()


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("SMART MICROGRID RL TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Create environment
    print("\n[Step 1] Creating environment...")
    env = create_demo_environment(num_days=30)  # 30 days for quick test
    print(f"✓ Environment created")
    
    # Step 2: Initialize trainer
    print("\n[Step 2] Initializing trainer...")
    trainer = MicrogridRLTrainer(env, output_dir='./microgrid_results')
    print(f"✓ Trainer initialized")
    
    # Step 3: Train DQN
    print("\n[Step 3] Training DQN...")
    dqn_model = trainer.train_dqn(
        total_timesteps=50000,  # Reduced for demo (normally 100k+)
        learning_rate=1e-4,
        batch_size=32,
        buffer_size=50000
    )
    print(f"✓ DQN training complete")
    
    # Step 4: Train PPO
    print("\n[Step 4] Training PPO...")
    ppo_model = trainer.train_ppo(
        total_timesteps=50000,  # Reduced for demo
        learning_rate=1e-4,
        batch_size=64,
        n_steps=2048
    )
    print(f"✓ PPO training complete")
    
    # Step 5: Compare controllers
    print("\n[Step 5] Comparing controllers...")
    comparison = trainer.compare_controllers(dqn_model, ppo_model)
    trainer.plot_comparison(comparison)
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: ./microgrid_results/")
    print(f"  - dqn_model.zip (trained DQN)")
    print(f"  - ppo_model.zip (trained PPO)")
    print(f"  - controller_comparison.csv (metrics table)")
    print(f"  - controller_comparison.png (visualization)")


if __name__ == "__main__":
    main()
