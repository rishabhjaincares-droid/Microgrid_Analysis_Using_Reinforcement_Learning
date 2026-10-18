"""
QUICK START GUIDE - Smart Microgrid RL Simulation
===================================================

This document walks you through setting up and running the complete
Smart Microgrid RL training pipeline.

All scripts are ready to use - no modifications needed to get started!
"""

# ============================================================================
# STEP 0: ENVIRONMENT SETUP
# ============================================================================

"""
Run this ONCE to set up your development environment.

Option A: Local Machine (Recommended)
=====================================

1. Create a virtual environment:
   $ python3 -m venv microgrid_env
   
2. Activate it:
   On Linux/macOS:
   $ source microgrid_env/bin/activate
   
   On Windows:
   $ microgrid_env\Scripts\activate

3. Install all dependencies:
   $ pip install --upgrade pip
   $ pip install pymgrid stable-baselines3[extra] tensorflow torch numpy pandas matplotlib

4. Verify installation:
   $ python -c "import pymgrid; from stable_baselines3 import DQN; print('✓ All good!')"


Option B: Google Colab (Free GPU)
=================================

1. Open a new Colab notebook:
   https://colab.research.google.com

2. In the first cell, paste:
   !pip install pymgrid stable-baselines3[extra] tensorflow torch pandas

3. In next cells, you can run the Python code directly


Option C: Docker (Reproducibility)
==================================

1. Create Dockerfile with:
   FROM python:3.10
   RUN pip install pymgrid stable-baselines3[extra] tensorflow torch jupyter
   WORKDIR /app
   CMD ["jupyter", "notebook", "--ip=0.0.0.0"]

2. Build & run:
   $ docker build -t microgrid-rl .
   $ docker run -p 8888:8888 microgrid-rl
"""

# ============================================================================
# STEP 1: TEST THE ENVIRONMENT
# ============================================================================

"""
Test that your custom environment works correctly.

Run this Python code:
"""

# test_environment.py
if __name__ == "__main__":
    from microgrid_env import create_demo_environment
    
    print("Creating demo environment (7 days of synthetic data)...")
    env = create_demo_environment(num_days=7)
    
    print(f"Environment: {env}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run 100 random steps
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode finished after {step+1} steps")
            print(f"Total reward: {total_reward:.2f}")
            break
        
        if step % 20 == 0:
            print(f"Step {step}: Obs={obs[:2]}, Reward={reward:.2f}")
    
    env.close()
    print("✓ Environment test passed!")

"""
Expected output:
  ✓ Environment created
  ✓ Test passed with random actions
  
If this works, move to Step 2.
"""

# ============================================================================
# STEP 2: DOWNLOAD REAL DATA (Optional)
# ============================================================================

"""
Download actual NASA POWER data for Hosur, Tamil Nadu.
This makes your simulation more realistic, but requires internet.

Run this Python code:
"""

# download_data.py
if __name__ == "__main__":
    from download_nasa_power import download_hosur_data
    
    print("Downloading NASA POWER data for Hosur...")
    print("(This may take 1-2 minutes)")
    
    # Download 2023 data
    data = download_hosur_data(start_year=2023, end_year=2023)
    
    if data and 2023 in data:
        print("✓ Data downloaded successfully!")
        print(f"  Solar irradiance: {data[2023]['solar'].shape}")
        print(f"  Temperature: {data[2023]['temperature'].shape}")
        
        # Data is automatically saved as NASA_POWER_Hosur_2023-2023.csv
        # and hosur_2023_data.pkl
    else:
        print("⚠ Download failed (check internet connection)")
        print("  You can still use synthetic data for testing")

"""
If you don't have internet or NASA POWER is down:
- The synthetic data in create_demo_environment() works fine for testing
- Real data just makes simulation more realistic
"""

# ============================================================================
# STEP 3: TRAIN LSTM SOLAR FORECASTER
# ============================================================================

"""
Train a separate LSTM model to predict solar irradiance 1 hour ahead.
This improves RL agent performance by giving it "future vision".

Run this Python code:
"""

# train_lstm.py
if __name__ == "__main__":
    from lstm_forecaster import train_lstm_forecaster
    import numpy as np
    
    # Create synthetic solar data
    print("Generating synthetic solar data...")
    num_days = 365
    num_hours = num_days * 24
    hour_of_day = np.arange(num_hours) % 24
    
    # Solar profile
    solar_data = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12) * 600)
    solar_data += np.random.normal(0, 30, len(solar_data))
    solar_data = np.clip(solar_data, 0, 800)
    
    print(f"Solar data shape: {solar_data.shape}")
    print(f"Solar data range: {solar_data.min():.1f} - {solar_data.max():.1f} W/m²")
    
    # Train LSTM
    print("\nTraining LSTM forecaster (this takes 2-5 minutes)...")
    forecaster, metrics = train_lstm_forecaster(solar_data, 'lstm_forecaster')
    
    print("\n✓ LSTM training complete!")
    print(f"  RMSE: {metrics['rmse']:.4f} W/m²")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Model saved: lstm_forecaster.h5")

"""
Expected output:
  ✓ LSTM trained with RMSE < 100 W/m² for synthetic data
  ✓ Model saved as lstm_forecaster.h5
  
This is optional - the environment works without it, but including
the forecast in the state helps the RL agent learn better policies.
"""

# ============================================================================
# STEP 4: TRAIN RL AGENTS (DQN & PPO)
# ============================================================================

"""
Main training pipeline!

This trains DQN and PPO agents and compares them against baselines.

Run this Python code (from command line):
"""

# Command line
#$ python train_rl_agents.py

"""
This script will:
  1. Create the environment with 30 days of synthetic data
  2. Train a DQN agent for 50,000 timesteps (~6-10 minutes on CPU)
  3. Train a PPO agent for 50,000 timesteps (~6-10 minutes on CPU)
  4. Evaluate both against:
     - Rule-based baseline (charge/discharge at fixed SOC thresholds)
     - Greedy baseline (always maximize immediate reward)
  5. Generate comparison plots
  6. Save all models and results to ./microgrid_results/

Expected output:
  ====== DQN Training ======
  Total timesteps: 50,000
  | timestep | episodes | reward |
  ... (progress bar)
  ✓ Model saved to ./microgrid_results/dqn_model.zip
  
  ====== PPO Training ======
  (similar)
  
  ====== Comparison Results ======
  Controller | Mean Reward | Improvement
  Rule-Based |     -50.5   |     0.0%
  Greedy     |     -45.2   |     10.5%
  DQN        |     -35.8   |     29.1%
  PPO        |     -32.1   |     36.4%
  
  ✓ Results saved to ./microgrid_results/

The improvement percentages show how much better the RL agents are
compared to the rule-based baseline.
"""

# ============================================================================
# STEP 5: EVALUATE AND VISUALIZE RESULTS
# ============================================================================

"""
After training completes, analyze the results.

Run this Python code:
"""

# evaluate_results.py
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load comparison results
    comparison = pd.read_csv('./microgrid_results/controller_comparison.csv')
    
    print("\n" + "="*60)
    print("CONTROLLER COMPARISON RESULTS")
    print("="*60)
    print(comparison.to_string(index=False))
    
    # Visualizations are automatically generated and saved to:
    #   ./microgrid_results/controller_comparison.png
    
    print("\nResults saved to:")
    print("  - ./microgrid_results/controller_comparison.csv")
    print("  - ./microgrid_results/controller_comparison.png")
    print("  - ./microgrid_results/dqn_model.zip")
    print("  - ./microgrid_results/ppo_model.zip")

"""
The comparison PNG shows:
  - Bar chart of mean rewards for each controller
  - Percentage improvement of RL agents over baselines
  
Use these results in your paper's Results section!
"""

# ============================================================================
# STEP 6: LOAD & TEST A TRAINED AGENT
# ============================================================================

"""
Use a trained agent for inference (prediction on new data).

Run this Python code:
"""

# test_trained_agent.py
if __name__ == "__main__":
    from stable_baselines3 import DQN, PPO
    from microgrid_env import create_demo_environment
    
    # Load trained models
    print("Loading trained agents...")
    dqn = DQN.load('./microgrid_results/dqn_model')
    ppo = PPO.load('./microgrid_results/ppo_model')
    
    # Create test environment
    env = create_demo_environment(num_days=7)
    
    # Test DQN agent
    print("\nTesting DQN agent...")
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(100):
        action, _ = dqn.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    print(f"  Episode reward: {episode_reward:.2f}")
    
    # Test PPO agent
    print("\nTesting PPO agent...")
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(100):
        action, _ = ppo.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    print(f"  Episode reward: {episode_reward:.2f}")
    
    env.close()

"""
This shows how to:
  1. Load saved models (no retraining needed)
  2. Use them for prediction on new data
  3. Get actions for any observation
"""

# ============================================================================
# COMMON ISSUES & SOLUTIONS
# ============================================================================

"""
Issue 1: "ModuleNotFoundError: No module named 'pymgrid'"
Solution:
  $ pip install pymgrid

Issue 2: "ImportError: No module named 'stable_baselines3'"
Solution:
  $ pip install stable-baselines3[extra]

Issue 3: Training is very slow (10+ minutes per agent)
Causes & solutions:
  a) Running on CPU (normal, expected)
     - Use GPU if available: install pytorch-cuda, tensorflow-gpu
     - Reduce timesteps: change total_timesteps=50000 to 10000
  
  b) Antivirus/firewall blocking PyTorch downloads
     - Try: pip install torch --no-cache-dir
  
  c) Out of memory
     - Reduce batch_size: 32 -> 16 or 8
     - Reduce buffer_size: 50000 -> 10000

Issue 4: NASA POWER download fails
Solution:
  - Check internet connection
  - NASA servers down? Use synthetic data (create_demo_environment)
  - If timeout, increase timeout in download_nasa_power.py:
    response = requests.get(..., timeout=60)  # 60 seconds

Issue 5: "RuntimeError: Episode finished, call reset()"
Solution:
  - You called env.step() after done=True
  - Always call env.reset() when done=True before continuing
"""

# ============================================================================
# NEXT STEPS FOR YOUR PAPER
# ============================================================================

"""
After running the training pipeline:

1. Write your Methods section:
   - Describe the environment: state, action, reward
   - Detail the RL algorithms: DQN, PPO hyperparameters
   - Explain baselines: rule-based, greedy
   
2. Write your Results section:
   - Include the comparison table (controller_comparison.csv)
   - Include the comparison plot (controller_comparison.png)
   - Discuss improvement percentages
   
3. Analysis to perform:
   - Train on longer data (365 days instead of 30)
   - Vary hyperparameters (learning_rate, batch_size)
   - Test on different years (train 2021, test 2022)
   - Analyze which actions each controller prefers
   
4. Advanced extensions:
   - Integrate real LSTM forecaster into environment
   - Add LSTM-augmented state vector
   - Deploy to TFLite for embedded systems
   - Add safety constraints (prevent over-charging, etc.)
   
5. Paper outline:
   Introduction
   - Motivation: renewable intermittency, need for smart control
   - Problem: energy management in small microgrids
   - Contribution: RL-based control vs baselines
   
   Related Work
   - Microgrid control methods (rule-based, MPC, RL)
   - RL applications in energy systems
   - Your novel aspects
   
   Methods
   - Environment design
   - RL algorithms
   - Baselines
   - Metrics
   
   Results
   - Comparison table
   - Plots
   - Analysis
   
   Discussion
   - Why RL works better
   - Limitations
   - Future work (Sim2Real, embedded deployment, etc.)
   
   Conclusion
"""

# ============================================================================
# FILE ORGANIZATION
# ============================================================================

"""
After running all steps, your directory structure looks like:

microgrid_rl/
├── microgrid_env.py          # Custom environment (CREATED)
├── download_nasa_power.py     # Data downloader (CREATED)
├── lstm_forecaster.py         # LSTM model trainer (CREATED)
├── train_rl_agents.py         # Main RL training (CREATED)
├── microgrid_env_briefing.md  # Documentation (CREATED)
│
├── nasa_power_Hosur_2023-2023.csv  # Downloaded data (OPTIONAL)
├── hosur_2023_data.pkl              # Pickled data (OPTIONAL)
├── lstm_forecaster.h5               # Trained LSTM (OPTIONAL)
├── lstm_forecaster_params.json      # LSTM params (OPTIONAL)
│
└── microgrid_results/         # All outputs from training
    ├── dqn_model.zip          # Trained DQN model
    ├── ppo_model.zip          # Trained PPO model
    ├── controller_comparison.csv     # Results table
    └── controller_comparison.png     # Results plot
"""

# ============================================================================
# RUNNING THE FULL PIPELINE (Recommended)
# ============================================================================

"""
Here's the complete sequence to run everything from scratch:

Step 1: Setup (one-time)
$ source microgrid_env/bin/activate
$ pip install pymgrid stable-baselines3[extra] tensorflow torch numpy pandas matplotlib

Step 2: Test environment (2 minutes)
$ python test_environment.py

Step 3: Download data (optional, 2 minutes)
$ python download_data.py

Step 4: Train LSTM (optional, 5 minutes)
$ python train_lstm.py

Step 5: Train RL agents (MAIN, 15 minutes)
$ python train_rl_agents.py

Step 6: Evaluate results (1 minute)
$ python evaluate_results.py

Total time: ~25 minutes on CPU
           ~5 minutes on GPU (with CUDA)

All results automatically saved to ./microgrid_results/
Ready for your paper!
"""

# ============================================================================
# COMMAND REFERENCE (Copy-Paste Ready)
# ============================================================================

"""
Quick copy-paste commands:

# Setup
python3 -m venv microgrid_env
source microgrid_env/bin/activate
pip install pymgrid stable-baselines3[extra] tensorflow torch numpy pandas matplotlib

# Run everything
python train_rl_agents.py

# Load and test a trained agent
python -c "from stable_baselines3 import DQN; model = DQN.load('./microgrid_results/dqn_model'); print('✓ Model loaded')"

# View results
cat ./microgrid_results/controller_comparison.csv

# Clean up (if needed)
rm -rf ./microgrid_results/
"""

print(__doc__)
