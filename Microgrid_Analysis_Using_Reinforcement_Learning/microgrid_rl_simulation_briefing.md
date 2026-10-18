# Smart Microgrid + RL Simulation — Complete Technical Briefing

## Table of Contents
1. [High-Level Architecture](#architecture)
2. [Simulation Frameworks & Tools](#frameworks)
3. [RL Training Libraries](#rl-libraries)
4. [Data Sources for India](#data-sources)
5. [Development Workflow](#workflow)
6. [Resource Links & Getting Started](#resources)

---

## Architecture

Your simulation stack will look like this:

```
┌─────────────────────────────────────────────────────────┐
│  NASA POWER Solar Irradiance Data (Tamil Nadu)          │
│  + Synthetic Load Profiles                               │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  PyMGrid Microgrid Environment (Physics Simulation)     │
│  - Solar Panel (IV curve modeling)                       │
│  - Battery (SOC, efficiency, degradation)               │
│  - Loads (controllable & non-controllable)              │
│  - Grid Connection (price signals)                      │
│  - State: [SOC, P_solar, P_load, price, temp, forecast]│
└──────────────────┬──────────────────────────────────────┘
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
┌──────────────────┐  ┌──────────────────────┐
│ LSTM Forecaster  │  │ RL Agent (DQN/PPO)   │
│ (trained offline)│  │ (trained in PyMGrid) │
└──────────────────┘  └──────────────────────┘
      │                         │
      └────────────┬────────────┘
                   ▼
    ┌──────────────────────────┐
    │ TensorFlow Lite Export    │
    │ (for embedded deployment) │
    └──────────────────────────┘
```

---

## Frameworks

### 1. **PyMGrid** — The Core Simulator
**What it is:** Open-source Python library for microgrid simulation, purpose-built for RL research.

**GitHub:** https://github.com/Total-RD/pymgrid  
**Paper:** arXiv 2011.08004 (NeurIPS 2020 Workshop)

**Why use it:**
- Abstracts physics (you don't rebuild PV models from scratch)
- Generates 600+ different microgrid configurations
- Built as an OpenAI Gym environment (native RL compatibility)
- Includes baseline controllers (rule-based, MPC, Q-learning) for comparison
- Pre-computed benchmark datasets: `pymgrid10` and `pymgrid25`

**Components PyMGrid provides:**
- `GensetModule` — diesel generator (diesel gen not needed for your small system, skip it)
- `BatteryModule` — lithium-ion or lead-acid with efficiency & degradation
- `RenewableModule` — solar PV with customizable power curves
- `LoadModule` — both fixed and controllable loads
- `GridModule` — grid connection with time-varying prices
- Automatic energy balance & unmet load penalties

**Key advantage:** PyMGrid is built *for* RL. It implements the MDP automatically — you define modules, it handles state transitions, rewards, and gym-compatible step() function.

**What to know:**
- Data comes from US datasets (DOE OpenEI, TMY3). You'll replace this with NASA POWER data for Tamil Nadu.
- Takes hourly timestep by default. You can set it to 30-minute or 15-minute intervals if needed.

---

### 2. **CityLearn** — Alternative (More Complex)
**GitHub:** https://github.com/intelligent-environments-lab/CityLearn  
**Status:** v1.0 released, actively maintained

**When to use it:** If you want a more sophisticated multi-building district-level simulation.

**Why NOT use it for your scope:** CityLearn is overkill for a single household/small microgrid. It's designed for multi-building demand response. PyMGrid is lighter and more directly applicable to your 10-20W scale project.

---

### 3. **EasyGrid** — Lightweight Custom Alternative
**GitHub:** https://github.com/YannBerthelot/easygrid  
**Status:** Newer, simpler than PyMGrid

**Advantage:** Fewer dependencies, easier to customize for your specific 1-2L tank system needs.

**Disadvantage:** Less mature, fewer pre-built baselines for comparison.

**Recommendation:** Start with PyMGrid, consider EasyGrid if you need ultra-lightweight or want to build custom modules.

---

## RL Libraries

### **Stable-Baselines3 (SB3)** — Your Training Framework
**GitHub:** https://github.com/DLR-RM/stable-baselines3  
**Docs:** https://stable-baselines3.readthedocs.io  
**Paper:** JMLR 2021

**What it is:** PyTorch-based RL implementations with production-ready algorithms.

**Why use SB3:**
- DQN and PPO are battle-tested (used in production energy management systems)
- Interfaces directly with Gym/Gymnasium environments (PyMGrid-compatible out of the box)
- 95% code coverage in unit tests — reliability matters for your paper
- Can export to TFLite for edge deployment (your ESP32/Pi)
- Active maintenance & community

**Algorithms you'll use:**
| Algorithm | Type | Best For | Action Space |
|-----------|------|----------|--------------|
| **DQN** | Value-based | Discrete actions, simple | Discrete only |
| **PPO** | Actor-Critic | Stable training, general purpose | Discrete & Continuous |
| **DDPG** | Actor-Critic | Continuous control | Continuous only |

**For your microgrid:** Start with **DQN** (simple, well-documented). Compare against **PPO** (more stable training). This comparison is your paper's algorithm evaluation section.

**Key hyperparameters you'll tune:**
- `learning_rate` — typically 1e-4 to 1e-3
- `batch_size` — 32 or 64 (small microgrid = small batches)
- `buffer_size` — experience replay memory (10,000 to 100,000)
- `exploration fraction` — epsilon decay schedule (0.1 to 0.2)
- `target_update_interval` — how often to update target network (1,000 to 10,000)

---

## Data Sources for India

### **NASA POWER API** — Your Primary Data Source

**Access:** https://power.larc.nasa.gov

**Coverage:** Global, including Tamil Nadu, with data since 1984.

**Parameters available:**
- `ALLSKY_SFC_SW_DWN` — Surface shortwave downwind radiation (global horizontal irradiance)
- `T2M` — 2m air temperature
- `RH2M` — Relative humidity
- `WS10M` — 10m wind speed
- `PS` — Surface pressure

**Temporal options:**
- Hourly (best for microgrid simulation)
- Daily
- Monthly
- Climatology (multi-year averages)

**Resolution:** 0.5° × 0.625° grid (about 50 km at equator). For Hosur/Tamil Nadu, you'll get one representative cell.

**Key advantage:** Freely available, no authentication needed for small queries, covers 40+ years of data enabling seasonal analysis.

**How to download:**
1. **Web interface:** https://power.larc.nasa.gov/data-access-viewer/ — point-and-click for quick exploration
2. **API:** REST-based, queryable from Python
3. **Python package:** `nasapower` (R package available too)

**Example API call (Python):**
```python
# Pseudo-code for illustration
import requests

response = requests.get(
    "https://power.larc.nasa.gov/api/v1/power/daily/point",
    params={
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,RH2M",
        "community": "RE",  # Renewable Energy archive
        "longitude": 78.14,  # Hosur approximate
        "latitude": 12.74,
        "start": "20200101",
        "end": "20231231",
        "format": "JSON"
    }
)
```

**Data format:** JSON or CSV with daily/hourly values.

---

### **TMY3 (Typical Meteorological Year)** — Alternative
- Used in PyMGrid's pre-built datasets
- US-focused, but methodology is sound
- Can use as fallback or for initial prototyping

---

### **NREL OpenEI Load Profiles**
**Link:** https://openei.org/datasets/dataset/commercial-and-residential-hourly-load-profiles-for-all-tmy3-locations

- US residential and commercial hourly load profiles
- Use as synthetic load generation for your simulation
- Create a simple load profile generator: baseline (night) + peak (morning/evening) + stochastic noise

---

## Development Workflow

### **Phase 1: Simulation Setup (Weeks 1-2)**

**Step 1: Install PyMGrid**
```bash
pip install pymgrid
# Check: imports gym (Gymnasium now, formerly OpenAI Gym)
```

**Step 2: Create a custom microgrid**
- Define battery: 12V, 7Ah capacity (roughly 84Wh)
- Define solar: 10W nominal (peak, use derating curves for real-world)
- Define load: constant baseline (100W) + controllable (fan, pump)
- Define grid connection: assume fixed grid price (₹8/kWh) + optional time-varying tariff

**Step 3: Download NASA POWER data**
- Query for Hosur coordinates (12.74°N, 78.14°E)
- Download 1 year of hourly data: `ALLSKY_SFC_SW_DWN`, `T2M`, `RH2M`
- Convert to CSV, validate (no missing values)

**Step 4: Generate synthetic load profile**
- Create a daily profile: baseline (night 2–4 AM = 50W, day = 150W, peak = 300W evening)
- Add stochastic noise (±20%)
- Repeat for 365 days (hourly resolution)

**Step 5: Create custom PyMGrid environment**
```python
from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, RenewableModule, LoadModule, GridModule

# Define components
battery = BatteryModule(
    min_capacity=0,
    max_capacity=84,  # 12V x 7Ah = 84Wh
    max_charge=50,    # W
    max_discharge=50, # W
    efficiency=0.95,
    init_soc=0.5
)

renewable = RenewableModule(time_series=your_solar_data)  # From NASA POWER
load = LoadModule(time_series=your_load_data, loss_load_cost=100)
grid = GridModule(time_series=electricity_prices)

microgrid = Microgrid([
    ("solar", renewable),
    ("battery", battery),
    ("load", load),
    ("grid", grid)
])
```

---

### **Phase 2: LSTM Forecasting (Weeks 2-3)**

**Goal:** Train a separate LSTM model to predict solar output 30 minutes ahead.

**Steps:**
1. Use your NASA POWER data (hourly)
2. Create sequences: input 24 historical hours → output next hour's irradiance
3. Train on 10 months, validate on 2 months
4. Evaluate: RMSE, MAE, R²
5. Export model as `.h5` or SavedModel format

**Framework:** TensorFlow/Keras or PyTorch
- Keras is simpler for time series
- Layer structure: Input(24) → LSTM(64) → Dense(32) → Output(1)
- Loss: MSE, Optimizer: Adam

**Integration:** Load the trained LSTM weights in your RL environment, use it to generate a forecast feature in the state vector.

---

### **Phase 3: RL Agent Training (Weeks 3-5)**

**Step 1: Define reward function**
```python
# Pseudocode
def compute_reward(action, state, next_state):
    energy_saved = (self.grid_imports[t-1] - self.grid_imports[t])
    grid_cost = self.electricity_prices[t] * self.grid_imports[t]
    battery_wear = degradation_factor * (self.battery.charge_cycles - last_cycles)
    unmet_load = 10 * max(0, required_load - supplied_load)
    
    reward = (α * energy_saved 
              - β * grid_cost 
              - γ * battery_wear 
              - δ * unmet_load)
    return reward
```

**Tune α, β, γ, δ through ablation study.**

**Step 2: Wrap environment for SB3**
```python
from stable_baselines3 import DQN, PPO

model = DQN(
    "MlpPolicy",                    # Fully connected neural network
    env=microgrid,                  # Your PyMGrid environment
    learning_rate=1e-4,
    buffer_size=50000,
    batch_size=32,
    exploration_fraction=0.15,
    target_update_interval=5000,
    verbose=1
)

model.learn(total_timesteps=100000)  # Train for 100k steps (~11 days of hourly data)
```

**Step 3: Compare baselines**
- Run the same environment with:
  - Rule-based controller (fixed thresholds)
  - Greedy controller (always maximize current reward)
  - Your DQN agent
  - Your PPO agent
- Metric: annual grid import cost, unmet load events, battery cycle count

**Step 4: Create comparison table (your research result)**
| Controller | Annual Cost (₹) | Unmet Load (%) | Battery Cycles |
|------------|-----------------|----------------|-----------------|
| Rule-based | 28,500 | 2.1 | 245 |
| Greedy | 26,800 | 3.4 | 312 |
| DQN | 24,200 | 0.8 | 198 |
| PPO | 23,900 | 0.6 | 189 |

---

### **Phase 4: Fine-tuning & Deployment (Week 5+)**

**Step 1: Test on unseen data**
- Train on 2020-2021 data
- Evaluate on 2022 data (different seasonal pattern)
- Measure transfer learning performance

**Step 2: Quantization for edge deployment**
```python
import tensorflow as tf

# Convert SB3 PyTorch model to TensorFlow
# Then quantize for TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
```

**Step 3: Export policy as C code (optional)**
- Decision trees can be hand-coded
- Neural networks can be converted to embedded-friendly formats

---

## Tools & Software You'll Need

### **Required (Free & Open Source)**
| Tool | Purpose | Installation |
|------|---------|--------------|
| Python 3.10+ | Base language | `sudo apt install python3` |
| PyMGrid | Microgrid simulation | `pip install pymgrid` |
| Stable-Baselines3 | RL algorithms | `pip install stable-baselines3[extra]` |
| PyTorch | Neural net backend | `pip install torch` |
| TensorFlow | LSTM, TFLite conversion | `pip install tensorflow` |
| NumPy, Pandas | Data handling | `pip install numpy pandas` |
| Matplotlib | Plotting results | `pip install matplotlib` |
| Jupyter | Interactive development | `pip install jupyter` |

### **Recommended (Free)**
| Tool | Purpose |
|------|---------|
| TensorBoard | RL training visualization |
| Gymnasium | Modern Gym fork (PyMGrid uses this) |
| RL Baselines3 Zoo | Pre-trained models & scripts |

### **Optional (For Embedded Deployment)**
| Tool | Purpose |
|------|---------|
| TensorFlow Lite | Model compression for ESP32 |
| TVM (Apache) | Further optimization for microcontrollers |
| ONNX | Cross-framework model conversion |

---

## Development Environment Setup

### **Option 1: Local Machine (Recommended for prototyping)**
```bash
# Create virtual environment
python3 -m venv microgrid_rl
source microgrid_rl/bin/activate  # On Windows: microgrid_rl\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install pymgrid stable-baselines3[extra] tensorflow torch numpy pandas matplotlib jupyter

# Verify
python -c "import pymgrid; print('PyMGrid OK')"
python -c "from stable_baselines3 import DQN; print('SB3 OK')"
```

### **Option 2: Google Colab (Free GPU)**
- Upload your NASA POWER CSV to Drive
- Mount Drive in Colab: `from google.colab import drive; drive.mount('/content/drive')`
- Install packages same as above
- Training will be faster with free GPU

### **Option 3: Docker Container (For reproducibility)**
```dockerfile
FROM python:3.10
RUN pip install pymgrid stable-baselines3[extra] tensorflow torch jupyter
WORKDIR /app
CMD ["jupyter", "notebook", "--ip=0.0.0.0"]
```

---

## Key Research Concepts to Understand

### **MDP Formulation**
Your environment is an MDP with:
- **State S:** `[SOC, P_solar, P_load, price, temp, LSTM_forecast]` (6 dimensions)
- **Action A:** `{charge, discharge, direct_supply, shed_load, grid_import}` (5 discrete actions)
- **Reward R:** Your designed function balancing cost, efficiency, battery life
- **Transition T:** Physics equations (battery ODE, energy balance constraint)

### **Experience Replay (DQN Key Innovation)**
Transition buffer stores (S, A, R, S') tuples. Instead of learning from consecutive steps (correlated), sample random mini-batches. Prevents catastrophic forgetting and improves stability.

### **Target Network (DQN Key Innovation)**
Two networks:
- **Online network:** Updated every step (noisy estimates)
- **Target network:** Updated every N steps (stable targets)

This decoupling prevents the "moving target" problem that crashes naive Q-learning.

### **Exploration-Exploitation Trade-off**
- **Epsilon-greedy:** With probability ε, take random action; with probability (1-ε), take best action
- ε starts high (explore broadly), decays over time (exploit learned policy)
- Controlled by `exploration_fraction` hyperparameter

---

## Recommended Learning Resources

### **RL Fundamentals (if new to the field)**
1. **Sutton & Barto** — "Reinforcement Learning: An Introduction" (free PDF available)
2. **DeepMind's Spinning Up in Deep RL** — https://spinningup.openai.com
3. **SB3 Documentation Guides** — https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

### **Microgrid-Specific RL Papers**
1. *"Deep Reinforcement Learning for Real-Time Economic Energy Management of Microgrid System Considering Uncertainties"* — Frontiers, 2023. Compares DQN, PPO, SAC as benchmarks.
2. *"Deep Reinforcement Learning Microgrid Optimization Strategy Considering Priority Flexible Demand Side"* — PMC, 2022. Shows PPO > DQN in convergence speed.
3. *"Day-Ahead Solar Irradiance Forecasting for Microgrids Using LSTM-RNN"* — Energies, 2019. LSTM forecasting improves microgrid operation by ~2%.

### **PyMGrid Specific**
- **Paper:** https://arxiv.org/abs/2011.08004
- **GitHub:** Complete example notebooks in `/examples`
- **Documentation:** https://pymgrid.readthedocs.io

---

## Deliverables Roadmap

### **Demonstration Level (What you're building for Phase-I Eval)**

**Week 1-2 Output:**
- Custom PyMGrid microgrid configuration
- NASA POWER data downloaded and validated
- Synthetic load profiles generated
- Rule-based and greedy baselines implemented

**Week 2-3 Output:**
- LSTM trained on solar irradiance
- Forecast accuracy metrics (RMSE, MAE)
- Integration of forecast into environment state

**Week 3-5 Output:**
- DQN agent trained (100k steps)
- PPO agent trained (100k steps)
- Comparison table: Rule vs Greedy vs DQN vs PPO
- Learning curves (reward over training steps)

**Week 5+ Output:**
- Cross-validation on unseen year of data
- TFLite model exported (~2-5 MB)
- Performance metrics for paper (annual cost savings %, battery life improvement %, etc.)

---

## Common Pitfalls & How to Avoid Them

### **Pitfall 1: Forgetting state normalization**
**Problem:** Neural networks train badly on raw sensor values (0-100 SOC, 0-1000 W power)
**Solution:** Normalize all state dimensions to [0, 1] before feeding to network
```python
state_normalized = np.array([
    soc / 100,                    # SOC in [0,1]
    power_solar / max_solar,      # Power in [0,1]
    power_load / max_load,        # Power in [0,1]
    electricity_price / 20,       # Price normalized
    temperature / 50,             # Temp normalized
    forecast / max_solar          # Forecast normalized
])
```

### **Pitfall 2: Reward engineering gone wrong**
**Problem:** Large reward weights cause agent to ignore other objectives
**Solution:** Start with equal weights (α=β=γ=δ=1), then ablation study
```python
# Log each reward component separately
reward_components = {
    'energy_saved': alpha * energy_saved,
    'grid_cost': -beta * grid_cost,
    'battery_wear': -gamma * battery_wear,
    'unmet_load': -delta * unmet_load
}
```

### **Pitfall 3: Training on tiny dataset**
**Problem:** 365 days = 8760 timesteps. One episode ≠ enough exploration
**Solution:** Train for 100k+ timesteps (11+ years worth), use experience replay
```python
model.learn(total_timesteps=100000)  # ~11 years of hourly data
```

### **Pitfall 4: Overfitting to training year**
**Problem:** Agent learns calendar patterns specific to 2020
**Solution:** Always validate on different year
```python
# Train: 2020-2021
# Test: 2022
# Measure transfer performance
```

---

## Next Steps

1. **This week:** Install PyMGrid, explore example notebooks
2. **Next week:** Download NASA POWER data, create custom microgrid
3. **Week 2:** Train LSTM forecaster
4. **Week 3+:** Train DQN/PPO agents, run comparisons
5. **End of Phase:** Write methods section of paper

---

## Quick Command Reference

```bash
# Create environment
python3 -m venv mg_env
source mg_env/bin/activate

# Install all dependencies
pip install pymgrid stable-baselines3[extra] tensorflow torch jupyter numpy pandas

# Start Jupyter notebook
jupyter notebook

# Run training (inside Jupyter or Python script)
from stable_baselines3 import DQN
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("dqn_microgrid")

# Load and test
model = DQN.load("dqn_microgrid")
obs, info = env.reset()
action, _ = model.predict(obs)
```

---

## Final Notes

- **Start simple:** Get PyMGrid working with dummy data first
- **Iterate:** Perfect state representation, then tune rewards, then train agents
- **Document everything:** Save hyperparameters, training curves, final weights
- **Research mindset:** Every failed experiment teaches something — document failures too

Your simulation is your "digital twin" — the fidelity of this determines how well your trained agent transfers to real hardware in Phase II.

**Good luck! 🚀**
