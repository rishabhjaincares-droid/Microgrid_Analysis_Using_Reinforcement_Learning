# ⚡ Smart Microgrid Energy Management System (RL & LSTM)

Welcome to the **Smart Microgrid Energy Management System** repository! This project implements an intelligent, data-driven control system for optimizing microgrid operations. It leverages Deep Reinforcement Learning (PPO, DQN) for decision-making and Long Short-Term Memory (LSTM) networks for forecasting energy generation and load demand. 

By integrating real-world meteorological data from NASA POWER, this project demonstrates how AI can be used to balance grid stability, minimize reliance on the main grid, and optimize battery storage (ESS) usage.

## 🚀 Features

* **Custom Microgrid Environment**: Built a custom simulation environment (`microgrid_env.py`, `custom_env.py`) compatible with OpenAI Gym/Gymnasium.
* **Deep Reinforcement Learning Agents**: Proximal Policy Optimization (PPO) and Deep Q-Networks (DQN) applied to energy management and battery dispatch (`train_rl_agents.py`, `train.py`).
* **LSTM Forecasting**: Neural network-based forecasting for future solar/wind generation and load demand (`lstm_forecaster.py`).
* **Real-World Data Integration**: Scripts to fetch and parse actual meteorological data directly from the NASA POWER API (`download_nasa_power.py`, `data_loader.py`).
* **Controller Benchmarking**: Compare the RL models against baseline, rule-based microgrid controllers (`controllers.py`, `evaluate.py`).
* **Interactive Dashboard**: A clean UI to visualize load, generation, state of charge (SoC), and agent actions (`dashboard.py`).
* **Automated Reporting**: Auto-generates summary PDF reports of microgrid performance over specified timeframes (`generate_microgrid_report_pdf.py`).

## 📁 Repository Structure

```text
├── QUICKSTART.py                       # 🚀 Entry point to quickly run the simulation
├── config.py                           # Global configuration parameters
├── microgrid_env.py / custom_env.py    # Custom Gym environments for the microgrid
├── data_loader.py                      # Data processing and preparation
├── download_nasa_power.py              # Script to fetch NASA weather data
├── load_simulator.py                   # Generates synthetic/simulated load profiles
├── lstm_forecaster.py                  # LSTM model for load and generation forecasting
├── controllers.py                      # Rule-based and baseline logic controllers
├── train_rl_agents.py / train.py       # Scripts for training PPO/DQN agents
├── evaluate.py                         # Evaluation and comparison of controllers
├── dashboard.py                        # Interactive Streamlit/Dash visualization dashboard
├── generate_microgrid_report_pdf.py    # Automated PDF technical report generator
├── requirements.txt                    # Python dependencies
├── ppo_model.zip / dqn_model.zip       # Pre-trained RL models
└── *.md / *.pdf                        # Project documentation and checklists
