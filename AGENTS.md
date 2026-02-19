# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning project implementing tabular RL algorithms for the CartPole-v1 environment from Gymnasium (formerly OpenAI Gym). The codebase is a single-file Python implementation focused on educational purposes and experimentation.

## Environment Setup

Conda environment setup:
```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate gym-simple-rl
```

## Core Commands

### Training
Train an agent with a specific algorithm:
```bash
python main.py train --algo qlearning --seeds 123
```

### Evaluation
Evaluate a trained model with visual rendering:
```bash
python main.py eval --model_path outputs/best_cartpole_model.npy
```

### Hyperparameter Tuning
Run Optuna hyperparameter optimization:
```bash
python main.py tune --study_name sarsa_study --seeds 123 456 789 --n_trials 100 --algo sarsa
```

Additional common arguments:
- `--n_timesteps`: Training duration (default: 5,000,000)
- `--trial_prune_interval`: Episodes between pruning checks (default: 500)

### TensorBoard Monitoring
View training metrics:
```bash
tensorboard --logdir=runs/cartpole
```

## Architecture

### Single-File Design
The entire implementation is in `main.py` (480 lines). This is intentional for educational clarity - keep all logic in this file.

### Supported Algorithms
Valid values for `--algo` parameter (stored in `VALID_ALGOS`):
- `qlearning`: Q-Learning (off-policy)
- `qlearning_lambda`: Q-Learning with eligibility traces
- `sarsa`: SARSA (on-policy)
- `expected_sarsa`: Expected SARSA
- `sarsa_lambda`: SARSA with eligibility traces
- `true_online_sarsa_lambda`: True Online SARSA(λ)

### Learning Function Architecture
Each algorithm has a dedicated `_learn__<algo_name>()` function that implements its specific update rule. The `_learn()` function dispatches to the appropriate implementation. All learning functions follow the same signature:
```python
def _learn__<algo>(config, env, state, action, reward, next_state, next_action, **kwargs)
```

Key kwargs passed to learning functions:
- `q_table`: The Q-value table being updated
- `eligibility_trace`: For algorithms using eligibility traces (λ methods)
- `epsilon`: Current exploration rate (for expected_sarsa)
- Previous step parameters can be returned and passed via kwargs for stateful algorithms

### State Discretization
CartPole has continuous state space which is discretized via `_discretize_state()`:
- Clips state dimensions to reasonable ranges (defined in `state_clips`)
- Maps to discrete buckets (defined in `state_buckets`)
- Default: (12, 24, 12, 24) buckets for [cart position, cart velocity, pole angle, pole angular velocity]

### Configuration System
Config dict in `main()` contains all hyperparameters. Tuned hyperparameters can be loaded from `rl/cartpole/hyperparams.json` if present. Config includes:
- Environment settings (`env_id`, `state_buckets`, `state_clips`)
- Learning parameters (`learning_rate`, `discount_factor`, `lambda`)
- Exploration parameters (`epsilon_min`, `epsilon_max`, `epsilon_decay_rate`)
- Training parameters (`mean_reward_window_size`, `max_mean_reward`)

### Training Loop Structure
1. Episode initialization: reset environment, discretize state, choose initial action
2. For eligibility trace algorithms: initialize trace to zeros
3. Step loop: execute action, observe next state/reward, choose next action, call learning function
4. Episode end: log metrics, decay epsilon, save best model, check stopping criteria
5. For Optuna trials: report metrics and check for pruning

### Optuna Integration
- MySQL storage backend: `OPTUNA_STORAGE_URI = "mysql://root@localhost/optuna"`
- Uses TPESampler and HyperbandPruner
- Multi-seed evaluation: trains with all provided seeds and averages results
- Trial pruning: checks `trial.should_prune()` at intervals to stop unpromising trials early
- Tensorboard logs per trial: `runs/cartpole/trial_{number}`

### Output Locations
- Best models: `outputs/best_cartpole_model.npy`
- TensorBoard logs: `runs/cartpole/` (subdirs: `train/` or `trial_{number}/` for tuning)
- Hyperparameters: `rl/cartpole/hyperparams.json` (loaded if exists)

## Key Implementation Details

### Epsilon-Greedy Action Selection
`_choose_action()` implements ε-greedy policy: random action with probability ε, otherwise greedy (argmax Q-values).

### TD Error and Updates
All algorithms use temporal difference learning:
- Calculate TD target (varies by algorithm)
- Compute TD error: target - current value
- Update Q-values scaled by learning rate

### Eligibility Traces
Three algorithms use eligibility traces to propagate credit faster:
- Increment trace for current state-action pair
- Update Q-values for all state-action pairs proportional to their trace
- Decay all traces by `discount_factor * lambda`

### Best Model Tracking
Model is saved when mean reward (over last 100 episodes) exceeds previous best. Training stops if mean reward reaches `max_mean_reward` (500 for CartPole).

## Dependencies

Key packages:
- `gymnasium`: Environment interface
- `numpy`: Q-table and numerical operations
- `optuna`: Hyperparameter optimization
- `tensorboard`: Metrics logging
- `matplotlib`: Plotting (optional, when `plot=True`)

## Important Notes

- **README.md must be kept up to date** with any significant project changes
- Environment seed is set at episode reset - there's a TODO comment questioning if this ensures reproducibility (main.py:257)
- Optuna storage URI is hardcoded - there's a TODO to make it configurable (TODO.md)
- Tuning mode uses 10% of specified timesteps to run trials faster
