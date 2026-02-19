> [!WARNING]
> ## Archived
> This project is archived and no longer maintained.
>
> It has been superseded by [gymsolve](https://github.com/tsilva/gymsolve), which builds on these foundations with a more complete and flexible approach.

<div align="center">
  <img src="logo.png" alt="gym-simple-rl" width="512"/>

  # gym-simple-rl

  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
  [![Gymnasium](https://img.shields.io/badge/Gymnasium-CartPole--v1-brightgreen.svg)](https://gymnasium.farama.org/)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![TensorBoard](https://img.shields.io/badge/TensorBoard-Enabled-orange.svg)](https://www.tensorflow.org/tensorboard)

  **🎮 Learn reinforcement learning by training tabular agents to balance CartPole 🤖**

  [Quick Start](#quick-start) · [Algorithms](#supported-algorithms) · [Hyperparameter Tuning](#hyperparameter-tuning)
</div>

## Overview

A hands-on implementation of classic tabular reinforcement learning algorithms for the CartPole-v1 environment. Designed for learning and experimentation with RL fundamentals—all algorithms in a single, readable Python file.

## Supported Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| `qlearning` | Off-policy | Classic Q-Learning with greedy target |
| `sarsa` | On-policy | State-Action-Reward-State-Action |
| `expected_sarsa` | On-policy | SARSA with expected value updates |
| `qlearning_lambda` | Off-policy | Q-Learning with eligibility traces |
| `sarsa_lambda` | On-policy | SARSA with eligibility traces |
| `true_online_sarsa_lambda` | On-policy | True Online SARSA(λ) for faster credit assignment |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tsilva/gym-simple-rl.git
cd gym-simple-rl

# Create and activate conda environment
conda env create -f environment.yml
conda activate gym-simple-rl
```

### Train an Agent

```bash
python main.py train --algo qlearning --seeds 123
```

### Evaluate a Trained Model

```bash
python main.py eval --model_path outputs/best_cartpole_model.npy
```

## Hyperparameter Tuning

Run Optuna-powered hyperparameter optimization with multi-seed evaluation:

```bash
python main.py tune --study_name sarsa_study --seeds 123 456 789 --n_trials 100 --algo sarsa
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_timesteps` | 5,000,000 | Training duration |
| `--n_trials` | 100 | Number of Optuna trials |
| `--trial_prune_interval` | 500 | Episodes between pruning checks |

## Monitoring

Track training progress with TensorBoard:

```bash
tensorboard --logdir=runs/cartpole
```

Metrics logged:
- Episode reward
- Mean reward (rolling 100 episodes)
- Best mean reward
- Exploration rate (epsilon)

## Architecture

```
main.py (480 lines)
├── State discretization (continuous → discrete buckets)
├── ε-greedy action selection
├── Learning functions (_learn__<algo>)
├── Training loop with early stopping
└── Optuna integration (TPESampler + HyperbandPruner)
```

### State Discretization

CartPole's continuous state space is discretized into `(12, 24, 12, 24)` buckets for cart position, cart velocity, pole angle, and angular velocity.

### Configuration

Default hyperparameters can be overridden via `rl/cartpole/hyperparams.json`:

```json
{
  "qlearning": {
    "learning_rate": 0.1,
    "discount_factor": 0.99,
    "epsilon_decay_rate": 0.995
  }
}
```

## Dependencies

- [Gymnasium](https://gymnasium.farama.org/) - Environment interface
- [NumPy](https://numpy.org/) - Q-table and numerical operations
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Metrics visualization
- [Matplotlib](https://matplotlib.org/) - Optional reward plotting

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built for learning and experimentation with assistance from LLMs</sub>
</div>
