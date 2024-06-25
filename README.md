# gym-simple-rl

This project implements various reinforcement learning algorithms to solve the CartPole environment from OpenAI Gym (now [Gymnasium](https://gymnasium.farama.org/index.html)). It supports training, evaluation, and hyperparameter tuning for different algorithms.

**Note**: This project was built for self-instructional and experimentation purposes with extensive assistance from various Large Language Models (LLMs).

## Table of Contents
- [Supported Algorithms](#supported-algorithms)
- [Installation](#installation)
- [Miniconda Setup](#miniconda-setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Key Components](#key-components)
- [Logging and Visualization](#logging-and-visualization)
- [Contributing](#contributing)
- [License](#license)

## Supported Algorithms

- Q-Learning
- SARSA
- Expected SARSA
- Q-Learning with Eligibility Traces (Q(λ))
- SARSA with Eligibility Traces (SARSA(λ))
- True Online SARSA(λ)

## Installation

1. Clone this repository:

```
git clone https://github.com/tsilva/gym-simple-rl.git
cd gym-simple-rl
```

2. Install Miniconda:
   - Visit the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) and download the appropriate installer for your operating system.
   - Follow the installation instructions for your platform.

3. Create a new Conda environment:

```
conda env create -f environment.yml
```

3. Activate the new environment:

```
conda activate gym-simple-rl
```

## Usage

The script can be run in three modes:

### Train

Train an agent using a specific algorithm:

```
python gym_simple_rl.py train --algo <algorithm_name> --seeds <seed_values>
```

Example:
```
python gym_simple_rl.py train --algo qlearning --seeds 123
```

### Evaluate

Evaluate a trained model:

```
python gym_simple_rl.py eval --model_path <path_to_model>
```

Example:
```
python gym_simple_rl.py eval --model_path output/best_cartpole_model.npy
```

### Tune

Perform hyperparameter optimization:

```
python gym_simple_rl.py tune --study_name <study_name> --seeds <seed_values> --n_trials <num_trials> --algo <algorithm_name>
```

Example:
```
python gym_simple_rl.py tune --study_name sarsa_study --seeds 123 456 789 --n_trials 100 --algo sarsa
```

Additional arguments:
- `--n_timesteps`: Number of timesteps for training (default: 500,000)
- `--trial_prune_interval`: Interval for pruning trials in Optuna (default: 500)

## Configuration

The script uses a configuration dictionary to set various parameters. You can modify these in the `config` dictionary within the script:

- Environment ID
- State discretization settings
- Learning parameters (learning rate, discount factor, etc.)
- Exploration parameters (epsilon min/max, decay rate)
- Algorithm-specific parameters (e.g., λ for eligibility trace methods)

## Key Components

1. **State Discretization**: Converts continuous state space to discrete for tabular methods.
2. **Action Selection**: Implements ε-greedy policy for exploration-exploitation balance.
3. **Learning Functions**: Separate functions for each supported algorithm.
4. **Training Loop**: Implements the main training process for all algorithms.
5. **Evaluation**: Renders the environment to visualize trained agent's performance.
6. **Hyperparameter Tuning**: Uses Optuna for optimizing hyperparameters.

## Logging and Visualization

- Console logging using the `logging` module.
- TensorBoard logging for training metrics.
- Optional plotting of mean rewards over episodes.

To view TensorBoard logs:

```
tensorboard --logdir=runs/cartpole
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
