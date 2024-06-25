import os
import random
import optuna
import logging
import gymnasium as gym
import numpy as np
import math
import argparse
import json
import shutil
from collections import deque
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

VALID_ALGOS = ['qlearning', 'qlearning_lambda', 'sarsa', 'expected_sarsa', 'sarsa_lambda', 'true_online_sarsa_lambda']
OPTUNA_STORAGE_URI = "mysql://root@localhost/optuna"
TENSORBOARD_RUNS_PATH = "runs/cartpole"

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def _discretize_state(config, env, state):
    state_buckets = config["state_buckets"]
    state_clips = config["state_clips"]
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    for i, clip in enumerate(state_clips):
        if clip is None: continue
        env_low[i] = clip[0]
        env_high[i] = clip[1]
    ratios = (state - env_low) / (env_high - env_low)
    discrete_state = (ratios * state_buckets).astype(int)
    discrete_state = np.clip(discrete_state, 0, np.array(state_buckets) - 1)
    return tuple(discrete_state)

def _choose_action(env, state, q_table, epsilon=0.0):
    if np.random.random() < epsilon: return env.action_space.sample()
    else: return np.argmax(q_table[state])

def _learn__qlearning(config, env, state, action, reward, next_state, next_action, **kwargs):
    q_table = kwargs["q_table"]
    discount_factor = config["discount_factor"]
    learning_rate = config["learning_rate"]

    # Calculate the target value (received reward + discounted value 
    # of next state when best action is taken according to current policy)
    best_next_action = np.argmax(q_table[next_state])
    best_next_value = q_table[next_state][best_next_action]
    td_target = reward + discount_factor * best_next_value

    # Calculate the error between the target value and the current value
    current_value = q_table[state][action]
    td_error = td_target - current_value
    
    # Update the value for the state action pair (scaled by 
    # learning rate to take small step in correct direction)
    td_update = learning_rate * td_error
    q_table[state][action] += td_update

def _learn__qlearning_lambda(config, env, state, action, reward, next_state, next_action, **kwargs):
    q_table = kwargs["q_table"]
    discount_factor = config["discount_factor"]
    learning_rate = config["learning_rate"]
    eligibility_trace = kwargs["eligibility_trace"]
    lambda_param = config["lambda"]

    # Calculate the target value (received reward + discounted value 
    # of next state when best action is taken according to current policy)
    best_next_action = np.argmax(q_table[next_state])
    best_next_state_value = q_table[next_state][best_next_action]
    td_target = reward + discount_factor * best_next_state_value

    # Calculate the error between the target value and the current value
    current_state_value = q_table[state][action]
    td_error = td_target - current_state_value

    # Increment the eligibility trace for the current state action pair
    # (marking that this action was taken in this state)
    eligibility_trace[state][action] += 1

    # Apply the update to all visited state action pairs 
    # in the eligibility trace, scaled by their current 
    # value (these values decay over time), by doing this
    # we propagate credit assignment to previous state 
    # action pairs faster
    td_update = learning_rate * td_error
    q_table += td_update * eligibility_trace

    # Decay the eligibility trace for all state action pairs
    eligibility_trace *= discount_factor * lambda_param

def _learn__sarsa(config, env, state, action, reward, next_state, next_action, **kwargs):
    q_table = kwargs["q_table"]
    discount_factor = config["discount_factor"]
    learning_rate = config["learning_rate"]

    # Calculate the target value (received reward + discounted reward of next state following current policy)
    next_state_value = q_table[next_state][next_action]
    td_target = reward + discount_factor * next_state_value

    # Calculate the error between the target value and the current value
    current_state_value = q_table[state][action]
    td_error = td_target - current_state_value

    # Update the value for the state action pair (scaled by 
    # learning rate to take small step in correct direction)
    td_update = learning_rate * td_error
    q_table[state][action] += td_update

def _learn__expected_sarsa(config, env, state, action, reward, next_state, next_action, **kwargs):
    q_table = kwargs["q_table"]
    epsilon = kwargs["epsilon"]
    discount_factor = config["discount_factor"]
    learning_rate = config["learning_rate"]

    # Calculate the probability of selecting 
    # each action in the next state
    num_actions = q_table.shape[-1]
    action_probs = np.ones(num_actions) * epsilon / num_actions
    best_next_action = np.argmax(q_table[next_state])
    action_probs[best_next_action] += (1.0 - epsilon)

    # Calculate the expected value of the next state by 
    # computed the weighted average of all values in next 
    # state according to current policy
    expected_q_value = np.sum(action_probs * q_table[next_state])

    # Calculate the target value (received reward + discounted average expected next action reward)
    td_target = reward + discount_factor * expected_q_value

    # Calculate the error between the target value and the current value
    current_state_value = q_table[state][action]
    td_error = td_target - current_state_value

    # Update the value for the state action pair (scaled by 
    # learning rate to take small step in correct direction)
    td_update = learning_rate * td_error
    q_table[state][action] += td_update

def _learn__sarsa_lambda(config, env, state, action, reward, next_state, next_action, **kwargs):
    q_table = kwargs["q_table"]
    discount_factor = config["discount_factor"]
    learning_rate = config["learning_rate"]
    eligibility_trace = kwargs["eligibility_trace"]
    lambda_param = config["lambda"]
    
    # Calculate the target value (received reward + discounted reward of next state following current policy)
    next_state_value = q_table[next_state][next_action]
    td_target = reward + discount_factor * next_state_value

    # Calculate the error between the target value and the current value
    current_state_value = q_table[state][action]
    td_error = td_target - current_state_value

    # Increment the eligibility trace for the current state action pair
    # (marking that this action was taken in this state)
    eligibility_trace[state][action] += 1

    # Apply the update to all visited state action pairs 
    # in the eligibility trace, scaled by their current 
    # value (these values decay over time), by doing this
    # we propagate credit assignment to previous state 
    # action pairs faster
    td_update = learning_rate * td_error
    td_update_trace = td_update * eligibility_trace
    q_table += td_update_trace

    # Decay the eligibility trace for all state action pairs
    # (we decay past experiences by the same factor we decay future ones
    # to be consistent with how the value function is updated, but we also
    # decay by an extra factor to control the trace decay independently)
    eligibility_trace *= discount_factor * lambda_param

def _learn__true_online_sarsa_lambda(config, env, state, action, reward, next_state, next_action, **kwargs):
    q_table = kwargs["q_table"]
    discount_factor = config["discount_factor"]
    learning_rate = config["learning_rate"]
    lambda_param = config["lambda"]
    eligibility_trace = kwargs["eligibility_trace"]
    old_state_value = kwargs.get("old_state_value", 0)
    
    # Calculate the target value (received reward + discounted reward of next state following current policy)
    next_state_value = q_table[next_state][next_action]
    td_target = reward + discount_factor * next_state_value

    # Calculate the error between the target value and the current value
    current_state_value = q_table[state][action]
    td_error = td_target - current_state_value
    
    # Update the eligibility trace for the current state-action pair
    eligibility_trace[state][action] += 1 - learning_rate * eligibility_trace[state][action]

    # Apply the update to the Q-values for all state-action pairs
    q_table += learning_rate * td_error * eligibility_trace

    # Adjust the Q-value for the current state-action pair to ensure true online update
    q_table[state][action] -= learning_rate * (current_state_value - old_state_value)

    # Decay the eligibility traces for all state-action pairs
    eligibility_trace *= discount_factor * lambda_param

    return {"old_state_value": next_state_value}

def _learn(config, env, algo, state, action, reward, next_state, next_action, **kwargs):
    return {
        "qlearning" : _learn__qlearning,
        "qlearning_lambda" : _learn__qlearning_lambda,
        "sarsa" : _learn__sarsa,
        "expected_sarsa" : _learn__expected_sarsa,
        "sarsa_lambda" : _learn__sarsa_lambda,
        "true_online_sarsa_lambda" : _learn__true_online_sarsa_lambda
    }[algo](config, env, state, action, reward, next_state, next_action, **kwargs)

def train(config, seed, algo, n_timesteps, plot=False, trial=None, trial_prune_interval=None):
    # Load the config params
    env_id = config["env_id"]
    epsilon_min = config["epsilon_min"]
    epsilon_max = config["epsilon_max"]
    epsilon_decay_rate = config["epsilon_decay_rate"]
    mean_reward_window_size = config["mean_reward_window_size"]
    max_mean_reward = config["max_mean_reward"]
    model_file_path = config.get("model_file_path")

    # Ensure model file path directory exists
    if model_file_path:
        model_file_dir = os.path.dirname(model_file_path)
        if not os.path.exists(model_file_dir):
            os.makedirs(model_file_dir)
    
    # Create a tensorboard logger
    tensorboard_path = f"{TENSORBOARD_RUNS_PATH}/trial_{trial.number}" if trial else f"{TENSORBOARD_RUNS_PATH}/train"
    writer = SummaryWriter(log_dir=tensorboard_path)

    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Create the environment (with the seed set for reproducibility)
    env = gym.make(env_id)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Initialize training variables
    epsilon = epsilon_max
    q_table = np.zeros(config["state_buckets"] + (env.action_space.n,))
    best_q_table = None
    mean_rewards = []
    episode_rewards_window = deque(maxlen=mean_reward_window_size)
    best_mean_reward = -float('inf')
    timestep = 0
    episode = 0

    # Train for X timesteps
    while timestep < n_timesteps:
        # Reset the environment and retrieve initial state and action
        state, _ = env.reset(seed=seed) # TODO: should we set the seed here? confirm states are always the same
        state = _discretize_state(config, env, state)
        action = _choose_action(env, state, q_table, epsilon=epsilon)
        
        # Initialize eligibility trace for algorithms that require it
        # (eligibility trace is a memory of all state action pairs
        # that have been visited and is used to propagate credit
        # assignment to previous state action pairs faster)
        eligibility_trace = np.zeros_like(q_table)
        
        # Train until episode is done or we reach the maximum number of timesteps
        done = False
        episode_reward = 0
        previous_learn_params = {}
        while not done and timestep < n_timesteps:
            # Perform the action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action)
            next_state = _discretize_state(config, env, next_state)

            # Choose the next action using the current policy
            # (some algorithms require the next action to be chosen 
            # before learning can be done, eg: sarsa)
            next_action = _choose_action(env, state, q_table, epsilon=epsilon)

            # Check if there is any non-zero value in the policy table
            previous_learn_params = _learn(
                config, env, algo, state, action, reward, next_state, next_action,
                q_table=q_table, 
                eligibility_trace=eligibility_trace,
                epsilon=epsilon,
                **(previous_learn_params if previous_learn_params else {})
            )

            # Current state and action are now the next state and action
            state = next_state
            action = next_action

            # Update the episode reward and advance timestep
            episode_reward += reward
            timestep += 1

        # Log to stats to tensorboard
        episode_rewards_window.append(episode_reward)
        mean_reward = np.mean(episode_rewards_window)
        mean_rewards.append(mean_reward)
        writer.add_scalar('reward/episode', episode_reward, episode)
        writer.add_scalar('reward/mean', mean_reward, episode)
        writer.add_scalar('reward/best_mean', best_mean_reward, episode)
        writer.add_scalar('exploration/epsilon', epsilon, episode)

        # Log stats to logger
        stats = (("Episode", episode), ("Timestep", timestep), ("Episode Reward", f"{episode_reward:.0f}"), ("Mean Reward", f"{mean_reward:.0f}"), ("Epsilon", f"{epsilon:.2f}"), ("Best Mean Reward", f"{best_mean_reward:.0f}"))
        logger.info(stats)

        # If this is the best model so far, store it
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_q_table = np.copy(q_table)
            if model_file_path: np.save(model_file_path, best_q_table)
            logger.info(f"New best model found at timestep {timestep} with mean reward {mean_reward}")

        # Stop training if we have reached the maximum reward
        if best_mean_reward >= max_mean_reward:
            logger.info("Stopping training as we have reached the maximum reward")
            break
        
        # If this training run is part of an Optuna study, 
        # report mean rewards and evaluate if trial 
        # should be pruned every couple of episodes
        if trial:
            trial.report(mean_reward, timestep)
            if not trial_prune_interval or episode % trial_prune_interval == 0:
                if trial.should_prune():
                    logger.info("Pruning unpromising trial")
                    raise optuna.exceptions.TrialPruned()
        
        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay_rate)

        # Increment episode counter
        episode += 1

    # Close the tensorboard writer
    writer.close()
    
    # In case plot flag is set, plot the mean rewards
    if plot:
        plt.plot(mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.title("Mean Reward vs Episode")
        plt.show()

    # Return the best q table and the best mean reward
    return best_q_table, best_mean_reward

def evaluate(config, q_table):
    env_id = config["env_id"]
    env = gym.make(env_id, render_mode="human")
    q_table = np.load(q_table) if isinstance(q_table, str) else q_table
    while True:
        state, _ = env.reset()
        state = _discretize_state(config, env, state)
        done = False
        episode_reward = 0
        while not done:
            action = _choose_action(env, state, q_table, epsilon=0.0)
            next_state, reward, done, _, _ = env.step(action)
            next_state = _discretize_state(config, env, next_state)
            state = next_state
            episode_reward += reward
        logger.info(f"Episode Reward: {episode_reward}")
        
def tune(config, study_name, seeds, n_trials, n_timesteps, algo=None, trial_prune_interval=None):
    # Remove the existing tensorboard logs for the study
    tensorboard_study_path = f"{TENSORBOARD_RUNS_PATH}/{study_name}"
    if os.path.exists(tensorboard_study_path):
        shutil.rmtree(tensorboard_study_path)

    # Define the objective function for the Optuna study
    _config = config
    def objective(trial):
        _algo = algo if algo else trial.suggest_categorical('algo', VALID_ALGOS)
        config = {
            **_config,
            "learning_rate": trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            "discount_factor": trial.suggest_float('discount_factor', 0.8, 1.0),
            "epsilon_min": trial.suggest_float('epsilon_min', 0.01, 0.1),
            "epsilon_decay_rate": trial.suggest_float('epsilon_decay_rate', 0.99, 1.0),
            "lambda": trial.suggest_float('lambda', 0.8, 1.0) if _algo in ["qlearning_lambda", "true_online_sarsa_lambda", "sarsa_lambda"] else None
        }

        # Train the agent for multiple seeds and average the 
        # rewards to ensure selected hyperparameters are robust
        mean_rewards = []
        for seed in seeds:
            _, mean_reward = train(config, seed, _algo, n_timesteps, trial=trial, trial_prune_interval=trial_prune_interval)
            mean_rewards.append(mean_reward)
        return np.mean(mean_rewards)
    
    # Run the optuna study
    study_name = f"study={study_name};algo={algo}"
    seed = seeds[0]
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=OPTUNA_STORAGE_URI,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    # Return the study result
    study_results = {
        "num_trials": len(study.trials),
        "num_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
    }
    logger.info(json.dumps(study_results, indent=4))
    return study_results

def main():
    # Read the command line arguments
    parser = argparse.ArgumentParser(description="Solve CartPole environment")
    parser.add_argument("mode", choices=["train", "eval", "tune"], help="Mode to run the script in")
    parser.add_argument("--algo", choices=VALID_ALGOS, help="Algorithm to use for training or tuning")
    parser.add_argument("--seeds", type=int, nargs="+", default=[123], help="Random seeds to use for training")
    parser.add_argument("--study_name", type=str, default="cartpole", help="Name of the Optuna study for tuning")
    parser.add_argument("--model_path", type=str, help="Path to the model file for evaluation")
    parser.add_argument("--n_timesteps", type=int, default=5_000_000, help="Number of timesteps for training")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for tuning")
    parser.add_argument("--trial_prune_interval", type=int, default=500, help="Interval for pruning trials in Optuna")
    args = parser.parse_args()
    
    # Initialize config
    config = {
        "env_id": 'CartPole-v1',
        "mean_reward_window_size": 100,
        "max_mean_reward": 500,
        "model_file_path": f'outputs/best_cartpole_model.npy',
        "state_clips": [None, (-1, 1), None, (-math.radians(50), math.radians(50))],
        "state_buckets": (12, 24, 12, 24),
        "algo": args.algo,
        "learning_rate": 0.1,
        "discount_factor": 0.99,
        "epsilon_min": 0.01,
        "epsilon_max": 1.0,
        "epsilon_decay_rate": 0.995,
        "lambda": 0.9
    }

    # Load tuned hyperparameters for selected algo if available
    hyperparams_path = f"rl/cartpole/hyperparams.json"
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
            algo_hyperparams = hyperparams.get(args.algo, {})
            config = {**config, **algo_hyperparams}

    # Train an agent
    if args.mode == "train":
        seed = args.seeds[0]
        if not args.algo: raise ValueError("--algo must be specified for training mode")
        best_q_table, _ = train(config, seed, args.algo, args.n_timesteps, plot=True)
        evaluate(config, best_q_table)
    # Evaluate an agent
    elif args.mode == "eval":
        if not args.model_path: raise ValueError("--model-path must be specified for evaluation mode")
        evaluate(config, args.model_path)
    # Tune training hyperparams
    elif args.mode == "tune":
        n_timesteps = int(args.n_timesteps / 10)
        tune(config, args.study_name, args.seeds, args.n_trials, n_timesteps, algo=args.algo, trial_prune_interval=args.trial_prune_interval)
    # Raise error for invalid mode
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()
