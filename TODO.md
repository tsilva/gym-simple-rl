## TODO

- [ ] Softcode Optuna storage
- [ ] Double check that Optuna is parallelizing with multiprocessing, otherwise implement multiprocessing support
- [ ] Verify that environment reproducibility is working (seed setting)
- [ ] Plot results for different algorithms
- [ ] Add support for additional environments: `FrozenLake-v0`, `Taxi-v3`, `CliffWalking-v0`
- [ ] Add target table support: eg. `Double Q-Learning`
- [ ] Add support for different exploration strategies
- [ ] Tune hyperparams and commit results
- [ ] Add wandb support (https://github.com/optuna/optuna-examples/blob/main/wandb/wandb_integration.py)
- [ ] Test wandb sweeps
- [ ] Test Ray Tune