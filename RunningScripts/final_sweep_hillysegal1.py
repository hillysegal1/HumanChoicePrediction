import wandb
YOUR_WANDB_USERNAME = "hilly-segal"
project = "NLP2024_PROJECT_hillysegal1"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "LSTM: SimFactor=0/4 for any features representation",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [True]},
        "architecture": {"values": ["LSTM"]},
        "seed": {"values": list(range(1, 3))},
        "ENV_LEARNING_RATE": {"values": [0.001, 0.01]},
        "hidden_dim": {"values": [32, 64]},
        "layers": {"values": [2, 4]}
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print("screen -dmS \"sweep_agent\" wandb agent {}/{}/{}".format(YOUR_WANDB_USERNAME, project, sweep_id))

