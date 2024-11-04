from typing import List, Tuple
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, r2_score
import json
from datetime import datetime
import os

Trajectory = List[Tuple[np.ndarray, np.ndarray]]


class LSTMTransitionModel(nn.Module):
    """
    LSTM-based model to predict the next observation for all agents based on their actions.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initializes the LSTMTransitionModel with given parameters.

        Parameters:
            input_size (int): The total dimension of input (combined actions of all agents).
            hidden_size (int): Number of features in LSTM hidden state.
            num_layers (int): Number of recurrent layers in LSTM.
            output_size (int): The total dimension of output (flattened observations of all agents).
        """
        super(LSTMTransitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Fully connected layer on last time step
        return out


def trajectories_to_traces(trajectories: List[List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]]):
    """
    Converts raw trajectories into traces suitable for model training, where each action and observation
    is a concatenated vector across all agents.

    Parameters:
        trajectories (List[List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]]):
            A list containing trajectories for each episode, where each trajectory is a list of tuples.
            Each tuple contains:
              - a dictionary of observations (flattened ndarray) for each agent
              - a dictionary of actions for each agent.

    Returns:
        List[Tuple[List[torch.Tensor],List[torch.Tensor]]]:
            A list where each element represents a couple of the actions and observations as tensors for each episode.
            These tensors are concatenated across agents, forming a single vector of actions and a single vector of observations.
    """
    traces: List[Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]] = []

    for episode in trajectories:
        actions = []
        observations = []

        for i in range(len(episode) - 1):

            action_dict = episode[i][1]
            observation_dict = episode[i+1][0]

            # Concatène les observations et actions de tous les agents en un seul vecteur pour chaque étape
            action_vector = np.concatenate(
                [action for action in action_dict.values()])
            observation_vector = np.concatenate(
                [observation for observation in observation_dict.values()])

            actions.append(action_vector)
            observations.append(observation_vector)

        traces.append((actions, observations))

    return traces


def get_transition_function_from_traces(trajectories: List[List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]], params=None):
    """
    Trains an LSTM model to predict the next observation vectors based on action vectors for all agents.

    Parameters:
        traces (List): A list of trajectories, where each trajectory is a list of tuples (action, next_observation).
        params (dict, optional): Hyperparameters for LSTM training. If None, Optuna is used to find optimal parameters.

    Returns:
        model (nn.Module): The trained LSTM model.
        statistics (dict): Dictionary containing training statistics.
    """

    traces = trajectories_to_traces(trajectories)

    # Split traces into training and test sets (80% training, 20% testing by episode)
    split_index = int(0.8 * len(traces))
    training_traces = traces[:split_index]
    test_traces = traces[split_index:]

    # Flatten the traces for each set into single datasets
    def flatten_traces(traces):
        actions = []
        observations = []
        for episode_actions, episode_observations in traces:
            actions.extend(episode_actions)
            observations.extend(episode_observations)
        return torch.tensor(np.array(actions), dtype=torch.float32), torch.tensor(np.array(observations), dtype=torch.float32)

    actions_train, observations_train = flatten_traces(training_traces)
    actions_test, observations_test = flatten_traces(test_traces)

    # Create DataLoader objects
    train_dataset = TensorDataset(actions_train, observations_train)
    test_dataset = TensorDataset(actions_test, observations_test)

    # Default hyperparameters for Optuna
    default_hpo_params = {
        'hidden_size': (32, 128),
        'num_layers': (1, 3),
        'batch_size': (16, 64),
        'learning_rate': (1e-4, 1e-2),
        'epochs': (10, 50)
    }

    # Optuna objective function
    def objective(trial):
        hidden_size = trial.suggest_int(
            "hidden_size", *default_hpo_params['hidden_size'])
        num_layers = trial.suggest_int(
            "num_layers", *default_hpo_params['num_layers'])
        batch_size = trial.suggest_int(
            "batch_size", *default_hpo_params['batch_size'])
        learning_rate = trial.suggest_float(
            "learning_rate", *default_hpo_params['learning_rate'], log=True)
        epochs = trial.suggest_int("epochs", *default_hpo_params['epochs'])

        # Check the size of train_dataset and test_dataset
        train_dataset_size = len(train_dataset)
        test_dataset_size = len(test_dataset)
        print(f"Train dataset size: {train_dataset_size}")
        print(f"Test dataset size: {test_dataset_size}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize the LSTM model with specified parameters
        input_size = actions_train.shape[1]
        output_size = observations_train.shape[1]
        model = LSTMTransitionModel(input_size=input_size, hidden_size=hidden_size,
                                    num_layers=num_layers, output_size=output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(epochs):
            for action_batch, obs_batch in train_loader:
                optimizer.zero_grad()
                # Add sequence dimension
                outputs = model(action_batch.unsqueeze(1))
                loss = criterion(outputs, obs_batch)
                loss.backward()
                optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for action_batch, obs_batch in val_loader:
                outputs = model(action_batch.unsqueeze(1))
                val_loss += criterion(outputs, obs_batch).item()

        return val_loss / len(val_loader)

    # Optimize hyperparameters with Optuna if params not provided
    if params is None:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
    else:
        best_params = params

    # Train final model with best parameters
    hidden_size = best_params.get("hidden_size", 64)
    num_layers = best_params.get("num_layers", 2)
    batch_size = best_params.get("batch_size", 32)
    learning_rate = best_params.get("learning_rate", 0.001)
    epochs = best_params.get("epochs", 20)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = LSTMTransitionModel(input_size=actions_train.shape[1], hidden_size=hidden_size,
                                num_layers=num_layers, output_size=observations_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Final training loop
    model.train()
    for epoch in range(epochs):
        for action_batch, obs_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(action_batch.unsqueeze(1))
            loss = criterion(outputs, obs_batch)
            loss.backward()
            optimizer.step()

    # Dossier contenant le fichier courant
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Save the model and compute metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{current_directory}/transition_model_{timestamp}.pth"
    stats_filename = f"{current_directory}/transition_model_stats_{timestamp}.json"

    torch.save(model.state_dict(), model_filename)

    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for action_batch, obs_batch in val_loader:
            outputs = model(action_batch.unsqueeze(1))
            predictions.append(outputs.numpy())
            targets.append(obs_batch.numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    mse = float(mean_squared_error(targets, predictions))
    r2 = float(r2_score(targets, predictions))
    statistics = {
        "mse": mse,
        "r2": r2,
        "params": best_params
    }

    with open(stats_filename, "w") as f:
        json.dump(statistics, f)

    return model, statistics

