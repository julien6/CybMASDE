import torch
import torch.nn as nn
import torch.optim as optim

# Methods to handle the creation of an environment model.

"""
Fonction pour arrondir les prédictions en fonction d'un epsilon
"""
def round_with_epsilon(tensor, epsilon=0.1):
    rounded_tensor = tensor.clone()  # Créer une copie du tenseur
    for i in range(tensor.size(0)):  # Pour chaque valeur dans le tenseur
        for j in range(tensor.size(1)):
            val = tensor[i][j].item()
            if abs(val - round(val)) <= epsilon:  # Si proche de l'entier
                rounded_tensor[i][j] = round(val)  # Arrondir à l'entier
            else:
                # Garder la valeur d'origine si pas proche
                rounded_tensor[i][j] = val
    return rounded_tensor

# Modèle RNN pour prédire les observations en fonction des actions


class RNNActionToObservationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNActionToObservationModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        hn = hn.view(-1, hn.size(2))
        return self.fc(hn)


def convert_traces_to_otf(traces):

    # Convertir les actions et observations en tenseurs
    actions = []
    observations = []

    for traj in traces:
        for i, obs_act in enumerate(traj):
            observation = obs_act[0]
            action = obs_act[1]
            if i < len(traj) - 1:
                actions.append(action)
            if i > 0:
                # Observation aplatie en vecteur 1D
                observations.append(torch.tensor(
                    observation).float().flatten())

    # Conversion en tenseur pour l'entraînement
    actions = torch.tensor(actions).unsqueeze(
        1).float()  # Ajout d'une dimension pour le RNN
    observations = torch.stack(observations)

    # Hyperparamètres
    input_size = 1  # Une action par entrée (car codée par un entier)
    hidden_size = 64
    output_size = observations[0].shape[0]  # 3x3 grid aplatie
    batch_size = 2
    n_epochs = 3000
    lr = 0.001

    # Modèle RNN pour associer action -> observation
    model = RNNActionToObservationModel(input_size, hidden_size, output_size)
    # Pour minimiser la différence entre observations prédites et vraies observations
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Diviser les données en mini-batchs
    dataset = torch.utils.data.TensorDataset(actions, observations)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    # Entraînement
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            # Ajouter la dimension séquentielle pour LSTM
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f'Epoch [{epoch + 1}/{n_epochs}], Loss: {running_loss / len(dataloader):.4f}')

    # Sauvegarder le modèle
    torch.save(model.state_dict(), "rnn_action_to_observation_model.pth")
    print("Modèle sauvegardé sous 'rnn_action_to_observation_model.pth'")

    # Prédire une observation à partir d'une action donnée
    # with torch.no_grad():
    #     test_action = torch.tensor([[0]]).float()  # Exemple d'action pour tester
    #     predicted_observation = model(test_action.unsqueeze(1))
    #     # print("Observation prédite (aplatie) pour l'action 2:", predicted_observation)
    #     print("Observation prédite (aplatie) pour l'action 0:", round_with_epsilon(predicted_observation))

    #     test_action = torch.tensor([[1]]).float()  # Exemple d'action pour tester
    #     predicted_observation = model(test_action.unsqueeze(1))
    #     # print("Observation prédite (aplatie) pour l'action 2:", predicted_observation)
    #     print("Observation prédite (aplatie) pour l'action 1:", round_with_epsilon(predicted_observation))


def collect_traces_in_environment():

    # from movingcompany.moving_company_v0 import parallel_env, raw_env

    # env = raw_env()
    # env.reset()
    # for i, agent in enumerate(env.possible_agents):
    #     print(f"agent {agent} plays 0, and got {env.observations[agent]}")
    pass


def add_otf_in_problem():
    pass


if __name__ == '__main__':

    trajectories = [
        [([[0, 1, 0], [1, 0, 2], [0, 1, 0]], 0), ([[0, 1, 0], [1, 0, 2],
                                                   [0, 1, 0]], 1), ([[1, 0, 1], [0, 2, 0], [0, 1, 2]], 5)],
        [([[0, 1, 1], [0, 0, 2], [2, 1, 0]], 2), ([[0, 1, 1], [0, 0, 2],
                                                   [2, 1, 0]], 3), ([[1, 0, 0], [2, 1, 2], [0, 0, 1]], 6)],
    ]

    convert_traces_to_otf(trajectories)
