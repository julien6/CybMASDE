import torch
import torch.nn as nn
import torch.optim as optim

# Fonction pour arrondir les prédictions en fonction d'un epsilon
def round_with_epsilon(tensor, epsilon=0.1):
    rounded_tensor = tensor.clone()  # Créer une copie du tenseur
    for i in range(tensor.size(0)):  # Pour chaque valeur dans le tenseur
        for j in range(tensor.size(1)):
            val = tensor[i][j].item()
            if abs(val - round(val)) <= epsilon:  # Si proche de l'entier
                rounded_tensor[i][j] = round(val)  # Arrondir à l'entier
            else:
                rounded_tensor[i][j] = val  # Garder la valeur d'origine si pas proche
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

# Exemple de trajectoires sous forme [(action1, observation1), (action2, observation2), ...]
# Les observations sont supposées être des grilles 3x3 (aplatie), les actions sont des entiers
trajectories = [
    [(0, [[0, 1, 0], [1, 0, 2], [0, 1, 0]]), (1, [[1, 0, 1], [0, 2, 0], [0, 1, 2]])],
    [(2, [[0, 1, 1], [0, 0, 2], [2, 1, 0]]), (3, [[1, 0, 0], [2, 1, 2], [0, 0, 1]])],
    # Ajouter plus de trajectoires ici
]

# Convertir les actions et observations en tenseurs
actions = []
observations = []

for traj in trajectories:
    for action, observation in traj:
        actions.append(action)
        observations.append(torch.tensor(observation).float().flatten())  # Observation aplatie en vecteur 1D

# Conversion en tenseur pour l'entraînement
actions = torch.tensor(actions).unsqueeze(1).float()  # Ajout d'une dimension pour le RNN
observations = torch.stack(observations)

# Hyperparamètres
input_size = 1  # Une action par entrée (car codée par un entier)
hidden_size = 64
output_size = 9  # 3x3 grid aplatie
batch_size = 2
n_epochs = 3000
lr = 0.001

# Modèle RNN pour associer action -> observation
model = RNNActionToObservationModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # Pour minimiser la différence entre observations prédites et vraies observations
optimizer = optim.Adam(model.parameters(), lr=lr)

# Diviser les données en mini-batchs
dataset = torch.utils.data.TensorDataset(actions, observations)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Entraînement
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))  # Ajouter la dimension séquentielle pour LSTM
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {running_loss / len(dataloader):.4f}')

# Sauvegarder le modèle
torch.save(model.state_dict(), "rnn_action_to_observation_model.pth")
print("Modèle sauvegardé sous 'rnn_action_to_observation_model.pth'")

# Prédire une observation à partir d'une action donnée
with torch.no_grad():
    test_action = torch.tensor([[0]]).float()  # Exemple d'action pour tester
    predicted_observation = model(test_action.unsqueeze(1))
    print("Observation prédite (aplatie) pour l'action 2:", predicted_observation)
    print("Observation prédite (aplatie) pour l'action 2:", round_with_epsilon(predicted_observation))

    test_action = torch.tensor([[1]]).float()  # Exemple d'action pour tester
    predicted_observation = model(test_action.unsqueeze(1))
    print("Observation prédite (aplatie) pour l'action 2:", predicted_observation)
    print("Observation prédite (aplatie) pour l'action 2:", round_with_epsilon(predicted_observation))
