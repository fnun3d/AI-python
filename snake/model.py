import os
import torch
import torch.nn as nn
import torch.optim as optim

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        os.makedirs(model_folder_path, exist_ok=True)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved at {file_path}")

    def load(self, file_name='model.pth'):
        file_path = os.path.join('./model', file_name)
        self.load_state_dict(torch.load(file_path))
        self.eval()
        print(f"Model loaded from {file_path}")

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = r + gamma * max(next_predicted Q value) -> only do this if not done
        target = pred.clone()
        for idx, d in enumerate(done):
            Q_new = reward[idx]
            if not d:
                Q_new = reward[idx] + self.gamma * self.model(next_state[idx]).max()

            target[idx, torch.argmax(action[idx]).item()] = Q_new

        # 3: train the neural network with the state and target (Q_new)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
