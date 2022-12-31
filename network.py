import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

class QNetwork(nn.Module):

    def __init__(self, action_dim, lr):
        super(QNetwork, self).__init__()
        # Input (4, 84, 84)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0) # 32 x 20 x 20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0) # 64 x 9 x 9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0) # 64 x 7 x 7
        self.fc1 = nn.Linear(3136, 512)
        init.xavier_uniform_(self.fc1.weight)
        init.uniform_(self.fc1.bias, -0.1, 0.1)
        self.fc2 = nn.Linear(512, action_dim)
        init.xavier_uniform_(self.fc2.weight)
        init.uniform_(self.fc2.bias, -0.1, 0.1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x