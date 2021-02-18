import torch
import torch.nn as nn

class Generative_Model(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, n_classes):
        super(Generative_Model, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.n_classes = n_classes

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(self.bn1(hidden))
        output = self.fc2(relu)
        output = self.relu(self.bn2(output))
        output = self.fc3(output)

        output_features = output[:, 0:-self.n_classes]
        output_labels = self.softmax(output[:, -self.n_classes:])
        output_total = torch.cat((output_features, output_labels), 1)
        return output_total