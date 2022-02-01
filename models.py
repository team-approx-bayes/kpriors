import torch
import torch.nn as nn
import numpy as np


# A linear model, such as for logistic regression
class LinearModel(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(LinearModel, self).__init__()

        # Upper layer
        self.upper = nn.Linear(D_in, D_out, bias=False)

        torch.nn.init.zeros_(self.upper.weight)

    # If output_range is not None, then only output some classes' values (cf a multi-head setup)
    def forward(self, x):
        h_act = x
        y_pred = self.upper(h_act)

        return y_pred

    # Return all parameters as a vector
    def return_parameters(self):
        num_params = sum([np.prod(p.size()) for p in self.parameters()])
        means = torch.zeros(num_params)

        start_ind = 0
        for p in self.parameters():
            num = np.prod(p.size())
            means[start_ind:start_ind+num] = p.data.reshape(-1)
            start_ind += num

        return means


# A deterministic MLP with hidden layer size: hidden_sizes[0], ..., hidden_sizes[-1]
class MLP(torch.nn.Module):
    def __init__(self, D_in, hidden_sizes, D_out, act_func="relu"):
        super(MLP, self).__init__()

        # Hidden layers
        self.linear = torch.nn.ModuleList()
        weight_matrix = [D_in] + hidden_sizes
        for i in range(len(hidden_sizes)):
            self.linear.append(nn.Linear(weight_matrix[i], weight_matrix[i+1]))

        # Upper layer
        self.upper = nn.Linear(hidden_sizes[-1], D_out)

        # Set activation function
        if act_func == "relu":
            self.act_function = torch.nn.ReLU()
        elif act_func == "sigmoid":
            self.act_function = torch.nn.Sigmoid()
        elif act_func == "tanh":
            self.act_function = torch.nn.Tanh()
        else:
            raise ValueError("Cannot yet implement activation %s" % act_func)


    # If output_range is not None, then only output some classes' values (cf a multi-head setup)
    def forward(self, x, output_range=None):
        x = x.squeeze()
        h_act = x
        for i in range(len(self.linear)):
            h_act = self.linear[i](h_act)
            h_act = self.act_function(h_act)

        y_pred = self.upper(h_act)

        return y_pred

    # Return all parameters as a vector
    def return_parameters(self):
        num_params = sum([np.prod(p.size()) for p in self.parameters()])
        means = torch.zeros(num_params)

        start_ind = 0
        for p in self.parameters():
            num = np.prod(p.size())
            means[start_ind:start_ind+num] = p.data.reshape(-1)
            start_ind += num

        return means
