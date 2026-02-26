import torch
import torch.nn as nn
import torch.nn.functional as F
# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


''' Swish activation '''
class Swish(nn.Module): # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


''' MLP '''
class MLP(nn.Module):
    def __init__(self, hidden_size, act_fun_name):
        super(MLP, self).__init__()
        # init encoder architecture
        self.linear_layers = self.init_layers(hidden_size)
        if act_fun_name == "Sigmoid":
            self.act_fun = nn.Sigmoid()
        if act_fun_name == "Tanh":
            self.act_fun = nn.Tanh()
        if act_fun_name == "Relu":
            self.act_fun = nn.ReLU(inplace=True)
        if act_fun_name == "Leaky_Relu":
            self.act_fun = nn.LeakyReLU(negative_slope=0.4, inplace=True)

    def init_layers(self, layer_dimensions):
        layers = []
        for i in range(len(layer_dimensions) - 1):
            linear_layer = self.linear_layer(layer_dimensions[i], layer_dimensions[i + 1])
            layers.append(linear_layer)
            self.add_module('linear_' + str(i), linear_layer)
        return layers

    def linear_layer(self, input_size, hidden_size):
        linear = nn.Linear(input_size, hidden_size, bias=True)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)
        return linear

    def forward(self, x):
        # Define the forward pass
        for i in range(len(self.linear_layers)):
            if i < len(self.linear_layers) - 1:
                x = self.act_fun(self.linear_layers[i](x))
            else:
                # x = self.act_fun(self.linear_layers[i](x))
                x = self.linear_layers[i](x)
        return x

    def embed(self, x):
        # Define the forward pass
        for i in range(len(self.linear_layers)-1):
            x = self.act_fun(self.linear_layers[i](x))
        return x


'''
### MLP ###
class IMG_MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(IMG_MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out

    def embed(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        # out = out.view(out.size(0), -1)
        return out
'''
