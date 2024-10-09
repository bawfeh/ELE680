import torch

class ANN(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=1, activation='relu', layers=[]):
        super(ANN, self).__init__()
        """
        - Create a neural network model that will serve as a agent network.
        - The model should take an input of dimension (number of states), and 
        return a relu activated output.
        ARGS:
            input_dim : number of input units | int (positive), default = 2
            output_dim : number of output units | int (positive), default = 1
            activation : activation function to use for output layer | str, default = 'relu'
            layers : list defining number of units per hidden layer | list <int> (positive), default = []
        """
        # Defining the layers
        self.mlp = torch.nn.Sequential()
        if len(layers):
            layer = torch.nn.Linear(input_dim, layers[0])
            self.mlp.add_module(module=layer, name='input')
            for j in range(1,len(layers)):
                self.mlp.add_module(module=torch.nn.ReLU(), name=f'relu_{j}')
                layer = torch.nn.Linear(layers[j-1], layers[j])
                self.mlp.add_module(module=layer, name=f'hidden_{j}')
            self.mlp.add_module(module=torch.nn.ReLU(), name=f'relu_{len(layers)}')
            output = torch.nn.Linear(layers[-1], output_dim)
            self.mlp.add_module(module=output, name='output')
        else:
            output = torch.nn.Linear(input_dim, output_dim)
            self.mlp.add_module(module=output, name='output')
        
        # Choose the activation function
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        # Forward pass through the network
        M = self.mlp(x)
        return M