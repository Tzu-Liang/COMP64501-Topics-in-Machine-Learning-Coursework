import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class Net(nn.Module):
    """
    Define your model here. Feel free to modify all code below, but do not change the class name. 
    This simple example is a feedforward neural network with one hidden layer.
    Please note that this example model does not achieve the required parameter count (101700).
    """
    def __init__(self, conv1=16, conv2=32, num_classes=10):
        super(Net, self).__init__()

        # We define the layers of our model here by instantiating layer objects.
        # Here we define two fully connected (linear) layers.
        # Here, output dimension O = (Input_dim + 2*padding - Kernel_size) / stride + 1
        # Output size (conv1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels = 1,
                             out_channels = conv1,
                             kernel_size = 3,
                             padding = 1,
                             stride = 1)
        # Fast training and better convergence
        
        # Output size (conv1, 14, 14)
        self.pool1 = nn.MaxPool2d(kernel_size = 2,
                                 stride = 2)

        # Output size (conv2, 14, 14)
        self.conv2 = nn.Conv2d(in_channels = conv1,
                             out_channels = conv2,
                             kernel_size = 3,
                             padding = 1,
                             stride = 1)
        
        # Output size (conv2, 7, 7)
        self.pool2 = nn.MaxPool2d(kernel_size = 2,
                                 stride = 2)                       
        
        # The output of the CNN must be flattened before the fully connected layers.
        # Calculate the size of the flattened layer: 7 * 7 * 64 = 3136
        self.fc1 = nn.Linear(conv2*7*7, num_classes)
        
        # Initialize weights (optional, but good practice)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        # nn.init module contains initialization methods:
        # https://docs.pytorch.org/docs/main/nn.init.html
        # This particular one is called Kaiming initialization (also known as
        # He initialization) described in He, K. et al. (2015)
        #nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        # The forward pass defines the connectivity of the layers defined in __init__.
        # Input shape: [Batch_size, 1, 28, 28]

        x = self.pool1(F.relu(self.conv1(x)))   # [B, conv1, 14, 14]
        x = self.pool2(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        # -1 automatically calculates the required batch size dimension
        x = x.view(x.size(0), -1)
        # Shape now: [Batch_size, 3136]
        x = self.fc1(x)
        # Shape now: [Batch_size, 10]
        return x
