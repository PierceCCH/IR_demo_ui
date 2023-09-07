import torch

class MLPManager(torch.nn.Module):
  """
  A PyTorch module representing a Multi-Layer Perceptron (MLP) manager.

  This class defines a neural network with multiple linear layers and
  tanh activation functions.

  Args:
      None

  Attributes:
      linear_in (torch.nn.Linear): The input linear layer mapping from
          640 nodes to 1024 nodes.
      hidden (torch.nn.Linear): The hidden linear layer mapping from
          1024 nodes to 1024 nodes.
      linear_out (torch.nn.Linear): The output linear layer mapping from
          1024 node to 640 nodes.
      tanh (torch.nn.Tanh): The tanh activation function.

  Methods:
      forward(x):
          Forward pass through the MLP manager.

  Example:
      mlp = MLPManager()
      input_data = torch.rand(1, 640)  # Example input data
      output = mlp(input_data)  # Forward pass through the MLP
  """

  def __init__(self):
    INPUT_DIM = 640
    OUTPUT_DIM = 640
    HIDDEN_LAYERS = 1024

    super(MLPManager, self).__init__()
    self.linear_in = torch.nn.Linear(INPUT_DIM, HIDDEN_LAYERS)
    self.hidden = torch.nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS)
    self.linear_out = torch.nn.Linear(HIDDEN_LAYERS, OUTPUT_DIM)
    self.tanh = torch.nn.Tanh()
    
  def forward(self, x):
    x = self.linear_in(x)
    x = self.tanh(x)
    x = self.hidden(x)
    x = self.tanh(x)
    x = self.linear_out(x)
    x = self.tanh(x)
    
    return x
