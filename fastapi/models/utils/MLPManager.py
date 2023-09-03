import torch

input_dim = 640
output_dim = 640
hidden_layers = 1024

class MLPManager(torch.nn.Module):
  def __init__(self):
    super(MLPManager, self).__init__()
    self.linear_in = torch.nn.Linear(input_dim, hidden_layers)
    self.hidden = torch.nn.Linear(hidden_layers, hidden_layers)
    self.linear_out = torch.nn.Linear(hidden_layers, output_dim)
    self.tanh = torch.nn.Tanh()
    
  def forward(self, x):
    x = self.linear_in(x)
    x = self.tanh(x)
    x = self.hidden(x)
    x = self.tanh(x)
    x = self.linear_out(x)
    x = self.tanh(x)
    
    return x
