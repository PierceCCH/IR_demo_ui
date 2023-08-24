# # V6, 512 to 1024 dimension
# import torch

# input_dim = 512
# output_dim = 1024
# hidden_layers = 1024

# class MLPManager(torch.nn.Module):
#   def __init__(self):
#     super(MLPManager, self).__init__()
#     self.linear_in = torch.nn.Linear(input_dim, hidden_layers)
#     self.hidden = torch.nn.Linear(hidden_layers, hidden_layers)
#     self.linear_out = torch.nn.Linear(hidden_layers, output_dim)
#     self.tanh = torch.nn.Tanh()
#     self.dropout = torch.nn.Dropout(0.5)
    
#   def forward(self, x):
#     x = self.linear_in(x)
#     x = self.dropout(x)
#     x = self.tanh(x)
#     x = self.hidden(x)
#     x = self.dropout(x)
#     x = self.tanh(x)
#     x = self.linear_out(x)
#     x = self.tanh(x)

#     return x.detach().cpu().numpy()

# # V5, 512 to 1024 dimension
# import torch

# input_dim = 512
# output_dim = 1024
# hidden_layers = 1024

# class MLPManager(torch.nn.Module):
#   def __init__(self):
#     super(MLPManager, self).__init__()
#     self.linear1 = torch.nn.Linear(input_dim, hidden_layers)
#     self.hidden = torch.nn.Linear(hidden_layers, hidden_layers)
#     self.linear2 = torch.nn.Linear(hidden_layers, output_dim)
#     self.tanh = torch.nn.Tanh()
#     self.dropout = torch.nn.Dropout(0.5)
    
#   def forward(self, x):
#     x = self.linear1(x)
#     x = self.tanh(x)
#     x = self.linear2(x)
#     x = self.tanh(x)

#     return x.detach().cpu().numpy()


# V3
import torch

input_dim = 512
output_dim = 512
hidden_layers = 1024

class MLPManager(torch.nn.Module):
  def __init__(self):
    super(MLPManager, self).__init__()
    self.input = torch.nn.Linear(input_dim, hidden_layers)
    self.hidden = torch.nn.Linear(hidden_layers, hidden_layers)
    self.output = torch.nn.Linear(hidden_layers, output_dim)
    self.tanh = torch.nn.Tanh()
    self.dropout = torch.nn.Dropout(0.5)
    
  def forward(self, x):
    x = self.input(x)
    x = self.dropout(x)
    x = self.hidden(x)
    x = self.dropout(x)
    x = self.output(x)
    x = self.tanh(x)

    return x.detach().cpu().numpy()
