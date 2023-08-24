from utils.ALIGNManager import ALIGNManager
from utils.CLIPManager import CLIPManager
from utils.MLPManager import MLPManager

from models.ram.models import tag2text
from models.ram.models import ram
from models.ram import inference_tag2text
from models.ram import inference_ram
from models.ram import get_transform

import torch
# import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")