from .utils.ALIGNManager import ALIGNManager
from .utils.MLPManager import MLPManager

from models.ram.models import ram
from models.ram.models import tag2text
from models.ram import inference_ram
from models.ram import inference_tag2text
from models.ram import get_transform

import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mlp_model = MLPManager().to(device)
mlp_model.load_state_dict(
    torch.load(os.path.join(os.path.dirname(__file__), 'weights/align_mlp_checkpoint.pth'), 
        map_location=device)
)

# MM Models
align_model = ALIGNManager(device)

RAM_WEIGHTS = os.path.join(os.path.dirname(__file__), 'weights/ram_swin_large_14m.pth')
T2T_WEIGHTS = os.path.join(os.path.dirname(__file__), 'weights/tag2text_swin_14m.pth')

# RAM
IMAGE_SIZE = 384
ram_model = ram(pretrained=RAM_WEIGHTS, image_size=IMAGE_SIZE, vit='swin_l')
ram_model = ram_model.to(device)
ram_model.eval()

# T2T
t2t_model = tag2text(pretrained=T2T_WEIGHTS, image_size=IMAGE_SIZE, vit='swin_b')
t2t_model = t2t_model.to(device)
t2t_model.threshold = 0.68 # value used in original repo
t2t_model.eval()


def generate_caption(image) -> str:
    """
    Generate caption from image using Recognize Anything Model (RAM) and Tag2Text (T2T) model.

    INPUT:
    ------------------------------------
    image (PIL.Image): 
                        Image to generate caption from.
    
    RETURNS:
    ------------------------------------
    caption (str):      
                        Generated caption.
    """
    transform = get_transform(image_size=IMAGE_SIZE)
    image = transform(image).unsqueeze(0).to(device)

    tags = inference_ram(image, ram_model)
    _, _, caption = inference_tag2text(image, t2t_model, tags) # [Model identified tags, RAM generated tags, caption]

    return caption, tags
    

def generate_image_query(query, model):
    """
    Generate image query from image using the specified model.

    INPUT:
    ------------------------------------
    query (PIL.Image):
                        Query image.
    
    model (int):
                        Model to use for query. If model involved hybrid search, a caption will be generated.

    RETURNS:
    ------------------------------------
    query_embedding (np.ndarray):
                        Embedding of the input image generated by the specified model.
    
    caption (str):
                        Caption generated by Recognize Anything Model (RAM) and Tag2Text (T2T) model.

    """

    if model == 2 or model == 3:
        caption, tags = generate_caption(query)
    else:
        caption = None
        tags = None
    return align_model.get_single_image_embedding(query), caption, tags
    

def generate_text_query(query, model):
    """
    Generate text query from text using the specified model.

    INPUT:
    ------------------------------------
    query (str):
                        Query text.
    
    model (int):
                        Model to use for query.
                        0 for ALIGN, 1 for ALIGN + MLP, 2 for ALIGN + MLP + Hybrid, 3 for ALIGN + Hybrid + Split.
                        If model consists of MLP, the text query is first passed through the MLP model to generate an embedding.
    
    RETURNS:
    ------------------------------------
    query_embedding (np.ndarray):
                        Embedding of the input text generated by the specified model. 
    
    query (str):
                        Query text.
    """
    if model == 1 or model == 2:
        tensor = mlp_model(torch.tensor(align_model.get_single_text_embedding(query))).detach().cpu().numpy()
        return tensor, query
    elif model == 0 or model == 3:
        return align_model.get_single_text_embedding(query), query
    else:
        raise KeyError("Invalid model selection")

print("Models loaded")