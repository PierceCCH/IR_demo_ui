from .utils.ALIGNManager import ALIGNManager
from .utils.CLIPManager import CLIPManager
from .utils.MLPManager import MLPManager

from models.ram.models import ram
from models.ram.models import tag2text
from models.ram import inference_ram
from models.ram import inference_tag2text
from models.ram import get_transform

from PIL import Image
import torch
import os
import io

RAM_WEIGHTS = os.path.join(os.path.dirname(__file__), 'weights/ram_swin_large_14m.pth')
T2T_WEIGHTS = os.path.join(os.path.dirname(__file__), 'weights/tag2text_swin_14m.pth')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

mlp_model = MLPManager()

# MM Models
align_model = ALIGNManager(device)
clip_model = CLIPManager(device)

# RAM
IMAGE_SIZE = 384
ram_model = ram(pretrained=RAM_WEIGHTS, image_size=IMAGE_SIZE, vit='swin_l')
ram_model = ram_model.to(device)
ram_model.eval()

# T2T
t2t_model = tag2text(pretrained=T2T_WEIGHTS, image_size=IMAGE_SIZE, vit='swin_b')
t2t_model = t2t_model.to(device)
t2t_model.threshold = 0.68
t2t_model.eval()


def resize(image):
    '''
    Reduce image size by factor of 3
    '''
    w, h = image.size
    return image.resize((w//3, h//3))

def generate_caption(ram, t2t, image_path, device):
    try:
        image = resize(Image.open(image_path))
        transform = get_transform(image_size=IMAGE_SIZE)
        image = transform(image).unsqueeze(0).to(device)

        tags = inference_ram(image, ram)
        _, _, caption = inference_tag2text(image, t2t, tags) # [Model identified tags, RAM generated tags, caption]

        # concat generated tags with caption
        res = caption + ' ' + tags.replace(' |', '')

        print(f"Generated text for {image_path}")
        return res
    
    except Exception as e:
        raise ValueError(f"Error generating caption: {e}")

def generate_query(modality: str, query, model=align_model, ram=ram_model, t2t=t2t_model, device=device):
    
    # TODO: Have variable model select

    if modality == "image":
        image = Image.open(query).convert('RGB')
        caption = generate_caption(ram, t2t, image, device)
        return model.get_single_image_embedding(image), caption
    
    elif modality == "text":
        return model.get_single_text_embedding(query), query
    
    else:
        raise ValueError(f"Invalid modality: {modality}")

print("Models loaded") # TODO: Replace with logger