import torch
from transformers import AlignProcessor, AlignModel

class ALIGNManager:
    def __init__(self, device):
        MODEL_VERSION = "kakaobrain/align-base"

        self.device = device
        self.model = AlignModel.from_pretrained(MODEL_VERSION).to(device)
        self.processor = AlignProcessor.from_pretrained(MODEL_VERSION)

    def get_model_info(self):
        return self.model, self.processor, self.tokenizer

    def get_single_text_embedding(self, text):
        inputs = self.processor(
                text = text,
                images = None,
                return_tensors="pt"
                ).to(self.device)
        text_embeddings = self.model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )
        return text_embeddings.cpu().detach().numpy()
    
    def get_single_image_embedding(self, my_image):
        image = self.processor(
                text = None,
                images = my_image,
                return_tensors="pt"
                )["pixel_values"].to(self.device)
        
        image_embedding = self.model.get_image_features(image)
        return image_embedding.cpu().detach().numpy()

    def add_new_tokens(self, new_tokens):
        new_tokens = list(set(new_tokens) - set(self.tokenizer.get_vocab().keys()))
        tokens_added = self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        return tokens_added
