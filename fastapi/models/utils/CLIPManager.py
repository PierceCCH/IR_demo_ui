from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

class CLIPManager:
    def __init__(self, device):
        MODEL_VERSION = "openai/clip-vit-base-patch32"

        self.device = device
        self.model = CLIPModel.from_pretrained(MODEL_VERSION).to(device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_VERSION)
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_VERSION)

    def get_model_info(self):
        return self.model, self.processor, self.tokenizer

    def get_single_text_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors = "pt").to(self.device)
        text_embeddings = self.model.get_text_features(**inputs)
        embedding_as_np = text_embeddings.cpu().detach().numpy()
        return embedding_as_np
    
    def get_single_image_embedding(self, my_image):
        image = self.processor(
                text = None,
                images = my_image,
                return_tensors="pt"
                )["pixel_values"].to(self.device)
        embedding = self.model.get_image_features(image)
        embedding_as_np = embedding.cpu().detach().numpy()
        return embedding_as_np
