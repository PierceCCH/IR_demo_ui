from transformers import AlignProcessor, AlignModel

class ALIGNManager:
    def __init__(self, device):
        """
        Initializes an ALIGNManager instance.

        Args:
        - device (torch.device): The device on which the ALIGN model should be loaded (e.g., 'cuda' for GPU, 'cpu' for CPU).

        This constructor initializes the ALIGN model and processor using the specified device.
        """
        MODEL_VERSION = "kakaobrain/align-base"

        self.device = device
        self.model = AlignModel.from_pretrained(MODEL_VERSION).to(device)
        self.processor = AlignProcessor.from_pretrained(MODEL_VERSION)

    def get_model_info(self):
        """
        Get information about the ALIGN model and processor.

        Returns:
        - Tuple[AlignModel, AlignProcessor]: A tuple containing the ALIGN model and processor used by this instance.
        """
        return self.model, self.processor, self.tokenizer

    def get_single_text_embedding(self, text: str):
        """
        Get the text embedding for a single text input.

        Args:
        - text (str): The input text for which the embedding should be generated.

        Returns:
        - numpy.ndarray: A NumPy array containing the text embedding.

        This method takes a single text input, processes it using the ALIGN processor, and returns the corresponding text embedding.
        """
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
        """
        Get the image embedding for a single image input.

        Args:
        - my_image (numpy.ndarray): The input image for which the embedding should be generated.

        Returns:
        - numpy.ndarray: A NumPy array containing the image embedding.

        This method takes a single image input, processes it using the ALIGN processor, and returns the corresponding image embedding.
        """
        image = self.processor(
                text = None,
                images = my_image,
                return_tensors="pt"
                )["pixel_values"].to(self.device)
        
        image_embedding = self.model.get_image_features(image)
        return image_embedding.cpu().detach().numpy()
