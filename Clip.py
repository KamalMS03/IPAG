import clip

import numpy as np
from PIL import Image

import torch

class Clip():
    def __init__(self, selected_model="ViT-B/32") -> None:
        self.model, self.preprocess = clip.load(selected_model)
        self.model.to("cpu")
    def measure_similarity(self, item_1, item_2):
        item_1_features = self.extract_features(item_1)
        item_2_features = self.extract_features(item_2)
        item_1_features /= item_1_features.norm(dim=-1, keepdim=True)
        item_2_features /= item_2_features.norm(dim=-1, keepdim=True)
        similarity = item_1_features.cpu().numpy() @ item_2_features.cpu().numpy().T
        return similarity

    def extract_features(self, item):
        if isinstance(item, str): return self.extract_features_test(item)
        return self.extract_features_image(item) 

    def extract_features_image(self, image):
        image = (image*255).astype(np.uint8)    
        image = Image.fromarray(image)
        image = self.preprocess(image)
        image = torch.tensor(np.expand_dims(image, axis=0))
        with torch.no_grad():
            image_features = self.model.encode_image(image).float()
        return image_features

    def extract_features_test(self, text):
        text = clip.tokenize([text])
        with torch.no_grad():
            text_features = self.model.encode_text(text).float()
        return text_features