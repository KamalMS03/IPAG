from transformers import BlipProcessor,BlipForConditionalGeneration
from utils import *


class IC():
  def __init__(self) -> None:
    self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


  def caption_image(self, image):
    inputs = self.processor(images=image, return_tensors="pt")
    out = self.model.generate(**inputs)
    result = self.processor.decode(out[0], skip_special_tokens=True)
    return result
