"""CharacterPredictor class"""
from typing import Tuple, Union

import numpy as np

from county_classifier.models.county_model import CnnModel
from county_classifier.datasets.dataset import Dataset

import county_classifier.util as util

class CountyImagePredictor:
    """Given an image of a single image of a county player, recognizes it."""
    def __init__(self):
        self.model = CnnModel()
        self.model.load_weights()
    
    def predict(self, image_filename)-> Tuple[str, float]:
        """Predict on a single image."""       
        
        if isinstance(image_filename, str):
            image=util.read_image(image_filename)
        else:
            image=image_filename
        return self.model.predict_on_image(image)
    
    def evaluate(self, dataset: Dataset):
        """Evaluate on test dataset."""
        loss, accuracy = self.model.evaluate(dataset)
        return loss, accuracy
    
