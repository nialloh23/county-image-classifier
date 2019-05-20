from typing import Callable, Dict, Tuple

import numpy as np


from county_classifier.models.base import Model
from county_classifier.datasets.county_dataset import GaaDataset
from county_classifier.networks.cnn_network import cnn_network

class CnnModel(Model):
    def __init__(self,
                 dataset_cls: type = GaaDataset,
                 network_fn: Callable = cnn_network,
                 dataset_args: Dict = None,
                 network_args: Dict = None):                
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)
        
 
    def predict_on_image(self, image: np.ndarray)-> Tuple[str, float]:
        x = np.expand_dims(image, axis=0)
      #  print('x is: {}'.format(x))
        image = np.vstack([x])
       # print('image is: {}'.format(image))
       # print('image shape is: {}'.format(image.shape))
        
        pred_raw = self.network.predict(image, batch_size=1).flatten()
        
        index_max_pred = np.argmax(pred_raw)
        confidence_of_prediction = pred_raw[index_max_pred]
        
        #class_labels = list(validation_generator.class_indices.keys())
        
        class_labels = ['dublin', 'galway', 'kerry', 'mayo']
        
        predicted_county = class_labels[index_max_pred]
        
        return predicted_county, confidence_of_prediction