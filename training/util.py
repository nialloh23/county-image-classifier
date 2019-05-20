from time import time
from typing import Optional
from datetime import datetime

import sys
sys.path.append('/home/jupyter/county_image_classifier')

import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import keras
import tensorflow as tf

import wandb
from wandb.keras import WandbCallback

from county_classifier.datasets.dataset import Dataset
from county_classifier.models.base import Model

import datetime, os


from typing import Tuple, Union
from keras.preprocessing import image


EARLY_STOPPING = False

def train_model(
        dataset: Dataset,
        model: Model,
        epochs: int,
        batch_size: int,
        gpu_ind: Optional[int] = None,
        use_wandb: bool = True) -> Model:
  
    """Train model."""
    callbacks = []
    
 #   if EARLY_STOPPING:
 #       early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
 #       callbacks.append(early_stopping)

    if use_wandb:
        wandb.init()
        wandb_callback = WandbCallback()  #creates callback that runs during training and sends log data to wandb
        callbacks.append(wandb_callback) #takes everything keras logs and sends it to console and wandb
    

    tensorboard_callback = TensorBoard(log_dir='./logs')
    callbacks.append(tensorboard_callback)
    
    model.network.summary()

    t = time()
    _history = model.fit(dataset=dataset, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    print('Training took {:2f} s'.format(time() - t))

    return model





def read_image_from_filename(image_filename):
    from keras.preprocessing import image
    test_image = image.load_img(image_filename, target_size=(300, 300))
    image = image.img_to_array(test_image)
    return image 