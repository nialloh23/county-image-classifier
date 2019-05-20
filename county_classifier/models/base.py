from typing import Callable, Dict, Optional
from keras.optimizers import RMSprop
from pathlib import Path

from county_classifier.datasets.county_dataset import GaaDataset

DIRNAME = Path(__file__).parents[1].resolve() / 'weights'


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""
    def __init__(self, dataset_cls: type, network_fn: Callable, dataset_args: Dict = None, network_args: Dict = None):
      
        self.name = 'test_weights_name'
              
        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)
        
        if network_args is None:
            network_args = {}
        self.network = network_fn(**network_args)
            
    
    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / '{named}_weights.h5').format(named=self.name)
    

    def fit(self, dataset, batch_size: int = 10, epochs: int = 2, callbacks: list = None):
                
        if callbacks is None:
            callbacks = []
      
        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())
        
       
        train_generator = dataset.train_datagen.flow_from_directory(
            str(dataset.train_dir),
            target_size=(300, 300),
            batch_size = 10,
            class_mode='categorical',
            color_mode='rgb',
        )
        
        
        validation_generator = dataset.validation_datagen.flow_from_directory(
            str(dataset.validation_dir),
            target_size=(300, 300),
            batch_size = batch_size,
            class_mode='categorical',
            color_mode='rgb',
        )
        
        self.network.fit_generator(
            generator=train_generator,
            epochs=epochs,
            steps_per_epoch=100,
            validation_data=validation_generator,
            validation_steps=20,
            callbacks=callbacks,
            verbose=1,
        )

    def evaluate(self, dataset, batch_size=10): 
        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())
        
        test_generator = dataset.test_datagen.flow_from_directory(
            str(dataset.test_dir),
            target_size=(300, 300),
            batch_size = batch_size,
            class_mode='categorical',
            color_mode='rgb',
        )
        loss, accuracy = self.network.evaluate_generator(test_generator,steps=20)
        return loss, accuracy

               
    def loss(self):
        return 'categorical_crossentropy'

    def optimizer(self):
        return RMSprop(lr=0.0001)
      
    def metrics(self):
        return ['accuracy']

    def load_weights(self):
        self.network.load_weights(self.weights_filename)
    
    def save_weights(self):
        self.network.save_weights(self.weights_filename)