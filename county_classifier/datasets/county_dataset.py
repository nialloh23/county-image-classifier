from keras.preprocessing.image import ImageDataGenerator

from county_classifier.datasets.dataset import Dataset

class GaaDataset:
    def __init__(self):  
        self.train_dir = Dataset.data_dirname() / 'raw' / 'train'
        self.validation_dir = Dataset.data_dirname() / 'raw' / 'validation'
        self.test_dir =  Dataset.data_dirname() / 'raw' / 'test'
        self.validation_datagen = ImageDataGenerator(rescale=1./255,)
        self.test_datagen = ImageDataGenerator(rescale=1./255,)
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

def main():
    dataset = GaaDataset()

if __name__ == '__main__':
    main()
