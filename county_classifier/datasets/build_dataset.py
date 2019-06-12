import json 
import glob
import urllib.request
import argparse
import random
import os
import toml
from pathlib import Path
from county_classifier.datasets.dataset import Dataset
import shutil

##### DIRECTORIES AND CONSTANTS ####

DATA_DIRECTORY = Dataset.data_dirname() / 'county_classifier'

dataset_path = DATA_DIRECTORY / 'raw'
dataset_path_folder = dataset_path.absolute()
dataset_path_as_string = dataset_path_folder.as_posix()

train_path = dataset_path / 'train'
train_path_folder = train_path.absolute()
train_path_as_string = train_path_folder.as_posix()

test_path = dataset_path / 'test'
test_path_folder = test_path.absolute()
test_path_as_string = test_path_folder.as_posix()

validation_path = dataset_path / 'validation'
validation_path_folder = validation_path.absolute()
validation_path_as_string = validation_path_folder.as_posix()

metadata_filepath = DATA_DIRECTORY / 'metadata.toml'
metadata = toml.load(metadata_filepath)
json_file = metadata['filename']
json_filepath =  DATA_DIRECTORY / json_file

#Specify Train/Valid/Test Splits
train_percentage = 70
validation_percentage = 15
test_percentage = 15
train = float(int(train_percentage) / 100)
validation = float(int(validation_percentage)  / 100)
test = float(int(test_percentage)  / 100)



##### UTILITY FUNCTIONS ####

def downloader(image_url , i):
    file_name = str(i)
    full_file_name = str(file_name) + '.jpg'
    urllib.request.urlretrieve(image_url,full_file_name)
    
def number_image_files():
    files = [f for f in dataset_path.glob("**/*.jpg")]
    directory_images = []
    
    for f in files:
        directory_images.append(f)
    return len(directory_images)

def list_images_urls():
    folder_path = DATA_DIRECTORY.absolute()
    folder_path_as_string = folder_path.as_posix()
    json_file_path = os.path.join(folder_path_as_string, json_file)
    with open(json_file_path) as file1: 
        lis = []
        for i in file1:
            lis.append(json.loads(i))
    return lis


##### CHECK IF DOWNLOAD NEEDED ####

def load_or_generate_data():
    if number_image_files() < len(list_images_urls()):
        print(number_image_files())
        print(len(list_images_urls()))
        extract_label_urls()
        make_directories()
        download_images()
    else:
        print('Image files already downloaded')
        
        
##### EXTRACT LABELS, MAKE DIRECTORIES & DOWNLOAD IMAGES ####

def extract_label_urls():
    folder_names = []
    label_to_urls = {}
    for i in list_images_urls():
        if i['annotation']['labels'][0] not in folder_names:
            folder_names.append(i['annotation']['labels'][0])
            label_to_urls[i['annotation']['labels'][0]] = [i['content']]
        else:
            label_to_urls[i['annotation']['labels'][0]].append(i['content'])
    return label_to_urls


def make_directories():
    if not os.path.exists(dataset_path_as_string):
        os.mkdir(dataset_path_as_string)
        os.chdir(dataset_path_as_string)
        os.mkdir("train")
        os.mkdir("validation")
        os.mkdir("test")
    else:
        shutil.rmtree(dataset_path_as_string, ignore_errors=True)
        os.mkdir(dataset_path_as_string)
        os.chdir(dataset_path_as_string)
        os.mkdir("train")
        os.mkdir("validation")
        os.mkdir("test")
    
    
def download_images():
    #Download Train Images
    os.chdir(train_path_as_string)
    print(os.getcwd())
    label_to_urls = extract_label_urls()
    for i in label_to_urls.keys():
        os.mkdir(str(i))
        os.chdir(str(i))
        k = 0
        for j in label_to_urls[i][:round(train*(len(label_to_urls[i])))]:
            #print(label_to_urls[i][:round(0.8*(len(label_to_urls[i])))])
            downloader(j , str(i)+str(k))
            k+=1
        os.chdir("../")
     
    #Download Validation Images
    os.chdir(validation_path_as_string)
    print(os.getcwd())
    for i in label_to_urls.keys():
        os.mkdir(str(i))
        os.chdir(str(i))
        k = 0
        for j in label_to_urls[i][round(train*(len(label_to_urls[i]))):round((train + validation)*(len(label_to_urls[i])))]:
            downloader(j , str(i)+str(k))
            k+=1
        os.chdir("../")        
        
     #Download Test Images
    os.chdir(test_path_as_string)
    print(os.getcwd())
    for i in label_to_urls.keys():
        os.mkdir(str(i))
        os.chdir(str(i))
        k = 0
        for j in label_to_urls[i][round((train + validation)*(len(label_to_urls[i]))):round((train + validation + test)*(len(label_to_urls[i])))]:
            downloader(j , str(i)+str(k))
            k+=1
        os.chdir("../")   
    

if __name__ == "__main__":
    load_or_generate_data()
    