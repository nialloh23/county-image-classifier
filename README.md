# county-image-classifier (Train, Evaluate, Predict, Deploy)

## Todo list:
- [x] Add project ReadMe
- [ ] Extend data annotation process by using DataTurks API
- [ ] Setup Jupyter notebooks to do basic error analysis (Confusion matrix, bias/variance/visualize errors)
- [ ] Add video/webcam to prediction input

## Setup
### 1. Check Out the Repo
If you already have the repo in your directory. Go into it and make sure you have the latest:  
```cd county-image-classifier```  
```git pull origin master```

If not, open a shell in your JupyterLab instance and run:  
```git clone https://github.com/nialloh23/county-image-classifier.git```  
```cd county-image-classifier```  

### 2. Setup the Python Enviornment
Setup a virtual environment for the project. Activate the environment and install the project package requirements.  
```python3 -m venv project_env```  
```source project_env/bin/activate```  
```pip install -r requirements.txt```  

## Training

### 1. Data preparation
+ Upload the images you wish to include in your training to Data Turks annotation tool to in the form of a text file of S3 URLs.  (at the moment I’m getting the creating the URLs in excel -> need to automate)  
```data/county_classifier/county_players_1_300.txt```

+ After classifying the images with labels download the resulting json file. This contains S3 URLs to the raw images and associated labels. (insert link)  
```data/county_classifier/gaa_county_classification.json```

+ Update the metadata.toml file with the name of the json file and SHA reference  
```data/county_classifier/metadata.toml```

+ When you eventually start training. If the dataset & labels have not been downloaded already they will be downloaded and split into train, valid and test folders ready for training using keras fit_generators. If the images & labels have already been downloaded this step will be skipped. 

### 2. Edit the Configuration File  
+ If you want to run a single experiment you can edit the configuration via the dictionary specification in the train_county_classifier.sh task.  

+ If you want to run a series of experiments in parallel you can edit the json configuration file in the experiments folder.  

+ In the config file you can specify the dataset, model and network to use. You can also set some network & training arguments (e.g. batch size, #epochs, filter size etc.)  

```python
{
    "experiment_group": "Sample Experiments",
    "experiments": [
        {
            "dataset": "GaaDataset",
            "model": "CnnModel",
            "network": "cnn_network",
            "network_args": {
                "kernel_size": 2
            },
            "train_args": {
                "batch_size": 10,
                "epochs": 2
            }
        },
        {
            "dataset": "GaaDataset",
            "model": "CnnModel",
            "network": "cnn_network",
            "network_args": {
                "kernel_size": 2
            },
            "train_args": {
                "batch_size": 20,
                "epochs": 2
            }
        },
    ]
}

```

### 3. Start the Training Process
```bash tasks/train_county_classifier.sh```  
 
 This shortcut command runs the following:  
 ```pipenv run training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp",  "train_args": {"batch_size": 256}}```  
 
By the end of this process, the code will write the weights of the trained model to file test_weights_name.h5 (or whatever name has been specified).  

If you wish to run multiple experiments in parallel you can run the following shortcut command:
```tasks/prepare_sample_experiments.sh```  

This will run the multiple experiments you have specified in the experiments config two at a time, and as soon as one finished, another one will start. Although you can't see output in the terminal, you can confirm that the experiments are running by going to Weights and Biases.  


 ## Evaluate  
 ### 2. Evaluate the performance  
To compare the performance of various experiments that you run and to visualize the training process and results visit the weights and biases application: https://www.wandb.com/
 
 ### 2. Evaluate the performance
 ```bash tasks/run_validation_tests```  
 
This evaluates the performance of the classification model across the entire set of validation data. It asserts that the accuracy must be above 0.2 and the time taken must be below 10s.

 ## Predict
 ```bash tasks/run_prediction_tests.sh ```  
 
This tests the classification model on a test image (e.g. Dubline_01.jpg) asserts that the prediction is correct and that the confidence level must be above 0.6.  

## Deploy  
 
### 1. Test the API
  ```bash tasks/test_api.sh ```  
  
Runs a test on the api to make sure it is (a) serving responses (b) serving the correct predictions using an assertion on a test image (e.g. classification = kerry).  

### 2. Build docker image
  ```bash tasks/build_api_docker.sh ```  
Build the docker image for the api. This docker image is then ready to deployed on multiple platforms. 

### 3. Deploy API to AWS Lambda
  ```bash tasks/deploy_api_to_lambda.sh```  
This tasks installs npm, freezes the package requirements and deploys the API package to AWS lambda:
  ```pip freeze -> api/requirements.txt
sed -i 's/tensorflow-gpu/tensorflow/' api/requirements.txt
cd api || exit 1
npm install
npx sls deploy -v
```  

### 4. Run Prediction via API

We can test out our API by running a few curl commands. We need to change the API_URL first though to point it at Lambda:
  ```
  export API_URL="https://rhnuvxmfmk.execute-api.us-west-2.amazonaws.com/dev"  
  (echo -n '{ "image": "data:image/jpg;base64,'$(base64 -w0 -i county_classifier/tests/support/dublin.jpg)'" }') |  
  curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' -d@-

  ```
  
## Monitoring
We can look at the requests our function is receiving in the AWS CloudWatch interface. It shows requests, errors, duration, and some other metrics. What it does not show is stuff that we care about specifically regarding machine learning: data and prediction distributions. This is why we added a few extra metrics to api/app.py, in predict(). Using these simple print statements, we can set up CloudWatch metrics by using the Log Metrics functionality.  

+ Log in to your AWS Console, and make sure you're in the us-west-2 region.  

+  Once you're in, click on 'Services' and go to 'CloudWatch' under 'Management Tools.' 

+ Click on 'Logs' in the left sidebar.   

+ Click on county-image-classifier. You'll some log streams. If you click on one, you'll see some logs for requests to your API. Each log entry starts with START and ends with REPORT.  

+ To view a dashboard of results click on the dashboard tab on the left hand side (this includes custom metrics we are tracking.  


## Push Changes to Gitub
### 1. Freeze requirments files
  ```pip freeze > requirements.txt ```  
  
### 2. Add, commit, push to Github
  ```git add .  git commit -m "note" git push origin master``` 

### 3. Check Circle CI checks pass in app & github
https://circleci.com/gh/nialloh23/county-image-classifier