#!/usr/bin/env python
import sys
sys.path.append('../')

from typing import Callable, Dict, Optional
import wandb
import argparse
import json

from util import train_model
import importlib
#from gpu_manager import GPUManager



def run_experiment(experiment_config: Dict, save_weights: bool, use_wandb: bool = True):
     
    datasets_module = importlib.import_module('county_classifier.datasets')
    dataset_class_ = getattr(datasets_module, experiment_config['dataset'])
    dataset_args = experiment_config.get('dataset_args', {})
    dataset = dataset_class_(**dataset_args) 
  
    networks_module = importlib.import_module('county_classifier.networks')
    network_fn_ = getattr(networks_module, experiment_config['network'])
    network_args = experiment_config.get('network_args', {})
    
    models_module = importlib.import_module('county_classifier.models')
    model_class_ = getattr(models_module, experiment_config['model'])
        
    
    experiment_config['train_args'] = {**experiment_config.get('train_args', {})}
    
    if use_wandb:
        wandb.init()  #initializes project in wandb
        wandb.config.update(experiment_config)  #sends our exp config file to wandb
    
    model = model_class_(
        network_fn=network_fn_,
        dataset_cls = dataset_class_,
        dataset_args= dataset_args,
        network_args= network_args
    )
    print(model)

    train_model(
        dataset,
        model,
        epochs=experiment_config['train_args']['epochs'],
        batch_size=experiment_config['train_args']['batch_size'],
        use_wandb=use_wandb
    )
    
 #   if save_weights:
    model.save_weights()

def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
#   parser.add_argument(
#        "--gpu",
#        type=int,
#        default=0,
#        help="Provide index of GPU to use."
#    )
    
    parser.add_argument(
        "--save",
        default=True,
        dest='save',
        action='store_true',
        help="If true, then final weights will be saved to canonical, version-controlled location"
    )
    
    parser.add_argument(
        "experiment_config",
        type=str,
        help="Experimenet JSON ('{\"dataset\": \"EmnistDataset\", \"model\": \"CharacterModel\", \"network\": \"mlp\"}'"
    )
    
    parser.add_argument(
        "--nowandb",
        default=False,
        action='store_true',
        help='If true, do not use wandb for this run'
    )

    args = parser.parse_args()
    return args       
        
   
        
def main():
    args = _parse_args()
    # Hide lines below until Lab 4
    #if args.gpu < 0:
    #    gpu_manager = GPUManager()
    #    args.gpu = gpu_manager.get_free_gpu()  # Blocks until one is available
    # Hide lines above until Lab 4

    experiment_config = json.loads(args.experiment_config)
#    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)
    run_experiment(experiment_config, args.save, args.nowandb)

if __name__ == '__main__':
    main()
