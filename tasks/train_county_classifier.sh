#!/bin/bash
python training/run_experiment.py --gpu=-1 --nowandb '{"dataset": "GaaDataset", "model": "CnnModel", "network": "cnn_network", "train_args": {"batch_size": 10, "epochs": 16}, "network_args": {"kernel_size": 3}}'
