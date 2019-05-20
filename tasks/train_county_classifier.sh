#!/bin/bash
python training/run_experiment.py --nowandb '{"dataset": "GaaDataset", "model": "CnnModel", "network": "cnn_network", "train_args": {"batch_size": 10, "epochs": 2}, "network_args": {"kernel_size": 3}}'
