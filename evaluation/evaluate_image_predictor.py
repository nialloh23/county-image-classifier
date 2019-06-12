"""Run validation test for CountyPredictor."""
import os
from pathlib import Path
from time import time
import unittest
import sys
sys.path.append('../')

from county_classifier.datasets.county_dataset import GaaDataset
from county_classifier.image_predictor import CountyImagePredictor


class TestEvaluateCountyPredictor(unittest.TestCase):
    def test_evaluate(self):
        predictor = CountyImagePredictor()
        dataset = GaaDataset()
        
        t = time()
        loss, accuracy = predictor.evaluate(dataset)
        time_taken = time() - t
        
        print('loss: {}, acc: {}, time_taken: {}'.format(loss, accuracy, time_taken))
        self.assertGreater(accuracy, 0.2)
        self.assertLess(time_taken, 15)
        

def main():
    evaluator = TestEvaluateCountyPredictor()
    evaluator.test_evaluate()

if __name__ == '__main__':
    main()