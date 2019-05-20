"""Tests for CountyPredictor class."""
from pathlib import Path
import unittest
import sys


sys.path.append('../')
from county_classifier.image_predictor import CountyImagePredictor




SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / 'support'
print(SUPPORT_DIRNAME)


class TestCountyPredictor(unittest.TestCase):
    def test_filename(self):
        predictor = CountyImagePredictor()

        for filename in SUPPORT_DIRNAME.glob('*.jpg'):
            print(filename)
            pred, conf = predictor.predict(str(filename))
            
            print('Prediction: {} at confidence: {} for image with county team {}'.format(pred, conf, filename.stem))
            self.assertEqual(pred, filename.stem)
            self.assertGreater(conf, 0.6)


def main():
    predictor = TestCountyPredictor()
    predictor.test_filename()

if __name__ == '__main__':
    main()