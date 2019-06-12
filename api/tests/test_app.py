"""Tests for web app."""
import os
from pathlib import Path
from unittest import TestCase
import base64

from api.app import app


#REPO_DIRNAME = Path(__file__).parents[2].resolve()
REPO_DIRNAME = '/home/jupyter/county_image_classifier'
SUPPORT_DIRNAME = '/home/jupyter/county_image_classifier/county_classifier/tests/support'  


class TestIntegrations(TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_index(self):
        response = self.app.get('/')
        assert response.get_data().decode() == 'Hello, world!'

    def test_predict(self):
        # with open(SUPPORT_DIRNAME / 'and came into the livingroom, where.png', 'rb') as f:
        with open('/home/jupyter/county_image_classifier/county_classifier/tests/support/dublin.jpg', 'rb') as f:
            b64_image = base64.b64encode(f.read())
        response = self.app.post('/v1/predict', json={
            'image': 'data:image/jpeg;base64,{}'.format(b64_image.decode())
        })
        json_data = response.get_json()
        self.assertEqual(json_data['pred'], 'dublin')
        
def main():
    tester = TestIntegrations()
    tester.setUp()
    tester.test_index()
    tester.test_predict()

if __name__ == '__main__':
    main()