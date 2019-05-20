""""Flask web server serving county_image_predictions."""
# From https://github.com/UnitedIncome/serverless-python-requirements

from flask import Flask, request, jsonify
from tensorflow.keras import backend

from county_classifier.image_predictor import CountyImagePredictor
import county_classifier.util as util

app = Flask(__name__)


# Tensorflow bug: https://github.com/keras-team/keras/issues/2397
#with backend.get_session().graph.as_default() as _:
 #   predictor = CountyImagePredictor()  # pylint: disable=invalid-name
    # predictor = LinePredictor(dataset_cls=IamLinesDataset)

#print(predictor.model.network.summary())    
    
    
@app.route('/')
def index():
    return 'Hello, world!'


@app.route('/v1/predict', methods=['GET', 'POST'])
def predict():
    image = _load_image()   #This image is a decoded numpy array.
   # with backend.get_session().graph.as_default() as _:
    predictor = CountyImagePredictor()
    pred, conf = predictor.predict(image)
    print("METRIC confidence {}".format(conf))
    print("METRIC mean_intensity {}".format(image.mean()))
    print("INFO pred {}".format(pred))
    return jsonify({'pred': str(pred), 'conf': float(conf)})


def _load_image():
    if request.method == 'POST':
        message = request.get_json()  #this seems to be an encoding of the image image in lots of crazy numbers and letters {'vfb1237213'}
        if message is None:
            return 'no json received'
        encoded = message['image']
        loaded_image = util.read_b64_image(encoded) #this function reads in the image and decodes it into a numpy array using CV2
        return loaded_image
    
    if request.method == 'GET':
        image_url = request.args.get('image_url')
        if image_url is None:
            return 'no image_url defined in query string'
        print("INFO url {}".format(image_url))
        return util.read_image(image_url)
    raise ValueError('Unsupported HTTP method')



def main():
    app.run(host='0.0.0.0', port=8000, debug=False) 

if __name__ == '__main__':
    main()
    
    
#TEST SCRIPTS
#export API_URL=http://0.0.0.0:8000
#(echo -n '{ "image": "data:image/jpg;base64,'$(base64 -w0 -i county_classifier/tests/support/dublin.jpg)'" }') |
# curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' -d@-

# docker run -p 8000:8000 --name api -it --rm county_classifier_api


#TEST SCRIPTS -> AWS LAMBDA FUNCTION
#export API_URL="https://rhnuvxmfmk.execute-api.us-west-2.amazonaws.com/dev"
#(echo -n '{ "image": "data:image/jpg;base64,'$(base64 -w0 -i county_classifier/tests/support/dublin.jpg)'" }') |
#curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' -d@-


