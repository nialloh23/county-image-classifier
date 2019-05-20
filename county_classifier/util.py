from typing import Optional

from typing import Tuple, Union
from keras.preprocessing import image
from urllib.request import urlopen, urlretrieve
from pathlib import Path
import numpy as np
import os
import hashlib
import tensorflow as tf
import io
from PIL import Image

def read_image(image_uri: Union[Path, str]) -> np.array:
    """Read image_uri."""
    
    def read_image_from_filename(image_filename):
        from keras.preprocessing import image
        test_image = image.load_img(image_filename, target_size=(300, 300))
        image = image.img_to_array(test_image)
        return image
    
    img = read_image_from_filename(image_uri)
    
  #  def read_image_from_url(image_url):
  #      url_response = urlopen(str(image_url))
  #      img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
  #      return cv2.imdecode(img_array, imread_flag)
    
  #  local_file = os.path.exists(image_uri)
    
   # try:
   #     img = None
   #     if local_file:
    #        img = read_image_from_filename(image_uri)
    #    else:
    #        img = read_image_from_url(image_uri, imread_flag)
    #    assert img is not None
   # except Exception as e:
    #    raise ValueError("Could not load image at {}: {}".format(image_uri, e))
    return img

def read_b64_image(encoded_b64_string):
    """Load base64-encoded images."""
    import base64
    try:
        _, b64_data = encoded_b64_string.split(',')
        decoded = base64.b64decode(b64_data)
        open_image = Image.open(io.BytesIO(decoded))
        resized_image = open_image.resize((300,300))
        image_array = image.img_to_array(resized_image)
        print(image_array)
        print(image_array.shape)
       # print(b64_data)
        #image_array = cv2.imdecode(np.frombuffer(base64.b64decode(b64_data), np.uint8), cv2.IMREAD_COLOR)
        
       # image_array = image.img_to_array(image.load_img(BytesIO(base64.b64decode(b64_data))))
      #  print(image_array)
        #print(image_array.shape)
      #  print(image_array.dtype)
        #image_array_numpy = image_array.np()
       # print('image array: {}'.format(image_array_numpy))
    # print(image_array)
    #    resized_image = cv2.resize(image_array, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)

      #  resized_image = tf.reshape(image_array, [300, 300, 3])
       # print('image array: {}'.format(resized_image))
        return image_array
    except Exception as e:
        raise ValueError("Could not load image from b64 {}: {}".format(encoded_b64_string, e))
