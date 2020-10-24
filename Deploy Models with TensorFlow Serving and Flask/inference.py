import tensorflow as tf
import numpy as np
import json
import requests
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SIZE=150
MODEL_URI = 'http://localhost:8501/v1/models/pets:predict'
CLASSES = ['Cat','Dog']

def get_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(SIZE, SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    data = json.dumps({"instances": image.tolist()})
    response = requests.post(MODEL_URI, data=data.encode())
    result = json.loads(response.text)
    prediction = result['predictions'][0]
    class_name = 'Cat' if prediction[0]<0.5 else 'Dog'
    return class_name
    # return result
    
#print(get_prediction('/home/sanket/Downloads/catvsdogs/static/dog.jpg'))
