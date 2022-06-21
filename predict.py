import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image

# Arguments input by user
parser = argparse.ArgumentParser()

parser.add_argument('input_image', default = './test_images/cautleya_spicata.jpg', type = str, action = 'store', help = 'Image path')
parser.add_argument('model', default = './my_saved_model.h5', type = str, action = 'store', help = 'Model path')
  # The following arguments are optional, therefore we set 'required = False'. 
parser.add_argument('--top_k', default = 1, type = int, required = False, action = 'store', help = 'Return first k classes with the highest probabilities')
parser.add_argument('--category_names', default = './label_map.json', type = str, required = False, action = 'store', help = 'Path to JSON file')
    
args = parser.parse_args()
    
# Passing arguments to variables
image_path_arg = args.input_image
model_arg = args.model
top_k_arg = args.top_k
category_names_arg = args.category_names

# Loading saved model
loaded_model = tf.keras.models.load_model(model_arg, custom_objects = {'L2': tf.keras.regularizers.L1L2, 'KerasLayer' : hub.KerasLayer}, compile = False)

# Image processing
def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [224, 224])
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    if top_k > 0:
        image = Image.open(image_path)
        image = np.asarray(image)
        image = process_image(image)
        image = np.expand_dims(image, axis = 0)

        probabilities = model.predict(image)
        result = tf.math.top_k(probabilities, top_k)
        top_k_probabilities = tf.squeeze(result.values).numpy()
        indices = tf.squeeze(result.indices).numpy()

        if top_k==1:
            classes = str(indices+1)
        else:
            classes = [str(x+1) for x in indices]

        return top_k_probabilities, classes
    else:
        print('-------------------------------')
        print('\nk must be a positive integer! Please try again...\n\n')
        exit()
    
if __name__ == '__main__':
    probs, classes = predict(image_path_arg, loaded_model, top_k_arg)
    probs = probs * 100
    
    with open(category_names_arg, 'r') as f:
        class_names = json.load(f)
    
    if top_k_arg == 1:
        names = class_names[classes]
        
        print('----------------------------------------------------\n\n')
        print('The image you have chosen depicts a(n):\n')
        print(f'{names.title()} with a probability of {probs:.2f}%.')
        print('\n')
    else:
        names = [class_names[label] for label in classes]
 
        print('----------------------------------------------------\n\n')
        print('The image you have chosen depicts a(n):\n')
        for prob, name in zip(probs, names):
            print(f'{name.title()} with a probability of {prob:.2f}%.')
        print('\n')
    