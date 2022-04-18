#!/usr/bin/env python
# coding: utf-8

# In[4]:


import keras
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def make_predictions(img, model_file):
    # load model here
    model = keras.models.load_model(model_file)
    
    # create the array of the shape for the keras model
    image = img
    data = np.ndarray(shape=(1,299,299,3), dtype=np.float32)
    size = (299,299)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    # change image to numpy array
    image_array = np.asarray(image)
    # normalize image
    normalized_image_array = image_array.astype(np.float32) / 255
    # load image into array
    data[0] = normalized_image_array
    
    preds = model.predict(data)
    return preds


# In[ ]:




