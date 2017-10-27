# --- API Controllers ---
from bo.Base import BO_Base
import scipy
# --- Logger ---
from logger import logger_error, logger_info, logger_warning
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

import numpy as np

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

import base64, json

class BO_Prediction(BO_Base):

    def __init__(self):
        pass

    def product_prediction(self, prediction_params, model):

        # ----> Machine Learning Code goes here

        products_dict = {'beans': 0, 'cake': 1, 'candy': 2, 'cereal': 3, 'chips': 4, 'chocolate': 5, 'coffee': 6, 'corn': 7, 'fish': 8,
                         'flour': 9, 'honey': 10, 'jam': 11, 'juice': 12, 'milk': 13, 'nuts': 14, 'oil': 15, 'pasta': 16, 'rice': 17,
                         'soda': 18, 'spices': 19, 'sugar': 20, 'tea': 21, 'tomatosauce': 22, 'vinegar': 23, 'water': 24 }
        
        reversed_prod_dict = {v: k for k, v in products_dict.items()}

        logger_info("Code block at Product Prediction fn.")
        with open("imageToSave.png", "wb") as fh:
            #fh.write(prediction_params["key_image"].decode('base64'))
            fh.write(base64.b64decode(prediction_params["key_image"]))
            fh.close()
       
        img = image.load_img('imageToSave.png', target_size=(227, 227))
        img = image.img_to_array(img)
        # img = np.array(img)

        images = []
        images.append(img)
        images = np.array(images)

        product_name = 'Oops, something went wrong'
        try:
            images = preprocess_input(images)
            predict = model.predict(images, batch_size=1)
            product_id = np.argmax(predict, axis=1)
            product_name = reversed_prod_dict[product_id[0]]
        except Exception as e:
            logger_error(str(e))

        finally:

            return product_name

       
