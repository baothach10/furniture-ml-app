import pickle
# !pip install colorthief

import tensorflow as tf
import os
import cv2
import imghdr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import GlobalMaxPooling2D,Input,Conv2D, BatchNormalization,AveragePooling2D,ReLU,Add,Dropout,Flatten,Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from numpy.linalg import norm
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from scipy.signal import find_peaks
import random
from distutils.dir_util import copy_tree
from itertools import chain
from skimage.feature import local_binary_pattern

import tensorflow as tf
from rembg import remove
from PIL import Image
import numpy as np
import io

class ImageRecommenderTask3:
    def __init__(self, cnn_model_path, feature_data_path,classifier_model_path):
        self.cnn_model = load_model(cnn_model_path)
        
        # Load feature data
        data = np.load(feature_data_path)
        self.extracted_features = data['features']
        self.file_paths = data['file_paths']
        self.categories = data['categories']
        self.styles = data['styles']
        self.knn_model = None
        self.classifer_model = load_model(classifier_model_path)

        self.style_dict = {
            0: 'Asian',
            1: 'Beach',
            2: 'Contemporary',
            3: 'Craftsman',
            4: 'Eclectic',
            5: 'Farmhouse',
            6: 'Industrial',
            7: 'Mediteranean',
            8: 'Midcentury',
            9: 'Modern',
            10: 'Rustic',
            11: 'Scandinavian',
            12: 'Southwestern',
            13: 'Traditional',
            14: 'Transitional',
            15: 'Tropical',
            16: 'Victorian'
        }

        # #build KNN model
        # self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='auto').fit(self.extracted_features)
        
    # def preprocess_image(self, img_path):
    #     img = tf.io.read_file(img_path)
    #     img = tf.image.decode_jpeg(img, channels=3)
    #     img = tf.image.resize(img, [224, 224])
    #     img = img / 255.0
    #     img = tf.expand_dims(img, axis=0)
    #     return img

    def preprocess_image(self, img_path):
        # Load image
        img = Image.open(img_path)
        
        # Remove background
        output = remove(img)
        
        # Convert to RGBA (to ensure it has an alpha channel)
        output = output.convert("RGBA")
        
        # Convert to numpy array
        data = np.array(output)
        
        # Create a white background
        white_bg = np.ones_like(data) * 255
        
        # Replace transparent areas with white
        white_bg[:, :, :3] = data[:, :, :3]
        
        # Where the alpha channel is 0 (transparent), use white
        alpha_channel = data[:, :, 3]
        white_bg[alpha_channel == 0] = [255, 255, 255, 255]
        
        # Convert back to Image
        result = Image.fromarray(white_bg, 'RGBA')
        
        # Resize the image to 224x224
        result = result.resize((224, 224))
        
        # Convert to RGB for model input
        result = result.convert("RGB")
        
        # Normalize the image
        result = np.array(result) / 255.0
        
        # Expand dimensions to match the expected input shape of the model
        result = np.expand_dims(result, axis=0)
        
        # Show the preprocessed image
        return result
                                           
    def extract_features(self, img_path):
        img = self.preprocess_image(img_path)
        features = self.cnn_model.predict(img).flatten()
        features = features / norm(features)
        return features
    
    def classify_style(self, img_path):
        img = self.preprocess_image(img_path)
        prediction = self.classifer_model.predict(img)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_style = self.style_dict[predicted_index]
        return predicted_style
    
    # def classify_style(self, img_path):
    #     return "Asian"

    def filter_by_style(self, style):
        filtered_indices = [i for i, s in enumerate(self.styles) if s == style]
        filtered_features = self.extracted_features[filtered_indices]
        filtered_file_paths = [self.file_paths[i] for i in filtered_indices]
        return filtered_features, filtered_file_paths

    def recommend_images(self, img_path, top_n=10, show_image=False):
        

        # Filter by style
        style = self.classify_style(img_path)
        filtered_features, filtered_file_paths = self.filter_by_style(style)


        features = self.extract_features(img_path)

        # Find similar images
        self.knn_model = NearestNeighbors(n_neighbors=top_n, metric='cosine', algorithm='auto')
        self.knn_model.fit(filtered_features)

        distances, indices = self.knn_model.kneighbors([features])

        if show_image:
            img = cv2.imread(img_path)
            RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(RGB_img)
            images = []
            for file in indices[0]:
                img = cv2.imread(filtered_file_paths[file])
                RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(RGB_img)

            fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 6))

            for i, ax in enumerate(axes.flat):
                ax.imshow(images[i])
                ax.axis('off')

            plt.tight_layout()
            plt.show()

        return [filtered_file_paths[idx] for idx in indices[0]],style

    def get_random_image(self, category, style):
        data_dir = 'Data'
        selected_dir = os.path.join(data_dir, category, style)
        return os.path.join(selected_dir, random.choice(os.listdir(selected_dir)))
    

# Save the ImageRecommender class instance
def save_recommender(cnn_model_path, feature_data_path, classifier,save_path):
    recommender = ImageRecommenderTask3(cnn_model_path, feature_data_path,classifier)
    with open(save_path, 'wb') as f:
        pickle.dump(recommender, f)

# Example usage
# cnn_model_path = r'./src/app/model/ResNet34Dropout_feature_extractor.h5'
# feature_data_path = r'./src/app/feature_extraction/features_resnet34_dropout_full.npz'
# classifier = r'./src/app/model/Task3.h5'
# save_path = r'./src/app/model/task3_recommender.pkl'
# save_recommender(cnn_model_path, feature_data_path, classifier,save_path)    


