
from fastapi import FastAPI, UploadFile, File, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.layers import GlobalMaxPooling2D,Input,Conv2D, BatchNormalization,AveragePooling2D,ReLU,Add,Dropout,Flatten,Dense,GlobalAveragePooling2D,Reshape,Multiply,Activation
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import io
from .utils.Image_recommender import ImageRecommender
import pickle
from pydantic import BaseModel
import json
import datetime
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
from .utils.Image_recommender_task3 import ImageRecommenderTask3



class SqueezeExcite(tf.keras.Layer):
    def __init__(self, input_filters, se_ratio, **kwargs):
        super(SqueezeExcite, self).__init__()
        self.se_ratio = se_ratio
        self.input_filters = input_filters
        self.se_filters = max(1, int(input_filters * se_ratio))

    def build(self, input_shape):
        self.global_avg_pool = GlobalAveragePooling2D()
        self.reshape = Reshape((1, 1, self.input_filters))
        self.conv1 = Conv2D(self.se_filters, 1, activation='swish', padding='same')
        self.conv2 = Conv2D(self.input_filters, 1, activation='sigmoid', padding='same')

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return Multiply()([inputs, x])




PREFIX = f"/api/v1"


category_dict = {
    0: 'beds',
    1: 'chairs',
    2: 'dressers',
    3: 'lamps',
    4: 'sofas',
    5: 'tables',
}

# Initialize the Category Classifier
model_path = r'./src/app/model/efficientB0.h5'
task_1_classifier = load_model(model_path,custom_objects={'SqueezeExcite': SqueezeExcite})

app = FastAPI(
    openapi_url=f"{PREFIX}/openapi.json",
    docs_url=f"{PREFIX}/docs",
    redoc_url=f"{PREFIX}/redoc",
)

def load_recommender(save_path):
    with open(save_path, 'rb') as f:
        recommender = pickle.load(f)
    return recommender

# Initialize the ImageRecommender Task 2
model_path_task2 = r'./src/app/model/task2_recommender.pkl'
recommender = load_recommender(model_path_task2)


# Initialize the ImageRecommender Task 3
model_path_task3 = r'./src/app/model/task3_recommender.pkl'
recommender_task3 = load_recommender(model_path_task3)
    
app.mount(
    "/static",
    StaticFiles(
        directory= "./static",
        html=False,
    ),
    name="static",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CreateImage(BaseModel):
    file_name: str

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/task1", status_code=status.HTTP_201_CREATED)
async def add_image(file: UploadFile = File(...)):
    if not file:
        return 

    contents = await file.read()
    
    try:
        # Read image from the uploaded file
        image = Image.open(io.BytesIO(contents))
        image_path = "static/uploaded_image.jpg"
        image.save(image_path)  # Save the uploaded image for processing

        # Preprocess the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize if required by the model

        # Predict the category
        prediction = task_1_classifier.predict(image)
        predicted_index = np.argmax(prediction, axis=1)[0]
        category = category_dict.get(predicted_index, "Unknown category")

        return category

    except Exception as e:
        return 

@app.post(
    "/task2",
    status_code=status.HTTP_201_CREATED,
)
async def add_image(
    file: UploadFile,
):
    if file:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_path = "static/uploaded_image.jpg"
        image.save(image_path)  # Save the uploaded image for processing

        recommendations = recommender.recommend_images(image_path, top_n=10)
        recommendations = [rec.replace('\\', '/') for rec in recommendations]
        recommendations = [os.path.join("static", path) for path in recommendations]

        return {"recommendations": recommendations}
    return {"error": "No file provided"}


@app.post(
    "/task3",
    status_code=status.HTTP_201_CREATED,
)
async def add_image(
    file: UploadFile,
):
    if file:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_path = "static/uploaded_image.jpg"
        image.save(image_path)

        recommendations,style = recommender_task3.recommend_images(image_path, top_n=10)
        recommendations = [rec.replace('\\', '/') for rec in recommendations]
        recommendations = [os.path.join("static", path) for path in recommendations]
        return {"recommendations": recommendations,"style":style}
