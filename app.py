from distutils.command import upload
from PIL import Image
import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow import keras
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import pickle as pkl


st.title('Reverse Image Search')


def save_uploaded_image(uploaded_file):
    try:
        with open(os.path.join('C:\\Users\\Jolly\\OneDrive\\Desktop\\uploads', uploaded_file.name), 'wb') as f:
            # print("\nbuffer",uploaded_file.getbuffer(),f)
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


feature_list = np.array(
    pkl.load(open('C:\\Users\\Jolly\\embeddings.pkl', 'rb')))
filenames = np.array(pkl.load(open('C:\\Users\\Jolly\\filenames.pkl', 'rb')))


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result/norm(result)

    return normalized_result


def recommend(path):
    model = ResNet50(weights='imagenet', include_top=False,
                     input_shape=(224, 224, 3))
    model.trainable = False
    model = keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])
    new_feature = extract_features(path, model)
    neighnors = NearestNeighbors(
        n_neighbors=6, algorithm='brute', metric='euclidean')
    neighnors.fit(feature_list)
    distances, indices = neighnors.kneighbors([new_feature])
    return indices


uploaded_file = st.file_uploader("choose an image")
if uploaded_file is not None:
    if save_uploaded_image(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        indices = recommend(os.path.join(
            'C:\\Users\\Jolly\\OneDrive\\Desktop\\uploads', uploaded_file.name))
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            img1 = Image.open(filenames[indices[0][1]])
            st.image(img1)
        with c2:
            img2 = Image.open(filenames[indices[0][2]])
            st.image(img2)
        with c3:
            img3 = Image.open(filenames[indices[0][3]])
            st.image(img3)
        with c4:
            img4 = Image.open(filenames[indices[0][4]])
            st.image(img4)
        with c5:
            img5 = Image.open(filenames[indices[0][5]])
            st.image(img5)
    else:
        st.header("oh oo! :(")
