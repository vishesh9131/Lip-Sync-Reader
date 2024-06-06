# app.py
import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
import streamlit as st
from PIL import Image
import gdown

# Streamlit app title
st.title("Lip Sync Reader")

# Define functions
def load_video(path: str) -> List[float]:
    st.write(f"Loading video from path: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.write(f"Failed to open video file: {path}")
        return []

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Total frames in video: {frame_count}")
    if frame_count == 0:
        st.write(f"No frames found in video: {path}")
        return frames

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            st.write(f"Failed to read frame from {path}")
            break
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    if len(frames) == 0:
        st.write(f"No frames were loaded from the video: {path}")
        return frames

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def load_data(path: str):
    path = bytes.decode(path.numpy())
    file_name, file_extension = os.path.splitext(path)
    
    if not file_name:
        raise ValueError("File name could not be extracted from the path.")
    
    if file_extension not in ['.mpg', '.mp4']:
        raise ValueError("No video file found with .mpg or .mp4 extension.")
    
    frames = load_video(path)
    
    if len(frames) == 0:
        raise ValueError("No frames were loaded from the video.")
    
    return frames

# Vocabulary setup
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# Model setup
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same', activation='relu'),
    tf.keras.layers.MaxPool3D((1, 2, 2)),
    tf.keras.layers.Conv3D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool3D((1, 2, 2)),
    tf.keras.layers.Conv3D(75, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool3D((1, 2, 2)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax')
])

# Compile model with legacy optimizer
from tensorflow.keras.optimizers import legacy as legacy_optimizers
model.compile(optimizer=legacy_optimizers.Adam(learning_rate=0.0001), loss=tf.keras.backend.ctc_batch_cost)

# Load weights
# url = 'https://drive.google.com/uc?id=1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y'
# output = 'checkpoints.zip'
# gdown.download(url, output, quiet=False)
# gdown.extractall('checkpoints.zip', 'models')
model.load_weights('models/checkpoint')

# File uploader
uploaded_file = st.file_uploader("Choose a video file...", type=["mpg", "mp4"])

if uploaded_file is not None:
    # Save uploaded file
    file_path = os.path.join("data", "s1", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load data
    try:
        sample = load_data(tf.convert_to_tensor(file_path))
    except ValueError as e:
        st.write(e)
        st.stop()
    
    # Normalize the frame for display
    if tf.size(sample) != 0:
        first_frame = sample[0].numpy().squeeze()
        first_frame_normalized = (first_frame - np.min(first_frame)) / (np.max(first_frame) - np.min(first_frame))
        
        # Display a frame from the video
        st.image(first_frame_normalized, caption="First frame of the video", use_column_width=True)
        
        # Make predictions
        yhat = model.predict(tf.expand_dims(sample, axis=0))
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
        
        # Display predictions
        st.write("Predicted Text:")
        for sentence in decoded:
            st.write(tf.strings.reduce_join([num_to_char(word) for word in sentence]).numpy().decode('utf-8'))
    else:
        st.write("No frames to display or predict.")