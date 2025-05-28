import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image


def main():
    st.header("How old are you according to a CNN 🥸👧")
    st.write(
        "Upload an image of yourself below to find out!")
    file = st.file_uploader("Upload Photo")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if file is not None:

        st.image(file, width=300)
        image = Image.open(file)
        image = tf.image.resize(image, [200, 200])
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)

        model = tf.keras.models.load_model("/CODE/Model\\agemodel2.h5")
        age = model.predict(image)
        st.markdown("## You're %i years old according to our model!" % age[0][0])


if __name__ == '__main__':
    main()
