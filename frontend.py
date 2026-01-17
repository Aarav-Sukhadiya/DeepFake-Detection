# frontend.py

import streamlit as st
import tempfile
import os

from src.image_infer import infer_image
from src.video_infer import infer_video

st.set_page_config(page_title="Deepfake Detection", layout="centered")

st.title("Deepfake Detection with Timestamp Localization")

st.write("Upload an image or video to analyze deepfake content.")

uploaded_file = st.file_uploader(
    "Choose an image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    suffix = uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    if suffix in ["jpg", "jpeg", "png"]:
        st.image(file_path, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            result = infer_image(file_path)

        st.subheader("Result (JSON)")
        st.json(result)

    else:
        st.video(file_path)

        with st.spinner("Analyzing video... This may take time on CPU."):
            result = infer_video(file_path)

        st.subheader("Result (JSON)")
        st.json(result)

    os.remove(file_path)
