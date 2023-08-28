from PIL import Image

import streamlit as st
import requests
import os

# Define FastAPI endpoint
BACKEND = "http://fastapi:8000/query_top_k_documents"


st.title("Multi-modal search demo")
st.write("Query a vector database for related documents and images.") # TODO: Add description


inputs_col, config_col = st.columns(2, gap="large")

with inputs_col:
    st.subheader("Inputs")
    modality = st.selectbox("Select query modality", ["Text", "Image"])

    if modality == "Text":
        text_input = st.text_area("Enter text query", placeholder="Enter text query here")
        image_file = None
    else:
        text_input = None
        image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

with config_col:
    st.subheader("Configuration")
    model = st.selectbox("Select model", ["ALIGN & MLP", "ALIGN"])
    num_results = st.slider("Number of results per modality", min_value=1, max_value=25, value=10)
    request_body = {
        "query": text_input, 
        "top_k": num_results, 
        "model": model
    }

    if st.button("Submit query"):
        with st.spinner("Sending query to FastAPI endpoint..."):
            try:
                if image_file is not None:
                    files = {
                        "image": image_file
                    }
                    response = requests.post(BACKEND, params=request_body, files=files)
                elif len(text_input) > 0:
                    response = requests.post(BACKEND, params=request_body)
                else:
                    st.error("No input provided.")
            except requests.exceptions.ConnectionError:
                st.error("Connection error. Is the backend running?")

st.divider()

with st.spinner("Waiting for results..."):
    text_results_col, image_results_col = st.columns(2, gap="large")

    try:
        st.write(response.json())
        text_results = response.json().get("text_results")['response']
        image_results = response.json().get("image_results")['response']

        # Display retrieved articles
        with text_results_col:
            st.subheader("Text results:")
            for article in text_results:
                document = article['response']['properties']
                doc_id = document['doc_id']
                article_path = document['article']
                score = article['score']
                text = document['text']

                with st.expander(f"ID: {doc_id} | {article_path} | Score: {score[:5]}"):
                    st.write(text)
        
        # Display retrieved images
        with image_results_col:
            st.subheader("Image results:")
            tabs = st.tabs([f"Image {str(i+1)}" for i in range(len(image_results))])
            for i, tab in enumerate(tabs):
                with tab:
                    document = image_results[i]['response']['properties']
                    score = image_results[i]['score']
                    doc_id = document['doc_id']
                    image = document['image']
                    caption = document['text']

                    st.subheader(f"ID: {doc_id} | Score: {score[:5]}")
                    img = Image.open(os.path.join("../data/m2e2/image/image", image))
                    st.image(img, caption=caption, use_column_width=True)

    except NameError:
        st.write("No results yet.")
