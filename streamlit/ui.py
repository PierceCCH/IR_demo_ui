from PIL import Image

import streamlit as st
import requests
import json
import os

# Define FastAPI endpoint
BACKEND = "http://fastapi:8000/query_top_k_documents"

# Define data paths
ARTICLES_PATH = "../data/m2e2/article"
IMAGES_PATH = "../data/m2e2/image/image"

# sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Search", "Articles", "Images", "Documentation"))


if page == "Search":
    st.title("Multi-modal search demo")
    st.write("Query a vector database for related documents and images.") # TODO: Add description

    inputs_col, config_col = st.columns(2, gap="large")

    with inputs_col:
        st.subheader("Inputs")
        modality = st.selectbox("Select query modality", ["Text", "Image"])

        if modality == "Text":
            text_input = st.text_input("Enter text query", placeholder="Enter text query here")
            image_file = None
        else:
            text_input = None
            image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    with config_col:
        st.subheader("Configuration")
        model = st.selectbox("Select model", ["ALIGN", "ALIGN + MLP", "ALIGN + Hybrid", "ALIGN + Hybrid + Split"])
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
                            "image_file": image_file
                        }
                        response = requests.post(BACKEND, params=request_body, files=files)
                        st.write(response.json())
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
                        img = Image.open(os.path.join(IMAGES_PATH, image))
                        st.image(img, caption=caption, use_column_width=True)

        except NameError:
            st.write("No results yet.")


elif page == "Articles":
    articles = os.listdir(ARTICLES_PATH)
    articles.sort()
    st.title("Articles")
    st.subheader("Browse all articles that are available in the database.")

    # For pagination
    num_entries = 10
    page_num = st.number_input("Page number", min_value=1, max_value=(len(articles) // num_entries) + 1, value=1)
    start_idx = (page_num - 1) * num_entries
    end_idx = start_idx + num_entries
    st.divider()

    # Display articles by page
    for article in articles[start_idx:end_idx]:
        with st.expander(article):
            with open(os.path.join(ARTICLES_PATH, article), "r") as f:
                st.write(f.read())


elif page == "Images":
    images = os.listdir(IMAGES_PATH)
    captions = json.loads(open(os.path.join("../data/m2e2/image", "image_url_caption.json"), "r").read())
    images.sort()

    st.title("Images")
    st.subheader("Browse all images that are available in the M2E2 dataset.")
    st.write("Dataset is stored locally under `/data/m2e2/`.")
    
    # For pagination
    num_entries = 10
    page_num = st.number_input("Page number", min_value=1, max_value=(len(images) // num_entries) + 1, value=1)
    start_idx = (page_num - 1) * num_entries
    end_idx = start_idx + num_entries
    page_images = images[start_idx:end_idx]
    st.divider()

    # Display images by page
    tabs = st.tabs([f"Image {str(i+1)}" for i in range(len(page_images))])
    for i, tab in enumerate(tabs):
        with tab:
            image_col, caption_col = st.columns(2)
            image = page_images[i]
            img = Image.open(os.path.join(IMAGES_PATH, image))
            with image_col:
                st.image(img, width=500)
            with caption_col:
                image_key = image[:-6] # This is specific to the M2E2 dataset
                image_index = image[-5]
                st.subheader(image)
                try:
                    caption = captions[image_key][image_index]['caption']
                    st.write(caption)
                except KeyError:
                    st.write("No caption available.")

elif page == "Documentation":
    st.title("Documentation")
    st.subheader("This is a demo for the Multi-modal Search project.")
    st.write("TODO: Add documentation")

else:
    st.error("Something went wrong. You're not supposed to be here!")