from PIL import Image

import streamlit as st
import requests
import json
import os
import copy

# Define FastAPI endpoint
BACKEND = "http://fastapi:8000/query_top_k_documents"

# Define data paths
ARTICLES_PATH = "../data/m2e2/article"
IMAGES_PATH = "../data/m2e2/image/image"

# Define model options
model_options = {
    "ALIGN + MLP": 1,
    "ALIGN + MLP + Hybrid": 2,
    "ALIGN + Hybrid + Split": 3
}

# sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Search", "Articles", "Images"))


if page == "Search":
    st.title("Multi-modal search demo")
    st.write("Query a vector database for related documents and images.")

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

            if image_file is not None:
                image_copy = copy.deepcopy(image_file)
                try:
                    img = Image.open(image_copy)
                    st.image(img, caption="Query image", width=300)
                except OSError:
                    st.error("Invalid image file. Please upload a valid image file.")
                    image_file = None

    with config_col:
        st.subheader("Configuration")
        model = st.selectbox("Select model", ["ALIGN + MLP", "ALIGN + MLP + Hybrid", "ALIGN + Hybrid + Split"])

        if model_options[model] == 2 or model_options[model] == 3:
            num_results = st.slider("Number of results per modality", min_value=1, max_value=20, value=10)
            alpha = st.select_slider(
                    "Weight of BM25 or vector search. 0 for pure keyword search, 1 for pure vector search.", 
                    options=[str(i/4) for i in range(5)], 
                    value='0.5')
        else:
            num_results = st.slider("Number of results", min_value=1, max_value=25, value=10)
            alpha = '0'
        
        request_body = {
            "text_query": text_input, 
            "top_k": num_results, 
            "model": model_options[model],
            "alpha": float(alpha)
        }

        if st.button("Submit query"):
            with st.spinner("Sending query to FastAPI endpoint..."):
                try:
                    if image_file is not None: # Image query
                        files = {
                            "image_file": image_file
                        }
                        response = requests.post(BACKEND, params=request_body, files=files)
                    elif len(text_input) > 0: # Text query
                        response = requests.post(BACKEND, params=request_body)
                    else:
                        st.error("No input provided.")
                except requests.exceptions.ConnectionError:
                    st.error("Connection error. Is the backend running?")
                except requests.exceptions.HTTPError as e:
                    st.error("HTTP error. Is the backend running?")

    st.divider()

    if model_options[model] == 3:
        with st.spinner("Waiting for results..."):
            try:
                text_results = response.json().get("text_results")['response']
                image_results = response.json().get("image_results")['response']
                
                # if response json contains query_text
                query_text = response.json().get("query_text")
                if query_text is not None:
                    st.write(f"Generated image caption: {query_text}")

                text_results_col, image_results_col = st.columns(2, gap="large")

                # Display retrieved articles
                with text_results_col:
                    st.subheader("Text results:")
                    if len(text_results) == 0:
                        st.write("No results found.")

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
                    if len(image_results) == 0:
                        st.write("No results found.")

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
                st.write("No results yet. Send a query to display results.")
            
            except (TypeError, AttributeError) as e:
                st.error(f"Something went terribly wrong: {e}")
    else:
        # models 1 and 2
        with st.spinner("Waiting for results..."):
            try:
                results = response.json().get("results")['response']
                query_text = response.json().get("query_text")

                st.subheader("Results:")
                if query_text is not None:
                    st.write(f"Generated image caption: {query_text}")
                    
                for i, result in enumerate(results):
                    if model_options[model] == 1:
                        score = result['certainty']
                    elif model_options[model] == 2:
                        score = result['score']
                    else:
                        raise ValueError(f"Invalid model option: {model_options[model]}")
            
                    document = result['response']['properties']
                    doc_id = document['doc_id']
                    content_path = document['content_path']
                    text = document['text']

                    with st.expander(f"ID: {doc_id} | {content_path} | Score: {score}"):
                        if os.path.exists(os.path.join(IMAGES_PATH, content_path)):
                            image = Image.open(os.path.join(IMAGES_PATH, content_path))
                            st.image(image, caption=text, width=500)

                        elif os.path.exists(os.path.join(ARTICLES_PATH, content_path)):
                            with open(os.path.join(ARTICLES_PATH, content_path), "r") as f:
                                st.write(f.read())
                        else:
                            st.write("No content available.")

            except NameError:
                st.write("No results yet. Send a query to display results.")
            except (TypeError, AttributeError) as e:
                st.error(f"Something went terribly wrong: {e}")


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

else:
    st.error("Something went wrong. You're not supposed to be here!")