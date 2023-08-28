import streamlit as st
import requests
import os

# interact with FastAPI endpoint
backend = "http://fastapi:8000/query_top_k_documents"

# construct UI layout
st.title("Multi-modal search demo")
st.write(
    """Query a vector database for related documents and images.
        This streamlit example uses a FastAPI service as backend.
        Visit this URL at `:8000/docs` for FastAPI documentation."""
)


inputs_col, config_col = st.columns(2, gap="large")

with inputs_col:
    text_input = st.text_input("Enter text query", placeholder="Enter text query here")
    image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

with config_col:
    model = st.selectbox("Select model", ["BERT", "ResNet50"])
    num_results = st.slider("Number of results per modality", min_value=1, max_value=25, value=10)
    request_body = {
        "query": text_input, 
        "top_k": num_results, 
        "model": "BERT"
    }

    if st.button("Submit query"):
        # Send query to FastAPI endpoint
        with st.spinner("Sending query to FastAPI endpoint..."):
            try:
                if image_file is not None:
                    files = {"image": image_file}
                    response = requests.post(backend, params=request_body, files=files)
                else:
                    response = requests.post(backend, params=request_body)
                st.success("Query sent successfully.")
            except requests.exceptions.ConnectionError:
                st.error("Connection error. Is the backend running?")

st.title("") # Add space before results

results_container = st.container()

with st.spinner("Waiting for results..."):
    with results_container:
        text_results_col, image_results_col = st.columns(2, gap="large")

        try:
            text_results = response.json().get("text_results")['response']
            image_results = response.json().get("image_results")['response']

            with text_results_col:
                st.subheader("Text results:")
                for article in text_results:
                    document = article['response']['properties']
                    score = article['score']
                    doc_id = document['doc_id']
                    text = document['text']

                    st.write(text)

            with image_results_col:
                st.subheader("Image results:")
                for image in image_results:
                    document = image['response']['properties']
                    score = image['score']
                    doc_id = document['doc_id']
                    image = document['image']
                    caption = document['text']

                    # open and display image
                    img = open(os.path.join("./data/m2e2/image/image", image), 'rb')
                    st.image(img, caption=caption, use_column_width=True)

        except NameError:
            st.write("No results yet.")
