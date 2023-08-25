import streamlit as st
import requests

# interact with FastAPI endpoint
backend = "http://fastapi:8000/"

# construct UI layout
st.title("Multi-modal search demo")

st.write(
    """Query a vector database for related documents and images.
        This streamlit example uses a FastAPI service as backend.
        Visit this URL at `:8000/docs` for FastAPI documentation."""
)

# Text input
text_input = st.text_input("Enter text query", "Enter text query here")

# Image upload
image_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

# Button for submitting query
if st.button("Submit query"):
    # Send query to FastAPI endpoint
    st.write("Sending query to FastAPI endpoint...")
    try:
        if image_file is not None:
            files = {"image": image_file}
            response = requests.post(backend + "image", files=files)
        else:
            response = requests.post(backend + "text", json={"text": text_input})
    except requests.exceptions.ConnectionError:
        st.error("Connection error. Is the backend running?")

# Display articles retrieved by FastAPI endpoint
if response is not None:
    st.write("Response received.")
    st.write(response.json())
    
    # Display images
    for image in response.json()["images"]:
        st.image(image)
    
    # Display articles
    for article in response.json()["articles"]:
        st.write(article)
    
