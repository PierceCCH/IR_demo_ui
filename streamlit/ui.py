import streamlit as st

# interact with FastAPI endpoint
backend = "http://fastapi:8000/segmentation"

# construct UI layout
st.title("Multi-modal search demo")

st.write(
    """Query a vector database for related documents and images.
        This streamlit example uses a FastAPI service as backend.
        Visit this URL at `:8000/docs` for FastAPI documentation."""
)

