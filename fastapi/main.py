from utils.WeaviateManager import VectorManager

from fastapi import FastAPI

app = FastAPI(
    title="Multi-modal search demo",
    description="""Query a vector database for related documents and images.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)

vec_db = VectorManager()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.delete("/delete_collection")
def delete_collection(collection: str):
    """Delete the collection in Weaviate"""
    try:
        res = vec_db.delete_collection(collection)
        return res
    except Exception as e:
        return e
