from WeaviateManager import VectorManager
from typing import Union, List

from fastapi import FastAPI

app = FastAPI(
    title="Multi-modal search demo",
    description="""Query a vector database for related documents and images.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)

vec_db = VectorManager()

# TODO: Define response schema

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/create_collection")
def create_collection(collection: str):
    """
    Create the collection in Weaviate

    Args:
        collection (str): Name of the collection

    Returns:

    """
    return {"response" :  NotImplementedError}
    

@app.delete("/delete_collection")
def delete_collection(collection: str):
    """
    Delete the collection in Weaviate

    Args:
        collection (str): Name of the collection
    
    Returns:

    """
    # TODO: Improve error catching
    try:
        res = vec_db.delete_collection(collection)
        return res
    except Exception as e:
        return e

    
@app.post("/add_document")
def add_document(collection: str, documents: Union[list, dict]):
    """
    Add a document to the collection
    
    Args:
        collection (str): Name of the collection

    Returns:

    """
    try:
        res = vec_db.create_document(collection, documents)
        return res
    except Exception as e:
        return e


@app.post("/batch_add_documents")
def batch_add_documents(collection: str, documents: Union[list, dict]):
    """
    Add a document to the collection
    
    Args:
        collection (str): Name of the collection

    Returns:

    """
    try:
        res = vec_db.batch_create_documents(collection, documents)
        return res
    except Exception as e:
        return e
    
    
@app.delete("/delete_document")
def delete_document(collection: str, document_id: str):
    """
    Delete a document from the collection
    
    Args:
        collection (str): Name of the collection
        document_id (str): ID of the document

    Returns:

    """
    try:
        res = vec_db.delete_document(collection, document_id)
        return res
    except Exception as e:
        return e
    

@app.get("/query_top_k_documents")
def query_top_k_documents(collection: str, query: str, top_k: int = 10):
    """
    Query the collection for the top k most similar documents
    
    Args:
        collection (str): Name of the collection
        query (str): Query string
        k (int): Number of results to return

    Returns:

    """
    try:
        query_embedding = vec_db.get_embedding(query)
        res = vec_db.get_top_k_by_hybrid(collection, query, query_embedding, top_k)
        return res
    except Exception as e:
        return e