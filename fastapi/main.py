from WeaviateManager import VectorManager
from models import generate_query
from typing import Union, List

from fastapi import FastAPI

app = FastAPI(
    title="Multi-modal search demo",
    description="""Query a vector database for related documents and images.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)

VecMgr = VectorManager()

# Schemas
article_db_schema = {
    "doc_id": "str",
    "text": "str",
    "article": "str"
}
image_db_schema = {
    "doc_id": "str",
    "text": "str",
    "image": "str"
}

# TODO: Define response schema

@app.get("/")
def read_root():
    return {"Hello": "World"}

'''
Collection management
'''
@app.post("/create_collection")
def create_collection(collection: str):
    """
    Create the collection in Weaviate

    Args:
        collection (str): Name of the collection

    Returns:

    """
    return {"response":  "Not implemented"}
    

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
        res = VecMgr.delete_collection(collection)
        return res
    except Exception as e:
        return e


'''
Document management
'''
@app.post("/add_document")
def add_document(collection: str, documents: Union[list, dict]):
    """
    Add a document to the collection
    
    Args:
        collection (str): Name of the collection

    Returns:

    """
    try:
        res = VecMgr.create_document(collection, documents)
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
        res = VecMgr.batch_create_documents(collection, documents)
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
        res = VecMgr.delete_document(collection, document_id)
        return res
    except Exception as e:
        return e
    

'''
Querying
'''
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
        query_embedding, query = generate_query(query)
        res = VecMgr.get_top_k_by_hybrid(collection, query, query_embedding, top_k)
        return res
    except Exception as e:
        return e
