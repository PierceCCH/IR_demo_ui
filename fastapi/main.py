from WeaviateManager import VectorManager
from models import generate_query
from typing import Optional
from PIL import Image
from io import BytesIO

from fastapi import FastAPI, UploadFile

app = FastAPI(
    title="Multi-modal search demo",
    description="""Query a vector database for related documents and images.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)

VecMgr = VectorManager()

TEXT_COLLECTION_NAME = 'ALIGN_M2E2_articles'
IMAGE_COLLECTION_NAME = 'ALIGN_M2E2_images'


@app.get("/")
def read_root():
    return {"Hello": "World"}

"""
Querying
"""
@app.post("/query_top_k_documents")
async def query_top_k_documents(image_file: Optional[UploadFile], query: Optional[str] = None, top_k: int = 10, model: str = "ALIGN"):
    """
    Query the collection for the top k most similar documents
    
    Args:
        collection (str): Name of the collection
        query (str): Query string
        k (int): Number of results to return

    Returns:
        response (dict): Dictionary containing the results
    """
    try:
        if query is not None:
            query_embedding, query_text = generate_query("text", query) # TODO: include model parameter
        else:
            image_content = await image_file.file.read()
            query_embedding, query_text = generate_query("image", image_content) # TODO: include model parameter

        text_res = VecMgr.get_top_k_by_hybrid(TEXT_COLLECTION_NAME, query_text, query_embedding, top_k)
        image_res = VecMgr.get_top_k_by_hybrid(IMAGE_COLLECTION_NAME, query_text, query_embedding, top_k)

        # TODO: include a cutoff for a certain score

        return {"text_results": text_res, "image_results": image_res}
    
    except Exception as e:
        return {f"Error handling query: {e}"}


'''

"""
Collection management
"""
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


"""
Document management
"""
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