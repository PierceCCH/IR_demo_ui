from WeaviateManager import VectorManager
from models import generate_image_query, generate_text_query
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

@app.get("/")
def read_root():
    return {"Hello": "World"}

"""
Querying
"""
@app.post("/query_top_k_documents")
async def query_top_k_documents(text_query: Optional[str] = None, top_k: int = 10, image_file: Optional[UploadFile] = None, model: int = 0, alpha: float = 0.5):
    """
    Queries both article and image collections in the vector database for the top k documents of each collection most similar to the query.
    
    INPUT: 
    ------------------------------------
    text_query (Optional[str]): 
                    Query string. Required only for text query.

    top_k (int): 
                    Number of results per modality to return.

    image_file (Optional[UploadFile]): 
                    Image file. Required only for image query.

    model (int):    
                    Model to use for query. 
                    0 for ALIGN, 1 for ALIGN + MLP, 2 for ALIGN + Hybrid, 3 for ALIGN + Hybrid + Split.

    alpha (float):  
                    Weight of BM25 or vector search. 
                    0 for pure keyword search, 1 for pure vector search.

    RETURNS: 
    ------------------------------------
        dict:       Dictionary of results or error.
                    example: {
                        "text_results": [
                            {
                                "doc_id": doc_id,
                                "score": score,
                                "text": text
                            },
                            ...
                        ],
                        "image_results": [
                            {
                                "doc_id": doc_id,
                                "score": score,
                                "image": image
                            },
                            ...
                        ]
                    }
    """
    if text_query is not None:
        query_embedding, query_text = generate_text_query(text_query, model)

    else:
        if not image_file:
            raise Exception("No image provided")
        
        try:
            image_content = await image_file.read()
            image_content = Image.open(BytesIO(image_content))

        except Exception as e:
            return {f"error: {e}"}
        query_embedding, query_text = generate_image_query(image_content, model)

    if model == 0:
        raise NotImplementedError("ALIGN model not implemented yet")
        # COLLECTION_NAME = 'ALIGN_M2E2_articles_images'
        # res = VecMgr.get_top_k(COLLECTION_NAME, query_embedding, top_k)

        # return {"results": res}
    
    elif model == 1:
        raise NotImplementedError("ALIGN + MLP model not implemented yet")
        # COLLECTION_NAME = 'ALIGN_MLP_M2E2_articles_images'
        # res = VecMgr.get_top_k(COLLECTION_NAME, query_embedding, top_k)

        # return {"results": res}
    
    elif model == 2:
        raise NotImplementedError("ALIGN + Hybrid model not implemented yet")
        # COLLECTION_NAME = 'ALIGN_Hybrid_M2E2_articles_images'
        # text_res = VecMgr.get_top_k_by_hybrid(TEXT_COLLECTION_NAME, query_text, query_embedding, top_k, alpha)
        # image_res = VecMgr.get_top_k_by_hybrid(IMAGE_COLLECTION_NAME, query_text, query_embedding, top_k, alpha)

        # return {"text_results": text_res, "image_results": image_res, "query_text": query_text}
    
    elif model == 3:
        TEXT_COLLECTION_NAME = 'ALIGN_M2E2_articles'
        IMAGE_COLLECTION_NAME = 'ALIGN_M2E2_images'

        text_res = VecMgr.get_top_k_by_hybrid(TEXT_COLLECTION_NAME, query_text, query_embedding, top_k, alpha)
        image_res = VecMgr.get_top_k_by_hybrid(IMAGE_COLLECTION_NAME, query_text, query_embedding, top_k, alpha)

        # TODO: include a cutoff for a certain score

        return {"text_results": text_res, "image_results": image_res, "query_text": query_text}

    else:
        raise Exception("Invalid model choice")
    


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