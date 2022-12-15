import weaviate
import torch
import os
import copy
from typing import Union
import numpy

class VectorManager:
    TYPE_MAP = {
        "int":["int"],
        "float":["number"],
        "double":["number"],
        "str": ["text"],
        "bool": ["boolean"],
        "datetime": ["date"],
        "list[int]":["int[]"],
        "list[str]":["text[]"],
        "list[float]": ["number[]"],
        "list[double]": ["number[]"],
    }
    
    def __init__(self) -> None:
        """
        Set up the connection

        INPUT: None
        ------------------------------------

        RETURNS: None
        ------------------------------------
        """
        self._client = weaviate.Client(f"http://{os.environ.get('WEAVIATE_HOST')}:{os.environ.get('WEAVIATE_C_PORT')}")
        
    def _traverse_map (self, map_dict):
        temp = []
        for k, v in map_dict.items():
            try:
                self.TYPE_MAP.get(v)
                temp.append({
                    "name": k,
                    "dataType": self.TYPE_MAP[v]
                })
            except:
                print(f'Invalid date type {v}')
                return None
        return temp
        
    def _id2uuid(self, collection_name: str, id_no: str) -> str:
        where_filter = {
            'operator': 'Equal',
            'valueText': id_no,
            'path': ["id_no"]
        }

        query_result =  self._client.query.get(collection_name, ["id_no", "_additional {id}"]).with_where(where_filter).do()
        if len(query_result['data']['Get'][collection_name]) < 1:
            print(f'id: {id_no} is not found')
            return None
        uuid = query_result['data']['Get'][collection_name][0]['_additional']['id']
        return uuid
    
    def _exists(self, collection_name: str, id_no: str) -> bool:
        where_filter = {
            'operator': 'Equal',
            'valueText': id_no,
            'path': ['id_no']
        }
        response = self._client.query.get(collection_name, ["id_no", "_additional {id}"]).with_where(where_filter).do()
        if response.get('errors'):
            return False
        if len(response['data']['Get'][collection_name]) > 1:
            print("There exist duplicated files")
            return True
        elif len(response['data']['Get'][collection_name]) == 1:
            return True
        return False

    def delete_collection(self, collection_name: str) -> None:
        try:
            self._client.schema.delete_class(collection_name)
            print('Successfully deleted')
        except Exception as e:
            if '400' in str(e):
                print('Collection does not exist so nothing to delete')
            else:
                print(f"Unknown error with error message -> {e}")
            
    def delete_document(self, collection_name: str, id_no: str) -> None:
        # Check if the id exist
        id_exists = self._exists(collection_name, id_no)
        
        if not id_exists:
            print("This id does not exist so no deletion is done")
            return
        
        uuid = self._id2uuid(collection_name, id_no)
        
        try:
            self._client.data_object.delete(uuid=uuid, class_name=collection_name)
        except Exception as e:
            print(f"Unknown error with error message -> {e}")

    def create_collection(self, collection_name: str, schema: dict) -> None:
        """
        create a collection of documents

        INPUT: 
        ------------------------------------
        collection_name:    Name of collection
                            example shape:  'Faces'
        schema:             Schema for each document
                            example: {
                                "id_no": ["text"],
                            }

        RETURNS: None
        ------------------------------------
        """
        # Ensure that there is a id_no for the schema
        if not schema.get('id_no'):
            print('Lack of id_no as an attribute in property')
            return
        
        # Extract the document information into a list
        properties = self._traverse_map(schema)
        if properties == None:
            return None

        # Proper formetting for creating the schema
        document_schema = {
            'class': collection_name,
            'vectorizer': 'none',
            'properties': properties
        }

        # Try to create schema. If exists, gracefully exit.
        try:
            self._client.schema.create_class(document_schema)
            print("Collection Successfully created")
        except Exception as e:
            if '422' in str(e):
                print("Collection has been created")
            else:
                print(f"Unknown error with error message -> {e}")


    def create_document(self, collection_name: str, properties: dict, embedding: torch.Tensor) -> None:
        """
        create a document in a specified collecton

        INPUT: 
        ------------------------------------
        collection_name:    Name of collection
                            example shape:  'Faces'
        properties:         Schema for the document
                            example: {
                                "id_no": 72671
                            }
        embedding:          embedding for the document
                            example: torch.Tensor([[
                                1, 2, ..., 512
                            ]])

        RETURNS: None
        ------------------------------------
        """
        # Check if the id_no attribute exist
        if not properties.get('id_no'):
            print('Lack of id_no as an attribute in property')
            return

        # Check if the id exist
        id_exists = self._exists(collection_name, properties['id_no'])
        
        if id_exists:
            print("This id already existed please use update instead")
            return

        # Create document
        try:
            self._client.data_object.create(
              properties,
              collection_name,
              vector = embedding
            )
        except Exception as e:
            if "vector lengths don't match"in str(e):
                print("Mistmatch vector length, creation failed")
                self.delete_document(collection_name, properties['id_no'])
            else:
                print(f'Error in creating. Please read error message -> {e}')

    def read_document(self, collection_name: str, id_no: str) -> dict:
        """
        Find a document in a specified collecton

        INPUT: 
        ------------------------------------
        collection_name:    Name of collection
                            example shape:  'Faces'
        id_no:              id of document
                            example: "72671"

        RETURNS: None
        ------------------------------------
        """
        if not self._exists(collection_name, id_no):
            print('Attempt to read a non-existent document. No reading is done')
            return
        
        # Create filter and search
        uuid = self._id2uuid(collection_name, id_no)
        return self._client.data_object.get_by_id(uuid = uuid, class_name = collection_name, with_vector = True)
    
    def get_top_k(self, collection_name: str, target_embedding: Union[list, numpy.ndarray, torch.Tensor], top_k: int = 1):
        if top_k < 1:
            print('Invalid top_k')
            return
        query_vector = {'vector': target_embedding}
        res = self._client.query.get(collection_name, ["id_no", "_additional {certainty, id}"]).with_near_vector(query_vector).do()
        if 'errors' in res:
            error_message = res['errors'][0]['message']
            if "vector lengths don't match" in error_message:
                print('Query with wrong embedding dimension')
            else:
                print(f'unknown error please read -> {error_message}')
            return
        try:
            limit = min(top_k, len(res['data']['Get'][collection_name]))
            top_id = res['data']['Get'][collection_name][:limit]
            top_results = [ self.read_document(collection_name, document['id_no']) for document in top_id]
            for document, res in zip(top_id, top_results):
                res['certainty'] = document['_additional']['certainty']
            return top_results
        except Exception as e:
            print(f"Unknown error with error message -> {e}")

    def update_document(self, collection_name: str, document: dict) -> None:
        """
        Update a document in a specified collecton

        INPUT: 
        ------------------------------------
        collection_name:    Name of collection
                            example shape:  'Faces'
        document:           dctionary format of the update document
                            {
                                'id_no': '4848272',
                                'vector': torch.Tensor([[
                                    1, 2, ..., 512
                                ]]),
                                'name': 'Trump'
                            }

        RETURNS: None
        ------------------------------------
        """
        if not document.get('id_no'):
            print('Lack of id_no result in ambiguous document to update')
            return
        if not self._exists(collection_name, document['id_no']):
            print('Attempt to update a non-existent document. No update is done')
            return
        if len(document.keys()) == 1:
            print(f'Only {document.keys()} is found which has nothing to update')
            return
        uuid = self._id2uuid(collection_name, document['id_no'])
        temp = copy.deepcopy(document)
        if 'vector' in document.keys():
            new_vector = temp['vector']
            del temp['id_no']
            del temp['vector']
            previous_vector = self._client.data_object.get_by_id(uuid = uuid, class_name = collection_name, with_vector = True)['vector']
            try:
                self._client.data_object.update(
                    data_object = temp, 
                    class_name = collection_name, 
                    uuid = uuid, 
                    vector = new_vector
                )
            except Exception as e:
                if '400' in str(e):
                    print("Unknown field(s) in document")
                elif "vector lengths don't match"in str(e):
                    print("Mistmatch vector length, updating failed")
                    document['vector'] = previous_vector
                    self.update_document(collection_name, document)
                else:
                    print(f"Unknown error with error message -> {e}")
        else:
            del temp['id_no']
            try:
                self._client.data_object.update(
                    data_object = temp, 
                    class_name = collection_name, 
                    uuid = uuid, 
                )
            except Exception as e:
                if '400' in str(e):
                    print(f"Error in document field(s) -> {e}")
                else:
                    print(f"Unknown error with error message -> {e}")
                    