# Multi-modal IR search demo

This is a demo of several multi-modal IR search experiments. The goal of this exploration was to see if the retrieval of both articles and images from a vector database using a single query is possible.

The demo is built using FastAPI, Streamlit, and Weaviate as the vector database.

Several potential methods to achieve this have been explored:
- Using an MLP to map text embeddings to image embeddings.
- Concatenating text and image embeddings.
- Using Weviate's hybrid search feature.
- Auto-generating captions for images for hybrid and semantic search.


# Setup
```
$ git clone
$ cd build
$ docker compose build
```

Before running the demo, make sure to include the weights for RAM, T2T and the MLP under /fastapi/models/weights. Then, run the following commands:


```
$ chmod +x start.sh
$ ./start.sh
```

View the demo at http://localhost:8501. FastAPI docs can be viewed at http://localhost:8000/docs.

# Todos:
- [ ] Build demo frontend to include:
    - [ ] Model selection 
    - [x] Image query search
    - [x] Text query search
    - [x] Top k selection
    - [x] Browse everything in database
- [ ] Logging and error handling
- [x] Query API
- [x] Setup environment for FastAPI server and weaviate database

# Architecture

