# Multi-modal IR search demo

This is a demo of several multi-modal IR search experiments.

# Setup
```
$ git clone
$ cd build
$ docker compose build
```

Running the demo.
```
$ chmod +x start.sh
$ ./start.sh
```

View the demo at http://localhost:8501. FastAPI docs can be viewed at http://localhost:8000/docs.

# Todos:
- [ ] Test query API
- [ ] Build demo frontend to include:
    - [ ] Image search
    - [ ] Text search
    - [ ] Model selection
    - [ ] Adding new data
    - [ ] Embedding visualization
- [ ] Error handling
- [x] Setup environment for FastAPI server and weaviate database