# Contributing

```bash
make install
make check
```

* `make check` runs ruff, the format check and the full test suite offline.
* Tests never touch the network and never need the real dataset or an LLM server.
* Dependencies change in `requirements*.in`. Run `make lock` and commit the regenerated `requirements*.txt`.
* Statistics belong in `core/`. The LLM layer in `app/` must never compute numbers in code, and the model is instructed not to compute them. The grounding check flags numbers it writes anyway.
