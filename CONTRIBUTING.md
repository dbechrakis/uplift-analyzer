# Contributing

```bash
make install
make check
```

* `make check` runs ruff, the format check and the full test suite offline.
* Tests never touch the network and never need the real dataset or an LLM server.
* Dependencies change in `requirements*.in`; run `make lock` and commit the regenerated `requirements*.txt`.
* Statistics belong in `core/`. The LLM layer in `app/` must never compute numbers.
