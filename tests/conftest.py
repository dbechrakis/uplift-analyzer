import socket

import pytest

from core.synthetic import generate_synthetic


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise ConnectionRefusedError("Network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(scope="session")
def synthetic_df():
    return generate_synthetic(n=40_000, seed=1)
