# Copyright 2026 Christopher Wilson
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from unittest.mock import Mock, call

import pytest

from blastradius.index import _embed_new_symbols


@pytest.fixture
def embedding_setup(monkeypatch):
    monkeypatch.setenv("BLASTRADIUS_EMBEDDING_ENDPOINT", "http://localhost:11434/")
    monkeypatch.setenv("BLASTRADIUS_EMBEDDING_MODEL", "test-model")
    monkeypatch.setenv("BLASTRADIUS_EMBEDDING_DIMS", "8")
    store = Mock()
    store.init_vectors.return_value = True
    store.symbols_needing_embeddings.return_value = [(7, "first"), (2, "second")]
    factory = Mock()
    provider = factory.return_value
    provider.embed.return_value = [[0.1], [0.2]]
    monkeypatch.setattr(
        "blastradius.semantic.provider.OpenAIEmbeddingProvider", factory
    )
    events = Mock()
    events.attach_mock(store, "store")
    events.attach_mock(factory, "provider")
    return store, factory, events


@pytest.mark.parametrize("name", ["ENDPOINT", "MODEL", "DIMS"])
@pytest.mark.parametrize("value", [None, ""])
def test_incomplete_configuration_skips(
    embedding_setup, monkeypatch, capsys, name, value
):
    key = f"BLASTRADIUS_EMBEDDING_{name}"
    if value is None:
        monkeypatch.delenv(key)
    else:
        monkeypatch.setenv(key, value)
    store, _, events = embedding_setup

    _embed_new_symbols(store)

    assert events.mock_calls == []
    assert capsys.readouterr().err == ""


def test_invalid_dimensions_skip(embedding_setup, monkeypatch, capsys):
    monkeypatch.setenv("BLASTRADIUS_EMBEDDING_DIMS", "eight")
    store, _, events = embedding_setup

    _embed_new_symbols(store)

    assert events.mock_calls == []
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("raw_dims,dims", [("8", 8), (" 8 ", 8), ("0", 0), ("-1", -1)])
def test_success_preserves_configuration_and_write_order(
    embedding_setup, monkeypatch, capsys, raw_dims, dims
):
    monkeypatch.setenv("BLASTRADIUS_EMBEDDING_DIMS", raw_dims)
    store, _, events = embedding_setup

    _embed_new_symbols(store)

    assert events.mock_calls == [
        call.store.init_vectors(dims),
        call.store.symbols_needing_embeddings(),
        call.provider(
            endpoint="http://localhost:11434/", model="test-model", dims=dims
        ),
        call.provider().embed(["first", "second"]),
        call.store.upsert_embeddings([(7, [0.1]), (2, [0.2])]),
        call.store.set_meta("embedding_model", "test-model"),
        call.store._conn.commit(),
    ]
    assert capsys.readouterr().err == "Embedded 2 symbol(s)\n"


def test_configuration_is_read_on_every_call(embedding_setup, monkeypatch):
    store, factory, _ = embedding_setup
    _embed_new_symbols(store)

    monkeypatch.setenv("BLASTRADIUS_EMBEDDING_ENDPOINT", "http://other:8080")
    monkeypatch.setenv("BLASTRADIUS_EMBEDDING_MODEL", "other-model")
    monkeypatch.setenv("BLASTRADIUS_EMBEDDING_DIMS", "16")
    _embed_new_symbols(store)

    assert factory.call_args_list == [
        call(endpoint="http://localhost:11434/", model="test-model", dims=8),
        call(endpoint="http://other:8080", model="other-model", dims=16),
    ]
    assert store.init_vectors.call_args_list == [call(8), call(16)]


@pytest.mark.parametrize("stop", ["vectors", "symbols"])
def test_nothing_to_embed_skips_provider(embedding_setup, capsys, stop):
    store, _, events = embedding_setup
    if stop == "vectors":
        store.init_vectors.return_value = False
        expected = [call.store.init_vectors(8)]
    else:
        store.symbols_needing_embeddings.return_value = []
        expected = [call.store.init_vectors(8), call.store.symbols_needing_embeddings()]

    _embed_new_symbols(store)

    assert events.mock_calls == expected
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "stage", ["construct", "embed", "upsert", "metadata", "commit"]
)
def test_embedding_failures_warn(embedding_setup, capsys, stage):
    store, factory, _ = embedding_setup
    operation = {
        "construct": factory,
        "embed": factory.return_value.embed,
        "upsert": store.upsert_embeddings,
        "metadata": store.set_meta,
        "commit": store._conn.commit,
    }[stage]
    operation.side_effect = RuntimeError("failed")

    _embed_new_symbols(store)

    assert capsys.readouterr().err == "Warning: embedding failed: failed\n"
    if stage in {"construct", "embed"}:
        store.upsert_embeddings.assert_not_called()
    if stage in {"construct", "embed", "upsert"}:
        store.set_meta.assert_not_called()
    if stage != "commit":
        store._conn.commit.assert_not_called()


@pytest.mark.parametrize("stage", ["init_vectors", "symbols_needing_embeddings"])
def test_preparation_failures_propagate(embedding_setup, capsys, stage):
    store, factory, _ = embedding_setup
    error = RuntimeError("failed")
    getattr(store, stage).side_effect = error

    with pytest.raises(RuntimeError) as raised:
        _embed_new_symbols(store)

    assert raised.value is error
    factory.assert_not_called()
    assert capsys.readouterr().err == ""
