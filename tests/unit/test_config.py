from dataclasses import FrozenInstanceError

import pytest

from pagedcore.config import (
    MODEL_ID,
    MODEL_REVISION,
    ConfigurationError,
    LogLevel,
    PagedCoreConfig,
    load_config,
)


def test_load_config_uses_documented_defaults() -> None:
    config = load_config({})

    assert config == PagedCoreConfig(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        context_limit=2048,
        min_new_tokens=1,
        default_max_new_tokens=128,
        max_new_tokens=256,
        block_tokens=16,
        kv_pool_mib=8192,
        workspace_margin_mib=1024,
        max_pending_requests=64,
        max_active_sequences=32,
        token_queue_size=32,
        bind_address="localhost",
        log_level=LogLevel.INFO,
    )


def test_load_config_applies_operational_overrides() -> None:
    config = load_config(
        {
            "PAGEDCORE_KV_POOL_MIB": "7168",
            "PAGEDCORE_WORKSPACE_MARGIN_MIB": "1536",
            "PAGEDCORE_MAX_PENDING_REQUESTS": "12",
            "PAGEDCORE_MAX_ACTIVE_SEQUENCES": "6",
            "PAGEDCORE_TOKEN_QUEUE_SIZE": "8",
            "PAGEDCORE_BIND_ADDRESS": "127.0.0.1",
            "PAGEDCORE_LOG_LEVEL": "debug",
        }
    )

    assert config.kv_pool_mib == 7168
    assert config.workspace_margin_mib == 1536
    assert config.max_pending_requests == 12
    assert config.max_active_sequences == 6
    assert config.token_queue_size == 8
    assert config.bind_address == "127.0.0.1"
    assert config.log_level is LogLevel.DEBUG


def test_config_is_immutable() -> None:
    config = PagedCoreConfig()

    with pytest.raises(FrozenInstanceError):
        config.kv_pool_mib = 4096


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PAGEDCORE_KV_POOL_MIB", "0"),
        ("PAGEDCORE_WORKSPACE_MARGIN_MIB", "-1"),
        ("PAGEDCORE_MAX_PENDING_REQUESTS", "0"),
        ("PAGEDCORE_MAX_ACTIVE_SEQUENCES", "0"),
        ("PAGEDCORE_TOKEN_QUEUE_SIZE", "0"),
    ],
)
def test_load_config_rejects_non_positive_operational_limits(name: str, value: str) -> None:
    with pytest.raises(ConfigurationError):
        load_config({name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PAGEDCORE_MODEL_ID", "another/model"),
        ("PAGEDCORE_MODEL_REVISION", "main"),
        ("PAGEDCORE_CONTEXT_LIMIT", "4096"),
        ("PAGEDCORE_MIN_NEW_TOKENS", "0"),
        ("PAGEDCORE_DEFAULT_MAX_NEW_TOKENS", "64"),
        ("PAGEDCORE_MAX_NEW_TOKENS", "512"),
        ("PAGEDCORE_BLOCK_TOKENS", "32"),
    ],
)
def test_load_config_rejects_changes_to_fixed_contract(name: str, value: str) -> None:
    with pytest.raises(ConfigurationError):
        load_config({name: value})


def test_load_config_rejects_non_integer_values() -> None:
    with pytest.raises(ConfigurationError, match="PAGEDCORE_KV_POOL_MIB must be an integer"):
        load_config({"PAGEDCORE_KV_POOL_MIB": "eight-gib"})


def test_load_config_rejects_invalid_bind_address() -> None:
    with pytest.raises(ConfigurationError, match="bind_address"):
        load_config({"PAGEDCORE_BIND_ADDRESS": " localhost "})


def test_load_config_rejects_unknown_log_level() -> None:
    with pytest.raises(ConfigurationError, match="PAGEDCORE_LOG_LEVEL"):
        load_config({"PAGEDCORE_LOG_LEVEL": "verbose"})
