from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "af8e934848d8dd00074cc2cd8a40a9b05c3b011e"


class ConfigurationError(ValueError):
    pass


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class PagedCoreConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    context_limit: int = 2048
    min_new_tokens: int = 1
    default_max_new_tokens: int = 128
    max_new_tokens: int = 256
    block_tokens: int = 16
    kv_pool_mib: int = 8192
    workspace_margin_mib: int = 1024
    max_pending_requests: int = 64
    max_active_sequences: int = 32
    token_queue_size: int = 32
    bind_address: str = "localhost"
    log_level: LogLevel = LogLevel.INFO

    def __post_init__(self) -> None:
        fixed_values = (
            ("model_id", self.model_id, MODEL_ID),
            ("model_revision", self.model_revision, MODEL_REVISION),
            ("context_limit", self.context_limit, 2048),
            ("min_new_tokens", self.min_new_tokens, 1),
            ("default_max_new_tokens", self.default_max_new_tokens, 128),
            ("max_new_tokens", self.max_new_tokens, 256),
            ("block_tokens", self.block_tokens, 16),
        )
        for name, value, expected in fixed_values:
            if value != expected:
                raise ConfigurationError(f"{name} is fixed at {expected!r}")

        positive_values = (
            ("kv_pool_mib", self.kv_pool_mib),
            ("workspace_margin_mib", self.workspace_margin_mib),
            ("max_pending_requests", self.max_pending_requests),
            ("max_active_sequences", self.max_active_sequences),
            ("token_queue_size", self.token_queue_size),
        )
        for name, value in positive_values:
            if value <= 0:
                raise ConfigurationError(f"{name} must be greater than zero")

        if not self.bind_address or self.bind_address != self.bind_address.strip():
            raise ConfigurationError(
                "bind_address must be a non-empty value without whitespace padding"
            )
        if not isinstance(self.log_level, LogLevel):
            raise ConfigurationError("log_level must be a supported LogLevel")


def _read_int(environment: Mapping[str, str], name: str, default: int) -> int:
    raw_value = environment.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error


def _read_log_level(environment: Mapping[str, str]) -> LogLevel:
    raw_value = environment.get("PAGEDCORE_LOG_LEVEL", LogLevel.INFO.value)
    try:
        return LogLevel(raw_value.upper())
    except ValueError as error:
        choices = ", ".join(level.value for level in LogLevel)
        raise ConfigurationError(f"PAGEDCORE_LOG_LEVEL must be one of: {choices}") from error


def load_config(environment: Mapping[str, str] | None = None) -> PagedCoreConfig:
    source = os.environ if environment is None else environment
    return PagedCoreConfig(
        model_id=source.get("PAGEDCORE_MODEL_ID", MODEL_ID),
        model_revision=source.get("PAGEDCORE_MODEL_REVISION", MODEL_REVISION),
        context_limit=_read_int(source, "PAGEDCORE_CONTEXT_LIMIT", 2048),
        min_new_tokens=_read_int(source, "PAGEDCORE_MIN_NEW_TOKENS", 1),
        default_max_new_tokens=_read_int(source, "PAGEDCORE_DEFAULT_MAX_NEW_TOKENS", 128),
        max_new_tokens=_read_int(source, "PAGEDCORE_MAX_NEW_TOKENS", 256),
        block_tokens=_read_int(source, "PAGEDCORE_BLOCK_TOKENS", 16),
        kv_pool_mib=_read_int(source, "PAGEDCORE_KV_POOL_MIB", 8192),
        workspace_margin_mib=_read_int(source, "PAGEDCORE_WORKSPACE_MARGIN_MIB", 1024),
        max_pending_requests=_read_int(source, "PAGEDCORE_MAX_PENDING_REQUESTS", 64),
        max_active_sequences=_read_int(source, "PAGEDCORE_MAX_ACTIVE_SEQUENCES", 32),
        token_queue_size=_read_int(source, "PAGEDCORE_TOKEN_QUEUE_SIZE", 32),
        bind_address=source.get("PAGEDCORE_BIND_ADDRESS", "localhost"),
        log_level=_read_log_level(source),
    )
