"""TinyLM: 1.67M parameter Nano Language Model for edge & embedded devices."""

from .model import (
    Config as TinyLMConfig,
    Config,
    TinyLM,
    Attention,
    Block,
    Block as TransformerBlock,
    KVBuf,
    KVBuf as KVCache,
    apply_rope,
    rope_tables,
    generate_story,
    generate_story_cached,
    decode_lines,
)

__version__ = "0.1.0"
__all__ = [
    "TinyLMConfig",
    "Config",
    "TinyLM",
    "Attention",
    "Block",
    "TransformerBlock",
    "KVBuf",
    "KVCache",
    "apply_rope",
    "rope_tables",
    "generate_story",
    "generate_story_cached",
    "decode_lines",
]
