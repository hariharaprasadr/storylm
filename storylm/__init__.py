"""StoryLM: a ~2M parameter Tiny Language Model for edge & embedded devices.

The shipped generation is v4 (2,024,192 params, 3,509-id closed vocabulary);
`storylm.vocab` pairs every checkpoint with the token table it was trained on.
"""

from .model import (
    Config as StoryLMConfig,
    Config,
    StoryLM,
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
    "StoryLMConfig",
    "Config",
    "StoryLM",
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
