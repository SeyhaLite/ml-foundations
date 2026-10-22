"""qwen_ft — Qwen Fine-Tuning from Zero toolkit.

Dependency-light core: this package intentionally imports only the standard
library + PyYAML at module level.  Heavy ML libraries (torch, transformers,
peft, trl) are imported lazily inside :mod:`qwen_ft.training` and
:mod:`qwen_ft.inference` so that validation, dataset checks, config checks and
documentation builds run on any machine — no GPU, no multi-GB downloads.
"""

from qwen_ft.config import (
    ConfigError,
    InferenceConfig,
    LoRAConfig,
    TrainingConfig,
    load_config,
    save_config,
)
from qwen_ft.data import (
    DataValidationError,
    split_dataset,
    validate_jsonl_file,
    validate_messages,
)
from qwen_ft.models import (
    ModelInfo,
    get_model_info,
    list_models,
    minimum_transformers_version,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "ConfigError",
    "DataValidationError",
    "InferenceConfig",
    "LoRAConfig",
    "ModelInfo",
    "TrainingConfig",
    "get_model_info",
    "list_models",
    "load_config",
    "minimum_transformers_version",
    "save_config",
    "split_dataset",
    "validate_jsonl_file",
    "validate_messages",
]
