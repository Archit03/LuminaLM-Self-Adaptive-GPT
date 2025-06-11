from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import torch
from omegaconf import OmegaConf, DataConfig
import os
from datetime import datetime
from enum import Enum

class DeviceType(Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps" #Apple Silicon
    TPU = "tpu" #Tensor Processing Unit

class PrecisionType(Enum):
    """Precision type enumeration"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    MIXED = "mixed"

class ModelConfig:
    """Core model config matching your Enhanced Self-Adaptive AI"""
    model_name: str = "gpt2"
    model_type: str = "gpt2"

    hidden_size: Optional[int] = None
    num_attention_layers: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    vocab_size: Optional[int] = None

    max_length:  int = 256
    max_new_tokens = int = 36

    temperature: float = 0.8
    top_p: float = 0.9
    top_k = int = 50
    do_sample: bool = True
    temperature_annealing: bool = True

