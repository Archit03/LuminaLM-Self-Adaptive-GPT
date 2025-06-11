"""
LuminaLM Self-Adaptive GPT - Core Components

This module contains the core components for self-adaptive language models,
inspired by Transformer² architecture with dynamic parameter adaptation.

The core module provides:
- Configuration classes for model and training setup
- Self-adaptive model architectures with SVD decomposition
- Episode management for reinforcement learning training
- Task-specific reward functions
- Dynamic parameter adaptation utilities
"""
from .config import (
    ModelConfig, 
    GRPOConfig, 
    CEMConfig, 
    AdaptionConfig,
    TaskConfig
)

from .models import(
    SelfAdaptiveGPT,
    ValueNetwork, 
    ExpertMixer, 
    ParameterAdapter
)

from .reward_functions import(
    BaseRewardFunction, 
    AccuracyReward, 
    PerplexityReward, 
    TaskSpecificReward, 
    get_reward_function
)

from .svd import (
    SVDDecomposer, 
    compose_parameters, 
    decompose_parameters,
    adaption_svd_rank
)

__all__ = [
    # Config Classes
    "ModelConfig", 
    "GRPOConfig", 
    "CEMConfig", 
    "AdaptionConfig", 
    "TaskConfig", 

    #Model classes
    "SelfAdaptiveGPT",
    "ValueNetwork", 
    "ExpertMixer",
    "ParameterAdapter", 

    #Reward function
    "Episode", 
    "EpisodeBuffer",
    "create_episode_from_batch", 
    
    #SVD utilities
    "SVDDecomposer", 
    "compose_parameters",
    "decompose_parameters", 
    "adaptive_svd_rank"
]

__version__ = "0.1.0"
__author__ = "EllanorAI Deep Learning Research"


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_GRPO_CONFIG = GRPOConfig()
DEFAULT_CEM_CONFIG = CEMConfig()
DEFAULT_ADAPTION_CONFIG = AdaptionConfig()

