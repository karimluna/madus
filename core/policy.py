"""Helper for task routing

ref: https://www.patronus.ai/ai-agent-development/ai-agent-routing"""

from dataclasses import dataclass
from typing import Literal

import psutil
import torch

Backend = Literal["local", "openai", "siglip", "colpali"]

@dataclass
class SystemState:
    ram_available: int   # bytes
    gpu_available: int   # bytes
    gpu_present: bool
#     ram_available = psutil.virtual_memory().available
#     gpu_present = torch.cuda.is_available()
#     gpu_available = torch.cuda.mem_get_info()[0]

@dataclass
class PolicyConfig:
    min_ram_local: int = 2 * 1024**3        # 2GB
    min_gpu_siglip: int = 2 * 1024**3       # 2GB VRAM
    min_gpu_colpali: int = 6 * 1024**3      # 6GB VRAM
    
    # max_local_tokens: int = 3000
    # max_batch_local: int = 8




def select_vision_backend(sys_state: SystemState, policy_config: PolicyConfig):
    """Dynamically selects the best image embedder"""
    ram_available = sys_state.ram_available
    gpu_present = sys_state.gpu_available

    gpu_available = 0
    if gpu_present:
        gpu_available = sys_state.gpu_available

    if ram_available < policy_config.min_gpu_siglip:
        if gpu_present and gpu_available > 0.75 * policy_config.min_gpu_siglip: # at least 25% more available ram
            return "siglip"
        else:
            return "disabled"
        
    if gpu_present:
        if gpu_available > policy_config.min_gpu_colpali:
            return "colpali"
        return "siglip"
    
    return "disabled"
