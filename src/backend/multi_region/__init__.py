"""
Multi-Region Deployment Module for Nexus Forge
Provides global infrastructure management and edge optimization
"""

from .data_sync import DataSynchronizer
from .edge_optimizer import EdgeOptimizer
from .load_balancer import GlobalLoadBalancer
from .models import EdgeLocation, Region, RegionConfig
from .region_manager import RegionManager

__all__ = [
    "RegionManager",
    "GlobalLoadBalancer",
    "DataSynchronizer",
    "EdgeOptimizer",
    "Region",
    "RegionConfig",
    "EdgeLocation",
]
