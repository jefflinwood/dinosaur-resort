# Managers package for AI Agent Dinosaur Simulator

from .metrics_manager import MetricsManager
from .event_manager import EventManager
# Note: HumanPlayerManager and RealTimeAgentChat should be imported directly to avoid circular imports

__all__ = ['MetricsManager', 'EventManager']