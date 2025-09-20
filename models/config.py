"""Configuration classes for the AI Agent Dinosaur Simulator."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from models.enums import AgentRole, DinosaurSpecies, PersonalityTrait


@dataclass
class Location:
    """Represents a location in the dinosaur resort."""
    x: float
    y: float
    zone: str
    description: str = ""


@dataclass
class AgentConfig:
    """Configuration for initializing agents."""
    staff_count: Dict[AgentRole, int] = field(default_factory=lambda: {
        AgentRole.PARK_RANGER: 2,
        AgentRole.VETERINARIAN: 1,
        AgentRole.SECURITY: 2,
        AgentRole.MAINTENANCE: 1
    })
    visitor_count: int = 5
    dinosaur_config: Dict[DinosaurSpecies, int] = field(default_factory=lambda: {
        DinosaurSpecies.TYRANNOSAURUS_REX: 1,
        DinosaurSpecies.TRICERATOPS: 2,
        DinosaurSpecies.VELOCIRAPTOR: 3,
        DinosaurSpecies.BRACHIOSAURUS: 1,
        DinosaurSpecies.STEGOSAURUS: 2
    })


@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    max_agents: int = 50
    simulation_speed: float = 1.0
    auto_resolve_timeout: int = 300  # seconds
    metrics_update_interval: int = 10  # seconds
    enable_logging: bool = True
    log_level: str = "INFO"


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI integration."""
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 500
    timeout: int = 30
    max_retries: int = 3


@dataclass
class AG2Config:
    """Configuration for ag2 framework."""
    max_round: int = 10
    human_input_mode: str = "NEVER"
    code_execution_config: bool = False
    system_message_template: str = "You are {role} in a dinosaur resort simulation. {personality_description}"