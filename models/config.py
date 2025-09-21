"""Configuration classes for the AI Agent Dinosaur Simulator."""

import os
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
        AgentRole.GUEST_RELATIONS: 1,
        AgentRole.PARK_RANGER: 1,
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
    max_agents: int = field(default_factory=lambda: int(os.getenv("SIMULATION_MAX_AGENTS", "50")))
    simulation_speed: float = field(default_factory=lambda: float(os.getenv("SIMULATION_SPEED", "1.0")))
    auto_resolve_timeout: int = field(default_factory=lambda: int(os.getenv("SIMULATION_AUTO_RESOLVE_TIMEOUT", "300")))  # seconds
    metrics_update_interval: int = field(default_factory=lambda: int(os.getenv("SIMULATION_METRICS_UPDATE_INTERVAL", "10")))  # seconds
    enable_logging: bool = field(default_factory=lambda: os.getenv("SIMULATION_ENABLE_LOGGING", "true").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("SIMULATION_LOG_LEVEL", "INFO"))
    
    def __post_init__(self):
        """Validate simulation configuration."""
        if self.max_agents <= 0:
            raise ValueError("Simulation max_agents must be positive.")
        if self.simulation_speed <= 0:
            raise ValueError("Simulation speed must be positive.")
        if self.auto_resolve_timeout <= 0:
            raise ValueError("Simulation auto_resolve_timeout must be positive.")
        if self.metrics_update_interval <= 0:
            raise ValueError("Simulation metrics_update_interval must be positive.")
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Simulation log_level must be one of {valid_log_levels}.")


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI integration."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4"))
    temperature: float = field(default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_TOKENS", "500")))
    timeout: int = field(default_factory=lambda: int(os.getenv("OPENAI_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_RETRIES", "3")))
    
    def __post_init__(self):
        """Validate OpenAI configuration."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("OpenAI temperature must be between 0 and 2.")
        if self.max_tokens <= 0:
            raise ValueError("OpenAI max_tokens must be positive.")
        if self.timeout <= 0:
            raise ValueError("OpenAI timeout must be positive.")
        if self.max_retries < 0:
            raise ValueError("OpenAI max_retries must be non-negative.")


@dataclass
class AG2Config:
    """Configuration for ag2 framework."""
    max_round: int = field(default_factory=lambda: int(os.getenv("AG2_MAX_ROUND", "10")))
    human_input_mode: str = field(default_factory=lambda: os.getenv("AG2_HUMAN_INPUT_MODE", "NEVER"))
    code_execution_config: bool = field(default_factory=lambda: os.getenv("AG2_CODE_EXECUTION_CONFIG", "false").lower() == "true")
    system_message_template: str = "You are {role} in a dinosaur resort simulation. {personality_description}"
    
    def __post_init__(self):
        """Validate ag2 configuration."""
        if self.max_round <= 0:
            raise ValueError("ag2 max_round must be positive.")
        valid_input_modes = ["ALWAYS", "NEVER", "TERMINATE"]
        if self.human_input_mode not in valid_input_modes:
            raise ValueError(f"ag2 human_input_mode must be one of {valid_input_modes}.")