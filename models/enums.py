"""Enums for the AI Agent Dinosaur Simulator."""

from enum import Enum, auto


class AgentRole(Enum):
    """Defines the different roles agents can have in the simulation."""
    PARK_RANGER = auto()
    VETERINARIAN = auto()
    SECURITY = auto()
    MAINTENANCE = auto()
    GUEST_RELATIONS = auto()
    TOURIST = auto()
    DINOSAUR = auto()


class AgentState(Enum):
    """Defines the current state of an agent."""
    IDLE = auto()
    ACTIVE = auto()
    RESPONDING_TO_EVENT = auto()
    COMMUNICATING = auto()
    UNAVAILABLE = auto()


class EventType(Enum):
    """Defines the types of events that can occur in the simulation."""
    DINOSAUR_ESCAPE = auto()
    DINOSAUR_ILLNESS = auto()
    DINOSAUR_AGGRESSIVE = auto()
    VISITOR_INJURY = auto()
    VISITOR_COMPLAINT = auto()
    VISITOR_EMERGENCY = auto()
    FACILITY_POWER_OUTAGE = auto()
    FACILITY_EQUIPMENT_FAILURE = auto()
    WEATHER_STORM = auto()
    WEATHER_EXTREME_TEMPERATURE = auto()
    CUSTOM = auto()


class ResolutionStatus(Enum):
    """Defines the resolution status of an event."""
    PENDING = auto()
    IN_PROGRESS = auto()
    RESOLVED = auto()
    ESCALATED = auto()
    FAILED = auto()


class DinosaurSpecies(Enum):
    """Defines the different dinosaur species in the simulation."""
    TYRANNOSAURUS_REX = auto()
    TRICERATOPS = auto()
    VELOCIRAPTOR = auto()
    BRACHIOSAURUS = auto()
    STEGOSAURUS = auto()
    PARASAUROLOPHUS = auto()


class PersonalityTrait(Enum):
    """Defines personality traits for agents."""
    CAUTIOUS = "cautious"
    BRAVE = "brave"
    FRIENDLY = "friendly"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    LEADERSHIP = "leadership"
    EMPATHY = "empathy"
    TECHNICAL = "technical"


class MessageType(Enum):
    """Defines the types of messages in human-AI communication."""
    HUMAN = auto()
    AI_AGENT = auto()
    SYSTEM = auto()