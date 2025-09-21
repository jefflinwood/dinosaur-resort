"""Core data models for the AI Agent Dinosaur Simulator."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from models.enums import AgentRole, AgentState, EventType, ResolutionStatus, DinosaurSpecies, MessageType
from models.config import Location


@dataclass
class Agent:
    """Represents an AI agent in the simulation."""
    id: str
    name: str
    role: AgentRole
    personality_traits: Dict[str, float] = field(default_factory=dict)
    current_state: AgentState = AgentState.IDLE
    location: Location = field(default_factory=lambda: Location(0.0, 0.0, "entrance"))
    capabilities: List[str] = field(default_factory=list)
    species: Optional[DinosaurSpecies] = None  # Only for dinosaur agents
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate agent data after initialization."""
        if not self.id:
            raise ValueError("Agent ID cannot be empty")
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        if self.role == AgentRole.DINOSAUR and self.species is None:
            raise ValueError("Dinosaur agents must have a species")
        if self.role != AgentRole.DINOSAUR and self.species is not None:
            raise ValueError("Only dinosaur agents can have a species")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.name,
            "personality_traits": self.personality_traits,
            "current_state": self.current_state.name,
            "location": {
                "x": self.location.x,
                "y": self.location.y,
                "zone": self.location.zone,
                "description": self.location.description
            },
            "capabilities": self.capabilities,
            "species": self.species.name if self.species else None,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Create agent from dictionary."""
        location_data = data["location"]
        location = Location(
            x=location_data["x"],
            y=location_data["y"],
            zone=location_data["zone"],
            description=location_data.get("description", "")
        )
        
        return cls(
            id=data["id"],
            name=data["name"],
            role=AgentRole[data["role"]],
            personality_traits=data["personality_traits"],
            current_state=AgentState[data["current_state"]],
            location=location,
            capabilities=data["capabilities"],
            species=DinosaurSpecies[data["species"]] if data["species"] else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"])
        )


@dataclass
class Event:
    """Represents an event in the simulation."""
    id: str
    type: EventType
    severity: int  # 1-10 scale
    location: Location
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    affected_agents: List[str] = field(default_factory=list)
    resolution_status: ResolutionStatus = ResolutionStatus.PENDING
    resolution_time: Optional[datetime] = None
    description: str = ""
    
    def __post_init__(self):
        """Validate event data after initialization."""
        if not self.id:
            raise ValueError("Event ID cannot be empty")
        if not (1 <= self.severity <= 10):
            raise ValueError("Event severity must be between 1 and 10")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.name,
            "severity": self.severity,
            "location": {
                "x": self.location.x,
                "y": self.location.y,
                "zone": self.location.zone,
                "description": self.location.description
            },
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
            "affected_agents": self.affected_agents,
            "resolution_status": self.resolution_status.name,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        location_data = data["location"]
        location = Location(
            x=location_data["x"],
            y=location_data["y"],
            zone=location_data["zone"],
            description=location_data.get("description", "")
        )
        
        return cls(
            id=data["id"],
            type=EventType[data["type"]],
            severity=data["severity"],
            location=location,
            parameters=data["parameters"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            affected_agents=data["affected_agents"],
            resolution_status=ResolutionStatus[data["resolution_status"]],
            resolution_time=datetime.fromisoformat(data["resolution_time"]) if data["resolution_time"] else None,
            description=data["description"]
        )


@dataclass
class MetricsSnapshot:
    """Represents a snapshot of resort metrics at a point in time."""
    visitor_satisfaction: float  # 0.0 to 1.0
    dinosaur_happiness: Dict[str, float] = field(default_factory=dict)  # agent_id -> happiness (0.0 to 1.0)
    facility_efficiency: float = 1.0  # 0.0 to 1.0
    safety_rating: float = 1.0  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate metrics data after initialization."""
        if not (0.0 <= self.visitor_satisfaction <= 1.0):
            raise ValueError("Visitor satisfaction must be between 0.0 and 1.0")
        if not (0.0 <= self.facility_efficiency <= 1.0):
            raise ValueError("Facility efficiency must be between 0.0 and 1.0")
        if not (0.0 <= self.safety_rating <= 1.0):
            raise ValueError("Safety rating must be between 0.0 and 1.0")
        for agent_id, happiness in self.dinosaur_happiness.items():
            if not (0.0 <= happiness <= 1.0):
                raise ValueError(f"Dinosaur happiness for {agent_id} must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "visitor_satisfaction": self.visitor_satisfaction,
            "dinosaur_happiness": self.dinosaur_happiness,
            "facility_efficiency": self.facility_efficiency,
            "safety_rating": self.safety_rating,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricsSnapshot':
        """Create metrics from dictionary."""
        return cls(
            visitor_satisfaction=data["visitor_satisfaction"],
            dinosaur_happiness=data["dinosaur_happiness"],
            facility_efficiency=data["facility_efficiency"],
            safety_rating=data["safety_rating"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class ChatMessage:
    """Represents a chat message in human-AI communication."""
    id: str
    sender_id: str
    sender_name: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: MessageType = MessageType.AI_AGENT
    conversation_id: str = ""
    
    def __post_init__(self):
        """Validate chat message data after initialization."""
        if not self.id:
            raise ValueError("Message ID cannot be empty")
        if not self.sender_id:
            raise ValueError("Sender ID cannot be empty")
        if not self.sender_name:
            raise ValueError("Sender name cannot be empty")
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chat message to dictionary for serialization."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type.name,
            "conversation_id": self.conversation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create chat message from dictionary."""
        return cls(
            id=data["id"],
            sender_id=data["sender_id"],
            sender_name=data["sender_name"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_type=MessageType[data["message_type"]],
            conversation_id=data["conversation_id"]
        )


@dataclass
class HumanAgent(Agent):
    """Represents a human player agent extending the base Agent model."""
    is_human_controlled: bool = field(default=True)
    chat_history: List[ChatMessage] = field(default_factory=list)
    conversation_access: List[str] = field(default_factory=list)  # Conversation IDs the human can see
    
    def __post_init__(self):
        """Validate human agent data after initialization."""
        # Call parent validation first
        super().__post_init__()
        
        # Validate human-specific fields
        if not isinstance(self.is_human_controlled, bool):
            raise ValueError("is_human_controlled must be a boolean")
        
        # Validate chat history
        for message in self.chat_history:
            if not isinstance(message, ChatMessage):
                raise ValueError("All chat history items must be ChatMessage instances")
        
        # Validate conversation access
        for conv_id in self.conversation_access:
            if not isinstance(conv_id, str):
                raise ValueError("All conversation access IDs must be strings")
    
    def add_chat_message(self, message: ChatMessage) -> None:
        """Add a chat message to the history.
        
        Args:
            message: ChatMessage to add
        """
        if not isinstance(message, ChatMessage):
            raise ValueError("Message must be a ChatMessage instance")
        
        self.chat_history.append(message)
        self.last_activity = datetime.now()
    
    def get_recent_messages(self, limit: int = 50) -> List[ChatMessage]:
        """Get recent chat messages.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of recent ChatMessage instances
        """
        return self.chat_history[-limit:] if self.chat_history else []
    
    def add_conversation_access(self, conversation_id: str) -> None:
        """Add access to a conversation.
        
        Args:
            conversation_id: ID of the conversation to grant access to
        """
        if not isinstance(conversation_id, str):
            raise ValueError("Conversation ID must be a string")
        
        if conversation_id not in self.conversation_access:
            self.conversation_access.append(conversation_id)
    
    def remove_conversation_access(self, conversation_id: str) -> None:
        """Remove access to a conversation.
        
        Args:
            conversation_id: ID of the conversation to remove access from
        """
        if conversation_id in self.conversation_access:
            self.conversation_access.remove(conversation_id)
    
    def has_conversation_access(self, conversation_id: str) -> bool:
        """Check if human has access to a conversation.
        
        Args:
            conversation_id: ID of the conversation to check
            
        Returns:
            True if human has access, False otherwise
        """
        return conversation_id in self.conversation_access
    
    def clear_chat_history(self) -> None:
        """Clear all chat history."""
        self.chat_history.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert human agent to dictionary for serialization."""
        # Get base agent dictionary
        base_dict = super().to_dict()
        
        # Add human-specific fields
        base_dict.update({
            "is_human_controlled": self.is_human_controlled,
            "chat_history": [message.to_dict() for message in self.chat_history],
            "conversation_access": self.conversation_access
        })
        
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HumanAgent':
        """Create human agent from dictionary."""
        # Extract location data
        location_data = data["location"]
        location = Location(
            x=location_data["x"],
            y=location_data["y"],
            zone=location_data["zone"],
            description=location_data.get("description", "")
        )
        
        # Extract chat history
        chat_history = [ChatMessage.from_dict(msg_data) for msg_data in data.get("chat_history", [])]
        
        return cls(
            id=data["id"],
            name=data["name"],
            role=AgentRole[data["role"]],
            personality_traits=data["personality_traits"],
            current_state=AgentState[data["current_state"]],
            location=location,
            capabilities=data["capabilities"],
            species=DinosaurSpecies[data["species"]] if data["species"] else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            is_human_controlled=data.get("is_human_controlled", True),
            chat_history=chat_history,
            conversation_access=data.get("conversation_access", [])
        )


@dataclass
class SimulationState:
    """Represents the current state of the simulation."""
    is_running: bool = False
    current_time: datetime = field(default_factory=datetime.now)
    active_events: List[Event] = field(default_factory=list)
    agent_count: int = 0
    current_metrics: Optional[MetricsSnapshot] = None
    human_player: Optional[HumanAgent] = None
    simulation_id: str = ""
    started_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate simulation state after initialization."""
        if self.agent_count < 0:
            raise ValueError("Agent count cannot be negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert simulation state to dictionary for serialization."""
        return {
            "is_running": self.is_running,
            "current_time": self.current_time.isoformat(),
            "active_events": [event.to_dict() for event in self.active_events],
            "agent_count": self.agent_count,
            "current_metrics": self.current_metrics.to_dict() if self.current_metrics else None,
            "human_player": self.human_player.to_dict() if self.human_player else None,
            "simulation_id": self.simulation_id,
            "started_at": self.started_at.isoformat() if self.started_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationState':
        """Create simulation state from dictionary."""
        active_events = [Event.from_dict(event_data) for event_data in data["active_events"]]
        current_metrics = MetricsSnapshot.from_dict(data["current_metrics"]) if data["current_metrics"] else None
        human_player = HumanAgent.from_dict(data["human_player"]) if data["human_player"] else None
        
        return cls(
            is_running=data["is_running"],
            current_time=datetime.fromisoformat(data["current_time"]),
            active_events=active_events,
            agent_count=data["agent_count"],
            current_metrics=current_metrics,
            human_player=human_player,
            simulation_id=data["simulation_id"],
            started_at=datetime.fromisoformat(data["started_at"]) if data["started_at"] else None
        )