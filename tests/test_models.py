"""Unit tests for core data models."""

import pytest
from datetime import datetime
from models.core import Agent, Event, MetricsSnapshot, SimulationState, ChatMessage, HumanAgent
from models.enums import AgentRole, AgentState, EventType, ResolutionStatus, DinosaurSpecies, MessageType
from models.config import Location


class TestAgent:
    """Test cases for Agent model."""
    
    def test_agent_creation_valid(self):
        """Test creating a valid agent."""
        location = Location(x=10.0, y=20.0, zone="main_area")
        agent = Agent(
            id="agent_1",
            name="John Ranger",
            role=AgentRole.PARK_RANGER,
            location=location,
            capabilities=["emergency_response", "wildlife_management"]
        )
        
        assert agent.id == "agent_1"
        assert agent.name == "John Ranger"
        assert agent.role == AgentRole.PARK_RANGER
        assert agent.current_state == AgentState.IDLE
        assert agent.location.x == 10.0
        assert agent.species is None
    
    def test_dinosaur_agent_creation(self):
        """Test creating a dinosaur agent with species."""
        location = Location(x=0.0, y=0.0, zone="paddock_1")
        agent = Agent(
            id="dino_1",
            name="Rex",
            role=AgentRole.DINOSAUR,
            species=DinosaurSpecies.TYRANNOSAURUS_REX,
            location=location
        )
        
        assert agent.species == DinosaurSpecies.TYRANNOSAURUS_REX
    
    def test_agent_validation_empty_id(self):
        """Test agent validation with empty ID."""
        location = Location(x=0.0, y=0.0, zone="test")
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            Agent(id="", name="Test", role=AgentRole.PARK_RANGER, location=location)
    
    def test_agent_validation_empty_name(self):
        """Test agent validation with empty name."""
        location = Location(x=0.0, y=0.0, zone="test")
        with pytest.raises(ValueError, match="Agent name cannot be empty"):
            Agent(id="test_1", name="", role=AgentRole.PARK_RANGER, location=location)
    
    def test_dinosaur_agent_without_species(self):
        """Test dinosaur agent validation without species."""
        location = Location(x=0.0, y=0.0, zone="test")
        with pytest.raises(ValueError, match="Dinosaur agents must have a species"):
            Agent(id="dino_1", name="Rex", role=AgentRole.DINOSAUR, location=location)
    
    def test_non_dinosaur_agent_with_species(self):
        """Test non-dinosaur agent validation with species."""
        location = Location(x=0.0, y=0.0, zone="test")
        with pytest.raises(ValueError, match="Only dinosaur agents can have a species"):
            Agent(
                id="ranger_1", 
                name="John", 
                role=AgentRole.PARK_RANGER, 
                species=DinosaurSpecies.TYRANNOSAURUS_REX,
                location=location
            )
    
    def test_agent_serialization(self):
        """Test agent to_dict and from_dict methods."""
        location = Location(x=15.5, y=25.5, zone="visitor_center", description="Main entrance")
        original_agent = Agent(
            id="agent_2",
            name="Dr. Smith",
            role=AgentRole.VETERINARIAN,
            personality_traits={"empathy": 0.9, "technical": 0.8},
            current_state=AgentState.ACTIVE,
            location=location,
            capabilities=["medical_treatment", "animal_behavior"]
        )
        
        # Test serialization
        agent_dict = original_agent.to_dict()
        assert agent_dict["id"] == "agent_2"
        assert agent_dict["role"] == "VETERINARIAN"
        assert agent_dict["location"]["x"] == 15.5
        
        # Test deserialization
        restored_agent = Agent.from_dict(agent_dict)
        assert restored_agent.id == original_agent.id
        assert restored_agent.role == original_agent.role
        assert restored_agent.location.x == original_agent.location.x
        assert restored_agent.personality_traits == original_agent.personality_traits


class TestEvent:
    """Test cases for Event model."""
    
    def test_event_creation_valid(self):
        """Test creating a valid event."""
        location = Location(x=30.0, y=40.0, zone="paddock_2")
        event = Event(
            id="event_1",
            type=EventType.DINOSAUR_ESCAPE,
            severity=8,
            location=location,
            description="T-Rex has escaped from paddock"
        )
        
        assert event.id == "event_1"
        assert event.type == EventType.DINOSAUR_ESCAPE
        assert event.severity == 8
        assert event.resolution_status == ResolutionStatus.PENDING
    
    def test_event_validation_empty_id(self):
        """Test event validation with empty ID."""
        location = Location(x=0.0, y=0.0, zone="test")
        with pytest.raises(ValueError, match="Event ID cannot be empty"):
            Event(id="", type=EventType.DINOSAUR_ESCAPE, severity=5, location=location)
    
    def test_event_validation_invalid_severity(self):
        """Test event validation with invalid severity."""
        location = Location(x=0.0, y=0.0, zone="test")
        with pytest.raises(ValueError, match="Event severity must be between 1 and 10"):
            Event(id="event_1", type=EventType.DINOSAUR_ESCAPE, severity=15, location=location)
        
        with pytest.raises(ValueError, match="Event severity must be between 1 and 10"):
            Event(id="event_2", type=EventType.DINOSAUR_ESCAPE, severity=0, location=location)
    
    def test_event_serialization(self):
        """Test event to_dict and from_dict methods."""
        location = Location(x=50.0, y=60.0, zone="medical_bay", description="Emergency area")
        original_event = Event(
            id="event_3",
            type=EventType.VISITOR_INJURY,
            severity=6,
            location=location,
            parameters={"injury_type": "minor_cut", "visitor_id": "visitor_5"},
            affected_agents=["medic_1", "security_2"],
            description="Visitor injured by fence"
        )
        
        # Test serialization
        event_dict = original_event.to_dict()
        assert event_dict["id"] == "event_3"
        assert event_dict["type"] == "VISITOR_INJURY"
        assert event_dict["parameters"]["injury_type"] == "minor_cut"
        
        # Test deserialization
        restored_event = Event.from_dict(event_dict)
        assert restored_event.id == original_event.id
        assert restored_event.type == original_event.type
        assert restored_event.parameters == original_event.parameters
        assert restored_event.affected_agents == original_event.affected_agents


class TestMetricsSnapshot:
    """Test cases for MetricsSnapshot model."""
    
    def test_metrics_creation_valid(self):
        """Test creating valid metrics."""
        metrics = MetricsSnapshot(
            visitor_satisfaction=0.85,
            dinosaur_happiness={"dino_1": 0.9, "dino_2": 0.7},
            facility_efficiency=0.95,
            safety_rating=0.8
        )
        
        assert metrics.visitor_satisfaction == 0.85
        assert metrics.dinosaur_happiness["dino_1"] == 0.9
        assert metrics.facility_efficiency == 0.95
        assert metrics.safety_rating == 0.8
    
    def test_metrics_validation_visitor_satisfaction(self):
        """Test metrics validation for visitor satisfaction."""
        with pytest.raises(ValueError, match="Visitor satisfaction must be between 0.0 and 1.0"):
            MetricsSnapshot(visitor_satisfaction=1.5)
        
        with pytest.raises(ValueError, match="Visitor satisfaction must be between 0.0 and 1.0"):
            MetricsSnapshot(visitor_satisfaction=-0.1)
    
    def test_metrics_validation_facility_efficiency(self):
        """Test metrics validation for facility efficiency."""
        with pytest.raises(ValueError, match="Facility efficiency must be between 0.0 and 1.0"):
            MetricsSnapshot(visitor_satisfaction=0.5, facility_efficiency=2.0)
    
    def test_metrics_validation_dinosaur_happiness(self):
        """Test metrics validation for dinosaur happiness."""
        with pytest.raises(ValueError, match="Dinosaur happiness for dino_1 must be between 0.0 and 1.0"):
            MetricsSnapshot(
                visitor_satisfaction=0.5,
                dinosaur_happiness={"dino_1": 1.5}
            )
    
    def test_metrics_serialization(self):
        """Test metrics to_dict and from_dict methods."""
        original_metrics = MetricsSnapshot(
            visitor_satisfaction=0.75,
            dinosaur_happiness={"rex_1": 0.8, "tri_1": 0.9},
            facility_efficiency=0.85,
            safety_rating=0.9
        )
        
        # Test serialization
        metrics_dict = original_metrics.to_dict()
        assert metrics_dict["visitor_satisfaction"] == 0.75
        assert metrics_dict["dinosaur_happiness"]["rex_1"] == 0.8
        
        # Test deserialization
        restored_metrics = MetricsSnapshot.from_dict(metrics_dict)
        assert restored_metrics.visitor_satisfaction == original_metrics.visitor_satisfaction
        assert restored_metrics.dinosaur_happiness == original_metrics.dinosaur_happiness


class TestSimulationState:
    """Test cases for SimulationState model."""
    
    def test_simulation_state_creation(self):
        """Test creating simulation state."""
        location = Location(x=0.0, y=0.0, zone="test")
        event = Event(id="test_event", type=EventType.DINOSAUR_ESCAPE, severity=5, location=location)
        metrics = MetricsSnapshot(visitor_satisfaction=0.8)
        human_player = HumanAgent(
            id="human_1",
            name="Player",
            role=AgentRole.PARK_RANGER,
            location=location
        )
        
        state = SimulationState(
            is_running=True,
            active_events=[event],
            agent_count=10,
            current_metrics=metrics,
            human_player=human_player,
            simulation_id="sim_1"
        )
        
        assert state.is_running is True
        assert len(state.active_events) == 1
        assert state.agent_count == 10
        assert state.current_metrics.visitor_satisfaction == 0.8
        assert state.human_player.name == "Player"
    
    def test_simulation_state_validation_negative_agent_count(self):
        """Test simulation state validation with negative agent count."""
        with pytest.raises(ValueError, match="Agent count cannot be negative"):
            SimulationState(agent_count=-1)
    
    def test_simulation_state_serialization(self):
        """Test simulation state to_dict and from_dict methods."""
        location = Location(x=10.0, y=20.0, zone="paddock")
        event = Event(id="event_1", type=EventType.FACILITY_POWER_OUTAGE, severity=7, location=location)
        metrics = MetricsSnapshot(visitor_satisfaction=0.6, facility_efficiency=0.3)
        human_player = HumanAgent(
            id="human_test",
            name="Test Player",
            role=AgentRole.VETERINARIAN,
            location=location
        )
        
        original_state = SimulationState(
            is_running=False,
            active_events=[event],
            agent_count=15,
            current_metrics=metrics,
            human_player=human_player,
            simulation_id="sim_test"
        )
        
        # Test serialization
        state_dict = original_state.to_dict()
        assert state_dict["is_running"] is False
        assert state_dict["agent_count"] == 15
        assert len(state_dict["active_events"]) == 1
        assert state_dict["human_player"]["name"] == "Test Player"
        
        # Test deserialization
        restored_state = SimulationState.from_dict(state_dict)
        assert restored_state.is_running == original_state.is_running
        assert restored_state.agent_count == original_state.agent_count
        assert len(restored_state.active_events) == len(original_state.active_events)
        assert restored_state.current_metrics.visitor_satisfaction == original_state.current_metrics.visitor_satisfaction
        assert restored_state.human_player.name == original_state.human_player.name
        assert restored_state.human_player.role == original_state.human_player.role
    
    def test_simulation_state_serialization_no_human_player(self):
        """Test simulation state serialization without human player."""
        location = Location(x=5.0, y=15.0, zone="entrance")
        event = Event(id="event_2", type=EventType.VISITOR_COMPLAINT, severity=3, location=location)
        metrics = MetricsSnapshot(visitor_satisfaction=0.9)
        
        original_state = SimulationState(
            is_running=True,
            active_events=[event],
            agent_count=8,
            current_metrics=metrics,
            human_player=None,  # No human player
            simulation_id="sim_no_human"
        )
        
        # Test serialization
        state_dict = original_state.to_dict()
        assert state_dict["human_player"] is None
        
        # Test deserialization
        restored_state = SimulationState.from_dict(state_dict)
        assert restored_state.human_player is None
        assert restored_state.agent_count == original_state.agent_count


class TestChatMessage:
    """Test cases for ChatMessage model."""
    
    def test_chat_message_creation_valid(self):
        """Test creating a valid chat message."""
        message = ChatMessage(
            id="msg_1",
            sender_id="human_1",
            sender_name="Player",
            content="Hello, I need help with the dinosaur situation!",
            message_type=MessageType.HUMAN,
            conversation_id="conv_1"
        )
        
        assert message.id == "msg_1"
        assert message.sender_id == "human_1"
        assert message.sender_name == "Player"
        assert message.content == "Hello, I need help with the dinosaur situation!"
        assert message.message_type == MessageType.HUMAN
        assert message.conversation_id == "conv_1"
        assert isinstance(message.timestamp, datetime)
    
    def test_chat_message_default_values(self):
        """Test chat message with default values."""
        message = ChatMessage(
            id="msg_2",
            sender_id="ai_agent_1",
            sender_name="Ranger Smith",
            content="I'm on my way to help!"
        )
        
        assert message.message_type == MessageType.AI_AGENT  # Default value
        assert message.conversation_id == ""  # Default value
        assert isinstance(message.timestamp, datetime)
    
    def test_chat_message_validation_empty_id(self):
        """Test chat message validation with empty ID."""
        with pytest.raises(ValueError, match="Message ID cannot be empty"):
            ChatMessage(
                id="",
                sender_id="human_1",
                sender_name="Player",
                content="Test message"
            )
    
    def test_chat_message_validation_empty_sender_id(self):
        """Test chat message validation with empty sender ID."""
        with pytest.raises(ValueError, match="Sender ID cannot be empty"):
            ChatMessage(
                id="msg_1",
                sender_id="",
                sender_name="Player",
                content="Test message"
            )
    
    def test_chat_message_validation_empty_sender_name(self):
        """Test chat message validation with empty sender name."""
        with pytest.raises(ValueError, match="Sender name cannot be empty"):
            ChatMessage(
                id="msg_1",
                sender_id="human_1",
                sender_name="",
                content="Test message"
            )
    
    def test_chat_message_validation_empty_content(self):
        """Test chat message validation with empty content."""
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            ChatMessage(
                id="msg_1",
                sender_id="human_1",
                sender_name="Player",
                content=""
            )
        
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            ChatMessage(
                id="msg_1",
                sender_id="human_1",
                sender_name="Player",
                content="   "  # Only whitespace
            )
    
    def test_chat_message_serialization(self):
        """Test chat message to_dict and from_dict methods."""
        original_message = ChatMessage(
            id="msg_3",
            sender_id="system",
            sender_name="System",
            content="Event notification: Dinosaur escape in Paddock A",
            message_type=MessageType.SYSTEM,
            conversation_id="emergency_conv"
        )
        
        # Test serialization
        message_dict = original_message.to_dict()
        assert message_dict["id"] == "msg_3"
        assert message_dict["sender_id"] == "system"
        assert message_dict["message_type"] == "SYSTEM"
        assert message_dict["conversation_id"] == "emergency_conv"
        
        # Test deserialization
        restored_message = ChatMessage.from_dict(message_dict)
        assert restored_message.id == original_message.id
        assert restored_message.sender_id == original_message.sender_id
        assert restored_message.sender_name == original_message.sender_name
        assert restored_message.content == original_message.content
        assert restored_message.message_type == original_message.message_type
        assert restored_message.conversation_id == original_message.conversation_id


class TestHumanAgent:
    """Test cases for HumanAgent model."""
    
    def test_human_agent_creation_valid(self):
        """Test creating a valid human agent."""
        location = Location(x=5.0, y=10.0, zone="visitor_center")
        message = ChatMessage(
            id="msg_1",
            sender_id="human_1",
            sender_name="Player",
            content="I'm ready to help!",
            message_type=MessageType.HUMAN
        )
        
        human_agent = HumanAgent(
            id="human_1",
            name="Player",
            role=AgentRole.PARK_RANGER,
            location=location,
            chat_history=[message],
            conversation_access=["conv_1", "conv_2"]
        )
        
        assert human_agent.id == "human_1"
        assert human_agent.name == "Player"
        assert human_agent.role == AgentRole.PARK_RANGER
        assert human_agent.is_human_controlled is True
        assert len(human_agent.chat_history) == 1
        assert len(human_agent.conversation_access) == 2
        assert "conv_1" in human_agent.conversation_access
    
    def test_human_agent_default_values(self):
        """Test human agent with default values."""
        location = Location(x=0.0, y=0.0, zone="entrance")
        human_agent = HumanAgent(
            id="human_2",
            name="Test Player",
            role=AgentRole.SECURITY,
            location=location
        )
        
        assert human_agent.is_human_controlled is True
        assert len(human_agent.chat_history) == 0
        assert len(human_agent.conversation_access) == 0
    
    def test_human_agent_inherits_agent_validation(self):
        """Test that human agent inherits base agent validation."""
        location = Location(x=0.0, y=0.0, zone="test")
        
        # Test empty ID validation
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            HumanAgent(id="", name="Player", role=AgentRole.TOURIST, location=location)
        
        # Test empty name validation
        with pytest.raises(ValueError, match="Agent name cannot be empty"):
            HumanAgent(id="human_1", name="", role=AgentRole.TOURIST, location=location)
    
    def test_human_agent_validation_invalid_chat_history(self):
        """Test human agent validation with invalid chat history."""
        location = Location(x=0.0, y=0.0, zone="test")
        
        with pytest.raises(ValueError, match="All chat history items must be ChatMessage instances"):
            HumanAgent(
                id="human_1",
                name="Player",
                role=AgentRole.TOURIST,
                location=location,
                chat_history=["not a ChatMessage"]
            )
    
    def test_human_agent_validation_invalid_conversation_access(self):
        """Test human agent validation with invalid conversation access."""
        location = Location(x=0.0, y=0.0, zone="test")
        
        with pytest.raises(ValueError, match="All conversation access IDs must be strings"):
            HumanAgent(
                id="human_1",
                name="Player",
                role=AgentRole.TOURIST,
                location=location,
                conversation_access=[123, "conv_1"]  # 123 is not a string
            )
    
    def test_human_agent_add_chat_message(self):
        """Test adding chat messages to human agent."""
        location = Location(x=0.0, y=0.0, zone="test")
        human_agent = HumanAgent(
            id="human_1",
            name="Player",
            role=AgentRole.TOURIST,
            location=location
        )
        
        message = ChatMessage(
            id="msg_1",
            sender_id="human_1",
            sender_name="Player",
            content="Test message"
        )
        
        initial_activity = human_agent.last_activity
        human_agent.add_chat_message(message)
        
        assert len(human_agent.chat_history) == 1
        assert human_agent.chat_history[0] == message
        assert human_agent.last_activity > initial_activity
    
    def test_human_agent_add_invalid_chat_message(self):
        """Test adding invalid chat message to human agent."""
        location = Location(x=0.0, y=0.0, zone="test")
        human_agent = HumanAgent(
            id="human_1",
            name="Player",
            role=AgentRole.TOURIST,
            location=location
        )
        
        with pytest.raises(ValueError, match="Message must be a ChatMessage instance"):
            human_agent.add_chat_message("not a ChatMessage")
    
    def test_human_agent_get_recent_messages(self):
        """Test getting recent messages from human agent."""
        location = Location(x=0.0, y=0.0, zone="test")
        human_agent = HumanAgent(
            id="human_1",
            name="Player",
            role=AgentRole.TOURIST,
            location=location
        )
        
        # Add multiple messages
        for i in range(5):
            message = ChatMessage(
                id=f"msg_{i}",
                sender_id="human_1",
                sender_name="Player",
                content=f"Message {i}"
            )
            human_agent.add_chat_message(message)
        
        # Test getting recent messages with limit
        recent = human_agent.get_recent_messages(limit=3)
        assert len(recent) == 3
        assert recent[0].content == "Message 2"  # Last 3 messages
        assert recent[2].content == "Message 4"
        
        # Test getting all messages
        all_messages = human_agent.get_recent_messages(limit=10)
        assert len(all_messages) == 5
        
        # Test empty chat history
        empty_agent = HumanAgent(
            id="human_2",
            name="Empty Player",
            role=AgentRole.TOURIST,
            location=location
        )
        empty_recent = empty_agent.get_recent_messages()
        assert len(empty_recent) == 0
    
    def test_human_agent_conversation_access_management(self):
        """Test conversation access management methods."""
        location = Location(x=0.0, y=0.0, zone="test")
        human_agent = HumanAgent(
            id="human_1",
            name="Player",
            role=AgentRole.TOURIST,
            location=location
        )
        
        # Test adding conversation access
        human_agent.add_conversation_access("conv_1")
        assert human_agent.has_conversation_access("conv_1")
        assert len(human_agent.conversation_access) == 1
        
        # Test adding duplicate conversation access
        human_agent.add_conversation_access("conv_1")
        assert len(human_agent.conversation_access) == 1  # Should not duplicate
        
        # Test adding multiple conversations
        human_agent.add_conversation_access("conv_2")
        human_agent.add_conversation_access("conv_3")
        assert len(human_agent.conversation_access) == 3
        
        # Test removing conversation access
        human_agent.remove_conversation_access("conv_2")
        assert not human_agent.has_conversation_access("conv_2")
        assert len(human_agent.conversation_access) == 2
        
        # Test removing non-existent conversation access
        human_agent.remove_conversation_access("conv_nonexistent")
        assert len(human_agent.conversation_access) == 2  # Should remain unchanged
    
    def test_human_agent_conversation_access_validation(self):
        """Test conversation access validation."""
        location = Location(x=0.0, y=0.0, zone="test")
        human_agent = HumanAgent(
            id="human_1",
            name="Player",
            role=AgentRole.TOURIST,
            location=location
        )
        
        with pytest.raises(ValueError, match="Conversation ID must be a string"):
            human_agent.add_conversation_access(123)
    
    def test_human_agent_clear_chat_history(self):
        """Test clearing chat history."""
        location = Location(x=0.0, y=0.0, zone="test")
        human_agent = HumanAgent(
            id="human_1",
            name="Player",
            role=AgentRole.TOURIST,
            location=location
        )
        
        # Add some messages
        for i in range(3):
            message = ChatMessage(
                id=f"msg_{i}",
                sender_id="human_1",
                sender_name="Player",
                content=f"Message {i}"
            )
            human_agent.add_chat_message(message)
        
        assert len(human_agent.chat_history) == 3
        
        # Clear history
        human_agent.clear_chat_history()
        assert len(human_agent.chat_history) == 0
    
    def test_human_agent_serialization(self):
        """Test human agent to_dict and from_dict methods."""
        location = Location(x=15.0, y=25.0, zone="security_office", description="Main security hub")
        message1 = ChatMessage(
            id="msg_1",
            sender_id="human_1",
            sender_name="Security Chief",
            content="Situation under control",
            message_type=MessageType.HUMAN,
            conversation_id="security_conv"
        )
        message2 = ChatMessage(
            id="msg_2",
            sender_id="ai_ranger",
            sender_name="Ranger AI",
            content="Thanks for the update",
            message_type=MessageType.AI_AGENT,
            conversation_id="security_conv"
        )
        
        original_human = HumanAgent(
            id="human_security",
            name="Security Chief",
            role=AgentRole.SECURITY,
            personality_traits={"leadership": 0.9, "brave": 0.8},
            current_state=AgentState.ACTIVE,
            location=location,
            capabilities=["threat_assessment", "emergency_response"],
            is_human_controlled=True,
            chat_history=[message1, message2],
            conversation_access=["security_conv", "emergency_conv"]
        )
        
        # Test serialization
        human_dict = original_human.to_dict()
        assert human_dict["id"] == "human_security"
        assert human_dict["role"] == "SECURITY"
        assert human_dict["is_human_controlled"] is True
        assert len(human_dict["chat_history"]) == 2
        assert len(human_dict["conversation_access"]) == 2
        assert human_dict["chat_history"][0]["content"] == "Situation under control"
        
        # Test deserialization
        restored_human = HumanAgent.from_dict(human_dict)
        assert restored_human.id == original_human.id
        assert restored_human.role == original_human.role
        assert restored_human.is_human_controlled == original_human.is_human_controlled
        assert len(restored_human.chat_history) == len(original_human.chat_history)
        assert len(restored_human.conversation_access) == len(original_human.conversation_access)
        assert restored_human.chat_history[0].content == original_human.chat_history[0].content
        assert restored_human.conversation_access == original_human.conversation_access