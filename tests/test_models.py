"""Unit tests for core data models."""

import pytest
from datetime import datetime
from models.core import Agent, Event, MetricsSnapshot, SimulationState
from models.enums import AgentRole, AgentState, EventType, ResolutionStatus, DinosaurSpecies
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
        
        state = SimulationState(
            is_running=True,
            active_events=[event],
            agent_count=10,
            current_metrics=metrics,
            simulation_id="sim_1"
        )
        
        assert state.is_running is True
        assert len(state.active_events) == 1
        assert state.agent_count == 10
        assert state.current_metrics.visitor_satisfaction == 0.8
    
    def test_simulation_state_validation_negative_agent_count(self):
        """Test simulation state validation with negative agent count."""
        with pytest.raises(ValueError, match="Agent count cannot be negative"):
            SimulationState(agent_count=-1)
    
    def test_simulation_state_serialization(self):
        """Test simulation state to_dict and from_dict methods."""
        location = Location(x=10.0, y=20.0, zone="paddock")
        event = Event(id="event_1", type=EventType.FACILITY_POWER_OUTAGE, severity=7, location=location)
        metrics = MetricsSnapshot(visitor_satisfaction=0.6, facility_efficiency=0.3)
        
        original_state = SimulationState(
            is_running=False,
            active_events=[event],
            agent_count=15,
            current_metrics=metrics,
            simulation_id="sim_test"
        )
        
        # Test serialization
        state_dict = original_state.to_dict()
        assert state_dict["is_running"] is False
        assert state_dict["agent_count"] == 15
        assert len(state_dict["active_events"]) == 1
        
        # Test deserialization
        restored_state = SimulationState.from_dict(state_dict)
        assert restored_state.is_running == original_state.is_running
        assert restored_state.agent_count == original_state.agent_count
        assert len(restored_state.active_events) == len(original_state.active_events)
        assert restored_state.current_metrics.visitor_satisfaction == original_state.current_metrics.visitor_satisfaction