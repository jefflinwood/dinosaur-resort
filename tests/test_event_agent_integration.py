"""Integration tests for event system and agent reactions."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from managers.event_manager import EventManager
from managers.agent_manager import AgentManager
from models.core import Event, Agent
from models.enums import EventType, ResolutionStatus, AgentRole, AgentState, DinosaurSpecies
from models.config import Location, OpenAIConfig, AG2Config, AgentConfig


class TestEventAgentIntegration:
    """Test integration between EventManager and AgentManager."""
    
    @pytest.fixture
    def mock_openai_config(self):
        """Create mock OpenAI configuration."""
        return OpenAIConfig(
            api_key="test-key",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150,
            timeout=30
        )
    
    @pytest.fixture
    def mock_ag2_config(self):
        """Create mock ag2 configuration."""
        return AG2Config(
            human_input_mode="NEVER",
            max_round=3,
            code_execution_config=False
        )
    
    @pytest.fixture
    def mock_agent_config(self):
        """Create mock agent configuration."""
        return AgentConfig(
            staff_count={
                AgentRole.PARK_RANGER: 2,
                AgentRole.VETERINARIAN: 1,
                AgentRole.SECURITY: 1
            },
            visitor_count=3,
            dinosaur_config={
                DinosaurSpecies.TYRANNOSAURUS_REX: 1,
                DinosaurSpecies.TRICERATOPS: 2
            }
        )
    
    @pytest.fixture
    def sample_location(self):
        """Create sample location."""
        return Location(10.0, 20.0, "main_area", "Main visitor area")
    
    @pytest.fixture
    def sample_agents(self, sample_location):
        """Create sample agents for testing."""
        return [
            Agent(
                id="ranger_1",
                name="Ranger Smith",
                role=AgentRole.PARK_RANGER,
                location=sample_location,
                capabilities=["wildlife_management", "visitor_safety"]
            ),
            Agent(
                id="vet_1",
                name="Dr. Johnson",
                role=AgentRole.VETERINARIAN,
                location=sample_location,
                capabilities=["medical_treatment", "health_assessment"]
            ),
            Agent(
                id="security_1",
                name="Officer Brown",
                role=AgentRole.SECURITY,
                location=sample_location,
                capabilities=["threat_assessment", "crowd_control"]
            )
        ]
    
    @pytest.fixture
    def mock_agent_manager(self, mock_openai_config, mock_ag2_config, mock_agent_config, sample_agents):
        """Create mock AgentManager with sample agents."""
        with patch('managers.agent_manager.AG2Integration'), \
             patch('managers.agent_manager.StaffAgentFactory'), \
             patch('managers.agent_manager.VisitorAgentFactory'), \
             patch('managers.agent_manager.DinosaurAgentFactory'):
            
            agent_manager = AgentManager(mock_openai_config, mock_ag2_config, mock_agent_config)
            
            # Mock the agents
            agent_manager.agents = {agent.id: agent for agent in sample_agents}
            agent_manager.agent_instances = {}
            agent_manager.agent_health = {}
            
            # Mock agent instances
            for agent in sample_agents:
                mock_instance = Mock()
                mock_instance.handle_event_notification.return_value = f"Agent {agent.name} responding to event"
                agent_manager.agent_instances[agent.id] = mock_instance
                agent_manager.agent_health[agent.id] = {
                    "status": "healthy",
                    "last_response_time": datetime.now(),
                    "response_count": 0,
                    "error_count": 0,
                    "communication_failures": 0
                }
            
            # Mock broadcast_event method
            def mock_broadcast_event(event):
                affected_agents = [agent.id for agent in sample_agents]
                individual_responses = {}
                
                for agent_id in affected_agents:
                    if event.severity >= 7:
                        individual_responses[agent_id] = f"Agent {agent_id} handling critical event - situation under control"
                    elif event.severity >= 5:
                        individual_responses[agent_id] = f"Agent {agent_id} responding to event - will monitor"
                    else:
                        individual_responses[agent_id] = f"Agent {agent_id} acknowledged event"
                
                return {
                    "event_id": event.id,
                    "affected_agents": affected_agents,
                    "individual_responses": individual_responses,
                    "group_response": None,
                    "broadcast_time": datetime.now().isoformat()
                }
            
            agent_manager.broadcast_event = Mock(side_effect=mock_broadcast_event)
            
            return agent_manager
    
    @pytest.fixture
    def event_manager_with_agents(self, mock_agent_manager):
        """Create EventManager with AgentManager integration."""
        event_manager = EventManager(mock_agent_manager)
        return event_manager
    
    def test_event_manager_initialization_with_agent_manager(self, mock_agent_manager):
        """Test EventManager initialization with AgentManager."""
        event_manager = EventManager(mock_agent_manager)
        
        assert event_manager._agent_manager is mock_agent_manager
        assert event_manager._event_responses == {}
        assert event_manager._resolution_callbacks == []
    
    def test_set_agent_manager(self, mock_agent_manager):
        """Test setting agent manager after initialization."""
        event_manager = EventManager()
        assert event_manager._agent_manager is None
        
        event_manager.set_agent_manager(mock_agent_manager)
        assert event_manager._agent_manager is mock_agent_manager
    
    def test_event_distribution_to_agents(self, event_manager_with_agents, sample_location):
        """Test event distribution to agents."""
        # Create test event with lower severity to avoid escalation
        event = event_manager_with_agents.create_event(
            event_type=EventType.DINOSAUR_ESCAPE,
            location=sample_location,
            parameters={"dinosaur_id": "trex_1", "enclosure_id": "enclosure_a"},
            severity=6  # Lower severity to avoid automatic escalation
        )
        
        # Distribute event
        result = event_manager_with_agents.distribute_event(event)
        
        # Verify distribution results
        assert result["event_id"] == event.id
        assert "affected_agents" in result
        assert "individual_responses" in result
        assert len(result["affected_agents"]) == 3  # All sample agents
        
        # Verify event is tracked
        assert event.id in event_manager_with_agents._active_events
        assert event.id in event_manager_with_agents._event_responses
        
        # Verify agent manager was called (may be called multiple times due to escalation logic)
        assert event_manager_with_agents._agent_manager.broadcast_event.called
    
    def test_event_resolution_detection(self, event_manager_with_agents, sample_location):
        """Test automatic event resolution detection."""
        # Create test event
        event = event_manager_with_agents.create_event(
            event_type=EventType.VISITOR_INJURY,
            location=sample_location,
            parameters={"visitor_id": "visitor_1", "injury_type": "minor_cut"},
            severity=5
        )
        
        # Mock agent responses that indicate resolution
        def mock_broadcast_with_resolution(event):
            return {
                "event_id": event.id,
                "affected_agents": ["ranger_1", "vet_1"],
                "individual_responses": {
                    "ranger_1": "Visitor injury treated and resolved - situation is safe",
                    "vet_1": "Medical treatment completed - visitor is stable and secure"
                },
                "group_response": None,
                "broadcast_time": datetime.now().isoformat()
            }
        
        event_manager_with_agents._agent_manager.broadcast_event.side_effect = mock_broadcast_with_resolution
        
        # Distribute event
        result = event_manager_with_agents.distribute_event(event)
        
        # The resolution detection should have resolved the event automatically
        # Check if event was resolved (no longer in active events)
        assert event.resolution_status == ResolutionStatus.RESOLVED
        assert event.id not in event_manager_with_agents._active_events
    
    def test_event_escalation_detection(self, event_manager_with_agents, sample_location):
        """Test automatic event escalation detection."""
        # Create test event
        event = event_manager_with_agents.create_event(
            event_type=EventType.DINOSAUR_AGGRESSIVE,
            location=sample_location,
            parameters={"dinosaur_id": "trex_1", "behavior_type": "territorial"},
            severity=6
        )
        
        # Mock agent responses that indicate need for escalation
        def mock_broadcast_with_escalation(event):
            return {
                "event_id": event.id,
                "affected_agents": ["ranger_1", "security_1"],
                "individual_responses": {
                    "ranger_1": "Dinosaur behavior is escalating - need backup immediately",
                    "security_1": "Cannot handle this situation alone - require emergency assistance"
                },
                "group_response": None,
                "broadcast_time": datetime.now().isoformat()
            }
        
        event_manager_with_agents._agent_manager.broadcast_event.side_effect = mock_broadcast_with_escalation
        
        # Distribute event
        result = event_manager_with_agents.distribute_event(event)
        
        # Verify event is still active but escalated
        assert event.id in event_manager_with_agents._active_events
        
        # Check if escalation was detected
        event_response_data = event_manager_with_agents.get_event_response_data(event.id)
        assert event_response_data is not None
        
        # Event should be escalated or have escalation indicators
        assert event.resolution_status == ResolutionStatus.ESCALATED or event_response_data.get("escalated", False)
    
    def test_resolution_callbacks(self, event_manager_with_agents, sample_location):
        """Test resolution callback functionality."""
        callback_called = False
        resolved_event = None
        
        def resolution_callback(event):
            nonlocal callback_called, resolved_event
            callback_called = True
            resolved_event = event
        
        # Add resolution callback
        event_manager_with_agents.add_resolution_callback(resolution_callback)
        
        # Create and distribute event
        event = event_manager_with_agents.create_event(
            event_type=EventType.FACILITY_EQUIPMENT_FAILURE,
            location=sample_location,
            parameters={"facility_id": "power_station", "equipment_type": "generator"},
            severity=4
        )
        
        # Distribute the event first
        event_manager_with_agents.distribute_event(event)
        
        # Force resolve the event
        event_manager_with_agents.force_resolve_event(event.id, ResolutionStatus.RESOLVED)
        
        # Verify callback was called
        assert callback_called
        assert resolved_event is not None
        assert resolved_event.id == event.id
        assert resolved_event.resolution_status == ResolutionStatus.RESOLVED
    
    def test_event_response_tracking(self, event_manager_with_agents, sample_location):
        """Test event response data tracking."""
        # Create test event
        event = event_manager_with_agents.create_event(
            event_type=EventType.WEATHER_STORM,
            location=sample_location,
            parameters={"storm_type": "thunderstorm", "intensity": "moderate"},
            severity=6
        )
        
        # Distribute event
        result = event_manager_with_agents.distribute_event(event)
        
        # Get response data
        response_data = event_manager_with_agents.get_event_response_data(event.id)
        
        assert response_data is not None
        assert response_data["event"] == event
        assert "agent_responses" in response_data
        assert "start_time" in response_data
        assert "last_update" in response_data
        assert response_data["resolution_attempts"] == 0
    
    def test_event_resolution_progress_tracking(self, event_manager_with_agents, sample_location):
        """Test event resolution progress tracking."""
        # Create test event
        event = event_manager_with_agents.create_event(
            event_type=EventType.VISITOR_COMPLAINT,
            location=sample_location,
            parameters={"visitor_id": "visitor_2", "complaint_type": "service_quality"},
            severity=3
        )
        
        # Distribute event
        event_manager_with_agents.distribute_event(event)
        
        # Check resolution progress
        progress = event_manager_with_agents.check_event_resolution_progress(event.id)
        
        assert progress["event_id"] == event.id
        assert progress["status"] == ResolutionStatus.PENDING.name
        assert progress["severity"] == 3
        assert "time_elapsed" in progress
        assert "affected_agents" in progress
        assert "responses_received" in progress
    
    def test_multiple_events_handling(self, event_manager_with_agents, sample_location):
        """Test handling multiple simultaneous events."""
        # Create multiple events
        events = []
        for i in range(3):
            event = event_manager_with_agents.create_event(
                event_type=EventType.VISITOR_EMERGENCY,
                location=sample_location,
                parameters={"visitor_id": f"visitor_{i}", "emergency_type": "medical"},
                severity=7 + i
            )
            events.append(event)
        
        # Distribute all events
        results = []
        for event in events:
            result = event_manager_with_agents.distribute_event(event)
            results.append(result)
        
        # Verify all events are tracked
        assert len(event_manager_with_agents._active_events) == 3
        assert len(event_manager_with_agents._event_responses) == 3
        
        # Verify each event has response data
        for event in events:
            response_data = event_manager_with_agents.get_event_response_data(event.id)
            assert response_data is not None
            assert response_data["event"] == event
    
    def test_agent_manager_event_resolution_check(self, mock_agent_manager, sample_location):
        """Test AgentManager's event resolution checking functionality."""
        # Create test event
        event = Event(
            id="test_event_1",
            type=EventType.DINOSAUR_ILLNESS,
            severity=5,
            location=sample_location,
            parameters={"dinosaur_id": "trex_1", "symptoms": "lethargy"}
        )
        
        # Mock conversation history with resolution indicators
        mock_agent_manager.conversation_history = [
            {
                "event_id": "test_event_1",
                "individual_responses": {
                    "vet_1": "Dinosaur illness treated successfully - patient is now healthy and stable",
                    "ranger_1": "Situation is resolved - dinosaur showing normal behavior"
                }
            }
        ]
        
        # Check resolution
        resolution_info = mock_agent_manager.check_event_resolution_by_agents(event)
        
        assert resolution_info["event_id"] == event.id
        assert resolution_info["overall_status"] == "resolved"
        assert len(resolution_info["resolution_indicators"]) > 0
        assert "treated" in resolution_info["resolution_indicators"]
        assert "resolved" in resolution_info["resolution_indicators"]
    
    def test_agent_manager_escalation_detection(self, mock_agent_manager, sample_location):
        """Test AgentManager's escalation detection functionality."""
        # Create test event
        event = Event(
            id="test_event_2",
            type=EventType.DINOSAUR_ESCAPE,
            severity=9,
            location=sample_location,
            parameters={"dinosaur_id": "trex_1", "enclosure_id": "enclosure_a"}
        )
        
        # Mock conversation history with escalation indicators
        mock_agent_manager.conversation_history = [
            {
                "event_id": "test_event_2",
                "individual_responses": {
                    "ranger_1": "Dinosaur escape situation is critical - need immediate backup",
                    "security_1": "Cannot handle this alone - require emergency assistance"
                }
            }
        ]
        
        # Check resolution
        resolution_info = mock_agent_manager.check_event_resolution_by_agents(event)
        
        assert resolution_info["event_id"] == event.id
        assert resolution_info["overall_status"] == "needs_escalation"
        assert len(resolution_info["escalation_indicators"]) > 0
        assert "need immediate" in resolution_info["escalation_indicators"]
        assert "backup" in resolution_info["escalation_indicators"]
    
    def test_agent_notification_of_resolution(self, mock_agent_manager, sample_location):
        """Test agent notification when event is resolved."""
        # Create resolved event
        event = Event(
            id="resolved_event",
            type=EventType.FACILITY_POWER_OUTAGE,
            severity=6,
            location=sample_location,
            parameters={"facility_id": "visitor_center", "affected_systems": ["lighting", "hvac"]},
            affected_agents=["maintenance_1", "security_1"],
            resolution_status=ResolutionStatus.RESOLVED,
            resolution_time=datetime.now()
        )
        
        # Mock send_message_to_agent
        mock_agent_manager.send_message_to_agent = Mock(return_value="Message received")
        
        # Notify agents of resolution
        mock_agent_manager.notify_event_resolution(event)
        
        # Verify all affected agents were notified
        assert mock_agent_manager.send_message_to_agent.call_count == len(event.affected_agents)
        
        # Verify message content
        calls = mock_agent_manager.send_message_to_agent.call_args_list
        for call in calls:
            args, kwargs = call
            agent_id, message, sender = args
            assert agent_id in event.affected_agents
            assert "EVENT RESOLVED" in message
            assert event.type.name.replace('_', ' ').title() in message
            assert sender == "EventSystem"
    
    def test_integration_error_handling(self, event_manager_with_agents, sample_location):
        """Test error handling in event-agent integration."""
        # Mock agent manager to raise exception
        event_manager_with_agents._agent_manager.broadcast_event.side_effect = Exception("Agent communication error")
        
        # Create test event
        event = event_manager_with_agents.create_event(
            event_type=EventType.CUSTOM,
            location=sample_location,
            parameters={"event_description": "Test custom event"},
            severity=5
        )
        
        # Distribute event (should handle error gracefully)
        result = event_manager_with_agents.distribute_event(event)
        
        # Verify error was handled
        assert "agent_error" in result
        assert result["agent_error"] == "Agent communication error"
        
        # Event should still be tracked
        assert event.id in event_manager_with_agents._active_events
        assert event.id in event_manager_with_agents._event_responses
    
    def test_event_manager_reset_with_integration(self, event_manager_with_agents, sample_location):
        """Test EventManager reset functionality with agent integration."""
        # Create and distribute event
        event = event_manager_with_agents.create_event(
            event_type=EventType.WEATHER_EXTREME_TEMPERATURE,
            location=sample_location,
            parameters={"temperature": "45", "temperature_type": "heat"},
            severity=5
        )
        
        event_manager_with_agents.distribute_event(event)
        
        # Verify event is tracked
        assert len(event_manager_with_agents._active_events) > 0
        assert len(event_manager_with_agents._event_responses) > 0
        
        # Reset event manager
        event_manager_with_agents.reset()
        
        # Verify all data is cleared
        assert len(event_manager_with_agents._active_events) == 0
        assert len(event_manager_with_agents._event_responses) == 0
        assert len(event_manager_with_agents._resolution_callbacks) == 0
        assert len(event_manager_with_agents._event_listeners) == 0


if __name__ == "__main__":
    pytest.main([__file__])