"""Integration tests for multi-agent communication scenarios using AgentManager."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

from managers.agent_manager import AgentManager
from models.core import Agent, Event, MetricsSnapshot
from models.config import OpenAIConfig, AG2Config, AgentConfig, Location
from models.enums import AgentRole, AgentState, EventType, ResolutionStatus, DinosaurSpecies, PersonalityTrait


class TestAgentManager:
    """Test suite for AgentManager multi-agent communication and orchestration."""
    
    @pytest.fixture
    def mock_agent_manager(self, mock_openai_config, mock_ag2_config, test_agent_config):
        """Create a mock agent manager for testing."""
        with patch('managers.agent_manager.AG2Integration') as mock_ag2_integration:
            # Mock the AG2Integration
            mock_integration = Mock()
            mock_integration.create_ag2_agent.return_value = Mock()
            mock_integration.create_group_chat.return_value = Mock()
            mock_integration.group_chat_manager = Mock()
            mock_integration.send_message_to_agent.return_value = "Test response"
            mock_integration.initiate_group_conversation.return_value = [
                {"role": "user", "content": "Test message", "name": "TestAgent"}
            ]
            mock_integration.get_agent_conversation_history.return_value = []
            mock_ag2_integration.return_value = mock_integration
            
            # Mock agent factories
            with patch('managers.agent_manager.BaseAgentConfig'), \
                 patch('managers.agent_manager.AgentFactory'), \
                 patch('managers.agent_manager.StaffAgentFactory'), \
                 patch('managers.agent_manager.VisitorAgentFactory'), \
                 patch('managers.agent_manager.DinosaurAgentFactory'):
                
                manager = AgentManager(mock_openai_config, mock_ag2_config, test_agent_config)
                
                # Mock agent creation methods
                manager._create_staff_agents = Mock(return_value=self._create_mock_staff_agents())
                manager._create_visitor_agents = Mock(return_value=self._create_mock_visitor_agents())
                manager._create_dinosaur_agents = Mock(return_value=self._create_mock_dinosaur_agents())
                manager._create_agent_instance = Mock(side_effect=self._create_mock_agent_instance)
                
                return manager
    
    def _create_mock_staff_agents(self) -> List[Agent]:
        """Create mock staff agents for testing."""
        return [
            Agent(
                id="ranger_1",
                name="Ranger_1",
                role=AgentRole.PARK_RANGER,
                personality_traits={PersonalityTrait.CAUTIOUS.value: 0.8},
                location=Location(0.0, 0.0, "entrance", "Main entrance")
            ),
            Agent(
                id="vet_1",
                name="Dr.Vet_1",
                role=AgentRole.VETERINARIAN,
                personality_traits={PersonalityTrait.ANALYTICAL.value: 0.9},
                location=Location(10.0, 10.0, "medical_center", "Veterinary clinic")
            ),
            Agent(
                id="security_1",
                name="Security_1",
                role=AgentRole.SECURITY,
                personality_traits={PersonalityTrait.BRAVE.value: 0.8},
                location=Location(5.0, 5.0, "security_station", "Security headquarters")
            )
        ]
    
    def _create_mock_visitor_agents(self) -> List[Agent]:
        """Create mock visitor agents for testing."""
        return [
            Agent(
                id="visitor_1",
                name="Visitor_1",
                role=AgentRole.TOURIST,
                personality_traits={PersonalityTrait.FRIENDLY.value: 0.7},
                location=Location(15.0, 15.0, "main_plaza", "Central plaza")
            ),
            Agent(
                id="visitor_2",
                name="Visitor_2",
                role=AgentRole.TOURIST,
                personality_traits={PersonalityTrait.CAUTIOUS.value: 0.6},
                location=Location(20.0, 20.0, "gift_shop", "Souvenir shop")
            )
        ]
    
    def _create_mock_dinosaur_agents(self) -> List[Agent]:
        """Create mock dinosaur agents for testing."""
        return [
            Agent(
                id="trex_1",
                name="Tyrannosaurus_Rex_1",
                role=AgentRole.DINOSAUR,
                species=DinosaurSpecies.TYRANNOSAURUS_REX,
                personality_traits={PersonalityTrait.BRAVE.value: 0.9},
                location=Location(100.0, 100.0, "carnivore_habitat", "T-Rex enclosure")
            ),
            Agent(
                id="trike_1",
                name="Triceratops_1",
                role=AgentRole.DINOSAUR,
                species=DinosaurSpecies.TRICERATOPS,
                personality_traits={PersonalityTrait.CAUTIOUS.value: 0.7},
                location=Location(200.0, 200.0, "herbivore_habitat", "Triceratops area")
            )
        ]
    
    def _create_mock_agent_instance(self, agent: Agent):
        """Create mock agent instance."""
        mock_instance = Mock()
        mock_instance.agent_model = agent
        mock_instance.handle_event_notification.return_value = f"Mock response from {agent.name}"
        mock_instance.update_state = Mock()
        mock_instance.update_location = Mock()
        
        # Add agent-specific attributes
        if agent.role == AgentRole.TOURIST:
            mock_instance.satisfaction_level = 0.7
        elif agent.role == AgentRole.DINOSAUR:
            mock_instance.happiness_level = 0.8
            mock_instance.stress_level = 0.2
        
        return mock_instance
    
    def test_agent_manager_initialization(self, mock_agent_manager):
        """Test AgentManager initialization."""
        assert mock_agent_manager is not None
        assert hasattr(mock_agent_manager, 'agents')
        assert hasattr(mock_agent_manager, 'agent_instances')
        assert hasattr(mock_agent_manager, 'group_chats')
        assert hasattr(mock_agent_manager, 'agent_health')
    
    def test_initialize_agents(self, mock_agent_manager):
        """Test agent initialization with different roles."""
        agents = mock_agent_manager.initialize_agents()
        
        # Verify agents were created
        assert len(agents) > 0
        assert len(mock_agent_manager.agents) > 0
        assert len(mock_agent_manager.agent_instances) > 0
        
        # Verify different agent types are present
        roles = [agent.role for agent in agents]
        assert AgentRole.PARK_RANGER in roles
        assert AgentRole.VETERINARIAN in roles
        assert AgentRole.SECURITY in roles
        assert AgentRole.TOURIST in roles
        assert AgentRole.DINOSAUR in roles
    
    def test_get_agent_states(self, mock_agent_manager):
        """Test getting agent states."""
        mock_agent_manager.initialize_agents()
        
        agent_states = mock_agent_manager.get_agent_states()
        
        assert isinstance(agent_states, dict)
        assert len(agent_states) > 0
        
        # Check state structure
        for agent_id, state in agent_states.items():
            assert 'id' in state
            assert 'name' in state
            assert 'role' in state
            assert 'current_state' in state
            assert 'location' in state
            assert 'health' in state
            assert 'is_responsive' in state
    
    def test_broadcast_event_to_agents(self, mock_agent_manager):
        """Test broadcasting events to relevant agents."""
        mock_agent_manager.initialize_agents()
        
        # Create test event
        event = Event(
            id="test_event_1",
            type=EventType.DINOSAUR_ESCAPE,
            severity=8,
            location=Location(100.0, 100.0, "carnivore_habitat", "T-Rex enclosure"),
            description="T-Rex has escaped from enclosure"
        )
        
        # Broadcast event
        result = mock_agent_manager.broadcast_event(event)
        
        assert 'event_id' in result
        assert 'affected_agents' in result
        assert 'individual_responses' in result
        assert result['event_id'] == event.id
        assert len(result['affected_agents']) > 0
        assert len(result['individual_responses']) > 0
    
    def test_determine_affected_agents_dinosaur_event(self, mock_agent_manager):
        """Test determining affected agents for dinosaur events."""
        mock_agent_manager.initialize_agents()
        
        event = Event(
            id="dino_event",
            type=EventType.DINOSAUR_ESCAPE,
            severity=7,
            location=Location(100.0, 100.0, "carnivore_habitat", "T-Rex area")
        )
        
        affected_agents = mock_agent_manager._determine_affected_agents(event)
        
        # Should include staff agents
        affected_agent_roles = [mock_agent_manager.agents[agent_id].role for agent_id in affected_agents]
        assert AgentRole.PARK_RANGER in affected_agent_roles
        assert AgentRole.VETERINARIAN in affected_agent_roles
        assert AgentRole.SECURITY in affected_agent_roles
    
    def test_determine_affected_agents_visitor_event(self, mock_agent_manager):
        """Test determining affected agents for visitor events."""
        mock_agent_manager.initialize_agents()
        
        event = Event(
            id="visitor_event",
            type=EventType.VISITOR_INJURY,
            severity=5,
            location=Location(15.0, 15.0, "main_plaza", "Central plaza")
        )
        
        affected_agents = mock_agent_manager._determine_affected_agents(event)
        
        # Should include relevant staff
        affected_agent_roles = [mock_agent_manager.agents[agent_id].role for agent_id in affected_agents]
        assert AgentRole.PARK_RANGER in affected_agent_roles
        assert AgentRole.SECURITY in affected_agent_roles
    
    def test_send_message_to_agent(self, mock_agent_manager):
        """Test sending direct messages to agents."""
        mock_agent_manager.initialize_agents()
        
        # Get first agent
        agent_id = list(mock_agent_manager.agents.keys())[0]
        
        response = mock_agent_manager.send_message_to_agent(
            agent_id, 
            "Test message", 
            "TestSender"
        )
        
        assert response is not None
        assert isinstance(response, str)
    
    def test_initiate_agent_conversation(self, mock_agent_manager):
        """Test initiating conversations between specific agents."""
        mock_agent_manager.initialize_agents()
        
        # Get two agents
        agent_ids = list(mock_agent_manager.agents.keys())[:2]
        
        messages = mock_agent_manager.initiate_agent_conversation(
            agent_ids,
            "Let's discuss the situation",
            "Coordinator"
        )
        
        assert isinstance(messages, list)
    
    def test_group_event_response_high_severity(self, mock_agent_manager):
        """Test group conversation initiation for high-severity events."""
        mock_agent_manager.initialize_agents()
        
        # Create high-severity event
        event = Event(
            id="critical_event",
            type=EventType.DINOSAUR_ESCAPE,
            severity=9,
            location=Location(100.0, 100.0, "carnivore_habitat", "T-Rex area"),
            description="Multiple dinosaurs have escaped"
        )
        
        # Get staff agents for group response
        staff_agents = [
            agent_id for agent_id, agent in mock_agent_manager.agents.items()
            if agent.role in [AgentRole.PARK_RANGER, AgentRole.VETERINARIAN, AgentRole.SECURITY]
        ]
        
        if staff_agents:
            group_response = mock_agent_manager._initiate_group_event_response(event, staff_agents)
            assert group_response is not None or len(staff_agents) == 0  # Allow for empty staff list in mock
    
    def test_agent_health_monitoring(self, mock_agent_manager):
        """Test agent health monitoring functionality."""
        mock_agent_manager.initialize_agents()
        
        # Check initial health
        health_report = mock_agent_manager.check_agent_health()
        
        assert isinstance(health_report, dict)
        assert len(health_report) > 0
        
        # Check health report structure
        for agent_id, health_info in health_report.items():
            assert 'agent_name' in health_info
            assert 'status' in health_info
            assert 'response_count' in health_info
            assert 'error_count' in health_info
            assert 'last_response_time' in health_info
    
    def test_update_agent_health_success(self, mock_agent_manager):
        """Test updating agent health on successful operations."""
        mock_agent_manager.initialize_agents()
        
        agent_id = list(mock_agent_manager.agents.keys())[0]
        
        # Update health with success
        mock_agent_manager._update_agent_health(agent_id, success=True)
        
        health_info = mock_agent_manager.agent_health[agent_id]
        assert health_info['status'] == 'healthy'
        assert health_info['response_count'] > 0
        assert health_info['communication_failures'] == 0
    
    def test_update_agent_health_failure(self, mock_agent_manager):
        """Test updating agent health on failed operations."""
        mock_agent_manager.initialize_agents()
        
        agent_id = list(mock_agent_manager.agents.keys())[0]
        
        # Update health with failure
        mock_agent_manager._update_agent_health(agent_id, success=False, error="Test error")
        
        health_info = mock_agent_manager.agent_health[agent_id]
        assert health_info['error_count'] > 0
        assert health_info['communication_failures'] > 0
        assert health_info['last_error'] == "Test error"
    
    def test_get_agents_by_role(self, mock_agent_manager):
        """Test filtering agents by role."""
        mock_agent_manager.initialize_agents()
        
        rangers = mock_agent_manager.get_agents_by_role(AgentRole.PARK_RANGER)
        tourists = mock_agent_manager.get_agents_by_role(AgentRole.TOURIST)
        dinosaurs = mock_agent_manager.get_agents_by_role(AgentRole.DINOSAUR)
        
        assert all(agent.role == AgentRole.PARK_RANGER for agent in rangers)
        assert all(agent.role == AgentRole.TOURIST for agent in tourists)
        assert all(agent.role == AgentRole.DINOSAUR for agent in dinosaurs)
    
    def test_get_agents_by_location(self, mock_agent_manager):
        """Test filtering agents by location."""
        mock_agent_manager.initialize_agents()
        
        # Test with a known location
        entrance_agents = mock_agent_manager.get_agents_by_location("entrance")
        
        assert all(agent.location.zone == "entrance" for agent in entrance_agents)
    
    def test_update_agent_location(self, mock_agent_manager):
        """Test updating agent locations."""
        mock_agent_manager.initialize_agents()
        
        agent_id = list(mock_agent_manager.agents.keys())[0]
        new_location = Location(50.0, 50.0, "new_zone", "New test location")
        
        success = mock_agent_manager.update_agent_location(agent_id, new_location)
        
        assert success is True
        assert mock_agent_manager.agents[agent_id].location.zone == "new_zone"
    
    def test_update_agent_state(self, mock_agent_manager):
        """Test updating agent states."""
        mock_agent_manager.initialize_agents()
        
        agent_id = list(mock_agent_manager.agents.keys())[0]
        new_state = AgentState.ACTIVE
        
        success = mock_agent_manager.update_agent_state(agent_id, new_state)
        
        assert success is True
        assert mock_agent_manager.agents[agent_id].current_state == new_state
    
    def test_conversation_history_recording(self, mock_agent_manager):
        """Test conversation history recording."""
        mock_agent_manager.initialize_agents()
        
        # Create test event and responses
        event = Event(
            id="history_test",
            type=EventType.VISITOR_COMPLAINT,
            severity=3,
            location=Location(0.0, 0.0, "entrance", "Main entrance")
        )
        
        individual_responses = {"agent_1": "Test response"}
        group_response = [{"role": "user", "content": "Group message"}]
        
        # Record communication
        mock_agent_manager._record_event_communication(event, individual_responses, group_response)
        
        # Check history
        history = mock_agent_manager.get_agent_conversations()
        assert len(history) > 0
        
        latest_record = history[-1]
        assert latest_record['event_id'] == event.id
        assert latest_record['individual_responses'] == individual_responses
        assert latest_record['group_response'] == group_response
    
    def test_reset_agent_conversations(self, mock_agent_manager):
        """Test resetting agent conversations."""
        mock_agent_manager.initialize_agents()
        
        # Add some conversation history
        mock_agent_manager.conversation_history.append({"test": "data"})
        
        # Reset conversations
        mock_agent_manager.reset_agent_conversations()
        
        # Verify reset
        assert len(mock_agent_manager.conversation_history) == 0
    
    def test_get_system_status(self, mock_agent_manager):
        """Test getting overall system status."""
        mock_agent_manager.initialize_agents()
        
        status = mock_agent_manager.get_system_status()
        
        assert 'total_agents' in status
        assert 'healthy_agents' in status
        assert 'unhealthy_agents' in status
        assert 'group_chats' in status
        assert 'conversation_records' in status
        assert 'ag2_integration_status' in status
        
        assert status['total_agents'] > 0
        assert status['ag2_integration_status'] == 'active'
    
    def test_get_agents_near_location(self, mock_agent_manager):
        """Test finding agents near a specific location."""
        mock_agent_manager.initialize_agents()
        
        # Test location
        test_location = Location(10.0, 10.0, "test_zone", "Test area")
        
        nearby_agents = mock_agent_manager._get_agents_near_location(test_location, radius=50.0)
        
        assert isinstance(nearby_agents, list)
        # Verify distance calculation (agents within 50 units)
        for agent_id in nearby_agents:
            agent = mock_agent_manager.agents[agent_id]
            distance = ((agent.location.x - test_location.x) ** 2 + (agent.location.y - test_location.y) ** 2) ** 0.5
            assert distance <= 50.0
    
    def test_create_event_message(self, mock_agent_manager):
        """Test event message creation."""
        event = Event(
            id="msg_test",
            type=EventType.FACILITY_POWER_OUTAGE,
            severity=6,
            location=Location(0.0, 0.0, "power_station", "Main power facility"),
            description="Power grid failure",
            parameters={"affected_systems": "lighting", "estimated_duration": "2 hours"}
        )
        
        message = mock_agent_manager._create_event_message(event)
        
        assert "MODERATE" in message  # Severity 6 should be "moderate" (4-6 range)
        assert "Facility Power Outage" in message  # Title case format
        assert "power_station" in message
        assert "Power grid failure" in message
        assert "affected_systems: lighting" in message


@pytest.fixture
def mock_openai_config():
    """Mock OpenAI configuration for testing."""
    return OpenAIConfig(
        api_key="test_key",
        model="gpt-4",
        temperature=0.7,
        max_tokens=500,
        timeout=30,
        max_retries=3
    )

@pytest.fixture
def mock_ag2_config():
    """Mock ag2 configuration for testing."""
    return AG2Config(
        max_round=5,
        human_input_mode="NEVER",
        code_execution_config=False
    )

@pytest.fixture
def test_agent_config():
    """Test agent configuration."""
    return AgentConfig(
        staff_count={
            AgentRole.PARK_RANGER: 1,
            AgentRole.VETERINARIAN: 1,
            AgentRole.SECURITY: 1,
            AgentRole.MAINTENANCE: 1
        },
        visitor_count=2,
        dinosaur_config={
            DinosaurSpecies.TYRANNOSAURUS_REX: 1,
            DinosaurSpecies.TRICERATOPS: 1
        }
    )


class TestAgentManagerIntegration:
    """Integration tests for complex multi-agent scenarios."""
    
    @pytest.fixture
    def integration_manager(self, mock_openai_config, mock_ag2_config, test_agent_config):
        """Create agent manager for integration testing."""
        with patch('managers.agent_manager.AG2Integration') as mock_ag2_integration:
            mock_integration = Mock()
            mock_integration.create_ag2_agent.return_value = Mock()
            mock_integration.create_group_chat.return_value = Mock()
            mock_integration.group_chat_manager = Mock()
            mock_integration.send_message_to_agent.return_value = "Integration response"
            mock_integration.initiate_group_conversation.return_value = [
                {"role": "assistant", "content": "Emergency response initiated", "name": "Ranger_1"},
                {"role": "assistant", "content": "Medical team standing by", "name": "Dr.Vet_1"},
                {"role": "assistant", "content": "Security perimeter established", "name": "Security_1"}
            ]
            mock_ag2_integration.return_value = mock_integration
            
            with patch('managers.agent_manager.BaseAgentConfig'), \
                 patch('managers.agent_manager.AgentFactory'), \
                 patch('managers.agent_manager.StaffAgentFactory'), \
                 patch('managers.agent_manager.VisitorAgentFactory'), \
                 patch('managers.agent_manager.DinosaurAgentFactory'):
                
                manager = AgentManager(mock_openai_config, mock_ag2_config, test_agent_config)
                
                # Create realistic agent setup
                manager.agents = self._create_integration_agents()
                manager.agent_instances = {
                    agent_id: self._create_integration_agent_instance(agent)
                    for agent_id, agent in manager.agents.items()
                }
                manager.agent_health = {
                    agent_id: manager._initialize_agent_health(agent)
                    for agent_id, agent in manager.agents.items()
                }
                
                return manager
    
    def _create_integration_agents(self) -> Dict[str, Agent]:
        """Create agents for integration testing."""
        return {
            "ranger_1": Agent(
                id="ranger_1", name="Chief Ranger", role=AgentRole.PARK_RANGER,
                location=Location(0.0, 0.0, "command_center", "Emergency command center")
            ),
            "vet_1": Agent(
                id="vet_1", name="Dr. Sarah", role=AgentRole.VETERINARIAN,
                location=Location(10.0, 10.0, "medical_center", "Veterinary clinic")
            ),
            "security_1": Agent(
                id="security_1", name="Security Chief", role=AgentRole.SECURITY,
                location=Location(5.0, 5.0, "security_station", "Security headquarters")
            ),
            "visitor_1": Agent(
                id="visitor_1", name="Tourist Family", role=AgentRole.TOURIST,
                location=Location(100.0, 100.0, "carnivore_viewing", "T-Rex viewing area")
            ),
            "trex_1": Agent(
                id="trex_1", name="Rexy", role=AgentRole.DINOSAUR, species=DinosaurSpecies.TYRANNOSAURUS_REX,
                location=Location(105.0, 105.0, "carnivore_habitat", "T-Rex enclosure")
            )
        }
    
    def _create_integration_agent_instance(self, agent: Agent):
        """Create mock agent instance for integration testing."""
        mock_instance = Mock()
        mock_instance.agent_model = agent
        
        # Create role-specific responses
        if agent.role == AgentRole.PARK_RANGER:
            mock_instance.handle_event_notification.return_value = (
                "Ranger responding: Initiating emergency protocols and coordinating with team."
            )
        elif agent.role == AgentRole.VETERINARIAN:
            mock_instance.handle_event_notification.return_value = (
                "Veterinarian responding: Preparing medical equipment and assessing situation."
            )
        elif agent.role == AgentRole.SECURITY:
            mock_instance.handle_event_notification.return_value = (
                "Security responding: Establishing perimeter and ensuring visitor safety."
            )
        elif agent.role == AgentRole.TOURIST:
            mock_instance.handle_event_notification.return_value = (
                "Tourist responding: We're scared! What should we do?"
            )
            mock_instance.satisfaction_level = 0.3  # Low due to emergency
        elif agent.role == AgentRole.DINOSAUR:
            mock_instance.handle_event_notification.return_value = (
                "Dinosaur responding: *Roars and shows signs of agitation*"
            )
            mock_instance.happiness_level = 0.4
            mock_instance.stress_level = 0.8
        
        return mock_instance
    
    def test_emergency_response_scenario(self, integration_manager):
        """Test complete emergency response scenario with multiple agents."""
        # Create emergency event
        emergency_event = Event(
            id="emergency_001",
            type=EventType.DINOSAUR_ESCAPE,
            severity=9,
            location=Location(105.0, 105.0, "carnivore_habitat", "T-Rex enclosure"),
            description="T-Rex has breached containment and is approaching visitor area",
            parameters={
                "dinosaur_species": "Tyrannosaurus Rex",
                "threat_level": "critical",
                "visitors_at_risk": 15
            }
        )
        
        # Broadcast emergency
        response = integration_manager.broadcast_event(emergency_event)
        
        # Verify emergency response
        assert response['event_id'] == emergency_event.id
        assert len(response['affected_agents']) > 0
        assert len(response['individual_responses']) > 0
        
        # Verify staff agents responded
        staff_responded = any(
            integration_manager.agents[agent_id].role in [AgentRole.PARK_RANGER, AgentRole.SECURITY, AgentRole.VETERINARIAN]
            for agent_id in response['affected_agents']
        )
        assert staff_responded
        
        # Check that group response was initiated for high severity
        assert response.get('group_response') is not None
    
    def test_multi_agent_coordination_workflow(self, integration_manager):
        """Test multi-agent coordination workflow."""
        # Step 1: Initial event
        initial_event = Event(
            id="coord_001",
            type=EventType.VISITOR_INJURY,
            severity=6,
            location=Location(100.0, 100.0, "carnivore_viewing", "T-Rex viewing area"),
            description="Visitor injured while observing dinosaurs"
        )
        
        # Broadcast initial event
        initial_response = integration_manager.broadcast_event(initial_event)
        
        # Step 2: Follow-up communication between agents
        ranger_id = None
        vet_id = None
        
        for agent_id, agent in integration_manager.agents.items():
            if agent.role == AgentRole.PARK_RANGER:
                ranger_id = agent_id
            elif agent.role == AgentRole.VETERINARIAN:
                vet_id = agent_id
        
        if ranger_id and vet_id:
            # Initiate coordination conversation
            coordination_messages = integration_manager.initiate_agent_conversation(
                [ranger_id, vet_id],
                "We need to coordinate medical response and visitor evacuation",
                "EmergencyCoordinator"
            )
            
            assert isinstance(coordination_messages, list)
        
        # Step 3: Check agent states after coordination
        agent_states = integration_manager.get_agent_states()
        
        # Verify agents are responsive
        for agent_id in initial_response['affected_agents']:
            if agent_id in agent_states:
                assert agent_states[agent_id]['is_responsive']
    
    def test_cascading_event_scenario(self, integration_manager):
        """Test scenario with cascading events affecting multiple agents."""
        # Primary event: Power outage
        primary_event = Event(
            id="cascade_001",
            type=EventType.FACILITY_POWER_OUTAGE,
            severity=7,
            location=Location(0.0, 0.0, "power_station", "Main power facility"),
            description="Main power grid failure affecting containment systems"
        )
        
        # Broadcast primary event
        primary_response = integration_manager.broadcast_event(primary_event)
        
        # Secondary event: Dinosaur agitation due to power failure
        secondary_event = Event(
            id="cascade_002",
            type=EventType.DINOSAUR_AGGRESSIVE,
            severity=8,
            location=Location(105.0, 105.0, "carnivore_habitat", "T-Rex enclosure"),
            description="Dinosaurs becoming agitated due to containment system failure",
            parameters={"related_event": "cascade_001"}
        )
        
        # Broadcast secondary event
        secondary_response = integration_manager.broadcast_event(secondary_event)
        
        # Verify both events were handled
        assert primary_response['event_id'] == primary_event.id
        assert secondary_response['event_id'] == secondary_event.id
        
        # Check that maintenance was involved in primary event
        maintenance_involved = any(
            integration_manager.agents[agent_id].role == AgentRole.MAINTENANCE
            for agent_id in primary_response['affected_agents']
            if agent_id in integration_manager.agents
        )
        
        # Check conversation history includes both events
        conversation_history = integration_manager.get_agent_conversations()
        event_ids = [record['event_id'] for record in conversation_history]
        assert primary_event.id in event_ids
        assert secondary_event.id in event_ids
    
    def test_agent_health_during_stress(self, integration_manager):
        """Test agent health monitoring during stressful scenarios."""
        # Create multiple rapid events to stress the system
        events = []
        for i in range(3):
            event = Event(
                id=f"stress_{i}",
                type=EventType.VISITOR_EMERGENCY,
                severity=5 + i,
                location=Location(i * 10.0, i * 10.0, f"zone_{i}", f"Emergency zone {i}"),
                description=f"Emergency situation {i}"
            )
            events.append(event)
        
        # Broadcast all events rapidly
        responses = []
        for event in events:
            response = integration_manager.broadcast_event(event)
            responses.append(response)
        
        # Check agent health after stress
        health_report = integration_manager.check_agent_health()
        
        # Verify health monitoring is working
        assert len(health_report) > 0
        
        # Check that agents have response counts
        for agent_id, health_info in health_report.items():
            assert 'response_count' in health_info
            assert 'error_count' in health_info
            assert 'status' in health_info
    
    def test_system_recovery_after_failure(self, integration_manager):
        """Test system recovery after agent communication failures."""
        # Simulate agent failure
        agent_id = list(integration_manager.agents.keys())[0]
        
        # Mark agent as failed
        integration_manager._update_agent_health(agent_id, success=False, error="Communication timeout")
        integration_manager._update_agent_health(agent_id, success=False, error="Communication timeout")
        integration_manager._update_agent_health(agent_id, success=False, error="Communication timeout")
        
        # Check that agent is marked as unhealthy
        health_report = integration_manager.check_agent_health()
        assert health_report[agent_id]['status'] == 'unhealthy'
        
        # Simulate recovery
        integration_manager._update_agent_health(agent_id, success=True)
        
        # Check that agent is marked as healthy again
        health_report = integration_manager.check_agent_health()
        assert health_report[agent_id]['status'] == 'healthy'
        assert health_report[agent_id]['communication_failures'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])