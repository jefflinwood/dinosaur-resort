"""Unit tests for Streamlit session state management."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from models.core import Agent, Event, MetricsSnapshot, SimulationState
from models.config import OpenAIConfig, AG2Config, SimulationConfig, AgentConfig, Location
from models.enums import AgentRole, EventType, ResolutionStatus, DinosaurSpecies
from ui.session_state import SessionStateManager


class TestSessionStateManager:
    """Test session state manager."""
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock streamlit session state."""
        with patch('ui.session_state.st') as mock_st:
            mock_st.session_state = {}
            yield mock_st
    
    @pytest.fixture
    def session_manager(self, mock_streamlit):
        """Create test session state manager."""
        return SessionStateManager()
    
    @pytest.fixture
    def test_agent(self):
        """Create test agent."""
        return Agent(
            id="test_agent_1",
            name="Test Ranger",
            role=AgentRole.PARK_RANGER,
            personality_traits={"cautious": 0.8},
            capabilities=["wildlife_management"]
        )
    
    @pytest.fixture
    def test_event(self):
        """Create test event."""
        return Event(
            id="test_event_1",
            type=EventType.DINOSAUR_ESCAPE,
            severity=8,
            location=Location(x=100.0, y=200.0, zone="main_area"),
            description="T-Rex has escaped from enclosure"
        )
    
    @pytest.fixture
    def test_metrics(self):
        """Create test metrics snapshot."""
        return MetricsSnapshot(
            visitor_satisfaction=0.8,
            dinosaur_happiness={"dino_1": 0.7, "dino_2": 0.9},
            facility_efficiency=0.85,
            safety_rating=0.6
        )
    
    def test_initialization(self, session_manager, mock_streamlit):
        """Test session state manager initialization."""
        # Check that session state was initialized
        assert mock_streamlit.session_state[SessionStateManager.INITIALIZED] is True
        assert isinstance(mock_streamlit.session_state[SessionStateManager.SIMULATION_STATE], SimulationState)
        assert mock_streamlit.session_state[SessionStateManager.AGENTS] == {}
        assert mock_streamlit.session_state[SessionStateManager.EVENTS] == []
        assert mock_streamlit.session_state[SessionStateManager.METRICS_HISTORY] == []
        assert mock_streamlit.session_state[SessionStateManager.CONVERSATION_HISTORY] == {}
    
    def test_get_set_operations(self, session_manager, mock_streamlit):
        """Test basic get/set operations."""
        # Test set
        session_manager.set("test_key", "test_value")
        assert mock_streamlit.session_state["test_key"] == "test_value"
        
        # Test get
        value = session_manager.get("test_key")
        assert value == "test_value"
        
        # Test get with default
        value = session_manager.get("nonexistent_key", "default")
        assert value == "default"
    
    def test_delete_operation(self, session_manager, mock_streamlit):
        """Test delete operation."""
        session_manager.set("test_key", "test_value")
        session_manager.delete("test_key")
        assert "test_key" not in mock_streamlit.session_state
    
    def test_clear_all(self, session_manager, mock_streamlit):
        """Test clearing all session state."""
        # Add some data
        session_manager.set("test_key", "test_value")
        session_manager.add_agent(Agent(id="test", name="Test", role=AgentRole.TOURIST))
        
        # Clear all
        session_manager.clear_all()
        
        # Should be reinitialized
        assert mock_streamlit.session_state[SessionStateManager.INITIALIZED] is True
        assert mock_streamlit.session_state[SessionStateManager.AGENTS] == {}
        assert "test_key" not in mock_streamlit.session_state
    
    def test_simulation_state_management(self, session_manager):
        """Test simulation state management."""
        # Get initial state
        state = session_manager.get_simulation_state()
        assert isinstance(state, SimulationState)
        assert state.is_running is False
        
        # Update state
        session_manager.update_simulation_state(is_running=True, agent_count=5)
        
        # Verify update
        updated_state = session_manager.get_simulation_state()
        assert updated_state.is_running is True
        assert updated_state.agent_count == 5
    
    def test_agent_management(self, session_manager, test_agent):
        """Test agent management operations."""
        # Initially no agents
        agents = session_manager.get_agents()
        assert len(agents) == 0
        
        # Add agent
        session_manager.add_agent(test_agent)
        
        # Verify agent was added
        agents = session_manager.get_agents()
        assert len(agents) == 1
        assert test_agent.id in agents
        assert agents[test_agent.id] == test_agent
        
        # Verify simulation state was updated
        sim_state = session_manager.get_simulation_state()
        assert sim_state.agent_count == 1
        
        # Get specific agent
        retrieved_agent = session_manager.get_agent(test_agent.id)
        assert retrieved_agent == test_agent
        
        # Update agent
        success = session_manager.update_agent(test_agent.id, name="Updated Name")
        assert success is True
        
        updated_agent = session_manager.get_agent(test_agent.id)
        assert updated_agent.name == "Updated Name"
        
        # Remove agent
        success = session_manager.remove_agent(test_agent.id)
        assert success is True
        
        agents = session_manager.get_agents()
        assert len(agents) == 0
        
        # Verify simulation state was updated
        sim_state = session_manager.get_simulation_state()
        assert sim_state.agent_count == 0
    
    def test_agent_management_nonexistent(self, session_manager):
        """Test agent operations with nonexistent agents."""
        # Get nonexistent agent
        agent = session_manager.get_agent("nonexistent")
        assert agent is None
        
        # Update nonexistent agent
        success = session_manager.update_agent("nonexistent", name="New Name")
        assert success is False
        
        # Remove nonexistent agent
        success = session_manager.remove_agent("nonexistent")
        assert success is False
    
    def test_event_management(self, session_manager, test_event):
        """Test event management operations."""
        # Initially no events
        events = session_manager.get_events()
        assert len(events) == 0
        
        # Add event
        session_manager.add_event(test_event)
        
        # Verify event was added
        events = session_manager.get_events()
        assert len(events) == 1
        assert events[0] == test_event
        
        # Verify simulation state was updated with active events
        sim_state = session_manager.get_simulation_state()
        assert len(sim_state.active_events) == 1
        assert sim_state.active_events[0] == test_event
        
        # Update event
        success = session_manager.update_event(test_event.id, severity=5)
        assert success is True
        
        events = session_manager.get_events()
        assert events[0].severity == 5
        
        # Update event to resolved status
        success = session_manager.update_event(test_event.id, resolution_status=ResolutionStatus.RESOLVED)
        assert success is True
        
        # Verify active events list was updated
        sim_state = session_manager.get_simulation_state()
        assert len(sim_state.active_events) == 0  # No longer active
    
    def test_event_management_nonexistent(self, session_manager):
        """Test event operations with nonexistent events."""
        success = session_manager.update_event("nonexistent", severity=5)
        assert success is False
    
    def test_metrics_management(self, session_manager, test_metrics):
        """Test metrics management operations."""
        # Initially no metrics
        history = session_manager.get_metrics_history()
        assert len(history) == 0
        
        latest = session_manager.get_latest_metrics()
        assert latest is None
        
        # Add metrics snapshot
        session_manager.add_metrics_snapshot(test_metrics)
        
        # Verify metrics were added
        history = session_manager.get_metrics_history()
        assert len(history) == 1
        assert history[0] == test_metrics
        
        latest = session_manager.get_latest_metrics()
        assert latest == test_metrics
        
        # Verify simulation state was updated
        sim_state = session_manager.get_simulation_state()
        assert sim_state.current_metrics == test_metrics
    
    def test_metrics_history_limit(self, session_manager):
        """Test metrics history size limit."""
        # Add more than 1000 metrics snapshots
        for i in range(1005):
            metrics = MetricsSnapshot(
                visitor_satisfaction=0.5,
                timestamp=datetime.now()
            )
            session_manager.add_metrics_snapshot(metrics)
        
        # Should be limited to 1000
        history = session_manager.get_metrics_history()
        assert len(history) == 1000
    
    def test_conversation_management(self, session_manager):
        """Test conversation history management."""
        agent_id = "test_agent"
        message = {
            "content": "Hello, world!",
            "role": "user",
            "timestamp": datetime.now().isoformat()
        }
        
        # Initially no conversations
        history = session_manager.get_conversation_history()
        assert len(history) == 0
        
        # Add conversation message
        session_manager.add_conversation_message(agent_id, message)
        
        # Verify message was added
        history = session_manager.get_conversation_history()
        assert agent_id in history
        assert len(history[agent_id]) == 1
        assert history[agent_id][0] == message
        
        # Clear specific agent conversation
        session_manager.clear_conversation_history(agent_id)
        history = session_manager.get_conversation_history()
        assert agent_id not in history
        
        # Add message again and clear all
        session_manager.add_conversation_message(agent_id, message)
        session_manager.clear_conversation_history()
        
        history = session_manager.get_conversation_history()
        assert len(history) == 0
    
    def test_conversation_message_limit(self, session_manager):
        """Test conversation message limit per agent."""
        agent_id = "test_agent"
        
        # Add more than 100 messages
        for i in range(105):
            message = {
                "content": f"Message {i}",
                "role": "user",
                "timestamp": datetime.now().isoformat()
            }
            session_manager.add_conversation_message(agent_id, message)
        
        # Should be limited to 100
        history = session_manager.get_conversation_history()
        assert len(history[agent_id]) == 100
        
        # Should keep the latest messages
        assert history[agent_id][0]["content"] == "Message 5"  # First kept message
        assert history[agent_id][-1]["content"] == "Message 104"  # Last message
    
    def test_configuration_management(self, session_manager):
        """Test configuration management."""
        # Test OpenAI config
        openai_config = session_manager.get_openai_config()
        assert openai_config is None  # Initially None
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            new_config = OpenAIConfig()
            session_manager.set_openai_config(new_config)
            
            retrieved_config = session_manager.get_openai_config()
            assert retrieved_config == new_config
        
        # Test ag2 config
        ag2_config = session_manager.get_ag2_config()
        assert isinstance(ag2_config, AG2Config)
        
        new_ag2_config = AG2Config()
        new_ag2_config.max_round = 15
        session_manager.set_ag2_config(new_ag2_config)
        
        retrieved_config = session_manager.get_ag2_config()
        assert retrieved_config.max_round == 15
        
        # Test simulation config
        sim_config = session_manager.get_simulation_config()
        assert isinstance(sim_config, SimulationConfig)
        
        new_sim_config = SimulationConfig()
        new_sim_config.max_agents = 100
        session_manager.set_simulation_config(new_sim_config)
        
        retrieved_config = session_manager.get_simulation_config()
        assert retrieved_config.max_agents == 100
        
        # Test agent config
        agent_config = session_manager.get_agent_config()
        assert isinstance(agent_config, AgentConfig)
        
        new_agent_config = AgentConfig()
        new_agent_config.visitor_count = 10
        session_manager.set_agent_config(new_agent_config)
        
        retrieved_config = session_manager.get_agent_config()
        assert retrieved_config.visitor_count == 10
    
    def test_utility_methods(self, session_manager, test_agent, test_event):
        """Test utility methods."""
        # Test is_initialized
        assert session_manager.is_initialized() is True
        
        # Add some data
        session_manager.add_agent(test_agent)
        session_manager.add_event(test_event)
        session_manager.add_metrics_snapshot(MetricsSnapshot(visitor_satisfaction=0.8))
        session_manager.add_conversation_message("agent1", {"content": "test"})
        session_manager.update_simulation_state(is_running=True)
        
        # Test get_session_info
        info = session_manager.get_session_info()
        
        assert info["initialized"] is True
        assert info["agent_count"] == 1
        assert info["event_count"] == 1
        assert info["metrics_history_count"] == 1
        assert "agent1" in info["conversation_agents"]
        assert info["simulation_running"] is True
    
    def test_session_state_persistence(self, mock_streamlit):
        """Test that session state persists across manager instances."""
        # Create first manager and add data
        manager1 = SessionStateManager()
        test_agent = Agent(id="test", name="Test", role=AgentRole.TOURIST)
        manager1.add_agent(test_agent)
        
        # Create second manager - should see the same data
        manager2 = SessionStateManager()
        agents = manager2.get_agents()
        
        assert len(agents) == 1
        assert "test" in agents
        assert agents["test"].name == "Test"