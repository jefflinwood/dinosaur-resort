"""Unit tests for SimulationManager class."""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from managers.simulation_manager import SimulationManager
from models.core import SimulationState, Event, Agent, MetricsSnapshot
from models.config import OpenAIConfig, AG2Config, SimulationConfig, AgentConfig, Location
from models.enums import EventType, ResolutionStatus, AgentRole, AgentState
from ui.session_state import SessionStateManager


class TestSimulationManager:
    """Test cases for SimulationManager class."""
    
    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session state manager."""
        mock_session = Mock(spec=SessionStateManager)
        
        # Default simulation state
        default_state = SimulationState(
            is_running=False,
            current_time=datetime.now(),
            active_events=[],
            agent_count=0,
            current_metrics=None,
            simulation_id="",
            started_at=None
        )
        mock_session.get_simulation_state.return_value = default_state
        mock_session.get_session_info.return_value = {"initialized": True}
        
        return mock_session
    
    @pytest.fixture
    def simulation_manager(self, mock_session_manager):
        """Create a SimulationManager instance for testing."""
        return SimulationManager(mock_session_manager)
    
    @pytest.fixture
    def openai_config(self):
        """Create OpenAI configuration for testing."""
        return OpenAIConfig(
            api_key="test-key",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )
    
    @pytest.fixture
    def ag2_config(self):
        """Create ag2 configuration for testing."""
        return AG2Config()
    
    @pytest.fixture
    def simulation_config(self):
        """Create simulation configuration for testing."""
        return SimulationConfig()
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig()
    
    def test_initialization(self, mock_session_manager):
        """Test SimulationManager initialization."""
        manager = SimulationManager(mock_session_manager)
        
        assert manager.session_manager == mock_session_manager
        assert manager.agent_manager is None
        assert manager.event_manager is None
        assert manager.metrics_manager is None
        assert manager.simulation_start_time is None
        assert manager.simulation_speed_multiplier == 1.0
        assert manager.last_update_time is None
        assert manager.event_processing_enabled is True
        assert manager.auto_resolution_enabled is True
    
    @patch('managers.simulation_manager.AgentManager')
    @patch('managers.simulation_manager.EventManager')
    @patch('managers.simulation_manager.MetricsManager')
    def test_start_simulation_success(self, mock_metrics_manager, mock_event_manager, 
                                    mock_agent_manager, simulation_manager, openai_config,
                                    ag2_config, simulation_config, agent_config):
        """Test successful simulation start."""
        # Mock agent manager initialization
        mock_agent_instance = Mock()
        mock_agent_instance.initialize_agents.return_value = [
            Agent(id="agent1", name="Test Agent", role=AgentRole.PARK_RANGER)
        ]
        mock_agent_manager.return_value = mock_agent_instance
        
        # Mock event manager
        mock_event_instance = Mock()
        mock_event_manager.return_value = mock_event_instance
        
        # Mock metrics manager
        mock_metrics_instance = Mock()
        mock_metrics_manager.return_value = mock_metrics_instance
        
        # Start simulation
        simulation_manager.start_simulation(
            openai_config=openai_config,
            ag2_config=ag2_config,
            simulation_config=simulation_config,
            agent_config=agent_config
        )
        
        # Verify managers were created
        assert simulation_manager.agent_manager is not None
        assert simulation_manager.event_manager is not None
        assert simulation_manager.metrics_manager is not None
        
        # Verify session state was updated
        simulation_manager.session_manager.update_simulation_state.assert_called()
        
        # Verify configurations were stored
        simulation_manager.session_manager.set_openai_config.assert_called_with(openai_config)
        simulation_manager.session_manager.set_ag2_config.assert_called_with(ag2_config)
        simulation_manager.session_manager.set_simulation_config.assert_called_with(simulation_config)
        simulation_manager.session_manager.set_agent_config.assert_called_with(agent_config)
    
    def test_start_simulation_without_openai_config(self, simulation_manager):
        """Test simulation start fails without OpenAI config."""
        simulation_manager.session_manager.get_openai_config.return_value = None
        
        with pytest.raises(ValueError, match="OpenAI configuration is required"):
            simulation_manager.start_simulation()
    
    @patch('managers.simulation_manager.AgentManager')
    @patch('managers.simulation_manager.EventManager')
    @patch('managers.simulation_manager.MetricsManager')
    def test_start_simulation_with_session_configs(self, mock_metrics_manager, mock_event_manager,
                                                 mock_agent_manager, simulation_manager, openai_config):
        """Test simulation start using configs from session state."""
        # Mock session state configs
        simulation_manager.session_manager.get_openai_config.return_value = openai_config
        simulation_manager.session_manager.get_ag2_config.return_value = AG2Config()
        simulation_manager.session_manager.get_simulation_config.return_value = SimulationConfig()
        simulation_manager.session_manager.get_agent_config.return_value = AgentConfig()
        
        # Mock agent manager
        mock_agent_instance = Mock()
        mock_agent_instance.initialize_agents.return_value = []
        mock_agent_manager.return_value = mock_agent_instance
        
        # Start simulation without explicit configs
        simulation_manager.start_simulation()
        
        # Verify managers were created
        assert simulation_manager.agent_manager is not None
        assert simulation_manager.event_manager is not None
        assert simulation_manager.metrics_manager is not None
    
    def test_pause_simulation(self, simulation_manager):
        """Test simulation pause."""
        # Mock running simulation
        running_state = SimulationState(is_running=True, simulation_id="test-id")
        simulation_manager.session_manager.get_simulation_state.return_value = running_state
        
        simulation_manager.pause_simulation()
        
        simulation_manager.session_manager.update_simulation_state.assert_called_with(is_running=False)
    
    def test_pause_simulation_not_running(self, simulation_manager):
        """Test pause when simulation is not running."""
        # Mock non-running simulation
        stopped_state = SimulationState(is_running=False)
        simulation_manager.session_manager.get_simulation_state.return_value = stopped_state
        
        simulation_manager.pause_simulation()
        
        # Should not update session state
        simulation_manager.session_manager.update_simulation_state.assert_not_called()
    
    def test_resume_simulation(self, simulation_manager):
        """Test simulation resume."""
        # Mock paused but initialized simulation
        paused_state = SimulationState(is_running=False, simulation_id="test-id")
        simulation_manager.session_manager.get_simulation_state.return_value = paused_state
        
        simulation_manager.resume_simulation()
        
        # Verify session state was updated
        call_args = simulation_manager.session_manager.update_simulation_state.call_args[1]
        assert call_args['is_running'] is True
        assert 'current_time' in call_args
    
    def test_resume_simulation_already_running(self, simulation_manager):
        """Test resume when simulation is already running."""
        # Mock running simulation
        running_state = SimulationState(is_running=True, simulation_id="test-id")
        simulation_manager.session_manager.get_simulation_state.return_value = running_state
        
        simulation_manager.resume_simulation()
        
        # Should not update session state
        simulation_manager.session_manager.update_simulation_state.assert_not_called()
    
    def test_resume_simulation_not_initialized(self, simulation_manager):
        """Test resume when simulation is not initialized."""
        # Mock uninitialized simulation
        uninitialized_state = SimulationState(is_running=False, simulation_id="")
        simulation_manager.session_manager.get_simulation_state.return_value = uninitialized_state
        
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            simulation_manager.resume_simulation()
    
    def test_stop_simulation(self, simulation_manager):
        """Test simulation stop."""
        # Mock managers
        simulation_manager.agent_manager = Mock()
        simulation_manager.event_manager = Mock()
        simulation_manager.event_manager.get_active_events.return_value = [
            Event(id="event1", type=EventType.DINOSAUR_ESCAPE, severity=5, 
                 location=Location(0, 0, "test"), resolution_status=ResolutionStatus.IN_PROGRESS)
        ]
        
        simulation_manager.stop_simulation()
        
        # Verify agent conversations were reset
        simulation_manager.agent_manager.reset_agent_conversations.assert_called_once()
        
        # Verify active events were failed
        simulation_manager.event_manager.update_event_status.assert_called_with(
            "event1", ResolutionStatus.FAILED
        )
        
        # Verify session state was updated
        simulation_manager.session_manager.update_simulation_state.assert_called_with(is_running=False)
        
        # Verify timing was reset
        assert simulation_manager.simulation_start_time is None
        assert simulation_manager.last_update_time is None
    
    def test_reset_simulation(self, simulation_manager):
        """Test simulation reset."""
        # Set up some state
        simulation_manager.agent_manager = Mock()
        simulation_manager.event_manager = Mock()
        simulation_manager.event_manager.get_active_events.return_value = []  # Mock empty list
        simulation_manager.metrics_manager = Mock()
        simulation_manager.simulation_start_time = datetime.now()
        simulation_manager.last_update_time = datetime.now()
        
        simulation_manager.reset_simulation()
        
        # Verify session state was cleared
        simulation_manager.session_manager.clear_all.assert_called_once()
        
        # Verify managers were reset
        assert simulation_manager.agent_manager is None
        assert simulation_manager.event_manager is None
        assert simulation_manager.metrics_manager is None
        
        # Verify timing was reset
        assert simulation_manager.simulation_start_time is None
        assert simulation_manager.last_update_time is None
    
    def test_trigger_event_success(self, simulation_manager):
        """Test successful event triggering."""
        # Mock initialized simulation
        initialized_state = SimulationState(simulation_id="test-id")
        simulation_manager.session_manager.get_simulation_state.return_value = initialized_state
        
        # Mock managers
        simulation_manager.event_manager = Mock()
        simulation_manager.agent_manager = Mock()
        
        # Mock event creation
        test_event = Event(
            id="test-event",
            type=EventType.DINOSAUR_ESCAPE,
            severity=5,
            location=Location(0, 0, "test")
        )
        simulation_manager.event_manager.create_event.return_value = test_event
        
        # Mock agent broadcast
        simulation_manager.agent_manager.broadcast_event.return_value = {
            'affected_agents': ['agent1', 'agent2']
        }
        
        # Trigger event
        event_id = simulation_manager.trigger_event(
            event_type="DINOSAUR_ESCAPE",
            parameters={"dinosaur_id": "dino1"},
            severity=7
        )
        
        assert event_id == "test-event"
        
        # Verify event was created
        simulation_manager.event_manager.create_event.assert_called_once()
        
        # Verify event was added to session state
        simulation_manager.session_manager.add_event.assert_called_with(test_event)
        
        # Verify event was distributed
        simulation_manager.event_manager.distribute_event.assert_called_with(test_event)
        
        # Verify event was broadcast to agents
        simulation_manager.agent_manager.broadcast_event.assert_called_with(test_event)
    
    def test_trigger_event_not_initialized(self, simulation_manager):
        """Test event triggering when simulation is not initialized."""
        # Mock uninitialized simulation
        uninitialized_state = SimulationState(simulation_id="")
        simulation_manager.session_manager.get_simulation_state.return_value = uninitialized_state
        
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            simulation_manager.trigger_event("DINOSAUR_ESCAPE", {})
    
    def test_trigger_event_no_event_manager(self, simulation_manager):
        """Test event triggering when event manager is not initialized."""
        # Mock initialized simulation but no event manager
        initialized_state = SimulationState(simulation_id="test-id")
        simulation_manager.session_manager.get_simulation_state.return_value = initialized_state
        simulation_manager.event_manager = None
        
        with pytest.raises(RuntimeError, match="Event manager not initialized"):
            simulation_manager.trigger_event("DINOSAUR_ESCAPE", {})
    
    def test_get_simulation_state(self, simulation_manager):
        """Test getting simulation state."""
        test_state = SimulationState(is_running=True, simulation_id="test")
        simulation_manager.session_manager.get_simulation_state.return_value = test_state
        
        result = simulation_manager.get_simulation_state()
        
        assert result == test_state
        simulation_manager.session_manager.get_simulation_state.assert_called_once()
    
    def test_update_simulation_time_running(self, simulation_manager):
        """Test simulation time update when running."""
        # Mock running simulation
        running_state = SimulationState(
            is_running=True,
            current_time=datetime.now() - timedelta(minutes=1)
        )
        simulation_manager.session_manager.get_simulation_state.return_value = running_state
        simulation_manager.last_update_time = datetime.now() - timedelta(seconds=30)
        
        simulation_manager.update_simulation_time()
        
        # Verify session state was updated
        simulation_manager.session_manager.update_simulation_state.assert_called()
        call_args = simulation_manager.session_manager.update_simulation_state.call_args[1]
        assert 'current_time' in call_args
    
    def test_update_simulation_time_not_running(self, simulation_manager):
        """Test simulation time update when not running."""
        # Mock stopped simulation
        stopped_state = SimulationState(is_running=False)
        simulation_manager.session_manager.get_simulation_state.return_value = stopped_state
        
        simulation_manager.update_simulation_time()
        
        # Should not update session state
        simulation_manager.session_manager.update_simulation_state.assert_not_called()
    
    def test_set_simulation_speed(self, simulation_manager):
        """Test setting simulation speed."""
        simulation_manager.set_simulation_speed(2.0)
        
        assert simulation_manager.simulation_speed_multiplier == 2.0
    
    def test_set_simulation_speed_invalid(self, simulation_manager):
        """Test setting invalid simulation speed."""
        with pytest.raises(ValueError, match="Speed multiplier must be positive"):
            simulation_manager.set_simulation_speed(0)
        
        with pytest.raises(ValueError, match="Speed multiplier must be positive"):
            simulation_manager.set_simulation_speed(-1)
    
    def test_get_active_events(self, simulation_manager):
        """Test getting active events."""
        # Mock event manager
        simulation_manager.event_manager = Mock()
        test_events = [
            Event(id="event1", type=EventType.DINOSAUR_ESCAPE, severity=5, location=Location(0, 0, "test"))
        ]
        simulation_manager.event_manager.get_active_events.return_value = test_events
        
        result = simulation_manager.get_active_events()
        
        assert result == test_events
        simulation_manager.event_manager.get_active_events.assert_called_once()
    
    def test_get_active_events_no_manager(self, simulation_manager):
        """Test getting active events when event manager is None."""
        simulation_manager.event_manager = None
        
        result = simulation_manager.get_active_events()
        
        assert result == []
    
    def test_get_event_history(self, simulation_manager):
        """Test getting event history."""
        # Mock event manager
        simulation_manager.event_manager = Mock()
        test_events = [
            Event(id="event1", type=EventType.DINOSAUR_ESCAPE, severity=5, location=Location(0, 0, "test"))
        ]
        simulation_manager.event_manager.get_event_history.return_value = test_events
        
        result = simulation_manager.get_event_history(limit=10)
        
        assert result == test_events
        simulation_manager.event_manager.get_event_history.assert_called_with(10)
    
    def test_get_agent_states(self, simulation_manager):
        """Test getting agent states."""
        # Mock agent manager
        simulation_manager.agent_manager = Mock()
        test_states = {"agent1": {"name": "Test Agent", "state": "IDLE"}}
        simulation_manager.agent_manager.get_agent_states.return_value = test_states
        
        result = simulation_manager.get_agent_states()
        
        assert result == test_states
        simulation_manager.agent_manager.get_agent_states.assert_called_once()
    
    def test_get_current_metrics(self, simulation_manager):
        """Test getting current metrics."""
        # Mock metrics manager
        simulation_manager.metrics_manager = Mock()
        test_metrics = MetricsSnapshot(visitor_satisfaction=0.8)
        simulation_manager.metrics_manager.get_current_metrics.return_value = test_metrics
        
        result = simulation_manager.get_current_metrics()
        
        assert result == test_metrics
        simulation_manager.metrics_manager.get_current_metrics.assert_called_once()
    
    def test_get_current_metrics_no_manager(self, simulation_manager):
        """Test getting current metrics when metrics manager is None."""
        simulation_manager.metrics_manager = None
        
        result = simulation_manager.get_current_metrics()
        
        assert result is None
    
    def test_is_running(self, simulation_manager):
        """Test checking if simulation is running."""
        # Mock running simulation
        running_state = SimulationState(is_running=True)
        simulation_manager.session_manager.get_simulation_state.return_value = running_state
        
        assert simulation_manager.is_running() is True
        
        # Mock stopped simulation
        stopped_state = SimulationState(is_running=False)
        simulation_manager.session_manager.get_simulation_state.return_value = stopped_state
        
        assert simulation_manager.is_running() is False
    
    def test_get_simulation_info(self, simulation_manager):
        """Test getting comprehensive simulation information."""
        # Mock simulation state
        test_state = SimulationState(
            is_running=True,
            simulation_id="test-id",
            started_at=datetime.now() - timedelta(minutes=5),
            current_time=datetime.now(),
            agent_count=3
        )
        simulation_manager.session_manager.get_simulation_state.return_value = test_state
        
        # Mock managers
        simulation_manager.agent_manager = Mock()
        simulation_manager.event_manager = Mock()
        simulation_manager.metrics_manager = Mock()
        
        simulation_manager.metrics_manager.get_metrics_summary.return_value = {"overall_score": 0.8}
        simulation_manager.event_manager.get_statistics.return_value = {"total_events": 5}
        simulation_manager.agent_manager.check_agent_health.return_value = {"agent1": {"status": "healthy"}}
        
        result = simulation_manager.get_simulation_info()
        
        assert result["is_running"] is True
        assert result["simulation_id"] == "test-id"
        assert result["agent_count"] == 3
        assert "uptime" in result
        assert result["managers_initialized"]["agent_manager"] is True
        assert result["managers_initialized"]["event_manager"] is True
        assert result["managers_initialized"]["metrics_manager"] is True
        assert "metrics" in result
        assert "event_statistics" in result
        assert "agent_health" in result
    
    def test_enable_auto_resolution(self, simulation_manager):
        """Test enabling/disabling auto resolution."""
        simulation_manager.enable_auto_resolution(True)
        assert simulation_manager.auto_resolution_enabled is True
        
        simulation_manager.enable_auto_resolution(False)
        assert simulation_manager.auto_resolution_enabled is False
    
    def test_enable_event_processing(self, simulation_manager):
        """Test enabling/disabling event processing."""
        simulation_manager.enable_event_processing(True)
        assert simulation_manager.event_processing_enabled is True
        
        simulation_manager.enable_event_processing(False)
        assert simulation_manager.event_processing_enabled is False
    
    def test_get_debug_info(self, simulation_manager):
        """Test getting debug information."""
        # Set up some state
        simulation_manager.simulation_start_time = datetime.now()
        simulation_manager.last_update_time = datetime.now()
        simulation_manager.agent_manager = Mock()
        
        # Mock simulation state
        test_state = SimulationState(is_running=True, simulation_id="test-id")
        simulation_manager.session_manager.get_simulation_state.return_value = test_state
        
        result = simulation_manager.get_debug_info()
        
        assert "simulation_manager" in result
        assert "managers" in result
        assert "session_state" in result
        
        sim_info = result["simulation_manager"]
        assert sim_info["initialized"] is True
        assert sim_info["running"] is True
        assert sim_info["simulation_speed"] == 1.0
        assert sim_info["auto_resolution_enabled"] is True
        assert sim_info["event_processing_enabled"] is True
    
    def test_check_event_timeouts(self, simulation_manager):
        """Test checking for event timeouts."""
        # Mock event manager with timed-out event
        simulation_manager.event_manager = Mock()
        old_event = Event(
            id="old-event",
            type=EventType.DINOSAUR_ESCAPE,
            severity=5,
            location=Location(0, 0, "test"),
            timestamp=datetime.now() - timedelta(minutes=15),  # 15 minutes ago
            resolution_status=ResolutionStatus.IN_PROGRESS
        )
        simulation_manager.event_manager.get_active_events.return_value = [old_event]
        
        # Call the private method
        simulation_manager._check_event_timeouts()
        
        # Verify event was auto-resolved
        simulation_manager.event_manager.update_event_status.assert_called_with(
            "old-event", ResolutionStatus.RESOLVED
        )
        simulation_manager.session_manager.update_event.assert_called()
    
    def test_update_agent_time_based_states(self, simulation_manager):
        """Test updating agent states based on time."""
        # Mock agents in session state
        test_agent = Agent(
            id="agent1",
            name="Test Agent",
            role=AgentRole.PARK_RANGER,
            current_state=AgentState.ACTIVE,
            last_activity=datetime.now() - timedelta(minutes=10)  # 10 minutes ago
        )
        simulation_manager.session_manager.get_agents.return_value = {"agent1": test_agent}
        
        # Call the private method
        simulation_manager._update_agent_time_based_states(timedelta(minutes=1))
        
        # Verify agent was updated to idle
        simulation_manager.session_manager.update_agent.assert_called_with(
            "agent1", current_state=AgentState.IDLE
        )
    
    def test_apply_time_based_metric_changes(self, simulation_manager):
        """Test applying time-based metric changes."""
        # Mock metrics manager
        simulation_manager.metrics_manager = Mock()
        test_metrics = MetricsSnapshot(
            visitor_satisfaction=0.7,
            facility_efficiency=0.8
        )
        simulation_manager.metrics_manager.get_current_metrics.return_value = test_metrics
        
        # Mock no active major events
        simulation_manager.event_manager = Mock()
        simulation_manager.event_manager.get_active_events.return_value = []
        
        # Call the private method with 1 minute
        simulation_manager._apply_time_based_metric_changes(timedelta(minutes=1))
        
        # Verify metrics were improved
        simulation_manager.metrics_manager.update_facility_efficiency.assert_called()
        simulation_manager.metrics_manager.update_visitor_satisfaction.assert_called()
    
    def test_handle_event_for_metrics(self, simulation_manager):
        """Test handling events for metrics updates."""
        # Mock metrics manager
        simulation_manager.metrics_manager = Mock()
        
        test_event = Event(
            id="test-event",
            type=EventType.DINOSAUR_ESCAPE,
            severity=7,
            location=Location(0, 0, "test"),
            affected_agents=["agent1", "agent2"]
        )
        
        # Call the private method
        simulation_manager._handle_event_for_metrics(test_event)
        
        # Verify metrics impact was applied
        simulation_manager.metrics_manager.apply_event_impact.assert_called_with(
            "DINOSAUR_ESCAPE", 7, ["agent1", "agent2"]
        )


if __name__ == "__main__":
    pytest.main([__file__])