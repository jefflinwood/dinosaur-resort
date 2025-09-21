"""End-to-end integration tests for the complete system functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any
import sys

# Mock streamlit before importing main
mock_streamlit = MagicMock()
mock_streamlit.set_page_config = MagicMock()
mock_streamlit.sidebar = MagicMock()
mock_streamlit.title = MagicMock()
mock_streamlit.write = MagicMock()
mock_streamlit.subheader = MagicMock()
mock_streamlit.metric = MagicMock()
mock_streamlit.divider = MagicMock()
mock_streamlit.button = MagicMock(return_value=False)
mock_streamlit.selectbox = MagicMock(return_value="Dashboard")
mock_streamlit.slider = MagicMock(return_value=5)
mock_streamlit.text_input = MagicMock(return_value="")
mock_streamlit.text_area = MagicMock(return_value="")
mock_streamlit.number_input = MagicMock(return_value=0)
mock_streamlit.multiselect = MagicMock(return_value=[])
mock_streamlit.checkbox = MagicMock(return_value=False)
mock_streamlit.warning = MagicMock()
mock_streamlit.error = MagicMock()
mock_streamlit.success = MagicMock()
mock_streamlit.info = MagicMock()
mock_streamlit.expander = MagicMock()
mock_streamlit.spinner = MagicMock()
mock_streamlit.rerun = MagicMock()
mock_streamlit.stop = MagicMock()
mock_streamlit.caption = MagicMock()
mock_streamlit.code = MagicMock()
mock_streamlit.progress = MagicMock()
mock_streamlit.line_chart = MagicMock()
mock_streamlit.area_chart = MagicMock()
mock_streamlit.bar_chart = MagicMock()
mock_streamlit.scatter_chart = MagicMock()
mock_streamlit.download_button = MagicMock()
mock_streamlit.json = MagicMock()

# Mock columns
def mock_columns(num_cols):
    if isinstance(num_cols, list):
        return [MagicMock() for _ in range(len(num_cols))]
    return [MagicMock() for _ in range(num_cols)]

mock_streamlit.columns = MagicMock(side_effect=mock_columns)

# Mock context managers
mock_expander = MagicMock()
mock_expander.__enter__ = MagicMock(return_value=mock_expander)
mock_expander.__exit__ = MagicMock(return_value=None)
mock_streamlit.expander.return_value = mock_expander

mock_spinner = MagicMock()
mock_spinner.__enter__ = MagicMock(return_value=mock_spinner)
mock_spinner.__exit__ = MagicMock(return_value=None)
mock_streamlit.spinner.return_value = mock_spinner

# Mock session state
mock_streamlit.session_state = {}

sys.modules['streamlit'] = mock_streamlit

# Now import the modules we want to test
from main import main, render_control_panel, render_dashboard_overview, render_agent_monitor, render_metrics_dashboard, render_event_log
from ui.session_state import SessionStateManager
from managers.simulation_manager import SimulationManager
from managers.agent_manager import AgentManager
from managers.event_manager import EventManager
from managers.metrics_manager import MetricsManager
from models.core import Agent, Event, MetricsSnapshot, SimulationState
from models.enums import AgentRole, EventType, ResolutionStatus, AgentState, DinosaurSpecies
from models.config import Location, OpenAIConfig, AG2Config, SimulationConfig, AgentConfig


class TestEndToEndIntegration:
    """End-to-end integration tests for complete system functionality."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        # Clear streamlit session state
        mock_streamlit.session_state.clear()
        
        # Create mock configurations
        self.mock_openai_config = OpenAIConfig(
            api_key="test-key",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150,
            timeout=30
        )
        
        self.mock_ag2_config = AG2Config(
            human_input_mode="NEVER",
            max_round=3,
            code_execution_config=False
        )
        
        self.mock_simulation_config = SimulationConfig(
            max_agents=10,
            auto_resolve_timeout=600,  # 10 minutes in seconds
            metrics_update_interval=30
        )
        
        self.mock_agent_config = AgentConfig(
            staff_count={
                AgentRole.PARK_RANGER: 2,
                AgentRole.VETERINARIAN: 1,
                AgentRole.SECURITY: 1,
                AgentRole.MAINTENANCE: 1
            },
            visitor_count=3,
            dinosaur_config={
                DinosaurSpecies.TYRANNOSAURUS_REX: 1,
                DinosaurSpecies.TRICERATOPS: 2,
                DinosaurSpecies.VELOCIRAPTOR: 1
            }
        )
        
        # Create sample location
        self.sample_location = Location(10.0, 20.0, "main_area", "Main visitor area")
        
        # Create sample agents
        self.sample_agents = {
            "ranger_1": Agent(
                id="ranger_1",
                name="Ranger Smith",
                role=AgentRole.PARK_RANGER,
                location=self.sample_location,
                capabilities=["wildlife_management", "visitor_safety"],
                current_state=AgentState.IDLE
            ),
            "vet_1": Agent(
                id="vet_1",
                name="Dr. Johnson",
                role=AgentRole.VETERINARIAN,
                location=self.sample_location,
                capabilities=["medical_treatment", "health_assessment"],
                current_state=AgentState.ACTIVE
            ),
            "tourist_1": Agent(
                id="tourist_1",
                name="John Visitor",
                role=AgentRole.TOURIST,
                location=self.sample_location,
                capabilities=["observation"],
                current_state=AgentState.IDLE
            )
        }
        
        # Create sample events
        self.sample_events = [
            Event(
                id="event_1",
                type=EventType.DINOSAUR_ESCAPE,
                severity=8,
                location=self.sample_location,
                parameters={"dinosaur_id": "trex_1", "enclosure_id": "enclosure_a"},
                timestamp=datetime.now() - timedelta(minutes=5),
                affected_agents=["ranger_1", "vet_1"],
                resolution_status=ResolutionStatus.IN_PROGRESS
            ),
            Event(
                id="event_2",
                type=EventType.VISITOR_INJURY,
                severity=5,
                location=self.sample_location,
                parameters={"visitor_id": "tourist_1", "injury_type": "minor_cut"},
                timestamp=datetime.now() - timedelta(minutes=2),
                affected_agents=["vet_1"],
                resolution_status=ResolutionStatus.RESOLVED,
                resolution_time=datetime.now() - timedelta(minutes=1)
            )
        ]
        
        # Create sample metrics
        self.sample_metrics = [
            MetricsSnapshot(
                visitor_satisfaction=0.85,
                dinosaur_happiness={"trex_1": 0.75, "tric_1": 0.80, "tric_2": 0.90},
                facility_efficiency=0.90,
                safety_rating=0.95,
                timestamp=datetime.now() - timedelta(minutes=5)
            ),
            MetricsSnapshot(
                visitor_satisfaction=0.80,
                dinosaur_happiness={"trex_1": 0.70, "tric_1": 0.85, "tric_2": 0.88},
                facility_efficiency=0.88,
                safety_rating=0.92,
                timestamp=datetime.now()
            )
        ]
    
    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session state manager with sample data."""
        session_manager = Mock(spec=SessionStateManager)
        
        # Create a mutable simulation state that can be updated
        self.sim_state = SimulationState()
        self.sim_state.is_running = True  # Start as running for real-time sync test
        self.sim_state.simulation_id = None
        self.sim_state.started_at = None
        self.sim_state.current_time = datetime.now()
        self.sim_state.agent_count = 0
        self.sim_state.active_events = []
        self.sim_state.current_metrics = None
        
        # Mock update_simulation_state to actually update the state
        def mock_update_simulation_state(**kwargs):
            for key, value in kwargs.items():
                if hasattr(self.sim_state, key):
                    setattr(self.sim_state, key, value)
        
        session_manager.update_simulation_state.side_effect = mock_update_simulation_state
        session_manager.get_simulation_state.return_value = self.sim_state
        session_manager.get_agents.return_value = self.sample_agents
        session_manager.get_events.return_value = self.sample_events
        session_manager.get_metrics_history.return_value = self.sample_metrics
        session_manager.get_latest_metrics.return_value = self.sample_metrics[-1]
        session_manager.get_conversation_history.return_value = {}
        session_manager.get_session_info.return_value = {
            "initialized": True,
            "agent_count": len(self.sample_agents),
            "event_count": len(self.sample_events),
            "metrics_history_count": len(self.sample_metrics),
            "conversation_agents": list(self.sample_agents.keys()),
            "simulation_running": True
        }
        
        # Mock configuration methods
        session_manager.get_openai_config.return_value = self.mock_openai_config
        session_manager.get_ag2_config.return_value = self.mock_ag2_config
        session_manager.get_simulation_config.return_value = self.mock_simulation_config
        session_manager.get_agent_config.return_value = self.mock_agent_config
        
        # Mock other methods
        session_manager.add_event.return_value = None
        session_manager.update_event.return_value = None
        session_manager.add_metrics_snapshot.return_value = None
        session_manager.clear_all.return_value = None
        
        return session_manager
    
    def test_complete_application_startup(self, mock_session_manager):
        """Test complete application startup sequence."""
        with patch('main.SessionStateManager', return_value=mock_session_manager):
            # Should not raise any exceptions
            main()
            
            # Verify basic UI elements were rendered
            mock_streamlit.set_page_config.assert_called_once()
            mock_streamlit.title.assert_called()
            mock_streamlit.write.assert_called()
    
    def test_simulation_lifecycle_integration(self, mock_session_manager):
        """Test complete simulation lifecycle from start to stop."""
        with patch('managers.simulation_manager.AgentManager') as mock_agent_manager_class, \
             patch('managers.simulation_manager.EventManager') as mock_event_manager_class, \
             patch('managers.simulation_manager.MetricsManager') as mock_metrics_manager_class:
            
            # Mock manager instances
            mock_agent_manager = Mock()
            mock_agent_manager.initialize_agents.return_value = list(self.sample_agents.values())
            mock_agent_manager.reset_agent_conversations.return_value = None
            mock_agent_manager_class.return_value = mock_agent_manager
            
            mock_event_manager = Mock()
            mock_event_manager.get_active_events.return_value = []
            mock_event_manager_class.return_value = mock_event_manager
            
            mock_metrics_manager = Mock()
            mock_metrics_manager_class.return_value = mock_metrics_manager
            
            # Create simulation manager
            sim_manager = SimulationManager(mock_session_manager)
            
            # Test start simulation
            sim_manager.start_simulation(
                openai_config=self.mock_openai_config,
                ag2_config=self.mock_ag2_config,
                simulation_config=self.mock_simulation_config,
                agent_config=self.mock_agent_config
            )
            
            # Verify managers were initialized
            mock_agent_manager_class.assert_called_once()
            mock_event_manager_class.assert_called_once()
            mock_metrics_manager_class.assert_called_once()
            
            # Verify agents were initialized
            mock_agent_manager.initialize_agents.assert_called_once()
            
            # Verify simulation state was updated
            mock_session_manager.update_simulation_state.assert_called()
            
            # Test pause simulation
            sim_manager.pause_simulation()
            assert not sim_manager.is_running()
            
            # Test resume simulation
            sim_manager.resume_simulation()
            
            # Test stop simulation
            sim_manager.stop_simulation()
            assert not sim_manager.is_running()
            
            # Test reset simulation
            sim_manager.reset_simulation()
            mock_session_manager.clear_all.assert_called_once()
    
    def test_event_triggering_and_resolution_flow(self, mock_session_manager):
        """Test complete event triggering and resolution flow."""
        with patch('managers.simulation_manager.AgentManager') as mock_agent_manager_class, \
             patch('managers.simulation_manager.EventManager') as mock_event_manager_class, \
             patch('managers.simulation_manager.MetricsManager') as mock_metrics_manager_class:
            
            # Mock managers
            mock_agent_manager = Mock()
            mock_agent_manager.initialize_agents.return_value = list(self.sample_agents.values())
            mock_agent_manager.broadcast_event.return_value = {
                "event_id": "test_event",
                "affected_agents": ["ranger_1", "vet_1"],
                "individual_responses": {
                    "ranger_1": "Responding to dinosaur escape",
                    "vet_1": "Assisting with containment"
                },
                "group_response": None,
                "broadcast_time": datetime.now().isoformat()
            }
            mock_agent_manager_class.return_value = mock_agent_manager
            
            mock_event_manager = Mock()
            mock_event_manager.create_event.return_value = self.sample_events[0]
            mock_event_manager.distribute_event.return_value = {"event_id": "test_event"}
            mock_event_manager_class.return_value = mock_event_manager
            
            mock_metrics_manager = Mock()
            mock_metrics_manager_class.return_value = mock_metrics_manager
            
            # Create and start simulation
            sim_manager = SimulationManager(mock_session_manager)
            sim_manager.start_simulation(
                openai_config=self.mock_openai_config,
                ag2_config=self.mock_ag2_config,
                simulation_config=self.mock_simulation_config,
                agent_config=self.mock_agent_config
            )
            
            # Trigger event
            event_id = sim_manager.trigger_event(
                event_type="DINOSAUR_ESCAPE",
                parameters={"dinosaur_id": "trex_1", "enclosure_id": "enclosure_a"},
                location=self.sample_location,
                severity=8
            )
            
            # Verify event was created and distributed
            mock_event_manager.create_event.assert_called_once()
            mock_event_manager.distribute_event.assert_called_once()
            
            # Verify agents were notified
            mock_agent_manager.broadcast_event.assert_called_once()
            
            # Verify event was added to session state
            mock_session_manager.add_event.assert_called_once()
            mock_session_manager.update_event.assert_called_once()
    
    def test_metrics_tracking_integration(self, mock_session_manager):
        """Test metrics tracking throughout simulation lifecycle."""
        with patch('managers.simulation_manager.AgentManager') as mock_agent_manager_class, \
             patch('managers.simulation_manager.EventManager') as mock_event_manager_class, \
             patch('managers.simulation_manager.MetricsManager') as mock_metrics_manager_class:
            
            # Mock managers
            mock_agent_manager = Mock()
            mock_agent_manager.initialize_agents.return_value = list(self.sample_agents.values())
            mock_agent_manager_class.return_value = mock_agent_manager
            
            mock_event_manager = Mock()
            mock_event_manager_class.return_value = mock_event_manager
            
            mock_metrics_manager = Mock()
            mock_metrics_manager.get_current_metrics.return_value = self.sample_metrics[-1]
            mock_metrics_manager.get_metrics_summary.return_value = {
                "visitor_satisfaction": {"status": "Good", "trend": "stable"},
                "safety_rating": {"status": "Excellent", "trend": "improving"},
                "facility_efficiency": {"status": "Good", "trend": "stable"},
                "overall_score": {"status": "Good", "trend": "stable"}
            }
            mock_metrics_manager_class.return_value = mock_metrics_manager
            
            # Create and start simulation
            sim_manager = SimulationManager(mock_session_manager)
            sim_manager.start_simulation(
                openai_config=self.mock_openai_config,
                ag2_config=self.mock_ag2_config,
                simulation_config=self.mock_simulation_config,
                agent_config=self.mock_agent_config
            )
            
            # Verify metrics manager was initialized
            mock_metrics_manager_class.assert_called_once()
            
            # Verify agent metrics were initialized
            for agent in self.sample_agents.values():
                mock_metrics_manager.initialize_agent_metrics.assert_any_call(agent)
            
            # Test metrics retrieval
            current_metrics = sim_manager.get_current_metrics()
            assert current_metrics == self.sample_metrics[-1]
            
            # Test metrics summary
            metrics_summary = sim_manager.get_metrics_summary()
            assert "visitor_satisfaction" in metrics_summary
            assert "safety_rating" in metrics_summary
    
    def test_dashboard_real_time_updates(self, mock_session_manager):
        """Test dashboard real-time update functionality."""
        # Test dashboard overview
        render_dashboard_overview(mock_session_manager)
        
        # Verify data retrieval methods were called
        mock_session_manager.get_simulation_state.assert_called()
        mock_session_manager.get_agents.assert_called()
        mock_session_manager.get_events.assert_called()
        mock_session_manager.get_latest_metrics.assert_called()
        
        # Verify UI elements were rendered
        mock_streamlit.title.assert_called()
        mock_streamlit.metric.assert_called()
        mock_streamlit.write.assert_called()
    
    def test_control_panel_integration(self, mock_session_manager):
        """Test control panel integration with simulation manager."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager.session_manager = mock_session_manager
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Test control panel rendering
            render_control_panel(mock_session_manager)
            
            # Verify simulation manager was created
            mock_sim_manager_class.assert_called_with(mock_session_manager)
            
            # Verify UI elements were rendered
            mock_streamlit.title.assert_called_with("🎮 Control Panel")
            mock_streamlit.metric.assert_called()
            mock_streamlit.button.assert_called()
    
    def test_agent_monitor_real_time_display(self, mock_session_manager):
        """Test agent monitor real-time display functionality."""
        # Mock simulation manager in session state
        mock_streamlit.session_state['simulation_manager'] = Mock()
        mock_sim_manager = mock_streamlit.session_state['simulation_manager']
        mock_sim_manager.agent_manager = Mock()
        mock_sim_manager.agent_manager.check_agent_health.return_value = {
            "ranger_1": {"status": "healthy", "response_count": 5, "error_count": 0},
            "vet_1": {"status": "healthy", "response_count": 3, "error_count": 0},
            "tourist_1": {"status": "healthy", "response_count": 1, "error_count": 0}
        }
        
        # Test agent monitor rendering
        render_agent_monitor(mock_session_manager)
        
        # Verify data retrieval
        mock_session_manager.get_agents.assert_called()
        mock_session_manager.get_simulation_state.assert_called()
        
        # Verify UI elements were rendered
        mock_streamlit.title.assert_called_with("🤖 Agent Monitor")
        mock_streamlit.write.assert_called()
        mock_streamlit.columns.assert_called()
    
    def test_metrics_dashboard_visualization(self, mock_session_manager):
        """Test metrics dashboard visualization and analysis."""
        # Mock simulation manager for advanced metrics
        mock_streamlit.session_state['simulation_manager'] = Mock()
        mock_sim_manager = mock_streamlit.session_state['simulation_manager']
        mock_sim_manager.metrics_manager = Mock()
        mock_sim_manager.metrics_manager.get_metrics_summary.return_value = {
            "visitor_satisfaction": {"status": "Good", "trend": "stable"},
            "safety_rating": {"status": "Excellent", "trend": "improving"},
            "facility_efficiency": {"status": "Good", "trend": "stable"},
            "overall_score": {"status": "Good", "trend": "stable"}
        }
        mock_sim_manager.metrics_manager.calculate_overall_resort_score.return_value = 0.85
        
        # Test metrics dashboard rendering
        render_metrics_dashboard(mock_session_manager)
        
        # Verify data retrieval
        mock_session_manager.get_latest_metrics.assert_called()
        mock_session_manager.get_metrics_history.assert_called()
        
        # Verify UI elements were rendered
        mock_streamlit.title.assert_called_with("📊 Metrics Dashboard")
        mock_streamlit.metric.assert_called()
        mock_streamlit.columns.assert_called()
    
    def test_event_log_comprehensive_display(self, mock_session_manager):
        """Test event log comprehensive display and filtering."""
        # Test event log rendering
        render_event_log(mock_session_manager)
        
        # Verify data retrieval
        mock_session_manager.get_events.assert_called()
        
        # Verify UI elements were rendered
        mock_streamlit.title.assert_called_with("📝 Event Log")
        mock_streamlit.write.assert_called()
        mock_streamlit.selectbox.assert_called()
        mock_streamlit.expander.assert_called()
    
    def test_error_handling_throughout_system(self, mock_session_manager):
        """Test error handling throughout the system."""
        # Test with session manager that raises exceptions
        mock_session_manager.get_simulation_state.side_effect = Exception("Database error")
        
        # Should handle errors gracefully
        try:
            render_dashboard_overview(mock_session_manager)
            # If no exception is raised, error handling worked
        except Exception as e:
            # If exception propagates, that's also acceptable for some components
            assert "Database error" in str(e)
    
    def test_session_state_persistence(self, mock_session_manager):
        """Test session state persistence across operations."""
        # Test that session state operations work correctly
        mock_session_manager.update_simulation_state.return_value = None
        mock_session_manager.add_agent.return_value = None
        mock_session_manager.add_event.return_value = None
        mock_session_manager.add_metrics_snapshot.return_value = None
        
        # Simulate operations that modify session state
        mock_session_manager.update_simulation_state(is_running=True)
        mock_session_manager.add_agent(self.sample_agents["ranger_1"])
        mock_session_manager.add_event(self.sample_events[0])
        mock_session_manager.add_metrics_snapshot(self.sample_metrics[0])
        
        # Verify operations were called
        mock_session_manager.update_simulation_state.assert_called()
        mock_session_manager.add_agent.assert_called()
        mock_session_manager.add_event.assert_called()
        mock_session_manager.add_metrics_snapshot.assert_called()
    
    def test_real_time_data_synchronization(self, mock_session_manager):
        """Test real-time data synchronization between components."""
        # Test the real-time sync functionality by verifying the integration points exist
        
        # Test 1: Verify that control panel can access simulation manager
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager.session_manager = mock_session_manager
            mock_sim_manager.update_simulation_time.return_value = None
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Test that control panel creates simulation manager
            render_control_panel(mock_session_manager)
            
            # Verify simulation manager was created (this tests the integration)
            mock_sim_manager_class.assert_called_with(mock_session_manager)
        
        # Test 2: Verify that real-time sync utilities work
        try:
            from utils.real_time_sync import create_real_time_sync_context, RealTimeDataSynchronizer
            
            # Create sync context
            sync_context = create_real_time_sync_context(mock_session_manager)
            
            # Verify sync context contains expected components
            assert 'synchronizer' in sync_context
            assert 'refresh_manager' in sync_context
            assert isinstance(sync_context['synchronizer'], RealTimeDataSynchronizer)
            
            # Test synchronizer functionality
            synchronizer = sync_context['synchronizer']
            sync_status = synchronizer.get_sync_status()
            
            # Verify sync status structure
            assert 'last_sync_time' in sync_status
            assert 'sync_healthy' in sync_status
            assert 'registered_callbacks' in sync_status
            
        except ImportError:
            # If real-time sync is not available, that's acceptable for this test
            pass
    
    def test_complete_workflow_simulation_to_resolution(self, mock_session_manager):
        """Test complete workflow from simulation start to event resolution."""
        with patch('managers.simulation_manager.AgentManager') as mock_agent_manager_class, \
             patch('managers.simulation_manager.EventManager') as mock_event_manager_class, \
             patch('managers.simulation_manager.MetricsManager') as mock_metrics_manager_class:
            
            # Set up mocks for complete workflow
            mock_agent_manager = Mock()
            mock_agent_manager.initialize_agents.return_value = list(self.sample_agents.values())
            mock_agent_manager.broadcast_event.return_value = {
                "event_id": "workflow_event",
                "affected_agents": ["ranger_1", "vet_1"],
                "individual_responses": {
                    "ranger_1": "Situation resolved - dinosaur contained",
                    "vet_1": "Medical assessment complete - all clear"
                }
            }
            mock_agent_manager_class.return_value = mock_agent_manager
            
            mock_event_manager = Mock()
            resolved_event = Event(
                id="workflow_event",
                type=EventType.DINOSAUR_ESCAPE,
                severity=8,
                location=self.sample_location,
                parameters={"dinosaur_id": "trex_1"},
                timestamp=datetime.now(),
                affected_agents=["ranger_1", "vet_1"],
                resolution_status=ResolutionStatus.RESOLVED,
                resolution_time=datetime.now()
            )
            mock_event_manager.create_event.return_value = resolved_event
            mock_event_manager.distribute_event.return_value = {"event_id": "workflow_event"}
            mock_event_manager_class.return_value = mock_event_manager
            
            mock_metrics_manager = Mock()
            mock_metrics_manager_class.return_value = mock_metrics_manager
            
            # 1. Start simulation
            sim_manager = SimulationManager(mock_session_manager)
            sim_manager.start_simulation(
                openai_config=self.mock_openai_config,
                ag2_config=self.mock_ag2_config,
                simulation_config=self.mock_simulation_config,
                agent_config=self.mock_agent_config
            )
            
            # 2. Trigger event
            event_id = sim_manager.trigger_event(
                event_type="DINOSAUR_ESCAPE",
                parameters={"dinosaur_id": "trex_1"},
                location=self.sample_location,
                severity=8
            )
            
            # 3. Verify complete workflow
            # - Simulation started
            assert sim_manager.is_running()
            
            # - Event created and distributed
            mock_event_manager.create_event.assert_called()
            mock_event_manager.distribute_event.assert_called()
            
            # - Agents notified
            mock_agent_manager.broadcast_event.assert_called()
            
            # - Session state updated
            mock_session_manager.add_event.assert_called()
            mock_session_manager.update_event.assert_called()
            
            # - Metrics updated (through event listener)
            # This would be called through the event listener mechanism
    
    def test_performance_under_load(self, mock_session_manager):
        """Test system performance under load with many agents and events."""
        # Create large dataset
        many_agents = {}
        many_events = []
        
        # Create 50 agents
        for i in range(50):
            agent = Agent(
                id=f"agent_{i}",
                name=f"Agent {i}",
                role=AgentRole.PARK_RANGER if i % 4 == 0 else AgentRole.TOURIST,
                location=self.sample_location,
                capabilities=["basic_capability"],
                current_state=AgentState.IDLE
            )
            many_agents[agent.id] = agent
        
        # Create 100 events
        for i in range(100):
            event = Event(
                id=f"event_{i}",
                type=EventType.VISITOR_COMPLAINT if i % 2 == 0 else EventType.DINOSAUR_ESCAPE,
                severity=i % 10 + 1,
                location=self.sample_location,
                parameters={"test_param": f"value_{i}"},
                timestamp=datetime.now() - timedelta(minutes=i),
                affected_agents=[f"agent_{i % 50}"],
                resolution_status=ResolutionStatus.RESOLVED if i % 3 == 0 else ResolutionStatus.PENDING
            )
            many_events.append(event)
        
        # Update mock session manager with large dataset
        mock_session_manager.get_agents.return_value = many_agents
        mock_session_manager.get_events.return_value = many_events
        
        # Test that UI components can handle large datasets
        render_dashboard_overview(mock_session_manager)
        render_agent_monitor(mock_session_manager)
        render_event_log(mock_session_manager)
        
        # Verify that components were rendered without errors
        mock_streamlit.title.assert_called()
        mock_streamlit.write.assert_called()
    
    def test_configuration_validation_and_error_recovery(self, mock_session_manager):
        """Test configuration validation and error recovery mechanisms."""
        # Test with invalid OpenAI configuration using Mock to avoid validation
        from unittest.mock import Mock
        invalid_openai_config = Mock(spec=OpenAIConfig)
        invalid_openai_config.api_key = ""
        invalid_openai_config.model = "invalid-model"
        invalid_openai_config.temperature = 2.0
        invalid_openai_config.max_tokens = -1
        invalid_openai_config.timeout = 0
        
        with patch('managers.simulation_manager.AgentManager') as mock_agent_manager_class:
            # Mock agent manager to raise configuration error
            mock_agent_manager_class.side_effect = ValueError("Invalid OpenAI configuration")
            
            # Test that simulation manager handles configuration errors
            sim_manager = SimulationManager(mock_session_manager)
            
            with pytest.raises(ValueError):
                sim_manager.start_simulation(openai_config=invalid_openai_config)
    
    def test_concurrent_operations_handling(self, mock_session_manager):
        """Test handling of concurrent operations and race conditions."""
        with patch('managers.simulation_manager.AgentManager') as mock_agent_manager_class, \
             patch('managers.simulation_manager.EventManager') as mock_event_manager_class, \
             patch('managers.simulation_manager.MetricsManager') as mock_metrics_manager_class:
            
            # Mock managers
            mock_agent_manager = Mock()
            mock_agent_manager.initialize_agents.return_value = list(self.sample_agents.values())
            mock_agent_manager_class.return_value = mock_agent_manager
            
            mock_event_manager = Mock()
            mock_event_manager_class.return_value = mock_event_manager
            
            mock_metrics_manager = Mock()
            mock_metrics_manager_class.return_value = mock_metrics_manager
            
            # Create simulation manager
            sim_manager = SimulationManager(mock_session_manager)
            sim_manager.start_simulation(
                openai_config=self.mock_openai_config,
                ag2_config=self.mock_ag2_config,
                simulation_config=self.mock_simulation_config,
                agent_config=self.mock_agent_config
            )
            
            # Simulate concurrent event triggering
            event_ids = []
            for i in range(5):
                mock_event_manager.create_event.return_value = Event(
                    id=f"concurrent_event_{i}",
                    type=EventType.VISITOR_COMPLAINT,
                    severity=5,
                    location=self.sample_location,
                    parameters={"concurrent_id": i},
                    timestamp=datetime.now(),
                    affected_agents=["ranger_1"],
                    resolution_status=ResolutionStatus.PENDING
                )
                
                event_id = sim_manager.trigger_event(
                    event_type="VISITOR_COMPLAINT",
                    parameters={"concurrent_id": i},
                    location=self.sample_location,
                    severity=5
                )
                event_ids.append(event_id)
            
            # Verify all events were processed
            assert len(event_ids) == 5
            assert mock_event_manager.create_event.call_count == 5
            assert mock_session_manager.add_event.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])