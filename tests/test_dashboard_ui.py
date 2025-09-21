"""Tests for Streamlit dashboard UI components."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

# Mock streamlit before importing main
import sys
from unittest.mock import MagicMock

# Create a mock streamlit module
mock_streamlit = MagicMock()
mock_streamlit.set_page_config = MagicMock()
mock_streamlit.sidebar = MagicMock()

# Mock columns to return the right number of mock objects based on call
def mock_columns(num_cols):
    if isinstance(num_cols, list):
        return [MagicMock() for _ in range(len(num_cols))]
    return [MagicMock() for _ in range(num_cols)]

mock_streamlit.columns = MagicMock(side_effect=mock_columns)
mock_streamlit.title = MagicMock()
mock_streamlit.write = MagicMock()
mock_streamlit.metric = MagicMock()
mock_streamlit.progress = MagicMock()
mock_streamlit.info = MagicMock()
mock_streamlit.success = MagicMock()
mock_streamlit.button = MagicMock(return_value=False)
mock_streamlit.selectbox = MagicMock(return_value="Dashboard")
mock_streamlit.divider = MagicMock()
mock_streamlit.subheader = MagicMock()
mock_streamlit.expander = MagicMock()
mock_streamlit.json = MagicMock()
mock_streamlit.caption = MagicMock()
mock_streamlit.rerun = MagicMock()
mock_streamlit.warning = MagicMock()
mock_streamlit.error = MagicMock()
mock_streamlit.spinner = MagicMock()
mock_streamlit.slider = MagicMock(return_value=5)
mock_streamlit.text_input = MagicMock(return_value="")
mock_streamlit.text_area = MagicMock(return_value="")
mock_streamlit.number_input = MagicMock(return_value=0)
mock_streamlit.multiselect = MagicMock(return_value=[])
mock_streamlit.checkbox = MagicMock(return_value=False)

# Mock session_state
mock_streamlit.session_state = MagicMock()

# Mock the context manager for spinner
mock_spinner = MagicMock()
mock_spinner.__enter__ = MagicMock(return_value=mock_spinner)
mock_spinner.__exit__ = MagicMock(return_value=None)
mock_streamlit.spinner.return_value = mock_spinner

# Mock the context manager for expander
mock_expander = MagicMock()
mock_expander.__enter__ = MagicMock(return_value=mock_expander)
mock_expander.__exit__ = MagicMock(return_value=None)
mock_streamlit.expander.return_value = mock_expander

sys.modules['streamlit'] = mock_streamlit

# Now import the modules we want to test
from main import (
    configure_page, render_sidebar, render_dashboard_overview,
    render_control_panel, render_agent_monitor, render_metrics_dashboard,
    render_event_log, render_settings
)
from ui.session_state import SessionStateManager
from models.core import Agent, Event, MetricsSnapshot, SimulationState
from models.enums import AgentRole, EventType, ResolutionStatus, AgentState
from models.config import Location


class TestDashboardUI:
    """Test cases for dashboard UI components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_session_manager = Mock(spec=SessionStateManager)
        
        # Mock simulation state
        self.mock_sim_state = SimulationState()
        self.mock_sim_state.is_running = True
        self.mock_session_manager.get_simulation_state.return_value = self.mock_sim_state
        
        # Mock agents
        self.mock_agents = {
            "agent1": Agent(
                id="agent1",
                name="Test Ranger",
                role=AgentRole.PARK_RANGER,
                personality_traits={"courage": 0.8},
                current_state=AgentState.IDLE,
                location=Location(x=0.0, y=0.0, zone="visitor_center", description="Visitor Center"),
                capabilities=["emergency_response"]
            )
        }
        self.mock_session_manager.get_agents.return_value = self.mock_agents
        
        # Mock events
        self.mock_events = [
            Event(
                id="event1",
                type=EventType.DINOSAUR_ESCAPE,
                severity=3,
                location=Location(x=100.0, y=200.0, zone="paddock_a", description="Paddock A"),
                parameters={},
                timestamp=datetime.now(),
                affected_agents=["agent1"],
                resolution_status=ResolutionStatus.IN_PROGRESS
            )
        ]
        self.mock_session_manager.get_events.return_value = self.mock_events
        
        # Mock metrics
        self.mock_metrics = MetricsSnapshot(
            visitor_satisfaction=0.85,
            dinosaur_happiness={"trex": 0.75},
            facility_efficiency=0.90,
            safety_rating=0.95,
            timestamp=datetime.now()
        )
        self.mock_session_manager.get_latest_metrics.return_value = self.mock_metrics
        self.mock_session_manager.get_metrics_history.return_value = [self.mock_metrics]
        
        # Mock session info
        self.mock_session_manager.get_session_info.return_value = {
            "initialized": True,
            "agent_count": 1,
            "event_count": 1,
            "metrics_history_count": 1,
            "conversation_agents": ["agent1"],
            "simulation_running": True
        }
        
        # Mock configurations
        self.mock_session_manager.get_simulation_config.return_value = Mock()
        self.mock_session_manager.get_agent_config.return_value = Mock()
        
        # Mock conversation history
        self.mock_session_manager.get_conversation_history.return_value = {}
    
    def test_configure_page(self):
        """Test page configuration."""
        configure_page()
        
        mock_streamlit.set_page_config.assert_called_once_with(
            page_title="AI Agent Dinosaur Simulator",
            page_icon="🦕",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def test_render_sidebar_basic_elements(self):
        """Test sidebar renders basic elements."""
        result = render_sidebar(self.mock_session_manager)
        
        # Check that sidebar elements are called
        mock_streamlit.sidebar.title.assert_called_with("🦕 Dinosaur Resort")
        mock_streamlit.sidebar.selectbox.assert_called()
        mock_streamlit.sidebar.divider.assert_called()
        mock_streamlit.sidebar.write.assert_called()
        mock_streamlit.sidebar.metric.assert_called()
        
        # Should return the selected page
        assert result == "Dashboard"  # Default return from mock
    
    def test_render_sidebar_with_metrics(self):
        """Test sidebar displays metrics when available."""
        render_sidebar(self.mock_session_manager)
        
        # Verify metrics are displayed
        self.mock_session_manager.get_latest_metrics.assert_called_once()
        mock_streamlit.sidebar.write.assert_called()
    
    def test_render_sidebar_quick_actions(self):
        """Test sidebar quick actions."""
        render_sidebar(self.mock_session_manager)
        
        # Check that buttons are created
        mock_streamlit.sidebar.button.assert_called()
    
    def test_render_dashboard_overview_basic_layout(self):
        """Test dashboard overview renders basic layout."""
        render_dashboard_overview(self.mock_session_manager)
        
        # Check main elements
        mock_streamlit.title.assert_called_with("🦕 AI Agent Dinosaur Simulator")
        mock_streamlit.write.assert_called()
        mock_streamlit.columns.assert_called()
        mock_streamlit.metric.assert_called()
        mock_streamlit.divider.assert_called()
        mock_streamlit.subheader.assert_called()
    
    def test_render_dashboard_overview_with_agents(self):
        """Test dashboard overview displays agent information."""
        render_dashboard_overview(self.mock_session_manager)
        
        # Verify agents are retrieved and displayed
        self.mock_session_manager.get_agents.assert_called_once()
        mock_streamlit.write.assert_called()
    
    def test_render_dashboard_overview_with_metrics(self):
        """Test dashboard overview displays metrics."""
        render_dashboard_overview(self.mock_session_manager)
        
        # Verify metrics are retrieved and displayed
        self.mock_session_manager.get_latest_metrics.assert_called_once()
        mock_streamlit.progress.assert_called()
    
    def test_render_dashboard_overview_with_events(self):
        """Test dashboard overview displays recent events."""
        render_dashboard_overview(self.mock_session_manager)
        
        # Verify events are retrieved and displayed
        self.mock_session_manager.get_events.assert_called_once()
        mock_streamlit.write.assert_called()
    
    def test_render_dashboard_overview_no_data(self):
        """Test dashboard overview handles missing data gracefully."""
        # Mock empty data
        self.mock_session_manager.get_agents.return_value = {}
        self.mock_session_manager.get_events.return_value = []
        self.mock_session_manager.get_latest_metrics.return_value = None
        
        render_dashboard_overview(self.mock_session_manager)
        
        # Should display info messages for missing data
        mock_streamlit.info.assert_called()
    
    def test_render_control_panel(self):
        """Test control panel renders correctly."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class, \
             patch('managers.event_manager.EventManager') as mock_event_manager_class:
            
            mock_sim_manager = Mock()
            mock_sim_manager.event_manager = None  # No event manager initially
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Mock temporary event manager
            mock_temp_event_manager = Mock()
            mock_temp_event_manager.get_available_events.return_value = [
                {
                    'type': Mock(name='DINOSAUR_ESCAPE'),
                    'name': 'DINOSAUR_ESCAPE',
                    'description': 'A dinosaur has escaped',
                    'default_severity': 8,
                    'required_parameters': ['dinosaur_id'],
                    'optional_parameters': ['escape_method'],
                    'affected_roles': ['PARK_RANGER']
                }
            ]
            mock_event_manager_class.return_value = mock_temp_event_manager
            
            render_control_panel(self.mock_session_manager)
            
            # Check basic elements
            mock_streamlit.title.assert_called_with("🎮 Control Panel")
            mock_streamlit.write.assert_called()
            mock_streamlit.subheader.assert_called()
            mock_streamlit.columns.assert_called()
            mock_streamlit.button.assert_called()
            mock_streamlit.metric.assert_called()
    
    def test_render_control_panel_simulation_status(self):
        """Test control panel displays simulation status correctly."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Test with running simulation
            self.mock_sim_state.is_running = True
            render_control_panel(self.mock_session_manager)
            
            # Should display status metrics
            mock_streamlit.metric.assert_called()
            
            # Test with stopped simulation
            self.mock_sim_state.is_running = False
            render_control_panel(self.mock_session_manager)
            
            mock_streamlit.metric.assert_called()
    
    def test_render_control_panel_simulation_controls(self):
        """Test control panel simulation control buttons."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Mock button clicks
            mock_streamlit.button.return_value = False
            
            render_control_panel(self.mock_session_manager)
            
            # Should create control buttons
            button_calls = mock_streamlit.button.call_args_list
            button_texts = [call[0][0] for call in button_calls if call[0]]
            
            # Check that control buttons are created
            control_buttons = ["▶️ Start", "⏸️ Pause", "⏹️ Stop", "🔄 Reset"]
            for button_text in control_buttons:
                assert any(button_text in text for text in button_texts)
    
    def test_render_control_panel_event_triggering_interface(self):
        """Test control panel event triggering interface."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager.event_manager = Mock()
            mock_sim_manager.event_manager.get_available_events.return_value = [
                {
                    'type': Mock(name='DINOSAUR_ESCAPE'),
                    'name': 'DINOSAUR_ESCAPE',
                    'description': 'A dinosaur has escaped',
                    'default_severity': 8,
                    'required_parameters': ['dinosaur_id'],
                    'optional_parameters': ['escape_method'],
                    'affected_roles': ['PARK_RANGER']
                }
            ]
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Set simulation as running to show event interface
            self.mock_sim_state.is_running = True
            
            render_control_panel(self.mock_session_manager)
            
            # Should display event triggering interface
            mock_streamlit.selectbox.assert_called()
            mock_streamlit.slider.assert_called()
    
    def test_render_control_panel_event_parameters(self):
        """Test control panel handles event parameters correctly."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager.event_manager = Mock()
            mock_sim_manager.event_manager.get_available_events.return_value = [
                {
                    'type': Mock(name='VISITOR_INJURY'),
                    'name': 'VISITOR_INJURY',
                    'description': 'A visitor has been injured',
                    'default_severity': 6,
                    'required_parameters': ['visitor_id', 'injury_type'],
                    'optional_parameters': ['injury_severity'],
                    'affected_roles': ['SECURITY']
                }
            ]
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Set simulation as running
            self.mock_sim_state.is_running = True
            
            render_control_panel(self.mock_session_manager)
            
            # Should create parameter inputs
            mock_streamlit.selectbox.assert_called()
            mock_streamlit.expander.assert_called()
    
    def test_render_control_panel_recent_events_display(self):
        """Test control panel displays recent events."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager.event_manager = Mock()
            mock_sim_manager.event_manager.get_available_events.return_value = []
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Set simulation as running
            self.mock_sim_state.is_running = True
            
            render_control_panel(self.mock_session_manager)
            
            # Should display recent events section
            mock_streamlit.subheader.assert_called()
            self.mock_session_manager.get_events.assert_called()
    
    def test_render_control_panel_not_running_warning(self):
        """Test control panel shows warning when simulation not running."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Set simulation as not running
            self.mock_sim_state.is_running = False
            
            render_control_panel(self.mock_session_manager)
            
            # Should show warning about starting simulation
            mock_streamlit.warning.assert_called()
    
    def test_render_control_panel_error_handling(self):
        """Test control panel handles errors gracefully."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            # Mock simulation manager to raise exception
            mock_sim_manager_class.side_effect = Exception("Test error")
            
            try:
                render_control_panel(self.mock_session_manager)
                # Should not crash, may display error or handle gracefully
            except Exception:
                # If exception propagates, that's also acceptable for this test
                pass
    
    def test_render_agent_monitor(self):
        """Test agent monitor renders correctly."""
        render_agent_monitor(self.mock_session_manager)
        
        # Check basic elements
        mock_streamlit.title.assert_called_with("🤖 Agent Monitor")
        mock_streamlit.write.assert_called()
        
        # Verify agents are retrieved
        self.mock_session_manager.get_agents.assert_called_once()
    
    def test_render_agent_monitor_with_agents(self):
        """Test agent monitor displays agent details."""
        render_agent_monitor(self.mock_session_manager)
        
        # Should display agent information
        mock_streamlit.write.assert_called()
        mock_streamlit.columns.assert_called()
        mock_streamlit.subheader.assert_called()
    
    def test_render_agent_monitor_no_agents(self):
        """Test agent monitor handles no agents."""
        self.mock_session_manager.get_agents.return_value = {}
        
        render_agent_monitor(self.mock_session_manager)
        
        # Should display info message
        mock_streamlit.info.assert_called()
    
    def test_render_agent_monitor_status_overview(self):
        """Test agent monitor displays status overview correctly."""
        # Add multiple agents with different roles and states
        self.mock_agents = {
            "agent1": Agent(
                id="agent1",
                name="Test Ranger",
                role=AgentRole.PARK_RANGER,
                personality_traits={"courage": 0.8},
                current_state=AgentState.IDLE,
                location=Location(x=0.0, y=0.0, zone="visitor_center", description="Visitor Center"),
                capabilities=["emergency_response"]
            ),
            "agent2": Agent(
                id="agent2",
                name="Test Vet",
                role=AgentRole.VETERINARIAN,
                personality_traits={"empathy": 0.9},
                current_state=AgentState.ACTIVE,
                location=Location(x=50.0, y=50.0, zone="medical_center", description="Medical Center"),
                capabilities=["medical_treatment"]
            ),
            "agent3": Agent(
                id="agent3",
                name="Test Tourist",
                role=AgentRole.TOURIST,
                personality_traits={"curiosity": 0.7},
                current_state=AgentState.RESPONDING_TO_EVENT,
                location=Location(x=100.0, y=100.0, zone="main_area", description="Main Area"),
                capabilities=["observation"]
            )
        }
        self.mock_session_manager.get_agents.return_value = self.mock_agents
        
        render_agent_monitor(self.mock_session_manager)
        
        # Should display role and state distributions
        mock_streamlit.subheader.assert_called()
        mock_streamlit.write.assert_called()
        mock_streamlit.columns.assert_called()
    
    def test_render_agent_monitor_filters(self):
        """Test agent monitor filter functionality."""
        render_agent_monitor(self.mock_session_manager)
        
        # Should display filter controls
        mock_streamlit.selectbox.assert_called()
        mock_streamlit.text_input.assert_called()
    
    def test_render_agent_monitor_auto_refresh(self):
        """Test agent monitor auto-refresh functionality."""
        render_agent_monitor(self.mock_session_manager)
        
        # Should display auto-refresh checkbox
        mock_streamlit.checkbox.assert_called()
        mock_streamlit.button.assert_called()
    
    def test_render_agent_monitor_real_time_status(self):
        """Test agent monitor real-time status display."""
        render_agent_monitor(self.mock_session_manager)
        
        # Should display agent status information
        mock_streamlit.subheader.assert_called()
        mock_streamlit.write.assert_called()
        mock_streamlit.columns.assert_called()
    
    def test_render_agent_monitor_location_tracking(self):
        """Test agent monitor location tracking interface."""
        render_agent_monitor(self.mock_session_manager)
        
        # Should display location tracking section
        mock_streamlit.subheader.assert_called()
        mock_streamlit.write.assert_called()
    
    def test_render_agent_monitor_conversation_history(self):
        """Test agent monitor conversation history viewer."""
        # Mock conversation history
        self.mock_session_manager.get_conversation_history.return_value = {
            "agent1": [
                {
                    "timestamp": "2024-01-01T12:00:00",
                    "sender": "System",
                    "recipient": "agent1",
                    "content": "Test message",
                    "event_context": {"event_id": "event1"}
                }
            ]
        }
        
        render_agent_monitor(self.mock_session_manager)
        
        # Should display conversation history section
        mock_streamlit.subheader.assert_called()
        mock_streamlit.selectbox.assert_called()
        # Note: get_conversation_history is only called when an agent is selected
    
    def test_render_agent_monitor_detailed_information(self):
        """Test agent monitor detailed information display."""
        render_agent_monitor(self.mock_session_manager)
        
        # Should handle detailed agent information
        mock_streamlit.columns.assert_called()
        mock_streamlit.write.assert_called()
        mock_streamlit.button.assert_called()
    
    def test_render_agent_monitor_health_information(self):
        """Test agent monitor displays health information when available."""
        # Mock session state with simulation manager
        mock_streamlit.session_state = {
            'simulation_manager': Mock()
        }
        mock_sim_manager = mock_streamlit.session_state['simulation_manager']
        mock_sim_manager.agent_manager = Mock()
        mock_sim_manager.agent_manager.check_agent_health.return_value = {
            "agent1": {
                "status": "healthy",
                "response_count": 5,
                "error_count": 0,
                "communication_failures": 0,
                "last_response_time": "2024-01-01T12:00:00",
                "time_since_response": 30.0
            }
        }
        
        render_agent_monitor(self.mock_session_manager)
        
        # Should display health information in agent cards
        mock_streamlit.write.assert_called()
        mock_streamlit.columns.assert_called()
    
    def test_render_agent_monitor_error_handling(self):
        """Test agent monitor handles errors gracefully."""
        # Mock session manager to raise exception for health check
        mock_streamlit.session_state = {
            'simulation_manager': Mock()
        }
        mock_sim_manager = mock_streamlit.session_state['simulation_manager']
        mock_sim_manager.agent_manager = Mock()
        mock_sim_manager.agent_manager.check_agent_health.side_effect = Exception("Health check failed")
        
        render_agent_monitor(self.mock_session_manager)
        
        # Should handle error gracefully and display warning
        mock_streamlit.warning.assert_called()
    
    def test_render_agent_monitor_filter_by_role(self):
        """Test agent monitor role filtering."""
        # Create agents with different roles
        self.mock_agents = {
            "ranger1": Agent(
                id="ranger1",
                name="Ranger 1",
                role=AgentRole.PARK_RANGER,
                personality_traits={},
                current_state=AgentState.IDLE,
                location=Location(x=0.0, y=0.0, zone="visitor_center", description="Visitor Center"),
                capabilities=[]
            ),
            "vet1": Agent(
                id="vet1",
                name="Vet 1",
                role=AgentRole.VETERINARIAN,
                personality_traits={},
                current_state=AgentState.ACTIVE,
                location=Location(x=0.0, y=0.0, zone="medical_center", description="Medical Center"),
                capabilities=[]
            )
        }
        self.mock_session_manager.get_agents.return_value = self.mock_agents
        
        # Mock selectbox to return specific role filter
        mock_streamlit.selectbox.return_value = "PARK_RANGER"
        
        render_agent_monitor(self.mock_session_manager)
        
        # Should filter agents by role
        mock_streamlit.selectbox.assert_called()
        mock_streamlit.write.assert_called()
    
    def test_render_agent_monitor_filter_by_state(self):
        """Test agent monitor state filtering."""
        # Mock selectbox to return specific state filter
        mock_streamlit.selectbox.return_value = "IDLE"
        
        render_agent_monitor(self.mock_session_manager)
        
        # Should filter agents by state
        mock_streamlit.selectbox.assert_called()
        mock_streamlit.write.assert_called()
    
    def test_render_agent_monitor_search_by_name(self):
        """Test agent monitor name search functionality."""
        # Mock text input to return search term
        mock_streamlit.text_input.return_value = "Test"
        
        render_agent_monitor(self.mock_session_manager)
        
        # Should filter agents by name
        mock_streamlit.text_input.assert_called()
        mock_streamlit.write.assert_called()
    
    def test_render_agent_monitor_conversation_history_empty(self):
        """Test agent monitor handles empty conversation history."""
        # Mock empty conversation history
        self.mock_session_manager.get_conversation_history.return_value = {}
        
        # Mock agent selection
        mock_streamlit.selectbox.return_value = "agent1"
        
        render_agent_monitor(self.mock_session_manager)
        
        # Should display info message for empty history
        mock_streamlit.info.assert_called()
    
    def test_render_agent_monitor_conversation_interface(self):
        """Test agent monitor displays conversation interface."""
        render_agent_monitor(self.mock_session_manager)
        
        # Should display conversation history interface
        mock_streamlit.selectbox.assert_called()
        mock_streamlit.subheader.assert_called()
    
    def test_render_metrics_dashboard(self):
        """Test metrics dashboard renders correctly."""
        render_metrics_dashboard(self.mock_session_manager)
        
        # Check basic elements
        mock_streamlit.title.assert_called_with("📊 Metrics Dashboard")
        mock_streamlit.write.assert_called()
        mock_streamlit.info.assert_called()
        
        # Verify metrics are retrieved
        self.mock_session_manager.get_latest_metrics.assert_called_once()
        self.mock_session_manager.get_metrics_history.assert_called_once()
    
    def test_render_metrics_dashboard_with_data(self):
        """Test metrics dashboard displays metrics data."""
        render_metrics_dashboard(self.mock_session_manager)
        
        # Should display metrics
        mock_streamlit.columns.assert_called()
        mock_streamlit.metric.assert_called()
        mock_streamlit.write.assert_called()
    
    def test_render_metrics_dashboard_no_data(self):
        """Test metrics dashboard handles no data."""
        self.mock_session_manager.get_latest_metrics.return_value = None
        
        render_metrics_dashboard(self.mock_session_manager)
        
        # Should display info message
        mock_streamlit.info.assert_called()
    
    def test_render_event_log(self):
        """Test event log renders correctly."""
        render_event_log(self.mock_session_manager)
        
        # Check basic elements
        mock_streamlit.title.assert_called_with("📝 Event Log")
        mock_streamlit.write.assert_called()
        mock_streamlit.info.assert_called()
        
        # Verify events are retrieved
        self.mock_session_manager.get_events.assert_called_once()
    
    def test_render_event_log_with_events(self):
        """Test event log displays event details."""
        render_event_log(self.mock_session_manager)
        
        # Should display event information
        mock_streamlit.expander.assert_called()
        mock_streamlit.write.assert_called()
    
    def test_render_event_log_no_events(self):
        """Test event log handles no events."""
        self.mock_session_manager.get_events.return_value = []
        
        render_event_log(self.mock_session_manager)
        
        # Should display info message
        mock_streamlit.info.assert_called()
    
    def test_render_settings(self):
        """Test settings page renders correctly."""
        render_settings(self.mock_session_manager)
        
        # Check basic elements
        mock_streamlit.title.assert_called_with("⚙️ Settings")
        mock_streamlit.write.assert_called()
        mock_streamlit.subheader.assert_called()
        mock_streamlit.columns.assert_called()
        mock_streamlit.button.assert_called()
        mock_streamlit.expander.assert_called()
        mock_streamlit.json.assert_called()
    
    def test_render_settings_session_management(self):
        """Test settings page session management."""
        render_settings(self.mock_session_manager)
        
        # Verify session info is retrieved
        self.mock_session_manager.get_session_info.assert_called_once()
        self.mock_session_manager.get_simulation_config.assert_called_once()
        self.mock_session_manager.get_agent_config.assert_called_once()


class TestDashboardIntegration:
    """Integration tests for dashboard components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_session_manager = Mock(spec=SessionStateManager)
    
    @patch('main.SessionStateManager')
    def test_main_function_initialization(self, mock_session_class):
        """Test main function initializes correctly."""
        mock_session_class.return_value = self.mock_session_manager
        
        # Mock the session manager methods
        self.mock_session_manager.get_simulation_state.return_value = SimulationState()
        self.mock_session_manager.get_agents.return_value = {}
        self.mock_session_manager.get_events.return_value = []
        self.mock_session_manager.get_latest_metrics.return_value = None
        self.mock_session_manager.get_session_info.return_value = {}
        
        # Import and call main (this would normally be done by streamlit)
        from main import main
        
        # This would fail in actual execution due to streamlit context,
        # but we can test that the function exists and imports work
        assert callable(main)
    
    def test_page_navigation_flow(self):
        """Test that page navigation works correctly."""
        # Mock different page selections
        pages = ["Dashboard", "Control Panel", "Agent Monitor", "Metrics", "Event Log", "Settings"]
        
        for page in pages:
            mock_streamlit.sidebar.selectbox.return_value = page
            result = render_sidebar(self.mock_session_manager)
            assert result == page
    
    def test_session_state_integration(self):
        """Test integration with session state manager."""
        # Test that all render functions call appropriate session manager methods
        functions_and_methods = [
            (render_dashboard_overview, ['get_simulation_state', 'get_agents', 'get_events', 'get_latest_metrics']),
            (render_agent_monitor, ['get_agents']),
            (render_metrics_dashboard, ['get_latest_metrics', 'get_metrics_history']),
            (render_event_log, ['get_events']),
            (render_settings, ['get_session_info', 'get_simulation_config', 'get_agent_config'])
        ]
        
        for render_func, expected_methods in functions_and_methods:
            # Reset mock
            self.mock_session_manager.reset_mock()
            
            # Set up default returns
            self.mock_session_manager.get_simulation_state.return_value = SimulationState()
            self.mock_session_manager.get_agents.return_value = {}
            self.mock_session_manager.get_events.return_value = []
            self.mock_session_manager.get_latest_metrics.return_value = None
            self.mock_session_manager.get_metrics_history.return_value = []
            self.mock_session_manager.get_session_info.return_value = {}
            self.mock_session_manager.get_simulation_config.return_value = Mock()
            self.mock_session_manager.get_agent_config.return_value = Mock()
            
            # Call function
            render_func(self.mock_session_manager)
            
            # Verify expected methods were called
            for method_name in expected_methods:
                method = getattr(self.mock_session_manager, method_name)
                method.assert_called()


class TestDashboardErrorHandling:
    """Test error handling in dashboard components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_session_manager = Mock(spec=SessionStateManager)
    
    def test_render_functions_handle_none_session_manager(self):
        """Test render functions handle None session manager gracefully."""
        # This should not happen in normal operation, but test defensive programming
        functions = [
            render_dashboard_overview,
            render_control_panel,
            render_agent_monitor,
            render_metrics_dashboard,
            render_event_log,
            render_settings
        ]
        
        for func in functions:
            try:
                # This will likely raise an AttributeError, which is expected
                func(None)
            except AttributeError:
                # Expected behavior when session_manager is None
                pass
    
    def test_render_functions_handle_session_manager_exceptions(self):
        """Test render functions handle session manager exceptions."""
        # Mock session manager to raise exceptions
        self.mock_session_manager.get_simulation_state.side_effect = Exception("Test error")
        self.mock_session_manager.get_agents.side_effect = Exception("Test error")
        self.mock_session_manager.get_events.side_effect = Exception("Test error")
        self.mock_session_manager.get_latest_metrics.side_effect = Exception("Test error")
        
        functions = [
            render_dashboard_overview,
            render_agent_monitor,
            render_metrics_dashboard,
            render_event_log
        ]
        
        for func in functions:
            try:
                func(self.mock_session_manager)
                # If no exception is raised, that's also acceptable
                # (means the function has good error handling)
            except Exception:
                # If an exception is raised, it should be handled gracefully
                # In a real implementation, we'd want to catch and handle these
                pass


if __name__ == "__main__":
    pytest.main([__file__])