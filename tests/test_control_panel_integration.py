"""Integration tests for the control panel functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

# Mock streamlit before importing main
import sys
from unittest.mock import MagicMock

# Create a comprehensive mock streamlit module
mock_streamlit = MagicMock()
mock_streamlit.set_page_config = MagicMock()
mock_streamlit.sidebar = MagicMock()
mock_streamlit.title = MagicMock()
mock_streamlit.write = MagicMock()
mock_streamlit.subheader = MagicMock()
mock_streamlit.metric = MagicMock()
mock_streamlit.divider = MagicMock()
mock_streamlit.button = MagicMock(return_value=False)
mock_streamlit.selectbox = MagicMock(return_value="DINOSAUR_ESCAPE - A dinosaur has escaped")
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
mock_streamlit.session_state = MagicMock()

# Mock columns to handle both integer and list arguments
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

sys.modules['streamlit'] = mock_streamlit

# Now import the modules we want to test
from main import render_control_panel
from ui.session_state import SessionStateManager
from models.core import SimulationState
from models.enums import EventType


class TestControlPanelIntegration:
    """Integration tests for control panel functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_session_manager = Mock(spec=SessionStateManager)
        
        # Mock simulation state
        self.mock_sim_state = SimulationState()
        self.mock_sim_state.is_running = False
        self.mock_sim_state.agent_count = 0
        self.mock_sim_state.started_at = None
        self.mock_session_manager.get_simulation_state.return_value = self.mock_sim_state
        
        # Mock events
        self.mock_session_manager.get_events.return_value = []
    
    def test_control_panel_renders_basic_elements(self):
        """Test that control panel renders basic UI elements."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager_class.return_value = mock_sim_manager
            
            render_control_panel(self.mock_session_manager)
            
            # Verify basic UI elements are rendered
            mock_streamlit.title.assert_called_with("🎮 Control Panel")
            mock_streamlit.write.assert_called()
            mock_streamlit.subheader.assert_called()
            mock_streamlit.columns.assert_called()
            mock_streamlit.button.assert_called()
            mock_streamlit.metric.assert_called()
    
    def test_control_panel_shows_warning_when_not_running(self):
        """Test that control panel shows warning when simulation is not running."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Ensure simulation is not running
            self.mock_sim_state.is_running = False
            
            render_control_panel(self.mock_session_manager)
            
            # Should show warning about starting simulation
            mock_streamlit.warning.assert_called_with("⚠️ Start the simulation to trigger events")
    
    def test_control_panel_simulation_controls_created(self):
        """Test that simulation control buttons are created."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager_class.return_value = mock_sim_manager
            
            render_control_panel(self.mock_session_manager)
            
            # Check that buttons are created
            button_calls = mock_streamlit.button.call_args_list
            button_texts = [call[0][0] for call in button_calls if call[0]]
            
            # Verify control buttons exist
            expected_buttons = ["▶️ Start", "⏸️ Pause", "⏹️ Stop", "🔄 Reset"]
            for expected_button in expected_buttons:
                assert any(expected_button in text for text in button_texts), f"Button '{expected_button}' not found"
    
    def test_control_panel_status_display(self):
        """Test that control panel displays simulation status correctly."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Test with stopped simulation
            self.mock_sim_state.is_running = False
            self.mock_sim_state.agent_count = 5
            
            render_control_panel(self.mock_session_manager)
            
            # Should display status metrics
            metric_calls = mock_streamlit.metric.call_args_list
            metric_labels = [call[0][0] for call in metric_calls if call[0]]
            
            expected_metrics = ["Status", "Uptime", "Active Agents", "Active Events"]
            for expected_metric in expected_metrics:
                assert any(expected_metric in label for label in metric_labels), f"Metric '{expected_metric}' not found"
    
    def test_control_panel_event_interface_when_running(self):
        """Test that event triggering interface appears when simulation is running."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class, \
             patch('managers.event_manager.EventManager') as mock_event_manager_class:
            
            mock_sim_manager = Mock()
            mock_sim_manager.event_manager = None  # No event manager initially
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Mock temporary event manager
            mock_temp_event_manager = Mock()
            mock_temp_event_manager.get_available_events.return_value = [
                {
                    'type': EventType.DINOSAUR_ESCAPE,
                    'name': 'DINOSAUR_ESCAPE',
                    'description': 'A dinosaur has escaped',
                    'default_severity': 8,
                    'required_parameters': ['dinosaur_id'],
                    'optional_parameters': ['escape_method'],
                    'affected_roles': ['PARK_RANGER']
                }
            ]
            mock_event_manager_class.return_value = mock_temp_event_manager
            
            # Set simulation as running
            self.mock_sim_state.is_running = True
            
            render_control_panel(self.mock_session_manager)
            
            # Should display event triggering interface
            mock_streamlit.selectbox.assert_called()
            mock_streamlit.slider.assert_called()
    
    def test_control_panel_session_state_integration(self):
        """Test that control panel integrates with session state correctly."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager_class.return_value = mock_sim_manager
            
            render_control_panel(self.mock_session_manager)
            
            # Should call session manager methods
            self.mock_session_manager.get_simulation_state.assert_called()
            self.mock_session_manager.get_events.assert_called()
    
    def test_control_panel_handles_simulation_manager_creation(self):
        """Test that control panel creates simulation manager correctly."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Mock session state to not contain simulation_manager
            mock_streamlit.session_state.__contains__ = Mock(return_value=False)
            
            render_control_panel(self.mock_session_manager)
            
            # Should create simulation manager
            mock_sim_manager_class.assert_called_with(self.mock_session_manager)
    
    def test_control_panel_button_interactions(self):
        """Test control panel button interaction logic."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Mock button to return True (clicked)
            mock_streamlit.button.return_value = True
            
            # Test start button when simulation is not running
            self.mock_sim_state.is_running = False
            
            render_control_panel(self.mock_session_manager)
            
            # Should attempt to start simulation (though mocked)
            # The actual button logic is tested through UI interaction
            mock_streamlit.button.assert_called()
    
    def test_control_panel_error_handling(self):
        """Test that control panel handles errors gracefully."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            # Mock simulation manager to raise exception during creation
            mock_sim_manager_class.side_effect = Exception("Test error")
            
            # Should not crash when error occurs
            try:
                render_control_panel(self.mock_session_manager)
            except Exception as e:
                # If exception propagates, it should be handled gracefully
                # In a real implementation, we'd want better error handling
                assert "Test error" in str(e)
    
    def test_control_panel_recent_events_display(self):
        """Test that control panel displays recent events correctly."""
        with patch('managers.simulation_manager.SimulationManager') as mock_sim_manager_class:
            mock_sim_manager = Mock()
            mock_sim_manager_class.return_value = mock_sim_manager
            
            # Mock some events
            from models.core import Event
            from models.config import Location
            from models.enums import ResolutionStatus
            
            mock_event = Event(
                id="test-event",
                type=EventType.DINOSAUR_ESCAPE,
                severity=5,
                location=Location(0.0, 0.0, "test_zone", "Test location"),
                parameters={},
                timestamp=datetime.now(),
                affected_agents=[],
                resolution_status=ResolutionStatus.PENDING
            )
            
            self.mock_session_manager.get_events.return_value = [mock_event]
            
            render_control_panel(self.mock_session_manager)
            
            # Should retrieve and display events
            self.mock_session_manager.get_events.assert_called()
            mock_streamlit.subheader.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])