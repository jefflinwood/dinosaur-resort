"""Real-time data synchronization utilities for the AI Agent Dinosaur Simulator."""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
from threading import Lock
import streamlit as st

from ui.session_state import SessionStateManager
from managers.simulation_manager import SimulationManager


class RealTimeDataSynchronizer:
    """Manages real-time data synchronization between simulation components and UI."""
    
    def __init__(self, session_manager: SessionStateManager):
        """Initialize the real-time data synchronizer.
        
        Args:
            session_manager: Session state manager instance
        """
        self.session_manager = session_manager
        self.logger = logging.getLogger(__name__)
        self._sync_lock = Lock()
        self._last_sync_time = datetime.now()
        self._sync_callbacks: List[Callable] = []
        self._error_count = 0
        self._max_errors = 10
        
    def register_sync_callback(self, callback: Callable) -> None:
        """Register a callback to be called during synchronization.
        
        Args:
            callback: Function to call during sync
        """
        self._sync_callbacks.append(callback)
        self.logger.debug(f"Registered sync callback: {callback.__name__}")
    
    def unregister_sync_callback(self, callback: Callable) -> None:
        """Unregister a sync callback.
        
        Args:
            callback: Function to remove from sync callbacks
        """
        if callback in self._sync_callbacks:
            self._sync_callbacks.remove(callback)
            self.logger.debug(f"Unregistered sync callback: {callback.__name__}")
    
    def sync_simulation_data(self, simulation_manager: Optional[SimulationManager] = None) -> Dict[str, Any]:
        """Synchronize simulation data between components.
        
        Args:
            simulation_manager: Optional simulation manager instance
            
        Returns:
            Dictionary with sync results and statistics
        """
        sync_start_time = time.time()
        sync_results = {
            "success": False,
            "sync_time": sync_start_time,
            "duration": 0.0,
            "components_synced": [],
            "errors": [],
            "data_updated": False
        }
        
        try:
            with self._sync_lock:
                # Get simulation manager from session state if not provided
                if simulation_manager is None:
                    simulation_manager = st.session_state.get('simulation_manager')
                
                if simulation_manager is None:
                    sync_results["errors"].append("No simulation manager available")
                    return sync_results
                
                # Update simulation time if running
                if simulation_manager.is_running():
                    try:
                        simulation_manager.update_simulation_time()
                        sync_results["components_synced"].append("simulation_time")
                    except Exception as e:
                        sync_results["errors"].append(f"Failed to update simulation time: {str(e)}")
                
                # Sync agent states
                try:
                    agent_states = simulation_manager.get_agent_states()
                    if agent_states:
                        # Update agent states in session state
                        for agent_id, state_info in agent_states.items():
                            self.session_manager.update_agent(agent_id, **state_info)
                        sync_results["components_synced"].append("agent_states")
                        sync_results["data_updated"] = True
                except Exception as e:
                    sync_results["errors"].append(f"Failed to sync agent states: {str(e)}")
                
                # Sync metrics
                try:
                    current_metrics = simulation_manager.get_current_metrics()
                    if current_metrics:
                        self.session_manager.add_metrics_snapshot(current_metrics)
                        sync_results["components_synced"].append("metrics")
                        sync_results["data_updated"] = True
                except Exception as e:
                    sync_results["errors"].append(f"Failed to sync metrics: {str(e)}")
                
                # Sync active events
                try:
                    active_events = simulation_manager.get_active_events()
                    if active_events:
                        # Update event statuses in session state
                        for event in active_events:
                            self.session_manager.update_event(
                                event.id,
                                resolution_status=event.resolution_status,
                                affected_agents=event.affected_agents
                            )
                        sync_results["components_synced"].append("active_events")
                        sync_results["data_updated"] = True
                except Exception as e:
                    sync_results["errors"].append(f"Failed to sync active events: {str(e)}")
                
                # Execute registered callbacks
                for callback in self._sync_callbacks:
                    try:
                        callback(simulation_manager, self.session_manager)
                        sync_results["components_synced"].append(f"callback_{callback.__name__}")
                    except Exception as e:
                        sync_results["errors"].append(f"Callback {callback.__name__} failed: {str(e)}")
                
                # Update sync statistics
                self._last_sync_time = datetime.now()
                sync_results["success"] = len(sync_results["errors"]) == 0
                sync_results["duration"] = time.time() - sync_start_time
                
                # Reset error count on successful sync
                if sync_results["success"]:
                    self._error_count = 0
                else:
                    self._error_count += 1
                
                self.logger.debug(f"Data sync completed in {sync_results['duration']:.3f}s, "
                                f"synced: {sync_results['components_synced']}")
                
        except Exception as e:
            sync_results["errors"].append(f"Critical sync error: {str(e)}")
            sync_results["duration"] = time.time() - sync_start_time
            self._error_count += 1
            self.logger.error(f"Critical error during data sync: {e}")
        
        return sync_results
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status.
        
        Returns:
            Dictionary with sync status information
        """
        return {
            "last_sync_time": self._last_sync_time.isoformat(),
            "time_since_last_sync": (datetime.now() - self._last_sync_time).total_seconds(),
            "error_count": self._error_count,
            "max_errors": self._max_errors,
            "sync_healthy": self._error_count < self._max_errors,
            "registered_callbacks": len(self._sync_callbacks)
        }
    
    def is_sync_healthy(self) -> bool:
        """Check if synchronization is healthy.
        
        Returns:
            True if sync is healthy, False otherwise
        """
        return self._error_count < self._max_errors
    
    def reset_error_count(self) -> None:
        """Reset the error count."""
        self._error_count = 0
        self.logger.info("Sync error count reset")


class AutoRefreshManager:
    """Manages automatic refresh functionality for Streamlit components."""
    
    def __init__(self, session_manager: SessionStateManager):
        """Initialize the auto-refresh manager.
        
        Args:
            session_manager: Session state manager instance
        """
        self.session_manager = session_manager
        self.logger = logging.getLogger(__name__)
        self._refresh_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_component(self, component_name: str, refresh_interval: int = 30,
                          condition_callback: Optional[Callable] = None) -> None:
        """Register a component for auto-refresh.
        
        Args:
            component_name: Name of the component
            refresh_interval: Refresh interval in seconds
            condition_callback: Optional callback to check if refresh should occur
        """
        self._refresh_configs[component_name] = {
            "interval": refresh_interval,
            "last_refresh": time.time(),
            "condition_callback": condition_callback,
            "refresh_count": 0
        }
        self.logger.debug(f"Registered component '{component_name}' for auto-refresh "
                         f"every {refresh_interval} seconds")
    
    def should_refresh(self, component_name: str) -> bool:
        """Check if a component should be refreshed.
        
        Args:
            component_name: Name of the component to check
            
        Returns:
            True if component should be refreshed
        """
        if component_name not in self._refresh_configs:
            return False
        
        config = self._refresh_configs[component_name]
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - config["last_refresh"] < config["interval"]:
            return False
        
        # Check condition callback if provided
        if config["condition_callback"]:
            try:
                if not config["condition_callback"]():
                    return False
            except Exception as e:
                self.logger.warning(f"Condition callback for '{component_name}' failed: {e}")
                return False
        
        return True
    
    def mark_refreshed(self, component_name: str) -> None:
        """Mark a component as refreshed.
        
        Args:
            component_name: Name of the component that was refreshed
        """
        if component_name in self._refresh_configs:
            config = self._refresh_configs[component_name]
            config["last_refresh"] = time.time()
            config["refresh_count"] += 1
            self.logger.debug(f"Component '{component_name}' marked as refreshed "
                            f"(count: {config['refresh_count']})")
    
    def get_refresh_status(self) -> Dict[str, Dict[str, Any]]:
        """Get refresh status for all registered components.
        
        Returns:
            Dictionary with refresh status for each component
        """
        status = {}
        current_time = time.time()
        
        for component_name, config in self._refresh_configs.items():
            time_since_refresh = current_time - config["last_refresh"]
            next_refresh_in = max(0, config["interval"] - time_since_refresh)
            
            status[component_name] = {
                "interval": config["interval"],
                "last_refresh": config["last_refresh"],
                "time_since_refresh": time_since_refresh,
                "next_refresh_in": next_refresh_in,
                "refresh_count": config["refresh_count"],
                "should_refresh": self.should_refresh(component_name)
            }
        
        return status
    
    def unregister_component(self, component_name: str) -> None:
        """Unregister a component from auto-refresh.
        
        Args:
            component_name: Name of the component to unregister
        """
        if component_name in self._refresh_configs:
            del self._refresh_configs[component_name]
            self.logger.debug(f"Unregistered component '{component_name}' from auto-refresh")


def create_real_time_sync_context(session_manager: SessionStateManager) -> Dict[str, Any]:
    """Create a real-time synchronization context for Streamlit components.
    
    Args:
        session_manager: Session state manager instance
        
    Returns:
        Dictionary with sync utilities and status
    """
    # Initialize synchronizer if not already in session state
    if 'data_synchronizer' not in st.session_state:
        st.session_state['data_synchronizer'] = RealTimeDataSynchronizer(session_manager)
    
    if 'auto_refresh_manager' not in st.session_state:
        st.session_state['auto_refresh_manager'] = AutoRefreshManager(session_manager)
    
    synchronizer = st.session_state['data_synchronizer']
    refresh_manager = st.session_state['auto_refresh_manager']
    
    # Perform sync if needed
    sync_results = None
    if 'simulation_manager' in st.session_state:
        simulation_manager = st.session_state['simulation_manager']
        if simulation_manager and simulation_manager.is_running():
            sync_results = synchronizer.sync_simulation_data(simulation_manager)
    
    return {
        "synchronizer": synchronizer,
        "refresh_manager": refresh_manager,
        "sync_results": sync_results,
        "sync_status": synchronizer.get_sync_status(),
        "refresh_status": refresh_manager.get_refresh_status()
    }


def apply_real_time_updates(component_name: str, session_manager: SessionStateManager,
                           refresh_interval: int = 30) -> bool:
    """Apply real-time updates to a Streamlit component.
    
    Args:
        component_name: Name of the component
        session_manager: Session state manager instance
        refresh_interval: Refresh interval in seconds
        
    Returns:
        True if component was refreshed, False otherwise
    """
    sync_context = create_real_time_sync_context(session_manager)
    refresh_manager = sync_context["refresh_manager"]
    
    # Register component if not already registered
    if component_name not in refresh_manager._refresh_configs:
        # Create condition callback based on simulation state
        def should_refresh_condition():
            sim_state = session_manager.get_simulation_state()
            return sim_state.is_running
        
        refresh_manager.register_component(
            component_name,
            refresh_interval,
            should_refresh_condition
        )
    
    # Check if refresh is needed
    if refresh_manager.should_refresh(component_name):
        refresh_manager.mark_refreshed(component_name)
        return True
    
    return False


def display_sync_status_indicator(sync_context: Dict[str, Any]) -> None:
    """Display synchronization status indicator in Streamlit.
    
    Args:
        sync_context: Sync context from create_real_time_sync_context
    """
    sync_status = sync_context["sync_status"]
    
    # Create status indicator
    if sync_status["sync_healthy"]:
        status_color = "🟢"
        status_text = "Sync Healthy"
    else:
        status_color = "🔴"
        status_text = f"Sync Issues ({sync_status['error_count']} errors)"
    
    # Display in sidebar or main area
    with st.sidebar:
        st.write(f"**Data Sync:** {status_color} {status_text}")
        
        # Show time since last sync
        time_since_sync = sync_status["time_since_last_sync"]
        if time_since_sync < 60:
            time_text = f"{time_since_sync:.0f}s ago"
        else:
            time_text = f"{time_since_sync/60:.1f}m ago"
        
        st.caption(f"Last sync: {time_text}")
        
        # Show refresh status for components
        refresh_status = sync_context["refresh_status"]
        if refresh_status:
            with st.expander("Component Refresh Status"):
                for component, status in refresh_status.items():
                    next_refresh = status["next_refresh_in"]
                    if next_refresh <= 0:
                        refresh_text = "Ready to refresh"
                    else:
                        refresh_text = f"Next: {next_refresh:.0f}s"
                    
                    st.write(f"**{component}:** {refresh_text}")