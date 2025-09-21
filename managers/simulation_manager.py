"""Simulation management core with session state for the AI Agent Dinosaur Simulator."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from models.core import SimulationState, Event, Agent, MetricsSnapshot
from models.config import OpenAIConfig, AG2Config, SimulationConfig, AgentConfig, Location
from models.enums import EventType, ResolutionStatus, AgentState
from ui.session_state import SessionStateManager
from managers.agent_manager import AgentManager
from managers.event_manager import EventManager
from managers.metrics_manager import MetricsManager


class SimulationManager:
    """Central orchestrator that coordinates all simulation activities with session state management."""
    
    def __init__(self, session_manager: SessionStateManager):
        """Initialize simulation manager.
        
        Args:
            session_manager: Session state manager for data persistence
        """
        self.session_manager = session_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize managers (will be set up when simulation starts)
        self.agent_manager: Optional[AgentManager] = None
        self.event_manager: Optional[EventManager] = None
        self.metrics_manager: Optional[MetricsManager] = None
        
        # Simulation clock and timing
        self.simulation_start_time: Optional[datetime] = None
        self.simulation_speed_multiplier: float = 1.0  # Real-time by default
        self.last_update_time: Optional[datetime] = None
        
        # Event processing
        self.event_processing_enabled: bool = True
        self.auto_resolution_enabled: bool = True
        
        self.logger.info("SimulationManager initialized with session state")
    
    def start_simulation(self, openai_config: Optional[OpenAIConfig] = None,
                        ag2_config: Optional[AG2Config] = None,
                        simulation_config: Optional[SimulationConfig] = None,
                        agent_config: Optional[AgentConfig] = None) -> None:
        """Start the simulation with the provided configurations.
        
        Args:
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
            simulation_config: Simulation parameters
            agent_config: Agent initialization configuration
        """
        try:
            self.logger.info("Starting simulation...")
            
            # Use provided configs or get from session state
            openai_config = openai_config or self.session_manager.get_openai_config()
            ag2_config = ag2_config or self.session_manager.get_ag2_config()
            simulation_config = simulation_config or self.session_manager.get_simulation_config()
            agent_config = agent_config or self.session_manager.get_agent_config()
            
            # If no OpenAI config exists, try to create one from environment variables
            if not openai_config:
                try:
                    from models.config import OpenAIConfig
                    openai_config = OpenAIConfig()  # This will read from environment variables
                    self.session_manager.set_openai_config(openai_config)
                    self.logger.info("Created OpenAI configuration from environment variables")
                except Exception as e:
                    raise ValueError(f"OpenAI configuration is required to start simulation. Please set OPENAI_API_KEY environment variable. Error: {e}")
            
            # Create default configs if they don't exist
            if not ag2_config:
                from models.config import AG2Config
                ag2_config = AG2Config()
                self.session_manager.set_ag2_config(ag2_config)
            
            if not simulation_config:
                from models.config import SimulationConfig
                simulation_config = SimulationConfig()
                self.session_manager.set_simulation_config(simulation_config)
            
            if not agent_config:
                from models.config import AgentConfig
                agent_config = AgentConfig()
                self.session_manager.set_agent_config(agent_config)
            
            # Initialize managers
            self.metrics_manager = MetricsManager(self.session_manager)
            self.event_manager = EventManager()
            self.agent_manager = AgentManager(openai_config, ag2_config, agent_config)
            
            # Set up event listener for metrics updates
            self.event_manager.add_event_listener(self._handle_event_for_metrics)
            
            # Initialize agents
            agents = self.agent_manager.initialize_agents(agent_config)
            
            # Initialize metrics for all agents
            for agent in agents:
                self.metrics_manager.initialize_agent_metrics(agent)
            
            # Update simulation state
            simulation_id = str(uuid.uuid4())
            self.simulation_start_time = datetime.now()
            self.last_update_time = self.simulation_start_time
            
            self.session_manager.update_simulation_state(
                is_running=True,
                simulation_id=simulation_id,
                started_at=self.simulation_start_time,
                current_time=self.simulation_start_time,
                agent_count=len(agents)
            )
            
            # Store configurations in session state
            self.session_manager.set_openai_config(openai_config)
            self.session_manager.set_ag2_config(ag2_config)
            self.session_manager.set_simulation_config(simulation_config)
            self.session_manager.set_agent_config(agent_config)
            
            self.logger.info(f"Simulation started with ID {simulation_id} and {len(agents)} agents")
        
        except Exception as e:
            self.logger.error(f"Error starting simulation: {e}")
            self.session_manager.update_simulation_state(is_running=False)
            raise
    
    def pause_simulation(self) -> None:
        """Pause the simulation."""
        if not self.is_running():
            self.logger.warning("Cannot pause simulation - not currently running")
            return
        
        self.session_manager.update_simulation_state(is_running=False)
        self.logger.info("Simulation paused")
    
    def resume_simulation(self) -> None:
        """Resume a paused simulation."""
        if self.is_running():
            self.logger.warning("Cannot resume simulation - already running")
            return
        
        if not self._is_initialized():
            self.logger.error("Cannot resume simulation - not initialized")
            raise RuntimeError("Simulation not initialized. Please start a new simulation.")
        
        self.session_manager.update_simulation_state(
            is_running=True,
            current_time=datetime.now()
        )
        self.last_update_time = datetime.now()
        self.logger.info("Simulation resumed")
    
    def stop_simulation(self) -> None:
        """Stop the simulation completely."""
        if self.agent_manager:
            # Reset agent conversations
            self.agent_manager.reset_agent_conversations()
        
        if self.event_manager:
            # Clear active events
            for event in self.event_manager.get_active_events():
                self.event_manager.update_event_status(event.id, ResolutionStatus.FAILED)
        
        self.session_manager.update_simulation_state(is_running=False)
        
        # Reset timing
        self.simulation_start_time = None
        self.last_update_time = None
        
        self.logger.info("Simulation stopped")
    
    def reset_simulation(self) -> None:
        """Reset the simulation to initial state."""
        self.logger.info("Resetting simulation...")
        
        # Stop current simulation
        self.stop_simulation()
        
        # Clear all session state data
        self.session_manager.clear_all()
        
        # Reset managers
        self.agent_manager = None
        self.event_manager = None
        self.metrics_manager = None
        
        # Reset timing
        self.simulation_start_time = None
        self.last_update_time = None
        
        self.logger.info("Simulation reset complete")
    
    def trigger_event(self, event_type: str, parameters: Dict[str, Any], 
                     location: Optional[Location] = None, severity: Optional[int] = None,
                     description: Optional[str] = None) -> str:
        """Trigger an event in the simulation.
        
        Args:
            event_type: Type of event to trigger
            parameters: Event parameters
            location: Event location (uses default if not provided)
            severity: Event severity (uses default if not provided)
            description: Event description (uses default if not provided)
            
        Returns:
            Event ID
        """
        if not self._is_initialized():
            raise RuntimeError("Simulation not initialized")
        
        if not self.event_manager:
            raise RuntimeError("Event manager not initialized")
        
        try:
            # Convert string event type to enum
            if isinstance(event_type, str):
                event_type_enum = EventType[event_type.upper()]
            else:
                event_type_enum = event_type
            
            # Use default location if not provided
            if location is None:
                location = Location(0.0, 0.0, "main_area", "Central area of the resort")
            
            # Create the event
            event = self.event_manager.create_event(
                event_type=event_type_enum,
                location=location,
                parameters=parameters,
                severity=severity,
                description=description
            )
            
            # Add to session state
            self.session_manager.add_event(event)
            
            # Distribute the event
            self.event_manager.distribute_event(event)
            
            # Broadcast to agents if agent manager is available
            if self.agent_manager:
                broadcast_result = self.agent_manager.broadcast_event(event)
                
                # Update event with affected agents
                affected_agents = broadcast_result.get('affected_agents', [])
                self.event_manager.update_event_status(
                    event.id, 
                    ResolutionStatus.IN_PROGRESS,
                    affected_agents
                )
                
                # Update event in session state
                self.session_manager.update_event(
                    event.id,
                    affected_agents=affected_agents,
                    resolution_status=ResolutionStatus.IN_PROGRESS
                )
            
            # Update simulation time
            self._update_simulation_time()
            
            self.logger.info(f"Triggered event {event.id} ({event_type_enum.name})")
            return event.id
        
        except Exception as e:
            self.logger.error(f"Error triggering event: {e}")
            raise
    
    def get_simulation_state(self) -> SimulationState:
        """Get current simulation state.
        
        Returns:
            Current simulation state
        """
        return self.session_manager.get_simulation_state()
    
    def update_simulation_time(self, time_delta: Optional[timedelta] = None) -> None:
        """Update simulation time and process time-based events.
        
        Args:
            time_delta: Time to advance (uses real time if not provided)
        """
        if not self.is_running():
            return
        
        current_time = datetime.now()
        
        if time_delta is None:
            # Calculate real time delta
            if self.last_update_time:
                real_delta = current_time - self.last_update_time
                # Apply simulation speed multiplier
                simulation_delta = real_delta * self.simulation_speed_multiplier
            else:
                simulation_delta = timedelta(0)
        else:
            simulation_delta = time_delta
        
        # Update simulation time
        sim_state = self.get_simulation_state()
        new_sim_time = sim_state.current_time + simulation_delta
        
        self.session_manager.update_simulation_state(current_time=new_sim_time)
        self.last_update_time = current_time
        
        # Process time-based updates
        self._process_time_based_updates(simulation_delta)
    
    def set_simulation_speed(self, multiplier: float) -> None:
        """Set simulation speed multiplier.
        
        Args:
            multiplier: Speed multiplier (1.0 = real time, 2.0 = 2x speed, etc.)
        """
        if multiplier <= 0:
            raise ValueError("Speed multiplier must be positive")
        
        self.simulation_speed_multiplier = multiplier
        self.logger.info(f"Simulation speed set to {multiplier}x")
    
    def get_active_events(self) -> List[Event]:
        """Get all active events.
        
        Returns:
            List of active events
        """
        if not self.event_manager:
            return []
        
        return self.event_manager.get_active_events()
    
    def get_event_history(self, limit: Optional[int] = None) -> List[Event]:
        """Get event history.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of historical events
        """
        if not self.event_manager:
            return []
        
        return self.event_manager.get_event_history(limit)
    
    def get_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all agents.
        
        Returns:
            Dictionary mapping agent_id to agent state information
        """
        if not self.agent_manager:
            return {}
        
        return self.agent_manager.get_agent_states()
    
    def get_current_metrics(self) -> Optional[MetricsSnapshot]:
        """Get current metrics snapshot.
        
        Returns:
            Current metrics or None if not available
        """
        if not self.metrics_manager:
            return None
        
        return self.metrics_manager.get_current_metrics()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get formatted metrics summary.
        
        Returns:
            Formatted metrics summary
        """
        if not self.metrics_manager:
            return {}
        
        return self.metrics_manager.get_metrics_summary()
    
    def is_running(self) -> bool:
        """Check if simulation is currently running.
        
        Returns:
            True if simulation is running
        """
        return self.get_simulation_state().is_running
    
    def get_simulation_info(self) -> Dict[str, Any]:
        """Get comprehensive simulation information.
        
        Returns:
            Dictionary with simulation status and statistics
        """
        sim_state = self.get_simulation_state()
        
        info = {
            "is_running": sim_state.is_running,
            "simulation_id": sim_state.simulation_id,
            "started_at": sim_state.started_at.isoformat() if sim_state.started_at else None,
            "current_time": sim_state.current_time.isoformat(),
            "agent_count": sim_state.agent_count,
            "active_events": len(sim_state.active_events),
            "simulation_speed": self.simulation_speed_multiplier,
            "uptime": None,
            "managers_initialized": {
                "agent_manager": self.agent_manager is not None,
                "event_manager": self.event_manager is not None,
                "metrics_manager": self.metrics_manager is not None
            }
        }
        
        # Calculate uptime if simulation is running
        if sim_state.started_at and self.is_running():
            uptime = datetime.now() - sim_state.started_at
            info["uptime"] = str(uptime).split('.')[0]  # Remove microseconds
        
        # Add metrics summary if available
        if self.metrics_manager:
            info["metrics"] = self.get_metrics_summary()
        
        # Add event statistics if available
        if self.event_manager:
            info["event_statistics"] = self.event_manager.get_statistics()
        
        # Add agent health if available
        if self.agent_manager:
            info["agent_health"] = self.agent_manager.check_agent_health()
        
        return info
    
    def _is_initialized(self) -> bool:
        """Check if simulation is initialized.
        
        Returns:
            True if simulation is initialized
        """
        sim_state = self.get_simulation_state()
        return bool(sim_state.simulation_id)
    
    def _update_simulation_time(self) -> None:
        """Update simulation time to current time."""
        self.session_manager.update_simulation_state(current_time=datetime.now())
        self.last_update_time = datetime.now()
    
    def _process_time_based_updates(self, time_delta: timedelta) -> None:
        """Process updates that occur over time.
        
        Args:
            time_delta: Time that has passed
        """
        if not self.is_running():
            return
        
        # Check for event resolution timeouts
        if self.event_manager and self.auto_resolution_enabled:
            self._check_event_timeouts()
        
        # Update agent states based on time
        if self.agent_manager:
            self._update_agent_time_based_states(time_delta)
        
        # Apply gradual metric changes
        if self.metrics_manager:
            self._apply_time_based_metric_changes(time_delta)
    
    def _check_event_timeouts(self) -> None:
        """Check for events that should timeout and be auto-resolved."""
        if not self.event_manager:
            return
        
        current_time = datetime.now()
        timeout_duration = timedelta(minutes=10)  # Events timeout after 10 minutes
        
        for event in self.event_manager.get_active_events():
            if event.resolution_status == ResolutionStatus.IN_PROGRESS:
                time_since_start = current_time - event.timestamp
                
                if time_since_start > timeout_duration:
                    # Auto-resolve the event
                    self.event_manager.update_event_status(event.id, ResolutionStatus.RESOLVED)
                    self.session_manager.update_event(
                        event.id,
                        resolution_status=ResolutionStatus.RESOLVED,
                        resolution_time=current_time
                    )
                    
                    self.logger.info(f"Auto-resolved event {event.id} due to timeout")
    
    def _update_agent_time_based_states(self, time_delta: timedelta) -> None:
        """Update agent states based on time passage.
        
        Args:
            time_delta: Time that has passed
        """
        # This could include things like:
        # - Agents becoming idle after periods of inactivity
        # - Periodic health checks
        # - Natural state transitions
        
        # For now, just update last activity times
        agents = self.session_manager.get_agents()
        current_time = datetime.now()
        
        for agent_id, agent in agents.items():
            # Agents become idle after 5 minutes of inactivity
            if agent.current_state == AgentState.ACTIVE:
                time_since_activity = current_time - agent.last_activity
                if time_since_activity > timedelta(minutes=5):
                    self.session_manager.update_agent(agent_id, current_state=AgentState.IDLE)
    
    def _apply_time_based_metric_changes(self, time_delta: timedelta) -> None:
        """Apply gradual metric changes over time.
        
        Args:
            time_delta: Time that has passed
        """
        # Apply small positive changes over time (things naturally improve)
        minutes_passed = time_delta.total_seconds() / 60.0
        
        if minutes_passed > 0:
            # Small improvements over time (very gradual)
            improvement_rate = 0.001 * minutes_passed  # 0.1% per minute
            
            # Only apply if no major events are active
            active_events = self.get_active_events()
            major_events = [e for e in active_events if e.severity >= 7]
            
            if not major_events and self.metrics_manager:
                # Gradual improvement in facility efficiency
                current_metrics = self.metrics_manager.get_current_metrics()
                if current_metrics.facility_efficiency < 1.0:
                    self.metrics_manager.update_facility_efficiency(min(improvement_rate, 0.01))
                
                # Gradual improvement in visitor satisfaction (if not at max)
                if current_metrics.visitor_satisfaction < 1.0:
                    self.metrics_manager.update_visitor_satisfaction('system', min(improvement_rate, 0.01))
    
    def _handle_event_for_metrics(self, event: Event) -> None:
        """Handle event for metrics updates.
        
        Args:
            event: Event that occurred
        """
        if not self.metrics_manager:
            return
        
        # Apply event impact to metrics
        self.metrics_manager.apply_event_impact(
            event.type.name,
            event.severity,
            event.affected_agents
        )
    
    def enable_auto_resolution(self, enabled: bool = True) -> None:
        """Enable or disable automatic event resolution.
        
        Args:
            enabled: Whether to enable auto-resolution
        """
        self.auto_resolution_enabled = enabled
        self.logger.info(f"Auto-resolution {'enabled' if enabled else 'disabled'}")
    
    def enable_event_processing(self, enabled: bool = True) -> None:
        """Enable or disable event processing.
        
        Args:
            enabled: Whether to enable event processing
        """
        self.event_processing_enabled = enabled
        self.logger.info(f"Event processing {'enabled' if enabled else 'disabled'}")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the simulation.
        
        Returns:
            Dictionary with debug information
        """
        return {
            "simulation_manager": {
                "initialized": self._is_initialized(),
                "running": self.is_running(),
                "simulation_speed": self.simulation_speed_multiplier,
                "auto_resolution_enabled": self.auto_resolution_enabled,
                "event_processing_enabled": self.event_processing_enabled,
                "start_time": self.simulation_start_time.isoformat() if self.simulation_start_time else None,
                "last_update": self.last_update_time.isoformat() if self.last_update_time else None
            },
            "managers": {
                "agent_manager": self.agent_manager is not None,
                "event_manager": self.event_manager is not None,
                "metrics_manager": self.metrics_manager is not None
            },
            "session_state": self.session_manager.get_session_info()
        }