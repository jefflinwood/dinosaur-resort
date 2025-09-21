"""Session state management for Streamlit dashboard."""

import logging
import os
import streamlit as st
from typing import Dict, List, Any, Optional, Type, TypeVar
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from models.core import Agent, Event, MetricsSnapshot, SimulationState
from models.config import OpenAIConfig, AG2Config, SimulationConfig, AgentConfig


T = TypeVar('T')


class SessionStateManager:
    """Manages Streamlit session state for the dinosaur simulator."""
    
    # Session state keys
    SIMULATION_STATE = "simulation_state"
    AGENTS = "agents"
    EVENTS = "events"
    METRICS_HISTORY = "metrics_history"
    CONVERSATION_HISTORY = "conversation_history"
    OPENAI_CONFIG = "openai_config"
    AG2_CONFIG = "ag2_config"
    SIMULATION_CONFIG = "simulation_config"
    AGENT_CONFIG = "agent_config"
    INITIALIZED = "initialized"
    
    def __init__(self):
        """Initialize session state manager."""
        self.logger = logging.getLogger(__name__)
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state with default values."""
        if not self.get(self.INITIALIZED, False):
            # Initialize simulation state
            self.set(self.SIMULATION_STATE, SimulationState())
            
            # Initialize collections
            self.set(self.AGENTS, {})
            self.set(self.EVENTS, [])
            self.set(self.METRICS_HISTORY, [])
            self.set(self.CONVERSATION_HISTORY, {})
            
            # Initialize configurations
            self.set(self.OPENAI_CONFIG, None)
            self.set(self.AG2_CONFIG, AG2Config())
            self.set(self.SIMULATION_CONFIG, SimulationConfig())
            self.set(self.AGENT_CONFIG, AgentConfig())
            
            # Mark as initialized
            self.set(self.INITIALIZED, True)
            
            self.logger.info("Session state initialized")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from session state.
        
        Args:
            key: Session state key
            default: Default value if key doesn't exist
            
        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in session state.
        
        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value
    
    def delete(self, key: str) -> None:
        """Delete key from session state.
        
        Args:
            key: Session state key to delete
        """
        if key in st.session_state:
            del st.session_state[key]
    
    def clear_all(self) -> None:
        """Clear all session state data."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._initialize_session_state()
        self.logger.info("Session state cleared and reinitialized")
    
    # Simulation State Management
    def get_simulation_state(self) -> SimulationState:
        """Get current simulation state.
        
        Returns:
            Current simulation state
        """
        return self.get(self.SIMULATION_STATE, SimulationState())
    
    def update_simulation_state(self, **kwargs) -> None:
        """Update simulation state with new values.
        
        Args:
            **kwargs: Fields to update in simulation state
        """
        current_state = self.get_simulation_state()
        
        # Update fields
        for field, value in kwargs.items():
            if hasattr(current_state, field):
                setattr(current_state, field, value)
        
        self.set(self.SIMULATION_STATE, current_state)
        self.logger.debug(f"Updated simulation state: {kwargs}")
    
    # Agent Management
    def get_agents(self) -> Dict[str, Agent]:
        """Get all agents.
        
        Returns:
            Dictionary of agent_id -> Agent
        """
        return self.get(self.AGENTS, {})
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to session state.
        
        Args:
            agent: Agent to add
        """
        agents = self.get_agents()
        agents[agent.id] = agent
        self.set(self.AGENTS, agents)
        
        # Update agent count in simulation state
        self.update_simulation_state(agent_count=len(agents))
        
        self.logger.info(f"Added agent {agent.name} to session state")
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from session state.
        
        Args:
            agent_id: ID of agent to remove
            
        Returns:
            True if agent was removed, False if not found
        """
        agents = self.get_agents()
        if agent_id in agents:
            del agents[agent_id]
            self.set(self.AGENTS, agents)
            
            # Update agent count in simulation state
            self.update_simulation_state(agent_count=len(agents))
            
            self.logger.info(f"Removed agent {agent_id} from session state")
            return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get a specific agent.
        
        Args:
            agent_id: ID of agent to get
            
        Returns:
            Agent if found, None otherwise
        """
        agents = self.get_agents()
        return agents.get(agent_id)
    
    def update_agent(self, agent_id: str, **kwargs) -> bool:
        """Update an agent's properties.
        
        Args:
            agent_id: ID of agent to update
            **kwargs: Fields to update
            
        Returns:
            True if agent was updated, False if not found
        """
        agents = self.get_agents()
        if agent_id in agents:
            agent = agents[agent_id]
            
            # Update fields
            for field, value in kwargs.items():
                if hasattr(agent, field):
                    setattr(agent, field, value)
            
            agents[agent_id] = agent
            self.set(self.AGENTS, agents)
            
            self.logger.debug(f"Updated agent {agent_id}: {kwargs}")
            return True
        return False
    
    # Event Management
    def get_events(self) -> List[Event]:
        """Get all events.
        
        Returns:
            List of events
        """
        return self.get(self.EVENTS, [])
    
    def add_event(self, event: Event) -> None:
        """Add an event to session state.
        
        Args:
            event: Event to add
        """
        events = self.get_events()
        events.append(event)
        self.set(self.EVENTS, events)
        
        # Update active events in simulation state
        active_events = [e for e in events if e.resolution_status.name in ['PENDING', 'IN_PROGRESS']]
        self.update_simulation_state(active_events=active_events)
        
        self.logger.info(f"Added event {event.id} to session state")
    
    def update_event(self, event_id: str, **kwargs) -> bool:
        """Update an event's properties.
        
        Args:
            event_id: ID of event to update
            **kwargs: Fields to update
            
        Returns:
            True if event was updated, False if not found
        """
        events = self.get_events()
        for i, event in enumerate(events):
            if event.id == event_id:
                # Update fields
                for field, value in kwargs.items():
                    if hasattr(event, field):
                        setattr(event, field, value)
                
                events[i] = event
                self.set(self.EVENTS, events)
                
                # Update active events in simulation state
                active_events = [e for e in events if e.resolution_status.name in ['PENDING', 'IN_PROGRESS']]
                self.update_simulation_state(active_events=active_events)
                
                self.logger.debug(f"Updated event {event_id}: {kwargs}")
                return True
        return False
    
    # Metrics Management
    def get_metrics_history(self) -> List[MetricsSnapshot]:
        """Get metrics history.
        
        Returns:
            List of metrics snapshots
        """
        return self.get(self.METRICS_HISTORY, [])
    
    def add_metrics_snapshot(self, metrics: MetricsSnapshot) -> None:
        """Add a metrics snapshot to history.
        
        Args:
            metrics: Metrics snapshot to add
        """
        history = self.get_metrics_history()
        history.append(metrics)
        
        # Keep only last 1000 snapshots to prevent memory issues
        if len(history) > 1000:
            history = history[-1000:]
        
        self.set(self.METRICS_HISTORY, history)
        
        # Update current metrics in simulation state
        self.update_simulation_state(current_metrics=metrics)
        
        self.logger.debug("Added metrics snapshot to history")
    
    def get_latest_metrics(self) -> Optional[MetricsSnapshot]:
        """Get the latest metrics snapshot.
        
        Returns:
            Latest metrics snapshot or None if no history
        """
        history = self.get_metrics_history()
        return history[-1] if history else None
    
    # Conversation Management
    def get_conversation_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get conversation history for all agents.
        
        Returns:
            Dictionary mapping agent_id to conversation messages
        """
        return self.get(self.CONVERSATION_HISTORY, {})
    
    def add_conversation_message(self, agent_id: str, message: Dict[str, Any]) -> None:
        """Add a conversation message for an agent.
        
        Args:
            agent_id: ID of the agent
            message: Message dictionary
        """
        history = self.get_conversation_history()
        if agent_id not in history:
            history[agent_id] = []
        
        history[agent_id].append(message)
        
        # Keep only last 100 messages per agent
        if len(history[agent_id]) > 100:
            history[agent_id] = history[agent_id][-100:]
        
        self.set(self.CONVERSATION_HISTORY, history)
        self.logger.debug(f"Added conversation message for agent {agent_id}")
    
    def clear_conversation_history(self, agent_id: Optional[str] = None) -> None:
        """Clear conversation history.
        
        Args:
            agent_id: Specific agent ID to clear, or None to clear all
        """
        if agent_id:
            history = self.get_conversation_history()
            if agent_id in history:
                del history[agent_id]
                self.set(self.CONVERSATION_HISTORY, history)
                self.logger.info(f"Cleared conversation history for agent {agent_id}")
        else:
            self.set(self.CONVERSATION_HISTORY, {})
            self.logger.info("Cleared all conversation history")
    
    # Configuration Management
    def get_openai_config(self) -> Optional[OpenAIConfig]:
        """Get OpenAI configuration.
        
        Returns:
            OpenAI configuration or None if not set
        """
        return self.get(self.OPENAI_CONFIG)
    
    def set_openai_config(self, config: OpenAIConfig) -> None:
        """Set OpenAI configuration.
        
        Args:
            config: OpenAI configuration
        """
        self.set(self.OPENAI_CONFIG, config)
        self.logger.info("Updated OpenAI configuration")
    
    def get_ag2_config(self) -> AG2Config:
        """Get ag2 configuration.
        
        Returns:
            ag2 configuration
        """
        return self.get(self.AG2_CONFIG, AG2Config())
    
    def set_ag2_config(self, config: AG2Config) -> None:
        """Set ag2 configuration.
        
        Args:
            config: ag2 configuration
        """
        self.set(self.AG2_CONFIG, config)
        self.logger.info("Updated ag2 configuration")
    
    def get_simulation_config(self) -> SimulationConfig:
        """Get simulation configuration.
        
        Returns:
            Simulation configuration
        """
        return self.get(self.SIMULATION_CONFIG, SimulationConfig())
    
    def set_simulation_config(self, config: SimulationConfig) -> None:
        """Set simulation configuration.
        
        Args:
            config: Simulation configuration
        """
        self.set(self.SIMULATION_CONFIG, config)
        self.logger.info("Updated simulation configuration")
    
    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration.
        
        Returns:
            Agent configuration
        """
        return self.get(self.AGENT_CONFIG, AgentConfig())
    
    def set_agent_config(self, config: AgentConfig) -> None:
        """Set agent configuration.
        
        Args:
            config: Agent configuration
        """
        self.set(self.AGENT_CONFIG, config)
        self.logger.info("Updated agent configuration")
    
    # Utility Methods
    def is_initialized(self) -> bool:
        """Check if session state is initialized.
        
        Returns:
            True if initialized
        """
        return self.get(self.INITIALIZED, False)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session state information for debugging.
        
        Returns:
            Dictionary with session state information
        """
        return {
            "initialized": self.is_initialized(),
            "agent_count": len(self.get_agents()),
            "event_count": len(self.get_events()),
            "metrics_history_count": len(self.get_metrics_history()),
            "conversation_agents": list(self.get_conversation_history().keys()),
            "simulation_running": self.get_simulation_state().is_running,
        }