"""Event management system for the AI Agent Dinosaur Simulator."""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from models.core import Event
from models.enums import EventType, ResolutionStatus
from models.config import Location

if TYPE_CHECKING:
    from managers.agent_manager import AgentManager


class EventManager:
    """Manages event creation, distribution, and resolution tracking."""
    
    def __init__(self, agent_manager: Optional['AgentManager'] = None):
        """Initialize the EventManager.
        
        Args:
            agent_manager: Optional AgentManager for event distribution
        """
        self._active_events: Dict[str, Event] = {}
        self._event_history: List[Event] = []
        self._event_listeners: List[Callable[[Event], None]] = []
        self._event_type_definitions = self._initialize_event_definitions()
        self._agent_manager = agent_manager
        self._event_responses: Dict[str, Dict[str, Any]] = {}  # event_id -> response data
        self._resolution_callbacks: List[Callable[[Event], None]] = []
    
    def _initialize_event_definitions(self) -> Dict[EventType, Dict[str, Any]]:
        """Initialize event type definitions with default parameters and descriptions."""
        return {
            EventType.DINOSAUR_ESCAPE: {
                "description": "A dinosaur has escaped from its enclosure",
                "default_severity": 8,
                "required_parameters": ["dinosaur_id", "enclosure_id"],
                "optional_parameters": ["escape_method", "last_seen_location"],
                "affected_roles": ["PARK_RANGER", "SECURITY", "VETERINARIAN"]
            },
            EventType.DINOSAUR_ILLNESS: {
                "description": "A dinosaur is showing signs of illness",
                "default_severity": 5,
                "required_parameters": ["dinosaur_id", "symptoms"],
                "optional_parameters": ["severity_level", "contagious"],
                "affected_roles": ["VETERINARIAN", "PARK_RANGER"]
            },
            EventType.DINOSAUR_AGGRESSIVE: {
                "description": "A dinosaur is displaying aggressive behavior",
                "default_severity": 7,
                "required_parameters": ["dinosaur_id", "behavior_type"],
                "optional_parameters": ["trigger_cause", "threat_level"],
                "affected_roles": ["PARK_RANGER", "SECURITY", "VETERINARIAN"]
            },
            EventType.VISITOR_INJURY: {
                "description": "A visitor has been injured",
                "default_severity": 6,
                "required_parameters": ["visitor_id", "injury_type"],
                "optional_parameters": ["injury_severity", "medical_attention_needed"],
                "affected_roles": ["SECURITY", "PARK_RANGER", "VETERINARIAN"]
            },
            EventType.VISITOR_COMPLAINT: {
                "description": "A visitor has filed a complaint",
                "default_severity": 3,
                "required_parameters": ["visitor_id", "complaint_type"],
                "optional_parameters": ["complaint_details", "requested_resolution"],
                "affected_roles": ["PARK_RANGER"]
            },
            EventType.VISITOR_EMERGENCY: {
                "description": "A visitor emergency situation has occurred",
                "default_severity": 9,
                "required_parameters": ["visitor_id", "emergency_type"],
                "optional_parameters": ["emergency_details", "immediate_danger"],
                "affected_roles": ["SECURITY", "PARK_RANGER", "VETERINARIAN"]
            },
            EventType.FACILITY_POWER_OUTAGE: {
                "description": "A power outage has occurred in a facility",
                "default_severity": 6,
                "required_parameters": ["facility_id", "affected_systems"],
                "optional_parameters": ["estimated_duration", "backup_power_available"],
                "affected_roles": ["MAINTENANCE", "SECURITY"]
            },
            EventType.FACILITY_EQUIPMENT_FAILURE: {
                "description": "Equipment has failed in a facility",
                "default_severity": 4,
                "required_parameters": ["facility_id", "equipment_type"],
                "optional_parameters": ["failure_cause", "repair_urgency"],
                "affected_roles": ["MAINTENANCE"]
            },
            EventType.WEATHER_STORM: {
                "description": "A storm is affecting the resort",
                "default_severity": 7,
                "required_parameters": ["storm_type", "intensity"],
                "optional_parameters": ["duration", "wind_speed", "precipitation"],
                "affected_roles": ["PARK_RANGER", "SECURITY", "MAINTENANCE"]
            },
            EventType.WEATHER_EXTREME_TEMPERATURE: {
                "description": "Extreme temperature conditions are affecting the resort",
                "default_severity": 5,
                "required_parameters": ["temperature", "temperature_type"],
                "optional_parameters": ["duration", "affected_areas"],
                "affected_roles": ["PARK_RANGER", "VETERINARIAN", "MAINTENANCE"]
            },
            EventType.CUSTOM: {
                "description": "A custom event defined by the user",
                "default_severity": 5,
                "required_parameters": ["event_description"],
                "optional_parameters": ["custom_parameters"],
                "affected_roles": []  # Will be determined based on event content
            }
        }
    
    def get_available_events(self) -> List[Dict[str, Any]]:
        """Get list of available event types with their definitions."""
        available_events = []
        for event_type, definition in self._event_type_definitions.items():
            available_events.append({
                "type": event_type,
                "name": event_type.name,
                "description": definition["description"],
                "default_severity": definition["default_severity"],
                "required_parameters": definition["required_parameters"],
                "optional_parameters": definition["optional_parameters"],
                "affected_roles": definition["affected_roles"]
            })
        return available_events
    
    def create_event(self, event_type: EventType, location: Location, 
                    parameters: Dict[str, Any], severity: Optional[int] = None,
                    description: Optional[str] = None) -> Event:
        """Create a new event with validation."""
        # Validate event type
        if event_type not in self._event_type_definitions:
            raise ValueError(f"Unknown event type: {event_type}")
        
        event_def = self._event_type_definitions[event_type]
        
        # Validate required parameters
        for required_param in event_def["required_parameters"]:
            if required_param not in parameters:
                raise ValueError(f"Missing required parameter '{required_param}' for event type {event_type.name}")
        
        # Use default severity if not provided
        if severity is None:
            severity = event_def["default_severity"]
        
        # Validate severity
        if not (1 <= severity <= 10):
            raise ValueError("Event severity must be between 1 and 10")
        
        # Generate unique event ID
        event_id = str(uuid.uuid4())
        
        # Use provided description or default
        if description is None:
            description = event_def["description"]
        
        # Create the event
        event = Event(
            id=event_id,
            type=event_type,
            severity=severity,
            location=location,
            parameters=parameters,
            timestamp=datetime.now(),
            affected_agents=[],  # Will be populated during distribution
            resolution_status=ResolutionStatus.PENDING,
            description=description
        )
        
        return event
    
    def distribute_event(self, event: Event) -> Dict[str, Any]:
        """Distribute an event to relevant listeners and agents, and track it.
        
        Args:
            event: Event to distribute
            
        Returns:
            Dictionary with distribution results including agent responses
        """
        # Add to active events
        self._active_events[event.id] = event
        
        # Add to event history
        self._event_history.append(event)
        
        # Initialize event response tracking
        self._event_responses[event.id] = {
            "event": event,
            "agent_responses": {},
            "group_response": None,
            "resolution_attempts": 0,
            "start_time": datetime.now(),
            "last_update": datetime.now()
        }
        
        # Notify all listeners
        for listener in self._event_listeners:
            try:
                listener(event)
            except Exception as e:
                # Log error but continue with other listeners
                print(f"Error notifying event listener: {e}")
        
        # Distribute to agents if agent manager is available
        distribution_result = {"event_id": event.id, "listeners_notified": len(self._event_listeners)}
        
        if self._agent_manager:
            try:
                agent_response = self._agent_manager.broadcast_event(event)
                distribution_result.update(agent_response)
                
                # Store agent responses for resolution tracking
                self._event_responses[event.id]["agent_responses"] = agent_response.get("individual_responses", {})
                self._event_responses[event.id]["group_response"] = agent_response.get("group_response")
                self._event_responses[event.id]["affected_agents"] = agent_response.get("affected_agents", [])
                
                # Update event with affected agents
                event.affected_agents = agent_response.get("affected_agents", [])
                
                # Start resolution monitoring
                self._monitor_event_resolution(event.id)
                
            except Exception as e:
                print(f"Error distributing event to agents: {e}")
                distribution_result["agent_error"] = str(e)
        
        return distribution_result
    
    def add_event_listener(self, listener: Callable[[Event], None]) -> None:
        """Add an event listener that will be notified when events are distributed."""
        if listener not in self._event_listeners:
            self._event_listeners.append(listener)
    
    def remove_event_listener(self, listener: Callable[[Event], None]) -> None:
        """Remove an event listener."""
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)
    
    def update_event_status(self, event_id: str, status: ResolutionStatus, 
                           affected_agents: Optional[List[str]] = None) -> bool:
        """Update the resolution status of an event."""
        if event_id not in self._active_events:
            return False
        
        event = self._active_events[event_id]
        event.resolution_status = status
        
        if affected_agents is not None:
            event.affected_agents = affected_agents
        
        # Set resolution time if event is resolved
        if status in [ResolutionStatus.RESOLVED, ResolutionStatus.FAILED]:
            event.resolution_time = datetime.now()
            # Remove from active events
            del self._active_events[event_id]
        
        return True
    
    def get_active_events(self) -> List[Event]:
        """Get all currently active events."""
        return list(self._active_events.values())
    
    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get a specific event by ID from active events or history."""
        # Check active events first
        if event_id in self._active_events:
            return self._active_events[event_id]
        
        # Check event history
        for event in self._event_history:
            if event.id == event_id:
                return event
        
        return None
    
    def get_event_history(self, limit: Optional[int] = None) -> List[Event]:
        """Get event history, optionally limited to recent events."""
        if limit is None:
            return self._event_history.copy()
        return self._event_history[-limit:] if limit > 0 else []
    
    def check_resolution_status(self, event_id: str) -> Optional[ResolutionStatus]:
        """Check the resolution status of a specific event."""
        event = self.get_event_by_id(event_id)
        return event.resolution_status if event else None
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all events (active and historical) of a specific type."""
        matching_events = []
        
        # Check active events
        for event in self._active_events.values():
            if event.type == event_type:
                matching_events.append(event)
        
        # Check historical events
        for event in self._event_history:
            if event.type == event_type and event.id not in self._active_events:
                matching_events.append(event)
        
        return matching_events
    
    def get_events_by_status(self, status: ResolutionStatus) -> List[Event]:
        """Get all events with a specific resolution status."""
        matching_events = []
        
        # Check active events
        for event in self._active_events.values():
            if event.resolution_status == status:
                matching_events.append(event)
        
        # Check historical events for resolved/failed events
        if status in [ResolutionStatus.RESOLVED, ResolutionStatus.FAILED]:
            for event in self._event_history:
                if event.resolution_status == status and event.id not in self._active_events:
                    matching_events.append(event)
        
        return matching_events
    
    def clear_event_history(self) -> None:
        """Clear the event history (for testing or reset purposes)."""
        self._event_history.clear()
    
    def set_agent_manager(self, agent_manager: 'AgentManager') -> None:
        """Set the agent manager for event distribution.
        
        Args:
            agent_manager: AgentManager instance to use for event distribution
        """
        self._agent_manager = agent_manager
    
    def add_resolution_callback(self, callback: Callable[[Event], None]) -> None:
        """Add a callback to be called when an event is resolved.
        
        Args:
            callback: Function to call when event resolution is detected
        """
        if callback not in self._resolution_callbacks:
            self._resolution_callbacks.append(callback)
    
    def remove_resolution_callback(self, callback: Callable[[Event], None]) -> None:
        """Remove a resolution callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self._resolution_callbacks:
            self._resolution_callbacks.remove(callback)
    
    def _monitor_event_resolution(self, event_id: str) -> None:
        """Monitor an event for resolution based on agent responses.
        
        Args:
            event_id: ID of the event to monitor
        """
        if event_id not in self._event_responses:
            return
        
        event_data = self._event_responses[event_id]
        event = event_data["event"]
        
        # Check if event should be automatically resolved based on agent responses
        resolution_detected = self._detect_event_resolution(event_id)
        
        if resolution_detected:
            self._resolve_event(event_id, ResolutionStatus.RESOLVED)
    
    def _detect_event_resolution(self, event_id: str) -> bool:
        """Detect if an event has been resolved based on agent responses.
        
        Args:
            event_id: ID of the event to check
            
        Returns:
            True if resolution is detected
        """
        if event_id not in self._event_responses:
            return False
        
        event_data = self._event_responses[event_id]
        event = event_data["event"]
        agent_responses = event_data["agent_responses"]
        
        # Simple resolution detection based on response keywords
        resolution_keywords = [
            "resolved", "handled", "completed", "fixed", "secured", "contained",
            "treated", "repaired", "evacuated", "safe", "under control"
        ]
        
        escalation_keywords = [
            "need help", "backup required", "escalate", "emergency", "critical",
            "cannot handle", "overwhelmed", "failed", "need backup", "need immediate"
        ]
        
        positive_responses = 0
        escalation_responses = 0
        total_responses = len(agent_responses)
        
        if total_responses == 0:
            return False
        
        for agent_id, response in agent_responses.items():
            if response and isinstance(response, str):
                response_lower = response.lower()
                
                # Check for resolution indicators
                if any(keyword in response_lower for keyword in resolution_keywords):
                    positive_responses += 1
                
                # Check for escalation indicators
                if any(keyword in response_lower for keyword in escalation_keywords):
                    escalation_responses += 1
        
        # Event is considered resolved if majority of agents indicate resolution
        resolution_threshold = max(1, total_responses // 2)
        
        if positive_responses >= resolution_threshold and escalation_responses == 0:
            return True
        
        # Check for escalation
        if escalation_responses > 0:
            self._escalate_event(event_id)
        
        return False
    
    def _resolve_event(self, event_id: str, status: ResolutionStatus) -> None:
        """Mark an event as resolved and notify callbacks.
        
        Args:
            event_id: ID of the event to resolve
            status: Resolution status to set
        """
        if event_id in self._active_events:
            event = self._active_events[event_id]
            event.resolution_status = status
            event.resolution_time = datetime.now()
            
            # Remove from active events
            del self._active_events[event_id]
            
            # Update response tracking
            if event_id in self._event_responses:
                self._event_responses[event_id]["resolution_time"] = datetime.now()
                self._event_responses[event_id]["final_status"] = status
            
            # Notify resolution callbacks
            for callback in self._resolution_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Error in resolution callback: {e}")
            
            print(f"Event {event_id} resolved with status: {status.name}")
    
    def _escalate_event(self, event_id: str) -> None:
        """Escalate an event when agents cannot handle it.
        
        Args:
            event_id: ID of the event to escalate
        """
        if event_id in self._active_events:
            event = self._active_events[event_id]
            event.resolution_status = ResolutionStatus.ESCALATED
            
            # Increase severity for escalated events
            if event.severity < 10:
                event.severity = min(10, event.severity + 2)
            
            # Update response tracking
            if event_id in self._event_responses:
                self._event_responses[event_id]["escalated"] = True
                self._event_responses[event_id]["escalation_time"] = datetime.now()
            
            print(f"Event {event_id} escalated due to agent responses")
            
            # Re-distribute escalated event to more agents
            if self._agent_manager:
                try:
                    escalated_response = self._agent_manager.broadcast_event(event)
                    if event_id in self._event_responses:
                        self._event_responses[event_id]["escalated_responses"] = escalated_response
                except Exception as e:
                    print(f"Error re-distributing escalated event: {e}")
    
    def get_event_response_data(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get response data for a specific event.
        
        Args:
            event_id: ID of the event
            
        Returns:
            Event response data or None if not found
        """
        return self._event_responses.get(event_id)
    
    def get_all_event_responses(self) -> Dict[str, Dict[str, Any]]:
        """Get response data for all events.
        
        Returns:
            Dictionary mapping event IDs to response data
        """
        return self._event_responses.copy()
    
    def force_resolve_event(self, event_id: str, status: ResolutionStatus = ResolutionStatus.RESOLVED) -> bool:
        """Manually force resolution of an event.
        
        Args:
            event_id: ID of the event to resolve
            status: Resolution status to set
            
        Returns:
            True if event was resolved, False if not found
        """
        # Check both active events and event responses (in case event was created but not distributed)
        if event_id in self._active_events or event_id in self._event_responses:
            # If event is not in active events, add it temporarily for resolution
            if event_id not in self._active_events and event_id in self._event_responses:
                event = self._event_responses[event_id]["event"]
                self._active_events[event_id] = event
            
            self._resolve_event(event_id, status)
            return True
        return False
    
    def check_event_resolution_progress(self, event_id: str) -> Dict[str, Any]:
        """Check the resolution progress of an event.
        
        Args:
            event_id: ID of the event to check
            
        Returns:
            Dictionary with resolution progress information
        """
        if event_id not in self._event_responses:
            return {"error": "Event not found"}
        
        event_data = self._event_responses[event_id]
        event = event_data["event"]
        
        progress_info = {
            "event_id": event_id,
            "status": event.resolution_status.name,
            "severity": event.severity,
            "affected_agents": len(event.affected_agents),
            "responses_received": len(event_data["agent_responses"]),
            "time_elapsed": (datetime.now() - event_data["start_time"]).total_seconds(),
            "escalated": event_data.get("escalated", False)
        }
        
        if event.resolution_time:
            progress_info["resolution_time"] = event.resolution_time.isoformat()
            progress_info["total_resolution_time"] = (event.resolution_time - event_data["start_time"]).total_seconds()
        
        return progress_info
    
    def reset(self) -> None:
        """Reset the event manager to initial state."""
        self._active_events.clear()
        self._event_history.clear()
        self._event_listeners.clear()
        self._event_responses.clear()
        self._resolution_callbacks.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about events."""
        total_events = len(self._event_history)
        active_count = len(self._active_events)
        
        # Count by status
        status_counts = {}
        for status in ResolutionStatus:
            status_counts[status.name] = len(self.get_events_by_status(status))
        
        # Count by type
        type_counts = {}
        for event_type in EventType:
            type_counts[event_type.name] = len(self.get_events_by_type(event_type))
        
        return {
            "total_events": total_events,
            "active_events": active_count,
            "status_distribution": status_counts,
            "type_distribution": type_counts
        }