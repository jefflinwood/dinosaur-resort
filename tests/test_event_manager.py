"""Unit tests for the EventManager class."""

import pytest
from datetime import datetime
from unittest.mock import Mock
from managers.event_manager import EventManager
from models.core import Event
from models.enums import EventType, ResolutionStatus
from models.config import Location


class TestEventManager:
    """Test cases for EventManager class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.event_manager = EventManager()
        self.test_location = Location(x=10.0, y=20.0, zone="main_area", description="Test location")
    
    def test_initialization(self):
        """Test EventManager initialization."""
        assert len(self.event_manager._active_events) == 0
        assert len(self.event_manager._event_history) == 0
        assert len(self.event_manager._event_listeners) == 0
        assert len(self.event_manager._event_type_definitions) == len(EventType)
    
    def test_get_available_events(self):
        """Test getting available event types."""
        available_events = self.event_manager.get_available_events()
        
        assert len(available_events) == len(EventType)
        
        # Check structure of returned events
        for event_info in available_events:
            assert "type" in event_info
            assert "name" in event_info
            assert "description" in event_info
            assert "default_severity" in event_info
            assert "required_parameters" in event_info
            assert "optional_parameters" in event_info
            assert "affected_roles" in event_info
            
            # Verify severity is valid
            assert 1 <= event_info["default_severity"] <= 10
    
    def test_create_event_valid(self):
        """Test creating a valid event."""
        parameters = {
            "dinosaur_id": "dino_001",
            "enclosure_id": "enc_001"
        }
        
        event = self.event_manager.create_event(
            event_type=EventType.DINOSAUR_ESCAPE,
            location=self.test_location,
            parameters=parameters
        )
        
        assert event.type == EventType.DINOSAUR_ESCAPE
        assert event.location == self.test_location
        assert event.parameters == parameters
        assert event.severity == 8  # Default severity for dinosaur escape
        assert event.resolution_status == ResolutionStatus.PENDING
        assert event.id is not None
        assert isinstance(event.timestamp, datetime)
    
    def test_create_event_with_custom_severity(self):
        """Test creating an event with custom severity."""
        parameters = {
            "dinosaur_id": "dino_001",
            "enclosure_id": "enc_001"
        }
        
        event = self.event_manager.create_event(
            event_type=EventType.DINOSAUR_ESCAPE,
            location=self.test_location,
            parameters=parameters,
            severity=5
        )
        
        assert event.severity == 5
    
    def test_create_event_with_custom_description(self):
        """Test creating an event with custom description."""
        parameters = {
            "dinosaur_id": "dino_001",
            "enclosure_id": "enc_001"
        }
        custom_description = "T-Rex has broken through the electric fence"
        
        event = self.event_manager.create_event(
            event_type=EventType.DINOSAUR_ESCAPE,
            location=self.test_location,
            parameters=parameters,
            description=custom_description
        )
        
        assert event.description == custom_description
    
    def test_create_event_missing_required_parameter(self):
        """Test creating an event with missing required parameters."""
        parameters = {
            "enclosure_id": "enc_001"
            # Missing dinosaur_id
        }
        
        with pytest.raises(ValueError, match="Missing required parameter 'dinosaur_id'"):
            self.event_manager.create_event(
                event_type=EventType.DINOSAUR_ESCAPE,
                location=self.test_location,
                parameters=parameters
            )
    
    def test_create_event_invalid_severity(self):
        """Test creating an event with invalid severity."""
        parameters = {
            "dinosaur_id": "dino_001",
            "enclosure_id": "enc_001"
        }
        
        with pytest.raises(ValueError, match="Event severity must be between 1 and 10"):
            self.event_manager.create_event(
                event_type=EventType.DINOSAUR_ESCAPE,
                location=self.test_location,
                parameters=parameters,
                severity=11
            )
    
    def test_distribute_event(self):
        """Test distributing an event."""
        parameters = {
            "dinosaur_id": "dino_001",
            "enclosure_id": "enc_001"
        }
        
        event = self.event_manager.create_event(
            event_type=EventType.DINOSAUR_ESCAPE,
            location=self.test_location,
            parameters=parameters
        )
        
        # Add a mock listener
        mock_listener = Mock()
        self.event_manager.add_event_listener(mock_listener)
        
        self.event_manager.distribute_event(event)
        
        # Check event is in active events
        assert event.id in self.event_manager._active_events
        assert self.event_manager._active_events[event.id] == event
        
        # Check event is in history
        assert event in self.event_manager._event_history
        
        # Check listener was called
        mock_listener.assert_called_once_with(event)
    
    def test_add_remove_event_listener(self):
        """Test adding and removing event listeners."""
        listener1 = Mock()
        listener2 = Mock()
        
        # Add listeners
        self.event_manager.add_event_listener(listener1)
        self.event_manager.add_event_listener(listener2)
        
        assert len(self.event_manager._event_listeners) == 2
        assert listener1 in self.event_manager._event_listeners
        assert listener2 in self.event_manager._event_listeners
        
        # Remove one listener
        self.event_manager.remove_event_listener(listener1)
        
        assert len(self.event_manager._event_listeners) == 1
        assert listener1 not in self.event_manager._event_listeners
        assert listener2 in self.event_manager._event_listeners
    
    def test_add_duplicate_listener(self):
        """Test adding the same listener twice."""
        listener = Mock()
        
        self.event_manager.add_event_listener(listener)
        self.event_manager.add_event_listener(listener)  # Add again
        
        assert len(self.event_manager._event_listeners) == 1
    
    def test_update_event_status(self):
        """Test updating event status."""
        parameters = {
            "dinosaur_id": "dino_001",
            "enclosure_id": "enc_001"
        }
        
        event = self.event_manager.create_event(
            event_type=EventType.DINOSAUR_ESCAPE,
            location=self.test_location,
            parameters=parameters
        )
        
        self.event_manager.distribute_event(event)
        
        # Update to in progress
        affected_agents = ["agent_001", "agent_002"]
        result = self.event_manager.update_event_status(
            event.id, 
            ResolutionStatus.IN_PROGRESS,
            affected_agents
        )
        
        assert result is True
        assert event.resolution_status == ResolutionStatus.IN_PROGRESS
        assert event.affected_agents == affected_agents
        assert event.id in self.event_manager._active_events
        
        # Update to resolved
        result = self.event_manager.update_event_status(event.id, ResolutionStatus.RESOLVED)
        
        assert result is True
        assert event.resolution_status == ResolutionStatus.RESOLVED
        assert event.resolution_time is not None
        assert event.id not in self.event_manager._active_events  # Removed from active
    
    def test_update_nonexistent_event_status(self):
        """Test updating status of non-existent event."""
        result = self.event_manager.update_event_status("nonexistent", ResolutionStatus.RESOLVED)
        assert result is False
    
    def test_get_active_events(self):
        """Test getting active events."""
        # Create and distribute multiple events
        events = []
        for i in range(3):
            parameters = {
                "dinosaur_id": f"dino_{i:03d}",
                "enclosure_id": f"enc_{i:03d}"
            }
            event = self.event_manager.create_event(
                event_type=EventType.DINOSAUR_ESCAPE,
                location=self.test_location,
                parameters=parameters
            )
            self.event_manager.distribute_event(event)
            events.append(event)
        
        active_events = self.event_manager.get_active_events()
        assert len(active_events) == 3
        
        for event in events:
            assert event in active_events
        
        # Resolve one event
        self.event_manager.update_event_status(events[0].id, ResolutionStatus.RESOLVED)
        
        active_events = self.event_manager.get_active_events()
        assert len(active_events) == 2
        assert events[0] not in active_events
    
    def test_get_event_by_id(self):
        """Test getting event by ID."""
        parameters = {
            "dinosaur_id": "dino_001",
            "enclosure_id": "enc_001"
        }
        
        event = self.event_manager.create_event(
            event_type=EventType.DINOSAUR_ESCAPE,
            location=self.test_location,
            parameters=parameters
        )
        
        self.event_manager.distribute_event(event)
        
        # Get active event
        retrieved_event = self.event_manager.get_event_by_id(event.id)
        assert retrieved_event == event
        
        # Resolve event and try to get it from history
        self.event_manager.update_event_status(event.id, ResolutionStatus.RESOLVED)
        retrieved_event = self.event_manager.get_event_by_id(event.id)
        assert retrieved_event == event
        
        # Try to get non-existent event
        retrieved_event = self.event_manager.get_event_by_id("nonexistent")
        assert retrieved_event is None
    
    def test_get_event_history(self):
        """Test getting event history."""
        # Initially empty
        history = self.event_manager.get_event_history()
        assert len(history) == 0
        
        # Create and distribute events
        events = []
        for i in range(5):
            parameters = {
                "dinosaur_id": f"dino_{i:03d}",
                "enclosure_id": f"enc_{i:03d}"
            }
            event = self.event_manager.create_event(
                event_type=EventType.DINOSAUR_ESCAPE,
                location=self.test_location,
                parameters=parameters
            )
            self.event_manager.distribute_event(event)
            events.append(event)
        
        # Get all history
        history = self.event_manager.get_event_history()
        assert len(history) == 5
        
        for event in events:
            assert event in history
        
        # Get limited history
        limited_history = self.event_manager.get_event_history(limit=3)
        assert len(limited_history) == 3
        assert limited_history == events[-3:]  # Last 3 events
    
    def test_check_resolution_status(self):
        """Test checking resolution status."""
        parameters = {
            "dinosaur_id": "dino_001",
            "enclosure_id": "enc_001"
        }
        
        event = self.event_manager.create_event(
            event_type=EventType.DINOSAUR_ESCAPE,
            location=self.test_location,
            parameters=parameters
        )
        
        self.event_manager.distribute_event(event)
        
        # Check initial status
        status = self.event_manager.check_resolution_status(event.id)
        assert status == ResolutionStatus.PENDING
        
        # Update and check status
        self.event_manager.update_event_status(event.id, ResolutionStatus.IN_PROGRESS)
        status = self.event_manager.check_resolution_status(event.id)
        assert status == ResolutionStatus.IN_PROGRESS
        
        # Check non-existent event
        status = self.event_manager.check_resolution_status("nonexistent")
        assert status is None
    
    def test_get_events_by_type(self):
        """Test getting events by type."""
        # Create events of different types
        escape_params = {"dinosaur_id": "dino_001", "enclosure_id": "enc_001"}
        illness_params = {"dinosaur_id": "dino_002", "symptoms": "lethargy"}
        
        escape_event = self.event_manager.create_event(
            EventType.DINOSAUR_ESCAPE, self.test_location, escape_params
        )
        illness_event = self.event_manager.create_event(
            EventType.DINOSAUR_ILLNESS, self.test_location, illness_params
        )
        
        self.event_manager.distribute_event(escape_event)
        self.event_manager.distribute_event(illness_event)
        
        # Get escape events
        escape_events = self.event_manager.get_events_by_type(EventType.DINOSAUR_ESCAPE)
        assert len(escape_events) == 1
        assert escape_event in escape_events
        
        # Get illness events
        illness_events = self.event_manager.get_events_by_type(EventType.DINOSAUR_ILLNESS)
        assert len(illness_events) == 1
        assert illness_event in illness_events
        
        # Get events of type that doesn't exist
        complaint_events = self.event_manager.get_events_by_type(EventType.VISITOR_COMPLAINT)
        assert len(complaint_events) == 0
    
    def test_get_events_by_status(self):
        """Test getting events by status."""
        # Create multiple events
        events = []
        for i in range(3):
            parameters = {
                "dinosaur_id": f"dino_{i:03d}",
                "enclosure_id": f"enc_{i:03d}"
            }
            event = self.event_manager.create_event(
                EventType.DINOSAUR_ESCAPE, self.test_location, parameters
            )
            self.event_manager.distribute_event(event)
            events.append(event)
        
        # All should be pending initially
        pending_events = self.event_manager.get_events_by_status(ResolutionStatus.PENDING)
        assert len(pending_events) == 3
        
        # Update one to in progress
        self.event_manager.update_event_status(events[0].id, ResolutionStatus.IN_PROGRESS)
        
        pending_events = self.event_manager.get_events_by_status(ResolutionStatus.PENDING)
        in_progress_events = self.event_manager.get_events_by_status(ResolutionStatus.IN_PROGRESS)
        
        assert len(pending_events) == 2
        assert len(in_progress_events) == 1
        assert events[0] in in_progress_events
        
        # Resolve one event
        self.event_manager.update_event_status(events[1].id, ResolutionStatus.RESOLVED)
        
        resolved_events = self.event_manager.get_events_by_status(ResolutionStatus.RESOLVED)
        assert len(resolved_events) == 1
        assert events[1] in resolved_events
    
    def test_clear_event_history(self):
        """Test clearing event history."""
        # Create and distribute events
        for i in range(3):
            parameters = {
                "dinosaur_id": f"dino_{i:03d}",
                "enclosure_id": f"enc_{i:03d}"
            }
            event = self.event_manager.create_event(
                EventType.DINOSAUR_ESCAPE, self.test_location, parameters
            )
            self.event_manager.distribute_event(event)
        
        assert len(self.event_manager._event_history) == 3
        
        self.event_manager.clear_event_history()
        
        assert len(self.event_manager._event_history) == 0
        # Active events should remain
        assert len(self.event_manager._active_events) == 3
    
    def test_reset(self):
        """Test resetting the event manager."""
        # Create and distribute events
        for i in range(3):
            parameters = {
                "dinosaur_id": f"dino_{i:03d}",
                "enclosure_id": f"enc_{i:03d}"
            }
            event = self.event_manager.create_event(
                EventType.DINOSAUR_ESCAPE, self.test_location, parameters
            )
            self.event_manager.distribute_event(event)
        
        # Add listeners
        self.event_manager.add_event_listener(Mock())
        
        assert len(self.event_manager._active_events) == 3
        assert len(self.event_manager._event_history) == 3
        assert len(self.event_manager._event_listeners) == 1
        
        self.event_manager.reset()
        
        assert len(self.event_manager._active_events) == 0
        assert len(self.event_manager._event_history) == 0
        assert len(self.event_manager._event_listeners) == 0
    
    def test_get_statistics(self):
        """Test getting event statistics."""
        # Initially empty
        stats = self.event_manager.get_statistics()
        assert stats["total_events"] == 0
        assert stats["active_events"] == 0
        
        # Create events of different types and statuses
        escape_params = {"dinosaur_id": "dino_001", "enclosure_id": "enc_001"}
        illness_params = {"dinosaur_id": "dino_002", "symptoms": "lethargy"}
        
        escape_event = self.event_manager.create_event(
            EventType.DINOSAUR_ESCAPE, self.test_location, escape_params
        )
        illness_event = self.event_manager.create_event(
            EventType.DINOSAUR_ILLNESS, self.test_location, illness_params
        )
        
        self.event_manager.distribute_event(escape_event)
        self.event_manager.distribute_event(illness_event)
        
        # Update one event status
        self.event_manager.update_event_status(escape_event.id, ResolutionStatus.RESOLVED)
        
        stats = self.event_manager.get_statistics()
        
        assert stats["total_events"] == 2
        assert stats["active_events"] == 1
        assert stats["status_distribution"]["PENDING"] == 1
        assert stats["status_distribution"]["RESOLVED"] == 1
        assert stats["type_distribution"]["DINOSAUR_ESCAPE"] == 1
        assert stats["type_distribution"]["DINOSAUR_ILLNESS"] == 1
    
    def test_listener_error_handling(self):
        """Test that listener errors don't break event distribution."""
        # Create a listener that raises an exception
        def failing_listener(event):
            raise Exception("Listener error")
        
        working_listener = Mock()
        
        self.event_manager.add_event_listener(failing_listener)
        self.event_manager.add_event_listener(working_listener)
        
        parameters = {
            "dinosaur_id": "dino_001",
            "enclosure_id": "enc_001"
        }
        
        event = self.event_manager.create_event(
            EventType.DINOSAUR_ESCAPE, self.test_location, parameters
        )
        
        # This should not raise an exception
        self.event_manager.distribute_event(event)
        
        # Working listener should still be called
        working_listener.assert_called_once_with(event)
        
        # Event should still be distributed properly
        assert event.id in self.event_manager._active_events
        assert event in self.event_manager._event_history