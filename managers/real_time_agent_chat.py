"""Real-time agent chat system for immediate agent interactions."""

import logging
import asyncio
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from queue import Queue, Empty
from dataclasses import dataclass
from enum import Enum

from models.core import Agent, Event, ChatMessage
from models.enums import AgentRole, AgentState, MessageType
from models.config import OpenAIConfig


class ChatPriority(Enum):
    """Priority levels for chat messages."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class QuickChatMessage:
    """Quick chat message for real-time agent communication."""
    id: str
    sender_id: str
    sender_name: str
    content: str
    timestamp: datetime
    priority: ChatPriority = ChatPriority.NORMAL
    event_id: Optional[str] = None
    recipients: Optional[List[str]] = None  # None means broadcast to all
    message_type: str = "agent_response"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name,
            "event_id": self.event_id,
            "recipients": self.recipients,
            "message_type": self.message_type
        }


class RealTimeAgentChat:
    """Manages real-time chat between agents with immediate responses."""
    
    def __init__(self, openai_config: OpenAIConfig):
        """Initialize real-time agent chat system.
        
        Args:
            openai_config: OpenAI configuration for generating responses
        """
        self.openai_config = openai_config
        self.logger = logging.getLogger(__name__)
        
        # Chat infrastructure
        self.message_queue = Queue()
        self.active_agents: Dict[str, Agent] = {}
        self.chat_history: List[QuickChatMessage] = []
        self.message_callbacks: List[Callable[[QuickChatMessage], None]] = []
        
        # Real-time processing
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.response_cache: Dict[str, List[str]] = {}  # Pre-generated responses
        
        # Performance tracking
        self.message_count = 0
        self.average_response_time = 0.0
        self.last_activity = datetime.now()
        
        # Initialize response templates
        self._initialize_response_templates()
        
        self.logger.info("RealTimeAgentChat initialized")
    
    def _initialize_response_templates(self) -> None:
        """Initialize quick response templates for different scenarios."""
        self.response_templates = {
            AgentRole.PARK_RANGER: {
                "dinosaur_escape": [
                    "Initiating containment protocol for escaped dinosaur!",
                    "Securing perimeter and evacuating visitors from danger zone.",
                    "Coordinating with security team for safe recapture.",
                    "All visitors please move to designated safe areas immediately."
                ],
                "visitor_injury": [
                    "Medical team dispatched to visitor location.",
                    "Securing area and providing first aid assistance.",
                    "Coordinating with veterinarian for any animal involvement."
                ],
                "facility_issue": [
                    "Assessing facility damage and visitor safety impact.",
                    "Coordinating with maintenance for immediate repairs.",
                    "Implementing backup safety protocols."
                ]
            },
            AgentRole.SECURITY: {
                "dinosaur_escape": [
                    "Security perimeter established around escape zone.",
                    "All exits secured, controlling visitor movement.",
                    "Backup security teams en route to assist.",
                    "Coordinating with rangers for containment strategy."
                ],
                "visitor_emergency": [
                    "Security responding to visitor emergency.",
                    "Area secured, emergency services notified.",
                    "Crowd control measures in effect."
                ],
                "facility_issue": [
                    "Security sweep of affected facility areas.",
                    "Access restrictions implemented for safety.",
                    "Monitoring for any security vulnerabilities."
                ]
            },
            AgentRole.VETERINARIAN: {
                "dinosaur_illness": [
                    "Immediate medical assessment of affected dinosaur.",
                    "Preparing treatment protocols and medications.",
                    "Isolating patient to prevent spread if contagious."
                ],
                "dinosaur_injury": [
                    "Emergency veterinary care being administered.",
                    "Stabilizing patient and assessing injury severity.",
                    "Coordinating with rangers for safe treatment access."
                ],
                "dinosaur_escape": [
                    "Preparing tranquilizer equipment for safe recapture.",
                    "Assessing dinosaur stress levels and behavior.",
                    "Ready to provide medical care post-recapture."
                ]
            },
            AgentRole.GUEST_RELATIONS: {
                "dinosaur_escape": [
                    "🍦 ATTENTION GUESTS: Free ice cream at the entrance plaza!",
                    "🎪 Special surprise show starting at the gift shop!",
                    "📸 Exclusive photo opportunities with our friendly staff dinosaurs!",
                    "🎁 Limited time 50% off all souvenirs while we enhance your experience!"
                ],
                "visitor_complaint": [
                    "We sincerely apologize and want to make this right immediately!",
                    "Complimentary fast-pass tickets and gift shop vouchers coming your way!",
                    "Let me personally ensure your visit becomes memorable for all the right reasons!"
                ],
                "facility_issue": [
                    "🎭 Surprise outdoor entertainment starting now on the main lawn!",
                    "🦕 Special behind-the-scenes dinosaur feeding demonstration!",
                    "🎪 Pop-up educational presentations while we enhance facilities!"
                ]
            },
            AgentRole.MAINTENANCE: {
                "facility_power_outage": [
                    "Emergency generators activated, restoring power systems.",
                    "Backup lighting and safety systems operational.",
                    "Estimated repair time: 15-30 minutes."
                ],
                "equipment_failure": [
                    "Technical team dispatched to equipment location.",
                    "Implementing backup systems and safety protocols.",
                    "Coordinating with operations for minimal disruption."
                ],
                "facility_damage": [
                    "Damage assessment complete, beginning immediate repairs.",
                    "Safety barriers installed, area secured for work.",
                    "Coordinating with security for visitor safety."
                ]
            }
        }
    
    def start_real_time_processing(self) -> None:
        """Start the real-time message processing thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_messages_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("Real-time chat processing started")
    
    def stop_real_time_processing(self) -> None:
        """Stop the real-time message processing."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self.logger.info("Real-time chat processing stopped")
    
    def register_agent(self, agent: Agent) -> None:
        """Register an agent for real-time chat.
        
        Args:
            agent: Agent to register
        """
        self.active_agents[agent.id] = agent
        self.logger.debug(f"Registered agent {agent.name} ({agent.role.name}) for real-time chat")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from real-time chat.
        
        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id in self.active_agents:
            agent_name = self.active_agents[agent_id].name
            del self.active_agents[agent_id]
            self.logger.debug(f"Unregistered agent {agent_name} from real-time chat")
    
    def add_message_callback(self, callback: Callable[[QuickChatMessage], None]) -> None:
        """Add a callback to be called when new messages are processed.
        
        Args:
            callback: Function to call with new messages
        """
        self.message_callbacks.append(callback)
    
    def trigger_event_response(self, event: Event, affected_agent_ids: List[str]) -> None:
        """Trigger immediate agent responses to an event.
        
        Args:
            event: Event that occurred
            affected_agent_ids: List of agent IDs that should respond
        """
        self.logger.info(f"Triggering real-time responses for event {event.id} ({event.type.name})")
        
        # Generate quick responses for each affected agent
        for agent_id in affected_agent_ids:
            if agent_id in self.active_agents:
                agent = self.active_agents[agent_id]
                response = self._generate_quick_response(agent, event)
                
                if response:
                    message = QuickChatMessage(
                        id=f"msg_{uuid.uuid4().hex[:8]}",
                        sender_id=agent.id,
                        sender_name=agent.name,
                        content=response,
                        timestamp=datetime.now(),
                        priority=self._get_message_priority(event),
                        event_id=event.id,
                        message_type="event_response"
                    )
                    
                    # Queue message for processing
                    self.message_queue.put(message)
    
    def _generate_quick_response(self, agent: Agent, event: Event) -> Optional[str]:
        """Generate a quick response for an agent to an event.
        
        Args:
            agent: Agent generating the response
            event: Event to respond to
            
        Returns:
            Quick response string or None
        """
        try:
            # Get response templates for this agent role
            role_templates = self.response_templates.get(agent.role, {})
            
            # Determine event category
            event_category = self._categorize_event(event)
            
            # Get appropriate templates
            templates = role_templates.get(event_category, [])
            
            if templates:
                # Use a simple rotation or random selection for variety
                template_index = (self.message_count + hash(agent.id)) % len(templates)
                base_response = templates[template_index]
                
                # Add some context if needed
                if event.severity >= 8:
                    base_response = f"🚨 URGENT: {base_response}"
                elif event.severity >= 6:
                    base_response = f"⚠️ {base_response}"
                
                return base_response
            else:
                # More interesting fallback responses based on role
                role_fallbacks = {
                    AgentRole.PARK_RANGER: [
                        f"Assessing situation at {event.location.zone} and coordinating response.",
                        f"Implementing safety protocols for {event.location.zone} area.",
                        f"Mobilizing ranger team to handle {event.type.name.replace('_', ' ').lower()}."
                    ],
                    AgentRole.SECURITY: [
                        f"Securing {event.location.zone} and establishing safety perimeter.",
                        f"All security units responding to {event.location.zone}.",
                        f"Implementing emergency lockdown procedures for {event.location.zone}."
                    ],
                    AgentRole.VETERINARIAN: [
                        f"Medical team en route to {event.location.zone}.",
                        f"Preparing emergency medical protocols for {event.location.zone}.",
                        f"Veterinary response team mobilizing to {event.location.zone}."
                    ],
                    AgentRole.GUEST_RELATIONS: [
                        f"🎪 Exciting activities starting near {event.location.zone}!",
                        f"🍿 Special entertainment beginning at the visitor center!",
                        f"📸 Unique photo opportunities available at the gift shop!"
                    ],
                    AgentRole.MAINTENANCE: [
                        f"Technical team responding to {event.location.zone}.",
                        f"Checking all systems in {event.location.zone} area.",
                        f"Maintenance protocols activated for {event.location.zone}."
                    ]
                }
                
                fallbacks = role_fallbacks.get(agent.role, [
                    f"Responding to situation at {event.location.zone}.",
                    f"Taking appropriate action for {event.type.name.replace('_', ' ').lower()}."
                ])
                
                # Use hash for consistent but varied responses
                fallback_index = hash(agent.id + event.id) % len(fallbacks)
                return fallbacks[fallback_index]
        
        except Exception as e:
            self.logger.error(f"Error generating quick response for {agent.name}: {e}")
            return f"Acknowledged {event.type.name.replace('_', ' ').lower()}. Taking appropriate action."
    
    def _categorize_event(self, event: Event) -> str:
        """Categorize an event for response template selection.
        
        Args:
            event: Event to categorize
            
        Returns:
            Event category string
        """
        event_type = event.type.name.lower()
        
        if "escape" in event_type:
            return "dinosaur_escape"
        elif "illness" in event_type:
            return "dinosaur_illness"
        elif "aggressive" in event_type:
            return "dinosaur_escape"  # Treat as escape scenario
        elif "injury" in event_type:
            if "visitor" in event_type:
                return "visitor_injury"
            else:
                return "dinosaur_injury"
        elif "complaint" in event_type:
            return "visitor_complaint"
        elif "emergency" in event_type:
            return "visitor_emergency"
        elif "power" in event_type:
            return "facility_power_outage"
        elif "equipment" in event_type:
            return "equipment_failure"
        elif "facility" in event_type:
            return "facility_issue"
        else:
            return "general"
    
    def _get_message_priority(self, event: Event) -> ChatPriority:
        """Get message priority based on event severity.
        
        Args:
            event: Event to assess
            
        Returns:
            ChatPriority level
        """
        if event.severity >= 9:
            return ChatPriority.URGENT
        elif event.severity >= 7:
            return ChatPriority.HIGH
        elif event.severity >= 4:
            return ChatPriority.NORMAL
        else:
            return ChatPriority.LOW
    
    def _process_messages_loop(self) -> None:
        """Main message processing loop running in separate thread."""
        self.logger.info("Starting real-time message processing loop")
        
        while self.is_running:
            try:
                # Get message from queue with timeout
                try:
                    message = self.message_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Process the message
                self._process_message(message)
                
                # Update statistics
                self.message_count += 1
                self.last_activity = datetime.now()
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                time.sleep(1.0)  # Longer delay on error
        
        self.logger.info("Real-time message processing loop stopped")
    
    def _process_message(self, message: QuickChatMessage) -> None:
        """Process a single chat message.
        
        Args:
            message: Message to process
        """
        try:
            # Add to chat history
            self.chat_history.append(message)
            
            # Keep history manageable
            if len(self.chat_history) > 1000:
                self.chat_history = self.chat_history[-500:]
            
            # Update agent state
            if message.sender_id in self.active_agents:
                agent = self.active_agents[message.sender_id]
                agent.current_state = AgentState.COMMUNICATING
                agent.last_activity = datetime.now()
            
            # Call registered callbacks (if any)
            for callback in self.message_callbacks:
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in message callback: {e}")
                    # Remove problematic callbacks to prevent repeated errors
                    self.message_callbacks.remove(callback)
            
            # Trigger follow-up responses if needed
            self._trigger_follow_up_responses(message)
            
            self.logger.debug(f"Processed message from {message.sender_name}: {message.content[:50]}...")
        
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    def _trigger_follow_up_responses(self, original_message: QuickChatMessage) -> None:
        """Trigger follow-up responses from other agents.
        
        Args:
            original_message: Original message that might trigger responses
        """
        # Only trigger follow-ups for high priority messages or specific scenarios
        if original_message.priority.value < ChatPriority.HIGH.value:
            return
        
        # Don't trigger follow-ups for follow-ups (prevent loops)
        if original_message.message_type == "follow_up":
            return
        
        # Find agents that should respond
        responding_agents = self._get_responding_agents(original_message)
        
        for agent_id in responding_agents:
            if agent_id in self.active_agents and agent_id != original_message.sender_id:
                agent = self.active_agents[agent_id]
                follow_up = self._generate_follow_up_response(agent, original_message)
                
                if follow_up:
                    follow_up_message = QuickChatMessage(
                        id=f"msg_{uuid.uuid4().hex[:8]}",
                        sender_id=agent.id,
                        sender_name=agent.name,
                        content=follow_up,
                        timestamp=datetime.now(),
                        priority=ChatPriority.NORMAL,
                        event_id=original_message.event_id,
                        message_type="follow_up"
                    )
                    
                    # Add small delay to make conversation feel natural
                    threading.Timer(1.0 + (hash(agent_id) % 3), 
                                  lambda: self.message_queue.put(follow_up_message)).start()
    
    def _get_responding_agents(self, message: QuickChatMessage) -> List[str]:
        """Get list of agents that should respond to a message.
        
        Args:
            message: Message to respond to
            
        Returns:
            List of agent IDs that should respond
        """
        responding_agents = []
        
        # Get sender's role
        sender_agent = self.active_agents.get(message.sender_id)
        if not sender_agent:
            return responding_agents
        
        sender_role = sender_agent.role
        
        # Define response patterns based on sender role
        response_patterns = {
            AgentRole.PARK_RANGER: [AgentRole.SECURITY, AgentRole.VETERINARIAN, AgentRole.GUEST_RELATIONS],
            AgentRole.SECURITY: [AgentRole.PARK_RANGER, AgentRole.GUEST_RELATIONS],
            AgentRole.VETERINARIAN: [AgentRole.PARK_RANGER, AgentRole.SECURITY],
            AgentRole.GUEST_RELATIONS: [AgentRole.PARK_RANGER, AgentRole.SECURITY],
            AgentRole.MAINTENANCE: [AgentRole.SECURITY, AgentRole.GUEST_RELATIONS]
        }
        
        target_roles = response_patterns.get(sender_role, [])
        
        # Find agents with target roles
        for agent_id, agent in self.active_agents.items():
            if agent.role in target_roles:
                responding_agents.append(agent_id)
        
        # Limit to 2-3 responses to avoid spam
        return responding_agents[:3]
    
    def _generate_follow_up_response(self, agent: Agent, original_message: QuickChatMessage) -> Optional[str]:
        """Generate a follow-up response from an agent.
        
        Args:
            agent: Agent generating the follow-up
            original_message: Original message to respond to
            
        Returns:
            Follow-up response string or None
        """
        try:
            # Get role-specific follow-up templates
            follow_up_templates = {
                AgentRole.SECURITY: [
                    "Security team standing by to assist.",
                    "Perimeter secured, ready for coordination.",
                    "All security protocols activated."
                ],
                AgentRole.GUEST_RELATIONS: [
                    "Implementing guest comfort measures now!",
                    "Positive messaging campaign activated!",
                    "Guest satisfaction protocols in effect!"
                ],
                AgentRole.VETERINARIAN: [
                    "Medical team ready to support as needed.",
                    "Health and safety protocols confirmed.",
                    "Standing by for any medical assistance."
                ],
                AgentRole.PARK_RANGER: [
                    "Coordinating response with all teams.",
                    "Wildlife safety measures confirmed.",
                    "All ranger units coordinated and ready."
                ],
                AgentRole.MAINTENANCE: [
                    "Technical support standing by.",
                    "All systems checked and operational.",
                    "Maintenance team ready to assist."
                ]
            }
            
            templates = follow_up_templates.get(agent.role, [])
            if templates:
                # Simple rotation for variety
                template_index = (self.message_count + hash(agent.id)) % len(templates)
                return templates[template_index]
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error generating follow-up response for {agent.name}: {e}")
            return None
    
    def get_recent_messages(self, count: int = 20) -> List[QuickChatMessage]:
        """Get recent chat messages.
        
        Args:
            count: Number of recent messages to return
            
        Returns:
            List of recent messages
        """
        return self.chat_history[-count:] if self.chat_history else []
    
    def get_messages_for_event(self, event_id: str) -> List[QuickChatMessage]:
        """Get all messages related to a specific event.
        
        Args:
            event_id: Event ID to filter by
            
        Returns:
            List of messages for the event
        """
        return [msg for msg in self.chat_history if msg.event_id == event_id]
    
    def clear_chat_history(self) -> None:
        """Clear all chat history."""
        self.chat_history.clear()
        self.message_count = 0
        self.logger.info("Chat history cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chat system statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_messages": len(self.chat_history),
            "active_agents": len(self.active_agents),
            "message_count": self.message_count,
            "last_activity": self.last_activity.isoformat(),
            "is_running": self.is_running,
            "queue_size": self.message_queue.qsize()
        }