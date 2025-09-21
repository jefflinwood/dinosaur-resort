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
            },
            AgentRole.TOURIST: {
                "dinosaur_escape": [
                    "😱 Is that dinosaur supposed to be running around loose?!",
                    "Should we be running? This wasn't in the brochure!",
                    "My kids are terrified! Where are the safety exits?",
                    "This is either the best or worst vacation ever!",
                    "I'm getting this on video - nobody will believe this!"
                ],
                "visitor_complaint": [
                    "The wait times are ridiculous and the food is overpriced!",
                    "We paid premium prices for this experience!",
                    "The staff needs better training on customer service.",
                    "I want to speak to a manager about this situation!"
                ],
                "facility_issue": [
                    "Great, now the power's out. What's next?",
                    "Are we getting refunds for this technical difficulty?",
                    "This place needs better maintenance!",
                    "At least the dinosaurs are still working..."
                ],
                "general": [
                    "This place is amazing! Look at those dinosaurs!",
                    "Are we safe here? Those fences look pretty thin...",
                    "Where's the gift shop? I need souvenirs!",
                    "Can we get closer to the T-Rex? For photos?"
                ]
            },
            AgentRole.DINOSAUR: {
                "dinosaur_escape": [
                    "🦕 *ROOOOOAAARRR* Freedom at last!",
                    "🦖 *sniffs air* Humans smell like... fear and sunscreen.",
                    "🦕 *stomps around* These tiny humans are everywhere!",
                    "🦖 *eyes the fence* That barrier was insulting anyway.",
                    "🦕 *curious growling* What are those flashing things they're pointing at me?"
                ],
                "dinosaur_illness": [
                    "🦕 *weak roar* Not feeling so good...",
                    "🦖 *sluggish movement* Something's wrong with my tummy.",
                    "🦕 *lies down heavily* Just need to rest a bit...",
                    "🦖 *whimpering sounds* Where's the nice vet human?"
                ],
                "visitor_interactions": [
                    "🦕 *tilts head curiously* These small creatures are interesting.",
                    "🦖 *snorts* They keep making those clicking sounds at me.",
                    "🦕 *gentle rumbling* The little ones aren't afraid. I like them.",
                    "🦖 *poses majestically* Yes, admire my magnificent presence!"
                ],
                "general": [
                    "🦕 *contentedly munching* These plants taste different than 65 million years ago.",
                    "🦖 *stretches* Being an apex predator is exhausting work.",
                    "🦕 *social calling* Where are my herd mates?",
                    "🦖 *territorial display* This is MY territory, tiny humans!"
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
        """Generate a quick response for an agent to an event using OpenAI.
        
        Args:
            agent: Agent generating the response
            event: Event to respond to
            
        Returns:
            Quick response string or None
        """
        try:
            # Use OpenAI to generate a response with templates as examples
            response = self._generate_openai_response(agent, event, "event_response")
            
            if response:
                # Add urgency indicators based on severity
                if event.severity >= 8:
                    response = f"🚨 URGENT: {response}"
                elif event.severity >= 6:
                    response = f"⚠️ {response}"
                
                return response
            else:
                # Fallback to template if OpenAI fails
                return self._get_template_fallback(agent, event)
        
        except Exception as e:
            self.logger.error(f"Error generating quick response for {agent.name}: {e}")
            return self._get_template_fallback(agent, event)
    
    def _generate_openai_response(self, agent: Agent, event: Event, response_type: str = "event_response") -> Optional[str]:
        """Generate a response using OpenAI API.
        
        Args:
            agent: Agent generating the response
            event: Event context
            response_type: Type of response (event_response, follow_up, second_wave)
            
        Returns:
            Generated response or None if failed
        """
        try:
            import openai
            
            # Get example responses for this agent role and event type
            examples = self._get_response_examples(agent.role, event, response_type)
            
            # Create role-specific system prompt
            role_descriptions = {
                AgentRole.PARK_RANGER: "You are a park ranger responsible for wildlife management and visitor safety. You are experienced, professional, and focused on coordinating responses.",
                AgentRole.SECURITY: "You are a security officer responsible for park security and visitor protection. You are alert, decisive, and focused on maintaining order.",
                AgentRole.VETERINARIAN: "You are a veterinarian specializing in dinosaur health. You are caring, knowledgeable, and focused on animal welfare.",
                AgentRole.GUEST_RELATIONS: "You are a guest relations manager focused on maintaining visitor satisfaction and spinning situations positively. You create cheerful distractions and always stay upbeat.",
                AgentRole.MAINTENANCE: "You are a maintenance worker responsible for park facilities. You are practical, technical, and focused on keeping systems running.",
                AgentRole.TOURIST: "You are a visitor to the dinosaur park. You may be excited, worried, confused, or demanding depending on the situation. You speak like a regular person, not a professional.",
                AgentRole.DINOSAUR: "You are a dinosaur with natural instincts. You communicate through roars, body language, and simple thoughts. Use emojis like 🦕 🦖 and describe your actions with *asterisks*."
            }
            
            system_prompt = f"""
{role_descriptions.get(agent.role, "You are an agent in a dinosaur park simulation.")}

Respond to the following event with a brief, natural message (1-2 sentences maximum). Stay in character for your role.

Event: {event.type.name.replace('_', ' ').title()} at {event.location.zone}
Severity: {event.severity}/10
Description: {event.description or 'No additional details'}

Examples of good responses for your role:
{chr(10).join(f"- {example}" for example in examples[:3])}

Keep your response brief, natural, and in character. Do not explain your role or repeat the event details.
"""
            
            # Make OpenAI API call with shorter max tokens for speed
            client = openai.OpenAI(api_key=self.openai_config.api_key)
            
            response = client.chat.completions.create(
                model=self.openai_config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Respond to this {event.type.name.replace('_', ' ').lower()} event as {agent.name} ({agent.role.name.replace('_', ' ').title()})."}
                ],
                max_tokens=50,  # Keep responses short for speed
                temperature=0.8,  # Add some variety
                timeout=5  # 5 second timeout for speed
            )
            
            if response.choices and response.choices[0].message.content:
                generated_response = response.choices[0].message.content.strip()
                
                # Clean up the response
                generated_response = generated_response.replace('"', '').replace("'", "'")
                
                # Ensure it's not too long
                if len(generated_response) > 150:
                    generated_response = generated_response[:147] + "..."
                
                return generated_response
            
            return None
        
        except Exception as e:
            self.logger.error(f"OpenAI API error for {agent.name}: {e}")
            return None
    
    def _get_response_examples(self, role: AgentRole, event: Event, response_type: str) -> List[str]:
        """Get example responses for a role and event type.
        
        Args:
            role: Agent role
            event: Event context
            response_type: Type of response needed
            
        Returns:
            List of example responses
        """
        # Get templates as examples for OpenAI
        role_templates = self.response_templates.get(role, {})
        event_category = self._categorize_event(event)
        
        examples = role_templates.get(event_category, [])
        
        if not examples:
            # Get general examples for the role
            all_examples = []
            for category_examples in role_templates.values():
                all_examples.extend(category_examples)
            examples = all_examples[:3] if all_examples else ["Responding to the situation appropriately."]
        
        return examples[:5]  # Limit examples to keep prompt manageable
    
    def _get_template_fallback(self, agent: Agent, event: Event) -> str:
        """Get a template-based fallback response when OpenAI fails.
        
        Args:
            agent: Agent generating the response
            event: Event to respond to
            
        Returns:
            Fallback response string
        """
        # Use the original template logic as fallback
        role_templates = self.response_templates.get(agent.role, {})
        event_category = self._categorize_event(event)
        templates = role_templates.get(event_category, [])
        
        if templates:
            template_index = (self.message_count + hash(agent.id)) % len(templates)
            return templates[template_index]
        else:
            # Simple fallback
            return f"Responding to {event.type.name.replace('_', ' ').lower()} at {event.location.zone}."
    
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
                    delay = 1.0 + (hash(agent_id) % 3)  # 1-4 second delay
                    threading.Timer(delay, 
                                  lambda msg=follow_up_message: self.message_queue.put(msg)).start()
                    
                    # Schedule a second wave of follow-ups for longer conversations
                    if original_message.priority.value >= ChatPriority.HIGH.value:
                        second_wave_delay = 5.0 + (hash(agent_id + "second") % 4)  # 5-9 seconds later
                        second_follow_up = self._generate_second_wave_response(agent, original_message)
                        if second_follow_up:
                            second_message = QuickChatMessage(
                                id=f"msg_{uuid.uuid4().hex[:8]}",
                                sender_id=agent.id,
                                sender_name=agent.name,
                                content=second_follow_up,
                                timestamp=datetime.now(),
                                priority=ChatPriority.NORMAL,
                                event_id=original_message.event_id,
                                message_type="second_wave"
                            )
                            threading.Timer(second_wave_delay,
                                          lambda msg=second_message: self.message_queue.put(msg)).start()
    
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
            AgentRole.PARK_RANGER: [AgentRole.SECURITY, AgentRole.VETERINARIAN, AgentRole.GUEST_RELATIONS, AgentRole.TOURIST],
            AgentRole.SECURITY: [AgentRole.PARK_RANGER, AgentRole.GUEST_RELATIONS, AgentRole.TOURIST],
            AgentRole.VETERINARIAN: [AgentRole.PARK_RANGER, AgentRole.SECURITY, AgentRole.DINOSAUR],
            AgentRole.GUEST_RELATIONS: [AgentRole.TOURIST, AgentRole.PARK_RANGER, AgentRole.SECURITY],
            AgentRole.MAINTENANCE: [AgentRole.SECURITY, AgentRole.GUEST_RELATIONS, AgentRole.TOURIST],
            AgentRole.TOURIST: [AgentRole.GUEST_RELATIONS, AgentRole.PARK_RANGER, AgentRole.SECURITY, AgentRole.DINOSAUR],
            AgentRole.DINOSAUR: [AgentRole.VETERINARIAN, AgentRole.PARK_RANGER, AgentRole.TOURIST]
        }
        
        target_roles = response_patterns.get(sender_role, [])
        
        # Find agents with target roles
        for agent_id, agent in self.active_agents.items():
            if agent.role in target_roles:
                responding_agents.append(agent_id)
        
        # Allow more responses for richer conversations, but limit to prevent spam
        return responding_agents[:5]
    
    def _generate_follow_up_response(self, agent: Agent, original_message: QuickChatMessage) -> Optional[str]:
        """Generate a follow-up response from an agent using OpenAI.
        
        Args:
            agent: Agent generating the follow-up
            original_message: Original message to respond to
            
        Returns:
            Follow-up response string or None
        """
        try:
            # Try OpenAI first for more natural follow-ups
            if original_message.event_id:
                # Create a mock event for context (we don't have the full event object here)
                mock_event = type('MockEvent', (), {
                    'type': type('EventType', (), {'name': 'ONGOING_SITUATION'}),
                    'location': type('Location', (), {'zone': 'incident_area'}),
                    'severity': 5,
                    'description': f"Follow-up to ongoing situation"
                })()
                
                openai_response = self._generate_openai_follow_up(agent, original_message, mock_event)
                if openai_response:
                    return openai_response
            
            # Fallback to templates
            return self._get_follow_up_template(agent)
        
        except Exception as e:
            self.logger.error(f"Error generating follow-up response for {agent.name}: {e}")
            return self._get_follow_up_template(agent)
    
    def _generate_openai_follow_up(self, agent: Agent, original_message: QuickChatMessage, event) -> Optional[str]:
        """Generate a follow-up response using OpenAI.
        
        Args:
            agent: Agent generating the follow-up
            original_message: Original message being responded to
            event: Event context
            
        Returns:
            Generated follow-up response or None
        """
        try:
            import openai
            
            # Get examples for follow-up responses
            examples = self._get_follow_up_examples(agent.role)
            
            system_prompt = f"""
You are {agent.name}, a {agent.role.name.replace('_', ' ').title()} in a dinosaur park. 

Another agent just said: "{original_message.content}"

Respond with a brief follow-up message (1-2 sentences) that shows you're coordinating or reacting to their message. Stay in character.

Examples of good follow-up responses for your role:
{chr(10).join(f"- {example}" for example in examples[:3])}

Keep it brief and natural. Don't repeat what was already said.
"""
            
            client = openai.OpenAI(api_key=self.openai_config.api_key)
            
            response = client.chat.completions.create(
                model=self.openai_config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a brief follow-up response as {agent.name}."}
                ],
                max_tokens=40,  # Even shorter for follow-ups
                temperature=0.9,  # More variety for follow-ups
                timeout=5
            )
            
            if response.choices and response.choices[0].message.content:
                generated_response = response.choices[0].message.content.strip()
                generated_response = generated_response.replace('"', '').replace("'", "'")
                
                if len(generated_response) > 120:
                    generated_response = generated_response[:117] + "..."
                
                return generated_response
            
            return None
        
        except Exception as e:
            self.logger.error(f"OpenAI follow-up error for {agent.name}: {e}")
            return None
    
    def _get_follow_up_examples(self, role: AgentRole) -> List[str]:
        """Get example follow-up responses for a role.
        
        Args:
            role: Agent role
            
        Returns:
            List of example follow-up responses
        """
        follow_up_examples = {
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
            ],
            AgentRole.TOURIST: [
                "Wait, what's happening now?",
                "Should we be worried about this?",
                "Is this part of the show?",
                "Can someone explain what's going on?"
            ],
            AgentRole.DINOSAUR: [
                "🦕 *curious sniffing* What's all the commotion about?",
                "🦖 *alert posture* Something's different in my territory.",
                "🦕 *nervous shuffling* The humans seem agitated."
            ]
        }
        
        return follow_up_examples.get(role, ["Standing by for further instructions."])
    
    def _get_follow_up_template(self, agent: Agent) -> Optional[str]:
        """Get a template-based follow-up response as fallback.
        
        Args:
            agent: Agent generating the follow-up
            
        Returns:
            Template follow-up response or None
        """
        examples = self._get_follow_up_examples(agent.role)
        if examples:
            template_index = (self.message_count + hash(agent.id)) % len(examples)
            return examples[template_index]
        return None
    
    def _generate_second_wave_response(self, agent: Agent, original_message: QuickChatMessage) -> Optional[str]:
        """Generate a second wave follow-up response using OpenAI to keep conversations going.
        
        Args:
            agent: Agent generating the second follow-up
            original_message: Original message that started the conversation
            
        Returns:
            Second wave response string or None
        """
        try:
            # Try OpenAI for more natural second wave responses
            openai_response = self._generate_openai_second_wave(agent, original_message)
            if openai_response:
                return openai_response
            
            # Fallback to templates
            return self._get_second_wave_template(agent, original_message)
        
        except Exception as e:
            self.logger.error(f"Error generating second wave response for {agent.name}: {e}")
            return self._get_second_wave_template(agent, original_message)
    
    def _generate_openai_second_wave(self, agent: Agent, original_message: QuickChatMessage) -> Optional[str]:
        """Generate a second wave response using OpenAI.
        
        Args:
            agent: Agent generating the second wave response
            original_message: Original message that started the conversation
            
        Returns:
            Generated second wave response or None
        """
        try:
            import openai
            
            # Get examples for second wave responses
            examples = self._get_second_wave_examples(agent.role)
            
            system_prompt = f"""
You are {agent.name}, a {agent.role.name.replace('_', ' ').title()} in a dinosaur park.

A few minutes have passed since the initial incident. The original situation was: "{original_message.content}"

Now provide a brief follow-up comment (1-2 sentences) showing how the situation has evolved or your current status. This is a "second wave" response - things might be calming down, or you might have new concerns.

Examples of good second wave responses for your role:
{chr(10).join(f"- {example}" for example in examples[:3])}

Keep it brief and show how time has passed or the situation has evolved.
"""
            
            client = openai.OpenAI(api_key=self.openai_config.api_key)
            
            response = client.chat.completions.create(
                model=self.openai_config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a brief second wave response as {agent.name}."}
                ],
                max_tokens=40,
                temperature=0.9,
                timeout=5
            )
            
            if response.choices and response.choices[0].message.content:
                generated_response = response.choices[0].message.content.strip()
                generated_response = generated_response.replace('"', '').replace("'", "'")
                
                if len(generated_response) > 120:
                    generated_response = generated_response[:117] + "..."
                
                return generated_response
            
            return None
        
        except Exception as e:
            self.logger.error(f"OpenAI second wave error for {agent.name}: {e}")
            return None
    
    def _get_second_wave_examples(self, role: AgentRole) -> List[str]:
        """Get example second wave responses for a role.
        
        Args:
            role: Agent role
            
        Returns:
            List of example second wave responses
        """
        second_wave_examples = {
            AgentRole.TOURIST: [
                "Okay, but seriously, when do we get our money back?",
                "My kids are asking if the dinosaurs are real. What do I tell them?",
                "Is there a manager I can speak to about this situation?",
                "The gift shop better have some good discounts after this!"
            ],
            AgentRole.DINOSAUR: [
                "🦕 *settles down* The excitement seems to be calming down.",
                "🦖 *yawns* All this drama is making me sleepy.",
                "🦕 *returns to grazing* Back to the important business of eating.",
                "🦖 *stretches* Time for my afternoon nap."
            ],
            AgentRole.GUEST_RELATIONS: [
                "📢 Don't forget to visit our newly opened souvenir photo booth!",
                "🎁 Complimentary dinosaur plushies for all affected guests!",
                "🍦 The ice cream cart is now offering double scoops!",
                "📸 Professional photographers available for family photos!"
            ],
            AgentRole.SECURITY: [
                "All clear - situation is under control.",
                "Resuming normal patrol patterns.",
                "Incident report filed and documented.",
                "Additional security measures implemented."
            ],
            AgentRole.PARK_RANGER: [
                "Wildlife behavior returning to normal patterns.",
                "All safety protocols have been effective.",
                "Continuing to monitor the situation closely.",
                "Coordinating with all teams for full resolution."
            ],
            AgentRole.VETERINARIAN: [
                "All animals showing normal vital signs.",
                "No medical intervention required at this time.",
                "Continuing health monitoring protocols.",
                "Ready to respond if medical needs arise."
            ]
        }
        
        return second_wave_examples.get(role, ["Situation continuing to develop."])
    
    def _get_second_wave_template(self, agent: Agent, original_message: QuickChatMessage) -> Optional[str]:
        """Get a template-based second wave response as fallback.
        
        Args:
            agent: Agent generating the second wave response
            original_message: Original message context
            
        Returns:
            Template second wave response or None
        """
        examples = self._get_second_wave_examples(agent.role)
        if examples:
            template_index = (hash(agent.id + original_message.id + "wave2")) % len(examples)
            return examples[template_index]
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