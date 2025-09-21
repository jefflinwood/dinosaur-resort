"""Human player manager system for role selection and agent creation."""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from models.core import HumanAgent, ChatMessage, Agent
from models.config import Location
from models.enums import AgentRole, AgentState, MessageType
from agents.base_agent import DinosaurAgent, BaseAgentConfig


class ConversationContext:
    """Context for human player conversations."""
    
    def __init__(self, conversation_id: str, participants: List[str], topic: str = ""):
        """Initialize conversation context.
        
        Args:
            conversation_id: Unique identifier for the conversation
            participants: List of participant agent IDs
            topic: Optional topic or subject of the conversation
        """
        self.conversation_id = conversation_id
        self.participants = participants
        self.topic = topic
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.is_active = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "participants": self.participants,
            "topic": self.topic,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create from dictionary."""
        context = cls(
            conversation_id=data["conversation_id"],
            participants=data["participants"],
            topic=data.get("topic", "")
        )
        context.created_at = datetime.fromisoformat(data["created_at"])
        context.last_activity = datetime.fromisoformat(data["last_activity"])
        context.is_active = data.get("is_active", True)
        return context


class HumanPlayerManager:
    """Manages human player participation in the simulation as an agent."""
    
    def __init__(self, base_config: BaseAgentConfig):
        """Initialize human player manager.
        
        Args:
            base_config: Base agent configuration for creating human agents
        """
        self.base_config = base_config
        self.logger = logging.getLogger(__name__)
        
        # Human player storage
        self.human_players: Dict[str, HumanAgent] = {}  # player_id -> HumanAgent
        self.active_conversations: Dict[str, ConversationContext] = {}  # conversation_id -> context
        self.role_permissions: Dict[AgentRole, List[str]] = self._initialize_role_permissions()
        
        # Timeout settings
        self.inactivity_timeout = timedelta(minutes=10)  # 10 minutes of inactivity
        self.last_activity_check = datetime.now()
        
        self.logger.info("HumanPlayerManager initialized")
    
    def _initialize_role_permissions(self) -> Dict[AgentRole, List[str]]:
        """Initialize role-based permissions for conversation access.
        
        Returns:
            Dictionary mapping roles to their conversation permissions
        """
        return {
            AgentRole.PARK_RANGER: [
                "staff_coordination", "emergency_response", "visitor_safety", 
                "dinosaur_management", "facility_operations"
            ],
            AgentRole.VETERINARIAN: [
                "staff_coordination", "emergency_response", "dinosaur_health", 
                "medical_treatment", "facility_operations"
            ],
            AgentRole.SECURITY: [
                "staff_coordination", "emergency_response", "visitor_safety", 
                "threat_assessment", "facility_security"
            ],
            AgentRole.MAINTENANCE: [
                "staff_coordination", "facility_operations", "equipment_repair", 
                "technical_support"
            ],
            AgentRole.TOURIST: [
                "visitor_interactions", "general_inquiries", "feedback"
            ],
            AgentRole.DINOSAUR: [
                "dinosaur_interactions", "environmental_response"
            ]
        }
    
    def create_human_agent(self, role: AgentRole, player_name: str, 
                          location: Optional[Location] = None) -> HumanAgent:
        """Create a human-controlled agent with the appropriate role characteristics.
        
        Args:
            role: Agent role for the human player
            player_name: Display name for the human player
            location: Optional initial location
            
        Returns:
            HumanAgent instance configured for the specified role
        """
        try:
            # Generate unique ID
            player_id = f"human_{uuid.uuid4().hex[:8]}"
            
            # Set default location if not provided
            if location is None:
                location = Location(0.0, 0.0, "entrance", "Main entrance area")
            
            # Get role-specific configuration
            personality_traits = self.base_config.get_default_personality(role)
            capabilities = self.base_config.get_default_capabilities(role)
            
            # Create human agent
            human_agent = HumanAgent(
                id=player_id,
                name=player_name,
                role=role,
                personality_traits=personality_traits,
                current_state=AgentState.IDLE,
                location=location,
                capabilities=capabilities,
                is_human_controlled=True,
                chat_history=[],
                conversation_access=[]
            )
            
            # Set up conversation access based on role
            self._setup_conversation_access(human_agent)
            
            # Store the human agent
            self.human_players[player_id] = human_agent
            
            self.logger.info(f"Created human agent {player_name} with role {role.name}")
            return human_agent
        
        except Exception as e:
            self.logger.error(f"Error creating human agent: {e}")
            raise
    
    def _setup_conversation_access(self, human_agent: HumanAgent) -> None:
        """Set up conversation access permissions for a human agent.
        
        Args:
            human_agent: Human agent to configure
        """
        role_permissions = self.role_permissions.get(human_agent.role, [])
        
        # Add role-specific conversation access
        for permission in role_permissions:
            human_agent.add_conversation_access(permission)
        
        # Add general access for all roles
        human_agent.add_conversation_access("general")
        
        self.logger.debug(f"Set up conversation access for {human_agent.name}: {role_permissions}")
    
    def update_human_role(self, player_id: str, new_role: AgentRole) -> bool:
        """Update a human player's role and permissions.
        
        Args:
            player_id: ID of the human player
            new_role: New role to assign
            
        Returns:
            True if role was updated successfully, False otherwise
        """
        try:
            if player_id not in self.human_players:
                self.logger.warning(f"Human player {player_id} not found")
                return False
            
            human_agent = self.human_players[player_id]
            old_role = human_agent.role
            
            # Update role
            human_agent.role = new_role
            
            # Update personality traits and capabilities
            human_agent.personality_traits = self.base_config.get_default_personality(new_role)
            human_agent.capabilities = self.base_config.get_default_capabilities(new_role)
            
            # Clear and reset conversation access
            human_agent.conversation_access.clear()
            self._setup_conversation_access(human_agent)
            
            # Update last activity
            human_agent.last_activity = datetime.now()
            
            self.logger.info(f"Updated human player {human_agent.name} role from {old_role.name} to {new_role.name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error updating human role for {player_id}: {e}")
            return False
    
    def process_human_input(self, player_id: str, message: str, 
                           context: ConversationContext) -> bool:
        """Process human player input and integrate with agent conversation system.
        
        Args:
            player_id: ID of the human player
            message: Message content from human player
            context: Conversation context
            
        Returns:
            True if message was processed successfully, False otherwise
        """
        try:
            if player_id not in self.human_players:
                self.logger.warning(f"Human player {player_id} not found")
                return False
            
            human_agent = self.human_players[player_id]
            
            # Check if human has access to this conversation
            if not self._has_conversation_access(human_agent, context):
                self.logger.warning(f"Human player {human_agent.name} denied access to conversation {context.conversation_id}")
                return False
            
            # Create chat message
            chat_message = ChatMessage(
                id=f"msg_{uuid.uuid4().hex[:8]}",
                sender_id=human_agent.id,
                sender_name=human_agent.name,
                content=message,
                timestamp=datetime.now(),
                message_type=MessageType.HUMAN,
                conversation_id=context.conversation_id
            )
            
            # Add to human agent's chat history
            human_agent.add_chat_message(chat_message)
            
            # Update conversation context
            context.last_activity = datetime.now()
            
            # Update human agent state
            human_agent.current_state = AgentState.COMMUNICATING
            human_agent.last_activity = datetime.now()
            
            self.logger.info(f"Processed input from human player {human_agent.name} in conversation {context.conversation_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error processing human input: {e}")
            return False
    
    def _has_conversation_access(self, human_agent: HumanAgent, context: ConversationContext) -> bool:
        """Check if human agent has access to a conversation.
        
        Args:
            human_agent: Human agent to check
            context: Conversation context
            
        Returns:
            True if human has access, False otherwise
        """
        # Check if human is a participant
        if human_agent.id in context.participants:
            return True
        
        # Check if human has explicit access to this conversation
        if human_agent.has_conversation_access(context.conversation_id):
            return True
        
        # Check role-based permissions
        role_permissions = self.role_permissions.get(human_agent.role, [])
        
        # Check if conversation topic matches role permissions
        if context.topic in role_permissions:
            return True
        
        # Check for general access only if topic is "general" or empty
        if (not context.topic or context.topic == "general") and human_agent.has_conversation_access("general"):
            return True
        
        return False
    
    def get_relevant_conversations(self, player_id: str) -> List[ConversationContext]:
        """Get conversations that are relevant to a human player based on their role.
        
        Args:
            player_id: ID of the human player
            
        Returns:
            List of relevant conversation contexts
        """
        if player_id not in self.human_players:
            return []
        
        human_agent = self.human_players[player_id]
        relevant_conversations = []
        
        for conversation_id, context in self.active_conversations.items():
            if self._has_conversation_access(human_agent, context) and context.is_active:
                relevant_conversations.append(context)
        
        # Sort by last activity (most recent first)
        relevant_conversations.sort(key=lambda x: x.last_activity, reverse=True)
        
        return relevant_conversations
    
    def create_conversation_context(self, participants: List[str], topic: str = "") -> ConversationContext:
        """Create a new conversation context.
        
        Args:
            participants: List of participant agent IDs
            topic: Optional conversation topic
            
        Returns:
            New conversation context
        """
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        context = ConversationContext(conversation_id, participants, topic)
        
        self.active_conversations[conversation_id] = context
        
        self.logger.info(f"Created conversation context {conversation_id} with {len(participants)} participants")
        return context
    
    def add_human_to_conversation(self, player_id: str, conversation_id: str) -> bool:
        """Add a human player to an existing conversation.
        
        Args:
            player_id: ID of the human player
            conversation_id: ID of the conversation
            
        Returns:
            True if human was added successfully, False otherwise
        """
        try:
            if player_id not in self.human_players:
                self.logger.warning(f"Human player {player_id} not found")
                return False
            
            if conversation_id not in self.active_conversations:
                self.logger.warning(f"Conversation {conversation_id} not found")
                return False
            
            human_agent = self.human_players[player_id]
            context = self.active_conversations[conversation_id]
            
            # Check if human has access to this conversation
            if not self._has_conversation_access(human_agent, context):
                self.logger.warning(f"Human player {human_agent.name} denied access to conversation {conversation_id}")
                return False
            
            # Add human to participants if not already there
            if human_agent.id not in context.participants:
                context.participants.append(human_agent.id)
                context.last_activity = datetime.now()
            
            # Grant explicit access to the conversation
            human_agent.add_conversation_access(conversation_id)
            
            self.logger.info(f"Added human player {human_agent.name} to conversation {conversation_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error adding human to conversation: {e}")
            return False
    
    def remove_human_from_conversation(self, player_id: str, conversation_id: str) -> bool:
        """Remove a human player from a conversation.
        
        Args:
            player_id: ID of the human player
            conversation_id: ID of the conversation
            
        Returns:
            True if human was removed successfully, False otherwise
        """
        try:
            if player_id not in self.human_players:
                return False
            
            if conversation_id not in self.active_conversations:
                return False
            
            human_agent = self.human_players[player_id]
            context = self.active_conversations[conversation_id]
            
            # Remove from participants
            if human_agent.id in context.participants:
                context.participants.remove(human_agent.id)
                context.last_activity = datetime.now()
            
            # Remove conversation access
            human_agent.remove_conversation_access(conversation_id)
            
            self.logger.info(f"Removed human player {human_agent.name} from conversation {conversation_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error removing human from conversation: {e}")
            return False
    
    def handle_player_inactivity(self, timeout_seconds: int = 600) -> List[str]:
        """Handle graceful simulation continuation when human players are inactive.
        
        Args:
            timeout_seconds: Inactivity timeout in seconds (default 10 minutes)
            
        Returns:
            List of player IDs that were marked as inactive
        """
        current_time = datetime.now()
        inactive_players = []
        
        for player_id, human_agent in self.human_players.items():
            time_since_activity = (current_time - human_agent.last_activity).total_seconds()
            
            if time_since_activity > timeout_seconds:
                # Mark as inactive
                if human_agent.current_state != AgentState.UNAVAILABLE:
                    human_agent.current_state = AgentState.UNAVAILABLE
                    inactive_players.append(player_id)
                    
                    self.logger.info(f"Human player {human_agent.name} marked as inactive after {time_since_activity:.0f} seconds")
        
        self.last_activity_check = current_time
        return inactive_players
    
    def reactivate_player(self, player_id: str) -> bool:
        """Reactivate an inactive human player.
        
        Args:
            player_id: ID of the human player
            
        Returns:
            True if player was reactivated, False otherwise
        """
        try:
            if player_id not in self.human_players:
                return False
            
            human_agent = self.human_players[player_id]
            
            # Update state and activity
            human_agent.current_state = AgentState.IDLE
            human_agent.last_activity = datetime.now()
            
            self.logger.info(f"Reactivated human player {human_agent.name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error reactivating player {player_id}: {e}")
            return False
    
    def get_human_player(self, player_id: str) -> Optional[HumanAgent]:
        """Get a human player by ID.
        
        Args:
            player_id: ID of the human player
            
        Returns:
            HumanAgent instance or None if not found
        """
        return self.human_players.get(player_id)
    
    def get_all_human_players(self) -> Dict[str, HumanAgent]:
        """Get all human players.
        
        Returns:
            Dictionary of player_id -> HumanAgent
        """
        return self.human_players.copy()
    
    def get_human_players_by_role(self, role: AgentRole) -> List[HumanAgent]:
        """Get human players with a specific role.
        
        Args:
            role: Agent role to filter by
            
        Returns:
            List of HumanAgent instances with the specified role
        """
        return [agent for agent in self.human_players.values() if agent.role == role]
    
    def get_active_human_players(self) -> List[HumanAgent]:
        """Get all active (non-inactive) human players.
        
        Returns:
            List of active HumanAgent instances
        """
        return [
            agent for agent in self.human_players.values() 
            if agent.current_state != AgentState.UNAVAILABLE
        ]
    
    def get_human_chat_history(self, player_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get chat history for a human player.
        
        Args:
            player_id: ID of the human player
            limit: Maximum number of messages to return
            
        Returns:
            List of recent chat messages
        """
        if player_id not in self.human_players:
            return []
        
        human_agent = self.human_players[player_id]
        return human_agent.get_recent_messages(limit)
    
    def clear_human_chat_history(self, player_id: str) -> bool:
        """Clear chat history for a human player.
        
        Args:
            player_id: ID of the human player
            
        Returns:
            True if history was cleared, False otherwise
        """
        try:
            if player_id not in self.human_players:
                return False
            
            human_agent = self.human_players[player_id]
            human_agent.clear_chat_history()
            
            self.logger.info(f"Cleared chat history for human player {human_agent.name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error clearing chat history for {player_id}: {e}")
            return False
    
    def remove_human_player(self, player_id: str) -> bool:
        """Remove a human player from the system.
        
        Args:
            player_id: ID of the human player to remove
            
        Returns:
            True if player was removed, False otherwise
        """
        try:
            if player_id not in self.human_players:
                return False
            
            human_agent = self.human_players[player_id]
            
            # Remove from all conversations
            for conversation_id in list(self.active_conversations.keys()):
                self.remove_human_from_conversation(player_id, conversation_id)
            
            # Remove from storage
            del self.human_players[player_id]
            
            self.logger.info(f"Removed human player {human_agent.name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error removing human player {player_id}: {e}")
            return False
    
    def get_conversation_participants(self, conversation_id: str) -> List[str]:
        """Get participants of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of participant agent IDs
        """
        if conversation_id not in self.active_conversations:
            return []
        
        return self.active_conversations[conversation_id].participants.copy()
    
    def close_conversation(self, conversation_id: str) -> bool:
        """Close an active conversation.
        
        Args:
            conversation_id: ID of the conversation to close
            
        Returns:
            True if conversation was closed, False otherwise
        """
        try:
            if conversation_id not in self.active_conversations:
                return False
            
            context = self.active_conversations[conversation_id]
            context.is_active = False
            context.last_activity = datetime.now()
            
            self.logger.info(f"Closed conversation {conversation_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error closing conversation {conversation_id}: {e}")
            return False
    
    def cleanup_inactive_conversations(self, max_age_hours: int = 24) -> int:
        """Clean up old inactive conversations.
        
        Args:
            max_age_hours: Maximum age in hours for keeping conversations
            
        Returns:
            Number of conversations cleaned up
        """
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=max_age_hours)
        
        conversations_to_remove = []
        
        for conversation_id, context in self.active_conversations.items():
            if not context.is_active and context.last_activity < cutoff_time:
                conversations_to_remove.append(conversation_id)
        
        # Remove old conversations
        for conversation_id in conversations_to_remove:
            del self.active_conversations[conversation_id]
        
        if conversations_to_remove:
            self.logger.info(f"Cleaned up {len(conversations_to_remove)} inactive conversations")
        
        return len(conversations_to_remove)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information.
        
        Returns:
            Dictionary with system status information
        """
        current_time = datetime.now()
        
        # Count players by status
        active_players = len(self.get_active_human_players())
        inactive_players = len([
            agent for agent in self.human_players.values() 
            if agent.current_state == AgentState.UNAVAILABLE
        ])
        
        # Count conversations
        active_conversations = len([
            context for context in self.active_conversations.values() 
            if context.is_active
        ])
        
        return {
            "total_human_players": len(self.human_players),
            "active_players": active_players,
            "inactive_players": inactive_players,
            "total_conversations": len(self.active_conversations),
            "active_conversations": active_conversations,
            "last_activity_check": self.last_activity_check.isoformat(),
            "current_time": current_time.isoformat()
        }