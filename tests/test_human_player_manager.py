"""Unit tests for human player management functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from models.core import HumanAgent, ChatMessage
from models.config import Location, OpenAIConfig, AG2Config
from models.enums import AgentRole, AgentState, MessageType
from managers.human_player_manager import HumanPlayerManager, ConversationContext
from agents.base_agent import BaseAgentConfig


class TestConversationContext:
    """Test cases for ConversationContext class."""
    
    def test_conversation_context_creation(self):
        """Test conversation context creation."""
        participants = ["agent_1", "agent_2", "human_1"]
        topic = "emergency_response"
        
        context = ConversationContext("conv_1", participants, topic)
        
        assert context.conversation_id == "conv_1"
        assert context.participants == participants
        assert context.topic == topic
        assert context.is_active is True
        assert isinstance(context.created_at, datetime)
        assert isinstance(context.last_activity, datetime)
    
    def test_conversation_context_serialization(self):
        """Test conversation context serialization and deserialization."""
        participants = ["agent_1", "human_1"]
        context = ConversationContext("conv_test", participants, "test_topic")
        
        # Test serialization
        context_dict = context.to_dict()
        assert context_dict["conversation_id"] == "conv_test"
        assert context_dict["participants"] == participants
        assert context_dict["topic"] == "test_topic"
        assert context_dict["is_active"] is True
        
        # Test deserialization
        restored_context = ConversationContext.from_dict(context_dict)
        assert restored_context.conversation_id == context.conversation_id
        assert restored_context.participants == context.participants
        assert restored_context.topic == context.topic
        assert restored_context.is_active == context.is_active


class TestHumanPlayerManager:
    """Test cases for HumanPlayerManager class."""
    
    @pytest.fixture
    def mock_base_config(self):
        """Create mock base agent configuration."""
        openai_config = Mock(spec=OpenAIConfig)
        ag2_config = Mock(spec=AG2Config)
        
        base_config = Mock(spec=BaseAgentConfig)
        base_config.get_default_personality.return_value = {"cautious": 0.7, "leadership": 0.8}
        base_config.get_default_capabilities.return_value = ["wildlife_management", "visitor_safety"]
        
        return base_config
    
    @pytest.fixture
    def human_player_manager(self, mock_base_config):
        """Create HumanPlayerManager instance for testing."""
        return HumanPlayerManager(mock_base_config)
    
    def test_human_player_manager_initialization(self, human_player_manager):
        """Test human player manager initialization."""
        assert isinstance(human_player_manager.human_players, dict)
        assert isinstance(human_player_manager.active_conversations, dict)
        assert isinstance(human_player_manager.role_permissions, dict)
        assert len(human_player_manager.human_players) == 0
        assert len(human_player_manager.active_conversations) == 0
        
        # Check role permissions are set up
        assert AgentRole.PARK_RANGER in human_player_manager.role_permissions
        assert AgentRole.TOURIST in human_player_manager.role_permissions
        assert "staff_coordination" in human_player_manager.role_permissions[AgentRole.PARK_RANGER]
    
    def test_create_human_agent(self, human_player_manager):
        """Test creating a human agent."""
        role = AgentRole.PARK_RANGER
        player_name = "Test Ranger"
        location = Location(10.0, 20.0, "ranger_station", "Ranger station")
        
        human_agent = human_player_manager.create_human_agent(role, player_name, location)
        
        assert isinstance(human_agent, HumanAgent)
        assert human_agent.name == player_name
        assert human_agent.role == role
        assert human_agent.location == location
        assert human_agent.is_human_controlled is True
        assert human_agent.current_state == AgentState.IDLE
        assert len(human_agent.conversation_access) > 0
        
        # Check that agent is stored
        assert human_agent.id in human_player_manager.human_players
        assert human_player_manager.human_players[human_agent.id] == human_agent
    
    def test_create_human_agent_default_location(self, human_player_manager):
        """Test creating a human agent with default location."""
        human_agent = human_player_manager.create_human_agent(AgentRole.TOURIST, "Test Tourist")
        
        assert human_agent.location.zone == "entrance"
        assert human_agent.location.description == "Main entrance area"
    
    def test_update_human_role(self, human_player_manager):
        """Test updating a human player's role."""
        # Create human agent
        human_agent = human_player_manager.create_human_agent(AgentRole.TOURIST, "Test Player")
        player_id = human_agent.id
        old_access = human_agent.conversation_access.copy()
        
        # Update role
        success = human_player_manager.update_human_role(player_id, AgentRole.PARK_RANGER)
        
        assert success is True
        assert human_agent.role == AgentRole.PARK_RANGER
        assert human_agent.conversation_access != old_access
        assert isinstance(human_agent.last_activity, datetime)
    
    def test_update_human_role_nonexistent_player(self, human_player_manager):
        """Test updating role for nonexistent player."""
        success = human_player_manager.update_human_role("nonexistent", AgentRole.SECURITY)
        assert success is False
    
    def test_process_human_input(self, human_player_manager):
        """Test processing human player input."""
        # Create human agent
        human_agent = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Test Ranger")
        player_id = human_agent.id
        
        # Create conversation context
        context = ConversationContext("conv_1", [player_id], "staff_coordination")
        human_player_manager.active_conversations[context.conversation_id] = context
        
        # Grant access to the conversation
        human_agent.add_conversation_access("staff_coordination")
        
        # Process input
        message = "I'm responding to the emergency situation."
        success = human_player_manager.process_human_input(player_id, message, context)
        
        assert success is True
        assert len(human_agent.chat_history) == 1
        assert human_agent.chat_history[0].content == message
        assert human_agent.chat_history[0].message_type == MessageType.HUMAN
        assert human_agent.current_state == AgentState.COMMUNICATING
    
    def test_process_human_input_no_access(self, human_player_manager):
        """Test processing human input without conversation access."""
        # Create human agent
        human_agent = human_player_manager.create_human_agent(AgentRole.TOURIST, "Test Tourist")
        player_id = human_agent.id
        
        # Create conversation context for staff coordination (tourist shouldn't have access)
        context = ConversationContext("conv_1", ["staff_1"], "staff_coordination")
        human_player_manager.active_conversations[context.conversation_id] = context
        
        # Process input (should fail)
        success = human_player_manager.process_human_input(player_id, "Test message", context)
        
        assert success is False
        assert len(human_agent.chat_history) == 0
    
    def test_process_human_input_nonexistent_player(self, human_player_manager):
        """Test processing input for nonexistent player."""
        context = ConversationContext("conv_1", [], "test")
        success = human_player_manager.process_human_input("nonexistent", "message", context)
        assert success is False
    
    def test_get_relevant_conversations(self, human_player_manager):
        """Test getting relevant conversations for a human player."""
        # Create human agent
        human_agent = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Test Ranger")
        player_id = human_agent.id
        
        # Create conversations
        context1 = ConversationContext("conv_1", [player_id], "staff_coordination")
        context2 = ConversationContext("conv_2", ["other_agent"], "visitor_interactions")
        context3 = ConversationContext("conv_3", [player_id], "emergency_response")
        
        human_player_manager.active_conversations["conv_1"] = context1
        human_player_manager.active_conversations["conv_2"] = context2
        human_player_manager.active_conversations["conv_3"] = context3
        
        # Get relevant conversations
        relevant = human_player_manager.get_relevant_conversations(player_id)
        
        # Should have access to conv_1 and conv_3 (participant or role-based access)
        relevant_ids = [conv.conversation_id for conv in relevant]
        assert "conv_1" in relevant_ids
        assert "conv_3" in relevant_ids
        # conv_2 should not be accessible to park ranger
    
    def test_create_conversation_context(self, human_player_manager):
        """Test creating a conversation context."""
        participants = ["agent_1", "human_1"]
        topic = "emergency_response"
        
        context = human_player_manager.create_conversation_context(participants, topic)
        
        assert context.participants == participants
        assert context.topic == topic
        assert context.is_active is True
        assert context.conversation_id in human_player_manager.active_conversations
    
    def test_add_human_to_conversation(self, human_player_manager):
        """Test adding a human player to a conversation."""
        # Create human agent
        human_agent = human_player_manager.create_human_agent(AgentRole.SECURITY, "Test Security")
        player_id = human_agent.id
        
        # Create conversation
        context = human_player_manager.create_conversation_context(["agent_1"], "emergency_response")
        conversation_id = context.conversation_id
        
        # Add human to conversation
        success = human_player_manager.add_human_to_conversation(player_id, conversation_id)
        
        assert success is True
        assert player_id in context.participants
        assert human_agent.has_conversation_access(conversation_id)
    
    def test_add_human_to_conversation_no_access(self, human_player_manager):
        """Test adding human to conversation without proper access."""
        # Create tourist (limited access)
        human_agent = human_player_manager.create_human_agent(AgentRole.TOURIST, "Test Tourist")
        player_id = human_agent.id
        
        # Create staff-only conversation
        context = human_player_manager.create_conversation_context(["staff_1"], "facility_operations")
        conversation_id = context.conversation_id
        
        # Try to add tourist (should fail)
        success = human_player_manager.add_human_to_conversation(player_id, conversation_id)
        
        assert success is False
        assert player_id not in context.participants
    
    def test_remove_human_from_conversation(self, human_player_manager):
        """Test removing a human player from a conversation."""
        # Create human agent
        human_agent = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Test Ranger")
        player_id = human_agent.id
        
        # Create conversation and add human
        context = human_player_manager.create_conversation_context([player_id], "staff_coordination")
        conversation_id = context.conversation_id
        human_agent.add_conversation_access(conversation_id)
        
        # Remove human from conversation
        success = human_player_manager.remove_human_from_conversation(player_id, conversation_id)
        
        assert success is True
        assert player_id not in context.participants
        assert not human_agent.has_conversation_access(conversation_id)
    
    def test_handle_player_inactivity(self, human_player_manager):
        """Test handling player inactivity."""
        # Create human agents
        active_agent = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Active Ranger")
        inactive_agent = human_player_manager.create_human_agent(AgentRole.SECURITY, "Inactive Security")
        
        # Make one agent inactive by setting old last_activity
        old_time = datetime.now() - timedelta(minutes=15)
        inactive_agent.last_activity = old_time
        
        # Handle inactivity with 10 minute timeout
        inactive_players = human_player_manager.handle_player_inactivity(timeout_seconds=600)
        
        assert len(inactive_players) == 1
        assert inactive_agent.id in inactive_players
        assert inactive_agent.current_state == AgentState.UNAVAILABLE
        assert active_agent.current_state != AgentState.UNAVAILABLE
    
    def test_reactivate_player(self, human_player_manager):
        """Test reactivating an inactive player."""
        # Create and deactivate human agent
        human_agent = human_player_manager.create_human_agent(AgentRole.VETERINARIAN, "Test Vet")
        player_id = human_agent.id
        human_agent.current_state = AgentState.UNAVAILABLE
        
        # Reactivate player
        success = human_player_manager.reactivate_player(player_id)
        
        assert success is True
        assert human_agent.current_state == AgentState.IDLE
        assert isinstance(human_agent.last_activity, datetime)
    
    def test_reactivate_nonexistent_player(self, human_player_manager):
        """Test reactivating nonexistent player."""
        success = human_player_manager.reactivate_player("nonexistent")
        assert success is False
    
    def test_get_human_player(self, human_player_manager):
        """Test getting a human player by ID."""
        human_agent = human_player_manager.create_human_agent(AgentRole.MAINTENANCE, "Test Maintenance")
        player_id = human_agent.id
        
        retrieved_agent = human_player_manager.get_human_player(player_id)
        assert retrieved_agent == human_agent
        
        # Test nonexistent player
        assert human_player_manager.get_human_player("nonexistent") is None
    
    def test_get_all_human_players(self, human_player_manager):
        """Test getting all human players."""
        agent1 = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Ranger 1")
        agent2 = human_player_manager.create_human_agent(AgentRole.TOURIST, "Tourist 1")
        
        all_players = human_player_manager.get_all_human_players()
        
        assert len(all_players) == 2
        assert agent1.id in all_players
        assert agent2.id in all_players
    
    def test_get_human_players_by_role(self, human_player_manager):
        """Test getting human players by role."""
        ranger1 = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Ranger 1")
        ranger2 = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Ranger 2")
        tourist = human_player_manager.create_human_agent(AgentRole.TOURIST, "Tourist 1")
        
        rangers = human_player_manager.get_human_players_by_role(AgentRole.PARK_RANGER)
        tourists = human_player_manager.get_human_players_by_role(AgentRole.TOURIST)
        
        assert len(rangers) == 2
        assert len(tourists) == 1
        assert ranger1 in rangers
        assert ranger2 in rangers
        assert tourist in tourists
    
    def test_get_active_human_players(self, human_player_manager):
        """Test getting active human players."""
        active_agent = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Active Ranger")
        inactive_agent = human_player_manager.create_human_agent(AgentRole.SECURITY, "Inactive Security")
        
        # Make one agent inactive
        inactive_agent.current_state = AgentState.UNAVAILABLE
        
        active_players = human_player_manager.get_active_human_players()
        
        assert len(active_players) == 1
        assert active_agent in active_players
        assert inactive_agent not in active_players
    
    def test_get_human_chat_history(self, human_player_manager):
        """Test getting human chat history."""
        human_agent = human_player_manager.create_human_agent(AgentRole.VETERINARIAN, "Test Vet")
        player_id = human_agent.id
        
        # Add some chat messages
        message1 = ChatMessage("msg_1", player_id, "Test Vet", "Hello", message_type=MessageType.HUMAN)
        message2 = ChatMessage("msg_2", player_id, "Test Vet", "How are you?", message_type=MessageType.HUMAN)
        
        human_agent.add_chat_message(message1)
        human_agent.add_chat_message(message2)
        
        # Get chat history
        history = human_player_manager.get_human_chat_history(player_id)
        
        assert len(history) == 2
        assert history[0] == message1
        assert history[1] == message2
    
    def test_clear_human_chat_history(self, human_player_manager):
        """Test clearing human chat history."""
        human_agent = human_player_manager.create_human_agent(AgentRole.SECURITY, "Test Security")
        player_id = human_agent.id
        
        # Add a chat message
        message = ChatMessage("msg_1", player_id, "Test Security", "Test message", message_type=MessageType.HUMAN)
        human_agent.add_chat_message(message)
        
        assert len(human_agent.chat_history) == 1
        
        # Clear history
        success = human_player_manager.clear_human_chat_history(player_id)
        
        assert success is True
        assert len(human_agent.chat_history) == 0
    
    def test_remove_human_player(self, human_player_manager):
        """Test removing a human player."""
        human_agent = human_player_manager.create_human_agent(AgentRole.MAINTENANCE, "Test Maintenance")
        player_id = human_agent.id
        
        # Add to a conversation
        context = human_player_manager.create_conversation_context([player_id], "test")
        
        assert player_id in human_player_manager.human_players
        assert player_id in context.participants
        
        # Remove player
        success = human_player_manager.remove_human_player(player_id)
        
        assert success is True
        assert player_id not in human_player_manager.human_players
        assert player_id not in context.participants
    
    def test_get_conversation_participants(self, human_player_manager):
        """Test getting conversation participants."""
        participants = ["agent_1", "agent_2", "human_1"]
        context = human_player_manager.create_conversation_context(participants, "test")
        
        retrieved_participants = human_player_manager.get_conversation_participants(context.conversation_id)
        
        assert retrieved_participants == participants
        
        # Test nonexistent conversation
        assert human_player_manager.get_conversation_participants("nonexistent") == []
    
    def test_close_conversation(self, human_player_manager):
        """Test closing a conversation."""
        context = human_player_manager.create_conversation_context(["agent_1"], "test")
        conversation_id = context.conversation_id
        
        assert context.is_active is True
        
        success = human_player_manager.close_conversation(conversation_id)
        
        assert success is True
        assert context.is_active is False
    
    def test_cleanup_inactive_conversations(self, human_player_manager):
        """Test cleaning up inactive conversations."""
        # Create conversations
        old_context = human_player_manager.create_conversation_context(["agent_1"], "old")
        new_context = human_player_manager.create_conversation_context(["agent_2"], "new")
        
        # Make one conversation old and inactive
        old_context.is_active = False
        old_context.last_activity = datetime.now() - timedelta(hours=25)
        
        # Make other conversation recent but inactive
        new_context.is_active = False
        new_context.last_activity = datetime.now() - timedelta(hours=1)
        
        # Cleanup with 24 hour threshold
        cleaned_count = human_player_manager.cleanup_inactive_conversations(max_age_hours=24)
        
        assert cleaned_count == 1
        assert old_context.conversation_id not in human_player_manager.active_conversations
        assert new_context.conversation_id in human_player_manager.active_conversations
    
    def test_get_system_status(self, human_player_manager):
        """Test getting system status."""
        # Create some human players
        active_agent = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Active Ranger")
        inactive_agent = human_player_manager.create_human_agent(AgentRole.SECURITY, "Inactive Security")
        inactive_agent.current_state = AgentState.UNAVAILABLE
        
        # Create conversations
        active_context = human_player_manager.create_conversation_context(["agent_1"], "active")
        inactive_context = human_player_manager.create_conversation_context(["agent_2"], "inactive")
        inactive_context.is_active = False
        
        status = human_player_manager.get_system_status()
        
        assert status["total_human_players"] == 2
        assert status["active_players"] == 1
        assert status["inactive_players"] == 1
        assert status["total_conversations"] == 2
        assert status["active_conversations"] == 1
        assert "last_activity_check" in status
        assert "current_time" in status
    
    def test_conversation_access_permissions(self, human_player_manager):
        """Test conversation access based on role permissions."""
        # Create different role agents
        ranger = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Test Ranger")
        tourist = human_player_manager.create_human_agent(AgentRole.TOURIST, "Test Tourist")
        
        # Create contexts with different topics
        emergency_context = ConversationContext("emergency", ["agent_1"], "emergency_response")
        visitor_context = ConversationContext("visitor", ["agent_2"], "visitor_interactions")
        
        # Test access permissions
        assert human_player_manager._has_conversation_access(ranger, emergency_context)
        assert not human_player_manager._has_conversation_access(tourist, emergency_context)
        assert not human_player_manager._has_conversation_access(ranger, visitor_context)
        assert human_player_manager._has_conversation_access(tourist, visitor_context)
    
    def test_conversation_access_participant(self, human_player_manager):
        """Test conversation access for participants."""
        human_agent = human_player_manager.create_human_agent(AgentRole.TOURIST, "Test Tourist")
        
        # Create context where human is a participant
        context = ConversationContext("test", [human_agent.id], "restricted_topic")
        
        # Should have access as participant even if topic is restricted
        assert human_player_manager._has_conversation_access(human_agent, context)
    
    def test_conversation_access_explicit(self, human_player_manager):
        """Test explicit conversation access."""
        human_agent = human_player_manager.create_human_agent(AgentRole.TOURIST, "Test Tourist")
        
        # Create context with restricted topic
        context = ConversationContext("test", ["other_agent"], "staff_coordination")
        
        # Should not have access initially
        assert not human_player_manager._has_conversation_access(human_agent, context)
        
        # Grant explicit access
        human_agent.add_conversation_access(context.conversation_id)
        
        # Should now have access
        assert human_player_manager._has_conversation_access(human_agent, context)


if __name__ == "__main__":
    pytest.main([__file__])