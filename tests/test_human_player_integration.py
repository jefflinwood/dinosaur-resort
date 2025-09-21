"""Integration tests for human player manager with existing system components."""

import pytest
from unittest.mock import Mock, patch
from models.core import HumanAgent, Event
from models.config import Location, OpenAIConfig, AG2Config
from models.enums import AgentRole, EventType, AgentState
from managers.human_player_manager import HumanPlayerManager, ConversationContext
from agents.base_agent import BaseAgentConfig


class TestHumanPlayerIntegration:
    """Integration tests for human player manager."""
    
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
    
    def test_human_player_role_based_event_access(self, human_player_manager):
        """Test that human players get appropriate access to events based on their role."""
        # Create different role human players
        ranger = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Test Ranger")
        tourist = human_player_manager.create_human_agent(AgentRole.TOURIST, "Test Tourist")
        security = human_player_manager.create_human_agent(AgentRole.SECURITY, "Test Security")
        
        # Create event-related conversations
        emergency_context = human_player_manager.create_conversation_context(
            ["ai_agent_1"], "emergency_response"
        )
        visitor_context = human_player_manager.create_conversation_context(
            ["ai_agent_2"], "visitor_interactions"
        )
        facility_context = human_player_manager.create_conversation_context(
            ["ai_agent_3"], "facility_operations"
        )
        
        # Test ranger access
        ranger_conversations = human_player_manager.get_relevant_conversations(ranger.id)
        ranger_conv_topics = [conv.topic for conv in ranger_conversations]
        assert "emergency_response" in ranger_conv_topics
        assert "facility_operations" in ranger_conv_topics
        
        # Test tourist access (should be limited)
        tourist_conversations = human_player_manager.get_relevant_conversations(tourist.id)
        tourist_conv_topics = [conv.topic for conv in tourist_conversations]
        assert "visitor_interactions" in tourist_conv_topics
        assert "emergency_response" not in tourist_conv_topics
        
        # Test security access
        security_conversations = human_player_manager.get_relevant_conversations(security.id)
        security_conv_topics = [conv.topic for conv in security_conversations]
        assert "emergency_response" in security_conv_topics
    
    def test_human_player_role_switching_scenario(self, human_player_manager):
        """Test a complete role switching scenario."""
        # Create human player as tourist
        human_agent = human_player_manager.create_human_agent(AgentRole.TOURIST, "Test Player")
        player_id = human_agent.id
        
        # Create conversations for different roles
        tourist_context = human_player_manager.create_conversation_context(
            [player_id], "visitor_interactions"
        )
        staff_context = human_player_manager.create_conversation_context(
            ["staff_1"], "staff_coordination"
        )
        
        # Initially, tourist should only access visitor conversations
        initial_conversations = human_player_manager.get_relevant_conversations(player_id)
        initial_topics = [conv.topic for conv in initial_conversations]
        assert "visitor_interactions" in initial_topics
        assert "staff_coordination" not in initial_topics
        
        # Switch to park ranger role
        success = human_player_manager.update_human_role(player_id, AgentRole.PARK_RANGER)
        assert success is True
        
        # Now should have access to staff conversations
        updated_conversations = human_player_manager.get_relevant_conversations(player_id)
        updated_topics = [conv.topic for conv in updated_conversations]
        assert "staff_coordination" in updated_topics
        
        # Verify role characteristics updated
        assert human_agent.role == AgentRole.PARK_RANGER
        assert "wildlife_management" in human_agent.capabilities
    
    def test_human_player_event_participation_workflow(self, human_player_manager):
        """Test complete workflow of human player participating in event response."""
        # Create human player as security guard
        security_agent = human_player_manager.create_human_agent(AgentRole.SECURITY, "Security Guard")
        player_id = security_agent.id
        
        # Simulate event occurring - create emergency conversation
        emergency_context = human_player_manager.create_conversation_context(
            ["ai_ranger", "ai_vet"], "emergency_response"
        )
        
        # Add human player to emergency response
        success = human_player_manager.add_human_to_conversation(player_id, emergency_context.conversation_id)
        assert success is True
        assert player_id in emergency_context.participants
        
        # Human player responds to emergency
        human_message = "I'm securing the area and evacuating visitors from the danger zone."
        success = human_player_manager.process_human_input(player_id, human_message, emergency_context)
        assert success is True
        
        # Verify message was recorded
        chat_history = human_player_manager.get_human_chat_history(player_id)
        assert len(chat_history) == 1
        assert chat_history[0].content == human_message
        assert chat_history[0].conversation_id == emergency_context.conversation_id
        
        # Verify human agent state updated
        assert security_agent.current_state == AgentState.COMMUNICATING
    
    def test_human_player_inactivity_handling(self, human_player_manager):
        """Test handling of inactive human players during events."""
        # Create multiple human players
        active_ranger = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Active Ranger")
        inactive_security = human_player_manager.create_human_agent(AgentRole.SECURITY, "Inactive Security")
        
        # Simulate one player becoming inactive
        from datetime import datetime, timedelta
        old_time = datetime.now() - timedelta(minutes=15)
        inactive_security.last_activity = old_time
        
        # Check system status before handling inactivity
        status_before = human_player_manager.get_system_status()
        assert status_before["active_players"] == 2
        assert status_before["inactive_players"] == 0
        
        # Handle inactivity (10 minute timeout)
        inactive_players = human_player_manager.handle_player_inactivity(timeout_seconds=600)
        
        # Verify inactive player was identified
        assert len(inactive_players) == 1
        assert inactive_security.id in inactive_players
        assert inactive_security.current_state == AgentState.UNAVAILABLE
        
        # Check system status after handling inactivity
        status_after = human_player_manager.get_system_status()
        assert status_after["active_players"] == 1
        assert status_after["inactive_players"] == 1
        
        # Reactivate the player
        success = human_player_manager.reactivate_player(inactive_security.id)
        assert success is True
        assert inactive_security.current_state == AgentState.IDLE
    
    def test_human_player_conversation_filtering(self, human_player_manager):
        """Test that conversation filtering works correctly for different scenarios."""
        # Create human players with different roles
        vet = human_player_manager.create_human_agent(AgentRole.VETERINARIAN, "Dr. Vet")
        maintenance = human_player_manager.create_human_agent(AgentRole.MAINTENANCE, "Maintenance Worker")
        
        # Create various conversation contexts
        medical_context = human_player_manager.create_conversation_context(
            ["ai_ranger"], "medical_treatment"
        )
        technical_context = human_player_manager.create_conversation_context(
            ["ai_security"], "technical_support"
        )
        general_context = human_player_manager.create_conversation_context(
            ["ai_tourist"], ""  # Empty topic should be accessible to all
        )
        
        # Test veterinarian access
        vet_conversations = human_player_manager.get_relevant_conversations(vet.id)
        vet_topics = [conv.topic for conv in vet_conversations]
        assert "medical_treatment" in vet_topics
        assert "technical_support" not in vet_topics
        assert "" in vet_topics  # General conversation
        
        # Test maintenance worker access
        maintenance_conversations = human_player_manager.get_relevant_conversations(maintenance.id)
        maintenance_topics = [conv.topic for conv in maintenance_conversations]
        assert "technical_support" in maintenance_topics
        assert "medical_treatment" not in maintenance_topics
        assert "" in maintenance_topics  # General conversation
    
    def test_human_player_system_cleanup(self, human_player_manager):
        """Test system cleanup functionality."""
        # Create human players and conversations
        player1 = human_player_manager.create_human_agent(AgentRole.PARK_RANGER, "Ranger 1")
        player2 = human_player_manager.create_human_agent(AgentRole.TOURIST, "Tourist 1")
        
        # Create conversations
        context1 = human_player_manager.create_conversation_context([player1.id], "test1")
        context2 = human_player_manager.create_conversation_context([player2.id], "test2")
        
        # Make one conversation old and inactive
        from datetime import datetime, timedelta
        context1.is_active = False
        context1.last_activity = datetime.now() - timedelta(hours=25)
        
        # Keep other conversation recent
        context2.is_active = False
        context2.last_activity = datetime.now() - timedelta(hours=1)
        
        # Test cleanup
        cleaned_count = human_player_manager.cleanup_inactive_conversations(max_age_hours=24)
        assert cleaned_count == 1
        
        # Verify old conversation was removed
        assert context1.conversation_id not in human_player_manager.active_conversations
        assert context2.conversation_id in human_player_manager.active_conversations
        
        # Test removing human players
        initial_count = len(human_player_manager.get_all_human_players())
        success = human_player_manager.remove_human_player(player1.id)
        assert success is True
        
        final_count = len(human_player_manager.get_all_human_players())
        assert final_count == initial_count - 1
        
        # Verify player was removed from remaining conversations
        remaining_participants = human_player_manager.get_conversation_participants(context2.conversation_id)
        assert player1.id not in remaining_participants


if __name__ == "__main__":
    pytest.main([__file__])