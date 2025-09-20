"""Unit tests for OpenAI API integration and ag2 framework."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from models.config import OpenAIConfig, AG2Config
from models.core import Agent
from models.enums import AgentRole, PersonalityTrait
from managers.ag2_integration import AG2Integration
from agents.base_agent import BaseAgentConfig, AgentFactory


class TestOpenAIConfig:
    """Test OpenAI configuration."""
    
    def test_openai_config_from_env(self):
        """Test OpenAI config creation from environment variables."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'OPENAI_MODEL': 'gpt-3.5-turbo',
            'OPENAI_TEMPERATURE': '0.5',
            'OPENAI_MAX_TOKENS': '1000',
            'OPENAI_TIMEOUT': '60',
            'OPENAI_MAX_RETRIES': '5'
        }):
            config = OpenAIConfig()
            
            assert config.api_key == 'test-key'
            assert config.model == 'gpt-3.5-turbo'
            assert config.temperature == 0.5
            assert config.max_tokens == 1000
            assert config.timeout == 60
            assert config.max_retries == 5
    
    def test_openai_config_defaults(self):
        """Test OpenAI config with default values."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=True):
            config = OpenAIConfig()
            
            assert config.api_key == 'test-key'
            assert config.model == 'gpt-4'
            assert config.temperature == 0.7
            assert config.max_tokens == 500
            assert config.timeout == 30
            assert config.max_retries == 3
    
    def test_openai_config_validation_missing_key(self):
        """Test OpenAI config validation with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIConfig()
    
    def test_openai_config_validation_invalid_temperature(self):
        """Test OpenAI config validation with invalid temperature."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'OPENAI_TEMPERATURE': '3.0'
        }):
            with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
                OpenAIConfig()
    
    def test_openai_config_validation_invalid_tokens(self):
        """Test OpenAI config validation with invalid max_tokens."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'OPENAI_MAX_TOKENS': '0'
        }):
            with pytest.raises(ValueError, match="max_tokens must be positive"):
                OpenAIConfig()


class TestAG2Config:
    """Test ag2 configuration."""
    
    def test_ag2_config_from_env(self):
        """Test ag2 config creation from environment variables."""
        with patch.dict(os.environ, {
            'AG2_MAX_ROUND': '15',
            'AG2_HUMAN_INPUT_MODE': 'ALWAYS',
            'AG2_CODE_EXECUTION_CONFIG': 'true'
        }):
            config = AG2Config()
            
            assert config.max_round == 15
            assert config.human_input_mode == 'ALWAYS'
            assert config.code_execution_config is True
    
    def test_ag2_config_defaults(self):
        """Test ag2 config with default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = AG2Config()
            
            assert config.max_round == 10
            assert config.human_input_mode == 'NEVER'
            assert config.code_execution_config is False
    
    def test_ag2_config_validation_invalid_max_round(self):
        """Test ag2 config validation with invalid max_round."""
        with patch.dict(os.environ, {'AG2_MAX_ROUND': '0'}):
            with pytest.raises(ValueError, match="max_round must be positive"):
                AG2Config()
    
    def test_ag2_config_validation_invalid_input_mode(self):
        """Test ag2 config validation with invalid human_input_mode."""
        with patch.dict(os.environ, {'AG2_HUMAN_INPUT_MODE': 'INVALID'}):
            with pytest.raises(ValueError, match="human_input_mode must be one of"):
                AG2Config()


class TestAG2Integration:
    """Test ag2 integration with OpenAI."""
    
    @pytest.fixture
    def openai_config(self):
        """Create test OpenAI config."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return OpenAIConfig()
    
    @pytest.fixture
    def ag2_config(self):
        """Create test ag2 config."""
        return AG2Config()
    
    @pytest.fixture
    def ag2_integration(self, openai_config, ag2_config):
        """Create test ag2 integration."""
        return AG2Integration(openai_config, ag2_config)
    
    @pytest.fixture
    def test_agent(self):
        """Create test agent."""
        return Agent(
            id="test_agent_1",
            name="Test Ranger",
            role=AgentRole.PARK_RANGER,
            personality_traits={
                PersonalityTrait.CAUTIOUS.value: 0.8,
                PersonalityTrait.LEADERSHIP.value: 0.7
            },
            capabilities=["wildlife_management", "visitor_safety"]
        )
    
    def test_ag2_integration_initialization(self, ag2_integration, openai_config):
        """Test ag2 integration initialization."""
        assert ag2_integration.openai_config == openai_config
        assert ag2_integration.llm_config is not None
        assert ag2_integration.llm_config["config_list"][0]["api_key"] == "test-key"
        assert len(ag2_integration.ag2_agents) == 0
    
    @patch('managers.ag2_integration.ConversableAgent')
    def test_create_ag2_agent(self, mock_conversable_agent, ag2_integration, test_agent):
        """Test creating ag2 agent from our agent model."""
        mock_agent_instance = Mock()
        mock_conversable_agent.return_value = mock_agent_instance
        
        result = ag2_integration.create_ag2_agent(test_agent)
        
        # Verify ConversableAgent was called with correct parameters
        mock_conversable_agent.assert_called_once()
        call_args = mock_conversable_agent.call_args
        
        assert call_args[1]["name"] == "Test Ranger"
        assert "park ranger" in call_args[1]["system_message"].lower()
        assert call_args[1]["llm_config"] == ag2_integration.llm_config
        assert call_args[1]["human_input_mode"] == "NEVER"
        
        # Verify agent was stored
        assert test_agent.id in ag2_integration.ag2_agents
        assert ag2_integration.ag2_agents[test_agent.id] == mock_agent_instance
        assert result == mock_agent_instance
    
    @patch('managers.ag2_integration.GroupChat')
    @patch('managers.ag2_integration.GroupChatManager')
    @patch('managers.ag2_integration.ConversableAgent')
    def test_create_group_chat(self, mock_conversable_agent, mock_group_chat_manager, 
                              mock_group_chat, ag2_integration, test_agent):
        """Test creating group chat with agents."""
        mock_agent_instance = Mock()
        mock_conversable_agent.return_value = mock_agent_instance
        mock_group_chat_instance = Mock()
        mock_group_chat.return_value = mock_group_chat_instance
        mock_manager_instance = Mock()
        mock_group_chat_manager.return_value = mock_manager_instance
        
        agents = [test_agent]
        result = ag2_integration.create_group_chat(agents)
        
        # Verify agent was created
        mock_conversable_agent.assert_called_once()
        
        # Verify group chat was created
        mock_group_chat.assert_called_once()
        call_args = mock_group_chat.call_args
        assert len(call_args[1]["agents"]) == 1
        assert call_args[1]["agents"][0] == mock_agent_instance
        
        # Verify group chat manager was created
        mock_group_chat_manager.assert_called_once()
        
        assert result == mock_group_chat_instance
    
    @patch('managers.ag2_integration.ConversableAgent')
    def test_send_message_to_agent(self, mock_conversable_agent, ag2_integration, test_agent):
        """Test sending message to specific agent."""
        mock_agent_instance = Mock()
        mock_agent_instance.generate_reply.return_value = "Test response"
        mock_conversable_agent.return_value = mock_agent_instance
        
        # Create agent first
        ag2_integration.create_ag2_agent(test_agent)
        
        # Send message
        response = ag2_integration.send_message_to_agent(test_agent.id, "Test message", "System")
        
        # Verify response
        assert response == "Test response"
        mock_agent_instance.generate_reply.assert_called_once()
        
        # Verify message format
        call_args = mock_agent_instance.generate_reply.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["content"] == "Test message"
        assert messages[0]["role"] == "user"
        assert messages[0]["name"] == "System"
    
    def test_send_message_to_nonexistent_agent(self, ag2_integration):
        """Test sending message to non-existent agent."""
        response = ag2_integration.send_message_to_agent("nonexistent", "Test message")
        assert response is None
    
    @patch('managers.ag2_integration.ConversableAgent')
    def test_broadcast_message(self, mock_conversable_agent, ag2_integration, test_agent):
        """Test broadcasting message to all agents."""
        mock_agent_instance = Mock()
        mock_agent_instance.generate_reply.return_value = "Test response"
        mock_conversable_agent.return_value = mock_agent_instance
        
        # Create agent
        ag2_integration.create_ag2_agent(test_agent)
        
        # Broadcast message
        responses = ag2_integration.broadcast_message("Broadcast message", "System")
        
        # Verify responses
        assert len(responses) == 1
        assert responses[test_agent.id] == "Test response"
    
    def test_get_agent_status(self, ag2_integration):
        """Test getting agent status information."""
        status = ag2_integration.get_agent_status()
        assert isinstance(status, dict)
        assert len(status) == 0  # No agents created yet
    
    def test_personality_description_generation(self, ag2_integration, test_agent):
        """Test personality description generation."""
        description = ag2_integration._generate_personality_description(test_agent)
        
        assert isinstance(description, str)
        assert "cautious" in description.lower()
        # Leadership is 0.7, so should be "somewhat"
        assert "somewhat" in description.lower()
    
    def test_personality_description_empty_traits(self, ag2_integration):
        """Test personality description with empty traits."""
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            role=AgentRole.TOURIST,
            personality_traits={}
        )
        
        description = ag2_integration._generate_personality_description(agent)
        assert description == "You have a balanced personality."


class TestBaseAgentConfig:
    """Test base agent configuration."""
    
    @pytest.fixture
    def openai_config(self):
        """Create test OpenAI config."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return OpenAIConfig()
    
    @pytest.fixture
    def ag2_config(self):
        """Create test ag2 config."""
        return AG2Config()
    
    @pytest.fixture
    def base_config(self, openai_config, ag2_config):
        """Create test base agent config."""
        return BaseAgentConfig(openai_config, ag2_config)
    
    def test_base_config_initialization(self, base_config):
        """Test base agent config initialization."""
        assert base_config.openai_config is not None
        assert base_config.ag2_config is not None
        assert base_config.ag2_integration is not None
        assert len(base_config.role_configs) == 6  # All agent roles
    
    def test_create_agent_with_config(self, base_config):
        """Test creating agent with role-specific config."""
        agent = base_config.create_agent_with_config(
            agent_id="test_ranger",
            name="Test Ranger",
            role=AgentRole.PARK_RANGER
        )
        
        assert agent.id == "test_ranger"
        assert agent.name == "Test Ranger"
        assert agent.role == AgentRole.PARK_RANGER
        assert PersonalityTrait.CAUTIOUS.value in agent.personality_traits
        assert agent.personality_traits[PersonalityTrait.CAUTIOUS.value] == 0.8
        assert "wildlife_management" in agent.capabilities
    
    def test_create_agent_with_custom_personality(self, base_config):
        """Test creating agent with custom personality traits."""
        custom_personality = {PersonalityTrait.BRAVE.value: 0.9}
        
        agent = base_config.create_agent_with_config(
            agent_id="test_agent",
            name="Test Agent",
            role=AgentRole.SECURITY,
            custom_personality=custom_personality
        )
        
        # Should have both default and custom traits
        assert PersonalityTrait.BRAVE.value in agent.personality_traits
        assert agent.personality_traits[PersonalityTrait.BRAVE.value] == 0.9
        assert PersonalityTrait.LEADERSHIP.value in agent.personality_traits
    
    def test_get_role_specific_prompt(self, base_config):
        """Test getting role-specific prompts."""
        prompt = base_config.get_role_specific_prompt(AgentRole.VETERINARIAN)
        
        assert isinstance(prompt, str)
        assert "veterinarian" in prompt.lower()
        assert "dinosaur health" in prompt.lower()
    
    def test_get_default_personality(self, base_config):
        """Test getting default personality for role."""
        personality = base_config.get_default_personality(AgentRole.PARK_RANGER)
        
        assert isinstance(personality, dict)
        assert PersonalityTrait.CAUTIOUS.value in personality
        assert personality[PersonalityTrait.CAUTIOUS.value] == 0.8
    
    def test_get_default_capabilities(self, base_config):
        """Test getting default capabilities for role."""
        capabilities = base_config.get_default_capabilities(AgentRole.MAINTENANCE)
        
        assert isinstance(capabilities, list)
        assert "equipment_repair" in capabilities
        assert "technical_support" in capabilities
    
    def test_validate_agent_config_valid(self, base_config):
        """Test validating valid agent configuration."""
        agent = base_config.create_agent_with_config(
            agent_id="test_agent",
            name="Test Agent",
            role=AgentRole.TOURIST
        )
        
        assert base_config.validate_agent_config(agent) is True
    
    def test_validate_agent_config_invalid_personality(self, base_config):
        """Test validating agent with invalid personality traits."""
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            role=AgentRole.TOURIST,
            personality_traits={"invalid_trait": 2.0}  # Invalid value > 1.0
        )
        
        assert base_config.validate_agent_config(agent) is False


class TestAgentFactory:
    """Test agent factory."""
    
    @pytest.fixture
    def openai_config(self):
        """Create test OpenAI config."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return OpenAIConfig()
    
    @pytest.fixture
    def ag2_config(self):
        """Create test ag2 config."""
        return AG2Config()
    
    @pytest.fixture
    def base_config(self, openai_config, ag2_config):
        """Create test base agent config."""
        return BaseAgentConfig(openai_config, ag2_config)
    
    @pytest.fixture
    def agent_factory(self, base_config):
        """Create test agent factory."""
        return AgentFactory(base_config)
    
    def test_create_park_ranger(self, agent_factory):
        """Test creating park ranger agent."""
        agent = agent_factory.create_park_ranger("Test Ranger")
        
        assert agent.name == "Test Ranger"
        assert agent.role == AgentRole.PARK_RANGER
        assert agent.id.startswith("ranger_")
        assert "wildlife_management" in agent.capabilities
    
    def test_create_park_ranger_default_name(self, agent_factory):
        """Test creating park ranger with default name."""
        agent = agent_factory.create_park_ranger()
        
        assert agent.name.startswith("Ranger_")
        assert agent.role == AgentRole.PARK_RANGER
    
    def test_create_veterinarian(self, agent_factory):
        """Test creating veterinarian agent."""
        agent = agent_factory.create_veterinarian("Dr. Smith")
        
        assert agent.name == "Dr. Smith"
        assert agent.role == AgentRole.VETERINARIAN
        assert agent.id.startswith("vet_")
        assert "medical_treatment" in agent.capabilities
    
    def test_create_security_guard(self, agent_factory):
        """Test creating security guard agent."""
        agent = agent_factory.create_security_guard("Security Joe")
        
        assert agent.name == "Security Joe"
        assert agent.role == AgentRole.SECURITY
        assert agent.id.startswith("security_")
        assert "threat_assessment" in agent.capabilities
    
    def test_create_maintenance_worker(self, agent_factory):
        """Test creating maintenance worker agent."""
        agent = agent_factory.create_maintenance_worker("Maintenance Mike")
        
        assert agent.name == "Maintenance Mike"
        assert agent.role == AgentRole.MAINTENANCE
        assert agent.id.startswith("maintenance_")
        assert "equipment_repair" in agent.capabilities
    
    def test_create_tourist(self, agent_factory):
        """Test creating tourist agent."""
        custom_traits = {PersonalityTrait.FRIENDLY.value: 0.9}
        agent = agent_factory.create_tourist("Tourist Tom", custom_traits)
        
        assert agent.name == "Tourist Tom"
        assert agent.role == AgentRole.TOURIST
        assert agent.id.startswith("tourist_")
        assert agent.personality_traits[PersonalityTrait.FRIENDLY.value] == 0.9
    
    def test_create_dinosaur(self, agent_factory):
        """Test creating dinosaur agent."""
        from models.enums import DinosaurSpecies
        
        agent = agent_factory.create_dinosaur("Rex", DinosaurSpecies.TYRANNOSAURUS_REX)
        
        assert agent.name == "Rex"
        assert agent.role == AgentRole.DINOSAUR
        assert agent.id.startswith("dinosaur_")
        assert agent.species == DinosaurSpecies.TYRANNOSAURUS_REX
    
    def test_agent_counter_increments(self, agent_factory):
        """Test that agent counter increments properly."""
        agent1 = agent_factory.create_park_ranger()
        agent2 = agent_factory.create_park_ranger()
        
        # Extract counter from IDs
        counter1 = int(agent1.id.split("_")[-1])
        counter2 = int(agent2.id.split("_")[-1])
        
        assert counter2 == counter1 + 1