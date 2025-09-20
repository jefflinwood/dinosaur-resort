"""Unit tests for the base agent system with ag2 and OpenAI integration."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from models.config import OpenAIConfig, AG2Config, Location
from models.core import Agent
from models.enums import AgentRole, AgentState, PersonalityTrait, DinosaurSpecies
from agents.base_agent import DinosaurAgent, BaseAgentConfig, AgentFactory


class TestDinosaurAgent:
    """Test DinosaurAgent class."""
    
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
    def test_agent(self):
        """Create test agent model."""
        return Agent(
            id="test_ranger_1",
            name="Test Ranger",
            role=AgentRole.PARK_RANGER,
            personality_traits={
                PersonalityTrait.CAUTIOUS.value: 0.8,
                PersonalityTrait.LEADERSHIP.value: 0.7,
                PersonalityTrait.EMPATHY.value: 0.6
            },
            capabilities=["wildlife_management", "visitor_safety", "emergency_response"],
            location=Location(10.0, 20.0, "main_gate", "Main entrance gate")
        )
    
    @pytest.fixture
    def dinosaur_test_agent(self):
        """Create test dinosaur agent model."""
        return Agent(
            id="test_dino_1",
            name="Rex",
            role=AgentRole.DINOSAUR,
            species=DinosaurSpecies.TYRANNOSAURUS_REX,
            personality_traits={
                PersonalityTrait.BRAVE.value: 0.9
            },
            capabilities=["instinctual_behavior", "environmental_response"],
            location=Location(50.0, 60.0, "habitat", "T-Rex habitat area")
        )
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    def test_dinosaur_agent_initialization(self, mock_super_init, test_agent, openai_config, ag2_config):
        """Test DinosaurAgent initialization."""
        mock_super_init.return_value = None
        
        dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
        
        # Verify initialization
        assert dinosaur_agent.agent_model == test_agent
        assert dinosaur_agent.openai_config == openai_config
        assert dinosaur_agent.ag2_config == ag2_config
        
        # Verify ConversableAgent was initialized with correct parameters
        mock_super_init.assert_called_once()
        call_args = mock_super_init.call_args[1]
        
        assert call_args["name"] == "Test Ranger"
        assert "park ranger" in call_args["system_message"].lower()
        assert call_args["llm_config"]["config_list"][0]["api_key"] == "test-key"
        assert call_args["human_input_mode"] == "NEVER"
        assert call_args["max_consecutive_auto_reply"] == 10
    
    def test_generate_system_message_park_ranger(self, test_agent, openai_config, ag2_config):
        """Test system message generation for park ranger."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
            system_message = dinosaur_agent._generate_system_message()
        
        assert "dinosaur resort simulation" in system_message.lower()
        assert "park ranger" in system_message.lower()
        assert "wildlife management" in system_message.lower()
        assert "quite cautious" in system_message.lower()
        assert "quite leadership" in system_message.lower()
        assert "main_gate" in system_message
        assert "wildlife_management" in system_message
    
    def test_generate_system_message_dinosaur(self, dinosaur_test_agent, openai_config, ag2_config):
        """Test system message generation for dinosaur."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(dinosaur_test_agent, openai_config, ag2_config)
            system_message = dinosaur_agent._generate_system_message()
        
        assert "dinosaur resort simulation" in system_message.lower()
        assert "dinosaur with natural instincts" in system_message.lower()
        assert "tyrannosaurus rex" in system_message.lower()
        assert "very brave" in system_message.lower()
        assert "habitat" in system_message
    
    def test_get_role_context(self, test_agent, openai_config, ag2_config):
        """Test role context generation."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
            context = dinosaur_agent._get_role_context()
        
        assert "park ranger" in context.lower()
        assert "wildlife management" in context.lower()
        assert "visitor safety" in context.lower()
    
    def test_generate_personality_description(self, test_agent, openai_config, ag2_config):
        """Test personality description generation."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
            description = dinosaur_agent._generate_personality_description()
        
        assert "quite cautious" in description.lower()
        assert "quite leadership" in description.lower()
        assert "somewhat empathy" in description.lower()
    
    def test_generate_personality_description_empty(self, openai_config, ag2_config):
        """Test personality description with empty traits."""
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            role=AgentRole.TOURIST,
            personality_traits={}
        )
        
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(agent, openai_config, ag2_config)
            description = dinosaur_agent._generate_personality_description()
        
        assert description == "You have a balanced personality."
    
    def test_generate_capabilities_description(self, test_agent, openai_config, ag2_config):
        """Test capabilities description generation."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
            description = dinosaur_agent._generate_capabilities_description()
        
        assert "wildlife_management" in description
        assert "visitor_safety" in description
        assert "emergency_response" in description
    
    def test_update_state(self, test_agent, openai_config, ag2_config):
        """Test agent state update."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
            
            assert dinosaur_agent.agent_model.current_state == AgentState.IDLE
            
            dinosaur_agent.update_state(AgentState.ACTIVE)
            assert dinosaur_agent.agent_model.current_state == AgentState.ACTIVE
    
    def test_update_location(self, test_agent, openai_config, ag2_config):
        """Test agent location update."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
            
            new_location = Location(100.0, 200.0, "visitor_center", "Main visitor center")
            dinosaur_agent.update_location(new_location)
            
            assert dinosaur_agent.agent_model.location == new_location
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.ConversableAgent.system_message', new_callable=lambda: property(lambda self: "Test system message"))
    def test_get_agent_info(self, mock_system_message, mock_super_init, test_agent, openai_config, ag2_config):
        """Test getting agent information."""
        mock_super_init.return_value = None
        
        dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
        
        info = dinosaur_agent.get_agent_info()
        
        assert info["id"] == "test_ranger_1"
        assert info["name"] == "Test Ranger"
        assert info["role"] == "PARK_RANGER"
        assert info["current_state"] == "IDLE"
        assert info["location"]["zone"] == "main_gate"
        assert info["personality_traits"] == test_agent.personality_traits
        assert info["capabilities"] == test_agent.capabilities
        assert info["species"] is None
        assert info["system_message"] == "Test system message"
    
    @patch('agents.base_agent.ConversableAgent.generate_reply')
    def test_handle_event_notification(self, mock_generate_reply, test_agent, openai_config, ag2_config):
        """Test handling event notifications."""
        mock_generate_reply.return_value = "I will respond to this emergency immediately."
        
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
            
            response = dinosaur_agent.handle_event_notification(
                "Dinosaur escape in sector 7",
                {"severity": 8, "location": "sector_7"}
            )
        
        assert response == "I will respond to this emergency immediately."
        mock_generate_reply.assert_called_once()
        
        # Verify message format
        call_args = mock_generate_reply.call_args[1]
        messages = call_args["messages"]
        assert len(messages) == 1
        assert "Dinosaur escape in sector 7" in messages[0]["content"]
        assert "severity: 8" in messages[0]["content"]
        assert messages[0]["name"] == "EventSystem"
    
    @patch('agents.base_agent.ConversableAgent.generate_reply')
    def test_handle_event_notification_error(self, mock_generate_reply, test_agent, openai_config, ag2_config):
        """Test handling event notification with error."""
        mock_generate_reply.side_effect = Exception("OpenAI API error")
        
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
            
            response = dinosaur_agent.handle_event_notification("Test event", {})
        
        assert "cannot respond properly" in response
    
    @patch('agents.base_agent.ConversableAgent.generate_reply')
    def test_communicate_with_agent(self, mock_generate_reply, test_agent, openai_config, ag2_config):
        """Test agent-to-agent communication."""
        mock_generate_reply.return_value = "I understand your message."
        
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
            
            response = dinosaur_agent.communicate_with_agent(
                "We need to coordinate our response",
                "Security_1"
            )
        
        assert response == "I understand your message."
        mock_generate_reply.assert_called_once()
        
        # Verify message format
        call_args = mock_generate_reply.call_args[1]
        messages = call_args["messages"]
        assert len(messages) == 1
        assert messages[0]["content"] == "We need to coordinate our response"
        assert messages[0]["name"] == "Security_1"
    
    @patch('agents.base_agent.ConversableAgent.generate_reply')
    def test_communicate_with_agent_error(self, mock_generate_reply, test_agent, openai_config, ag2_config):
        """Test agent communication with error."""
        mock_generate_reply.side_effect = Exception("Communication error")
        
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur_agent = DinosaurAgent(test_agent, openai_config, ag2_config)
            
            response = dinosaur_agent.communicate_with_agent("Test message", "TestAgent")
        
        assert "trouble communicating" in response


class TestBaseAgentConfig:
    """Test BaseAgentConfig class."""
    
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
    
    def test_initialization(self, base_config):
        """Test BaseAgentConfig initialization."""
        assert base_config.openai_config is not None
        assert base_config.ag2_config is not None
        assert base_config.ag2_integration is not None
        assert len(base_config.role_configs) == 6  # All agent roles
    
    def test_role_configs_structure(self, base_config):
        """Test role configurations structure."""
        for role in AgentRole:
            assert role in base_config.role_configs
            config = base_config.role_configs[role]
            assert "capabilities" in config
            assert "personality_defaults" in config
            assert "system_prompt_additions" in config
            assert isinstance(config["capabilities"], list)
            assert isinstance(config["personality_defaults"], dict)
            assert isinstance(config["system_prompt_additions"], str)
    
    def test_create_agent_with_config_park_ranger(self, base_config):
        """Test creating park ranger with config."""
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
        assert agent.location.zone == "entrance"
    
    def test_create_agent_with_custom_personality(self, base_config):
        """Test creating agent with custom personality."""
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
        # Default traits should still be present
        assert PersonalityTrait.LEADERSHIP.value in agent.personality_traits
    
    def test_create_agent_with_custom_capabilities(self, base_config):
        """Test creating agent with custom capabilities."""
        custom_capabilities = ["custom_skill_1", "custom_skill_2"]
        
        agent = base_config.create_agent_with_config(
            agent_id="test_agent",
            name="Test Agent",
            role=AgentRole.MAINTENANCE,
            custom_capabilities=custom_capabilities
        )
        
        assert agent.capabilities == custom_capabilities
    
    def test_create_agent_with_custom_location(self, base_config):
        """Test creating agent with custom location."""
        custom_location = Location(100.0, 200.0, "custom_zone", "Custom area")
        
        agent = base_config.create_agent_with_config(
            agent_id="test_agent",
            name="Test Agent",
            role=AgentRole.TOURIST,
            location=custom_location
        )
        
        assert agent.location == custom_location
    
    def test_create_dinosaur_agent_with_species(self, base_config):
        """Test creating dinosaur agent with species."""
        agent = base_config.create_agent_with_config(
            agent_id="test_dino",
            name="Rex",
            role=AgentRole.DINOSAUR,
            species=DinosaurSpecies.TYRANNOSAURUS_REX
        )
        
        assert agent.species == DinosaurSpecies.TYRANNOSAURUS_REX
        assert agent.role == AgentRole.DINOSAUR
    
    @patch('agents.base_agent.DinosaurAgent')
    def test_create_dinosaur_agent_instance(self, mock_dinosaur_agent, base_config):
        """Test creating DinosaurAgent instance."""
        mock_instance = Mock()
        mock_dinosaur_agent.return_value = mock_instance
        
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            role=AgentRole.PARK_RANGER
        )
        
        result = base_config.create_dinosaur_agent(agent)
        
        mock_dinosaur_agent.assert_called_once_with(
            agent, base_config.openai_config, base_config.ag2_config
        )
        assert result == mock_instance
    
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
        
        # Should be a copy, not the original
        personality[PersonalityTrait.CAUTIOUS.value] = 0.5
        original = base_config.get_default_personality(AgentRole.PARK_RANGER)
        assert original[PersonalityTrait.CAUTIOUS.value] == 0.8
    
    def test_get_default_capabilities(self, base_config):
        """Test getting default capabilities for role."""
        capabilities = base_config.get_default_capabilities(AgentRole.MAINTENANCE)
        
        assert isinstance(capabilities, list)
        assert "equipment_repair" in capabilities
        assert "technical_support" in capabilities
        
        # Should be a copy, not the original
        capabilities.append("new_skill")
        original = base_config.get_default_capabilities(AgentRole.MAINTENANCE)
        assert "new_skill" not in original
    
    def test_validate_agent_config_valid(self, base_config):
        """Test validating valid agent configuration."""
        agent = base_config.create_agent_with_config(
            agent_id="test_agent",
            name="Test Agent",
            role=AgentRole.TOURIST
        )
        
        assert base_config.validate_agent_config(agent) is True
    
    def test_validate_agent_config_invalid_personality_value(self, base_config):
        """Test validating agent with invalid personality trait value."""
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            role=AgentRole.TOURIST,
            personality_traits={"invalid_trait": 2.0}  # Invalid value > 1.0
        )
        
        assert base_config.validate_agent_config(agent) is False
    
    def test_validate_agent_config_invalid_personality_negative(self, base_config):
        """Test validating agent with negative personality trait value."""
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            role=AgentRole.TOURIST,
            personality_traits={"trait": -0.5}  # Invalid negative value
        )
        
        assert base_config.validate_agent_config(agent) is False
    
    def test_validate_agent_config_invalid_capability_type(self, base_config):
        """Test validating agent with invalid capability type."""
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            role=AgentRole.TOURIST,
            capabilities=[123, "valid_capability"]  # Invalid non-string capability
        )
        
        assert base_config.validate_agent_config(agent) is False
    
    def test_validate_agent_config_exception(self, base_config):
        """Test validating agent configuration with exception."""
        # Create agent with invalid data that will cause exception
        agent = Mock()
        agent.name = "Test Agent"
        agent.personality_traits = None  # This will cause exception when iterating
        
        assert base_config.validate_agent_config(agent) is False


class TestAgentFactory:
    """Test AgentFactory class."""
    
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
    
    def test_initialization(self, agent_factory, base_config):
        """Test AgentFactory initialization."""
        assert agent_factory.base_config == base_config
        assert agent_factory._agent_counter == 0
    
    def test_create_park_ranger_with_name(self, agent_factory):
        """Test creating park ranger with custom name."""
        agent = agent_factory.create_park_ranger("Ranger Smith")
        
        assert agent.name == "Ranger Smith"
        assert agent.role == AgentRole.PARK_RANGER
        assert agent.id == "ranger_1"
        assert "wildlife_management" in agent.capabilities
        assert PersonalityTrait.CAUTIOUS.value in agent.personality_traits
    
    def test_create_park_ranger_default_name(self, agent_factory):
        """Test creating park ranger with default name."""
        agent = agent_factory.create_park_ranger()
        
        assert agent.name == "Ranger_1"
        assert agent.role == AgentRole.PARK_RANGER
        assert agent.id == "ranger_1"
    
    def test_create_park_ranger_with_location(self, agent_factory):
        """Test creating park ranger with custom location."""
        location = Location(50.0, 60.0, "patrol_area", "Patrol zone")
        agent = agent_factory.create_park_ranger("Ranger Bob", location)
        
        assert agent.location == location
    
    def test_create_veterinarian(self, agent_factory):
        """Test creating veterinarian."""
        agent = agent_factory.create_veterinarian("Dr. Johnson")
        
        assert agent.name == "Dr. Johnson"
        assert agent.role == AgentRole.VETERINARIAN
        assert agent.id == "vet_1"
        assert "medical_treatment" in agent.capabilities
        assert PersonalityTrait.ANALYTICAL.value in agent.personality_traits
    
    def test_create_veterinarian_default_name(self, agent_factory):
        """Test creating veterinarian with default name."""
        agent = agent_factory.create_veterinarian()
        
        assert agent.name == "Dr._1"
        assert agent.role == AgentRole.VETERINARIAN
    
    def test_create_security_guard(self, agent_factory):
        """Test creating security guard."""
        agent = agent_factory.create_security_guard("Officer Davis")
        
        assert agent.name == "Officer Davis"
        assert agent.role == AgentRole.SECURITY
        assert agent.id == "security_1"
        assert "threat_assessment" in agent.capabilities
        assert PersonalityTrait.BRAVE.value in agent.personality_traits
    
    def test_create_security_guard_default_name(self, agent_factory):
        """Test creating security guard with default name."""
        agent = agent_factory.create_security_guard()
        
        assert agent.name == "Security_1"
        assert agent.role == AgentRole.SECURITY
    
    def test_create_maintenance_worker(self, agent_factory):
        """Test creating maintenance worker."""
        agent = agent_factory.create_maintenance_worker("Mike the Mechanic")
        
        assert agent.name == "Mike the Mechanic"
        assert agent.role == AgentRole.MAINTENANCE
        assert agent.id == "maintenance_1"
        assert "equipment_repair" in agent.capabilities
        assert PersonalityTrait.TECHNICAL.value in agent.personality_traits
    
    def test_create_maintenance_worker_default_name(self, agent_factory):
        """Test creating maintenance worker with default name."""
        agent = agent_factory.create_maintenance_worker()
        
        assert agent.name == "Maintenance_1"
        assert agent.role == AgentRole.MAINTENANCE
    
    def test_create_tourist(self, agent_factory):
        """Test creating tourist."""
        custom_traits = {PersonalityTrait.FRIENDLY.value: 0.9}
        agent = agent_factory.create_tourist("Tourist Tom", custom_traits)
        
        assert agent.name == "Tourist Tom"
        assert agent.role == AgentRole.TOURIST
        assert agent.id == "tourist_1"
        assert agent.personality_traits[PersonalityTrait.FRIENDLY.value] == 0.9
        assert "observation" in agent.capabilities
    
    def test_create_tourist_default_name(self, agent_factory):
        """Test creating tourist with default name."""
        agent = agent_factory.create_tourist()
        
        assert agent.name == "Tourist_1"
        assert agent.role == AgentRole.TOURIST
    
    def test_create_dinosaur(self, agent_factory):
        """Test creating dinosaur."""
        custom_traits = {PersonalityTrait.BRAVE.value: 0.8}
        agent = agent_factory.create_dinosaur(
            "Rex", 
            DinosaurSpecies.TYRANNOSAURUS_REX, 
            custom_traits
        )
        
        assert agent.name == "Rex"
        assert agent.role == AgentRole.DINOSAUR
        assert agent.id == "dinosaur_1"
        assert agent.species == DinosaurSpecies.TYRANNOSAURUS_REX
        assert agent.personality_traits[PersonalityTrait.BRAVE.value] == 0.8
        assert "instinctual_behavior" in agent.capabilities
        assert agent.location.zone == "habitat"
    
    def test_create_dinosaur_with_location(self, agent_factory):
        """Test creating dinosaur with custom location."""
        location = Location(100.0, 150.0, "trex_enclosure", "T-Rex enclosure")
        agent = agent_factory.create_dinosaur(
            "Rexy", 
            DinosaurSpecies.TYRANNOSAURUS_REX,
            location=location
        )
        
        assert agent.location == location
    
    @patch('agents.base_agent.BaseAgentConfig.create_dinosaur_agent')
    def test_create_dinosaur_agent_instance(self, mock_create_dinosaur_agent, agent_factory):
        """Test creating DinosaurAgent instance."""
        mock_instance = Mock()
        mock_create_dinosaur_agent.return_value = mock_instance
        
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            role=AgentRole.PARK_RANGER
        )
        
        result = agent_factory.create_dinosaur_agent_instance(agent)
        
        mock_create_dinosaur_agent.assert_called_once_with(agent)
        assert result == mock_instance
    
    def test_agent_counter_increments(self, agent_factory):
        """Test that agent counter increments properly."""
        agent1 = agent_factory.create_park_ranger()
        agent2 = agent_factory.create_veterinarian()
        agent3 = agent_factory.create_security_guard()
        
        assert agent1.id == "ranger_1"
        assert agent2.id == "vet_2"
        assert agent3.id == "security_3"
    
    def test_mixed_agent_creation(self, agent_factory):
        """Test creating multiple different types of agents."""
        ranger = agent_factory.create_park_ranger("Ranger Alpha")
        vet = agent_factory.create_veterinarian("Dr. Beta")
        tourist = agent_factory.create_tourist("Tourist Gamma")
        dino = agent_factory.create_dinosaur("Delta", DinosaurSpecies.VELOCIRAPTOR)
        
        # Verify all agents are different
        agents = [ranger, vet, tourist, dino]
        ids = [agent.id for agent in agents]
        names = [agent.name for agent in agents]
        roles = [agent.role for agent in agents]
        
        assert len(set(ids)) == 4  # All IDs unique
        assert len(set(names)) == 4  # All names unique
        assert len(set(roles)) == 4  # All roles unique
        
        # Verify counter incremented correctly
        assert ranger.id == "ranger_1"
        assert vet.id == "vet_2"
        assert tourist.id == "tourist_3"
        assert dino.id == "dinosaur_4"