"""Unit tests for dinosaur agent implementations."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from agents.dinosaur_agents import (
    DinosaurSpeciesAgent, TyrannosaurusRexAgent, TriceratopsAgent, VelociraptorAgent,
    BrachiosaurusAgent, StegosaurusAgent, ParasaurolophusAgent, DinosaurAgentFactory
)
from models.core import Agent
from models.config import OpenAIConfig, AG2Config, Location
from models.enums import AgentRole, AgentState, PersonalityTrait, DinosaurSpecies


@pytest.fixture
def openai_config():
    """Create OpenAI configuration for testing."""
    return OpenAIConfig(
        api_key="test-key",
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=150,
        timeout=30
    )


@pytest.fixture
def ag2_config():
    """Create ag2 configuration for testing."""
    return AG2Config(
        human_input_mode="NEVER",
        max_round=3,
        code_execution_config=False
    )


@pytest.fixture
def trex_agent_model():
    """Create T-Rex agent model for testing."""
    return Agent(
        id="trex_001",
        name="Rexy",
        role=AgentRole.DINOSAUR,
        personality_traits={
            PersonalityTrait.BRAVE.value: 0.9,
            PersonalityTrait.CAUTIOUS.value: 0.2,
            PersonalityTrait.ANALYTICAL.value: 0.7
        },
        capabilities=["instinctual_behavior", "environmental_response", "apex_predator"],
        location=Location(10.0, 20.0, "carnivore_habitat", "Large carnivore habitat"),
        species=DinosaurSpecies.TYRANNOSAURUS_REX
    )


@pytest.fixture
def triceratops_agent_model():
    """Create Triceratops agent model for testing."""
    return Agent(
        id="triceratops_001",
        name="Trike",
        role=AgentRole.DINOSAUR,
        personality_traits={
            PersonalityTrait.CAUTIOUS.value: 0.7,
            PersonalityTrait.EMPATHY.value: 0.8,
            PersonalityTrait.FRIENDLY.value: 0.6
        },
        capabilities=["instinctual_behavior", "environmental_response", "herd_behavior"],
        location=Location(15.0, 25.0, "herbivore_habitat", "Herbivore grazing area"),
        species=DinosaurSpecies.TRICERATOPS
    )


@pytest.fixture
def velociraptor_agent_model():
    """Create Velociraptor agent model for testing."""
    return Agent(
        id="velociraptor_001",
        name="Blue",
        role=AgentRole.DINOSAUR,
        personality_traits={
            PersonalityTrait.ANALYTICAL.value: 0.9,
            PersonalityTrait.CREATIVE.value: 0.8,
            PersonalityTrait.BRAVE.value: 0.7
        },
        capabilities=["instinctual_behavior", "environmental_response", "pack_hunting", "problem_solving"],
        location=Location(5.0, 15.0, "raptor_enclosure", "Secure raptor habitat"),
        species=DinosaurSpecies.VELOCIRAPTOR
    )


class TestDinosaurSpeciesAgent:
    """Test cases for base DinosaurSpeciesAgent."""
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    def test_dinosaur_species_agent_initialization(self, mock_super_init, trex_agent_model, openai_config, ag2_config):
        """Test dinosaur species agent initialization."""
        mock_super_init.return_value = None
        
        dinosaur = DinosaurSpeciesAgent(trex_agent_model, openai_config, ag2_config)
        
        assert dinosaur.agent_model == trex_agent_model
        assert dinosaur.happiness_level == 0.7  # Default happiness
        assert dinosaur.health_status == "healthy"
        assert dinosaur.hunger_level == 0.3
        assert dinosaur.stress_level == 0.2
        assert 0.0 <= dinosaur.social_needs <= 1.0
        assert dinosaur.territory_size > 0
        assert dinosaur.logger.name.endswith("TYRANNOSAURUS_REX.Rexy")
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_react_to_environment(self, mock_generate_reply, mock_super_init,
                                trex_agent_model, openai_config, ag2_config):
        """Test environmental reaction."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "I sense changes in my territory. The weather affects my comfort."
        
        dinosaur = DinosaurSpeciesAgent(trex_agent_model, openai_config, ag2_config)
        dinosaur.update_state = Mock()
        
        environmental_factors = {
            "weather": "storm",
            "food_availability": 0.8,
            "visitor_density": 0.6,
            "available_space": 0.9
        }
        
        result = dinosaur.react_to_environment(environmental_factors)
        
        assert "reaction" in result
        assert "happiness_change" in result
        assert "stress_change" in result
        assert "new_happiness" in result
        assert "new_stress" in result
        assert "behavioral_state" in result
        
        dinosaur.update_state.assert_any_call(AgentState.RESPONDING_TO_EVENT)
        dinosaur.update_state.assert_any_call(AgentState.IDLE)
    
    def test_calculate_environmental_happiness_change(self, trex_agent_model, openai_config, ag2_config):
        """Test environmental happiness change calculation."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur = DinosaurSpeciesAgent(trex_agent_model, openai_config, ag2_config)
            
            # Test positive factors
            positive_factors = {
                "weather": "sunny",
                "food_availability": 0.8,
                "visitor_density": 0.3,
                "available_space": 0.9
            }
            change = dinosaur._calculate_environmental_happiness_change(positive_factors)
            assert change > 0
            
            # Test negative factors
            negative_factors = {
                "weather": "storm",
                "food_availability": 0.2,
                "visitor_density": 0.9,
                "available_space": 0.3
            }
            change = dinosaur._calculate_environmental_happiness_change(negative_factors)
            assert change < 0
    
    def test_calculate_environmental_stress_change(self, trex_agent_model, openai_config, ag2_config):
        """Test environmental stress change calculation."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur = DinosaurSpeciesAgent(trex_agent_model, openai_config, ag2_config)
            
            # Test stress-inducing factors
            stressful_factors = {
                "noise_level": 0.8,
                "visitor_proximity": 0.9,
                "disruptions": 3,
                "medical_care": False
            }
            change = dinosaur._calculate_environmental_stress_change(stressful_factors)
            assert change > 0
            
            # Test stress-reducing factors
            calming_factors = {
                "noise_level": 0.2,
                "visitor_proximity": 0.1,
                "disruptions": 0,
                "medical_care": True
            }
            change = dinosaur._calculate_environmental_stress_change(calming_factors)
            assert change < 0
    
    def test_determine_behavioral_state(self, trex_agent_model, openai_config, ag2_config):
        """Test behavioral state determination."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur = DinosaurSpeciesAgent(trex_agent_model, openai_config, ag2_config)
            
            # Test different mood combinations
            dinosaur.stress_level = 0.9
            assert dinosaur._determine_behavioral_state() == "highly_agitated"
            
            dinosaur.stress_level = 0.7
            assert dinosaur._determine_behavioral_state() == "stressed"
            
            dinosaur.stress_level = 0.3
            dinosaur.happiness_level = 0.9
            assert dinosaur._determine_behavioral_state() == "content"
            
            dinosaur.happiness_level = 0.7
            assert dinosaur._determine_behavioral_state() == "calm"
            
            dinosaur.happiness_level = 0.2
            assert dinosaur._determine_behavioral_state() == "unhappy"
            
            dinosaur.happiness_level = 0.5
            assert dinosaur._determine_behavioral_state() == "neutral"
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_interact_with_humans(self, mock_generate_reply, mock_super_init,
                                trex_agent_model, openai_config, ag2_config):
        """Test human interaction."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "I notice humans nearby. They seem small and cautious."
        
        dinosaur = DinosaurSpeciesAgent(trex_agent_model, openai_config, ag2_config)
        dinosaur.update_state = Mock()
        
        human_info = {
            "type": "visitor",
            "behavior": "observing",
            "distance": 10.0,
            "group_size": 3
        }
        
        result = dinosaur.interact_with_humans(human_info)
        
        assert "interaction" in result
        assert "happiness_change" in result
        assert "stress_change" in result
        assert "threat_level" in result
        
        dinosaur.update_state.assert_any_call(AgentState.COMMUNICATING)
        dinosaur.update_state.assert_any_call(AgentState.IDLE)
    
    def test_assess_threat_level(self, trex_agent_model, openai_config, ag2_config):
        """Test threat level assessment."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur = DinosaurSpeciesAgent(trex_agent_model, openai_config, ag2_config)
            
            # Test different threat scenarios
            high_threat = {"behavior": "aggressive", "distance": 0.5}
            assert dinosaur._assess_threat_level(high_threat) == "high"
            
            medium_threat = {"behavior": "loud_noises", "distance": 2.0}
            assert dinosaur._assess_threat_level(medium_threat) == "medium"
            
            low_threat = {"behavior": "observing", "distance": 5.0}
            assert dinosaur._assess_threat_level(low_threat) == "low"
            
            no_threat = {"behavior": "observing", "distance": 15.0}
            assert dinosaur._assess_threat_level(no_threat) == "none"
    
    def test_update_health_status(self, trex_agent_model, openai_config, ag2_config):
        """Test health status update."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur = DinosaurSpeciesAgent(trex_agent_model, openai_config, ag2_config)
            
            initial_happiness = dinosaur.happiness_level
            initial_stress = dinosaur.stress_level
            
            # Test getting sick
            dinosaur.update_health_status("sick", "viral infection")
            assert dinosaur.health_status == "sick"
            assert dinosaur.happiness_level < initial_happiness
            assert dinosaur.stress_level > initial_stress
            
            # Test recovery
            dinosaur.update_health_status("healthy", "treatment successful")
            assert dinosaur.health_status == "healthy"
            assert dinosaur.happiness_level > 0.0  # Should improve
    
    def test_get_mood_report(self, trex_agent_model, openai_config, ag2_config):
        """Test mood report generation."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            dinosaur = DinosaurSpeciesAgent(trex_agent_model, openai_config, ag2_config)
            
            report = dinosaur.get_mood_report()
            
            assert report["dinosaur_id"] == "trex_001"
            assert report["name"] == "Rexy"
            assert report["species"] == "TYRANNOSAURUS_REX"
            assert "happiness_level" in report
            assert "stress_level" in report
            assert "health_status" in report
            assert "behavioral_state" in report
            assert "current_location" in report


class TestTyrannosaurusRexAgent:
    """Test cases for TyrannosaurusRexAgent."""
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    def test_trex_initialization(self, mock_super_init, trex_agent_model, openai_config, ag2_config):
        """Test T-Rex agent initialization."""
        mock_super_init.return_value = None
        
        trex = TyrannosaurusRexAgent(trex_agent_model, openai_config, ag2_config)
        
        assert trex.social_needs == 0.2  # Solitary
        assert trex.territory_size == 500.0  # Large territory
    
    def test_trex_environmental_reactions(self, trex_agent_model, openai_config, ag2_config):
        """Test T-Rex specific environmental reactions."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            trex = TyrannosaurusRexAgent(trex_agent_model, openai_config, ag2_config)
            
            # T-Rex should like hot weather
            hot_weather = {"weather": "hot", "visitor_density": 0.3}
            change = trex._calculate_environmental_happiness_change(hot_weather)
            assert change > 0
            
            # T-Rex should dislike crowds
            crowded = {"weather": "clear", "visitor_density": 0.8}
            change = trex._calculate_environmental_happiness_change(crowded)
            assert change < 0
    
    def test_trex_threat_assessment(self, trex_agent_model, openai_config, ag2_config):
        """Test T-Rex threat assessment (more aggressive)."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            trex = TyrannosaurusRexAgent(trex_agent_model, openai_config, ag2_config)
            
            # T-Rex should be more likely to see threats
            moderate_distance = {"behavior": "observing", "distance": 12.0}
            threat_level = trex._assess_threat_level(moderate_distance)
            assert threat_level in ["low", "medium"]  # More aggressive than base


class TestTriceratopsAgent:
    """Test cases for TriceratopsAgent."""
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    def test_triceratops_initialization(self, mock_super_init, triceratops_agent_model, openai_config, ag2_config):
        """Test Triceratops agent initialization."""
        mock_super_init.return_value = None
        
        trike = TriceratopsAgent(triceratops_agent_model, openai_config, ag2_config)
        
        assert trike.social_needs == 0.8  # Herd animal
        assert trike.territory_size == 200.0  # Moderate territory
    
    def test_triceratops_environmental_reactions(self, triceratops_agent_model, openai_config, ag2_config):
        """Test Triceratops specific environmental reactions."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            trike = TriceratopsAgent(triceratops_agent_model, openai_config, ag2_config)
            
            # Triceratops should like good vegetation
            good_vegetation = {"vegetation_quality": 0.8, "visitor_density": 0.5}
            change = trike._calculate_environmental_happiness_change(good_vegetation)
            assert change > 0
            
            # Should be more tolerant of moderate crowds
            moderate_crowd = {"visitor_density": 0.6}
            change = trike._calculate_environmental_happiness_change(moderate_crowd)
            assert change >= 0  # Should be neutral or positive
    
    def test_triceratops_human_interaction(self, triceratops_agent_model, openai_config, ag2_config):
        """Test Triceratops human interaction (more docile)."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            trike = TriceratopsAgent(triceratops_agent_model, openai_config, ag2_config)
            
            # Should be more positive with calm humans
            calm_interaction = {"behavior": "calm_observation", "distance": 8.0}
            change = trike._calculate_interaction_happiness_change(calm_interaction)
            assert change > 0


class TestVelociraptorAgent:
    """Test cases for VelociraptorAgent."""
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    def test_velociraptor_initialization(self, mock_super_init, velociraptor_agent_model, openai_config, ag2_config):
        """Test Velociraptor agent initialization."""
        mock_super_init.return_value = None
        
        raptor = VelociraptorAgent(velociraptor_agent_model, openai_config, ag2_config)
        
        assert raptor.social_needs == 0.9  # Pack hunters
        assert raptor.territory_size == 150.0  # Moderate territory
    
    def test_velociraptor_environmental_reactions(self, velociraptor_agent_model, openai_config, ag2_config):
        """Test Velociraptor specific environmental reactions."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            raptor = VelociraptorAgent(velociraptor_agent_model, openai_config, ag2_config)
            
            # Velociraptors should like enrichment
            enriched_environment = {"environmental_enrichment": 0.8, "activity_opportunities": 0.7}
            change = raptor._calculate_environmental_happiness_change(enriched_environment)
            assert change > 0
            
            # Should dislike lack of stimulation
            boring_environment = {"environmental_enrichment": 0.2, "activity_opportunities": 0.3}
            change = raptor._calculate_environmental_happiness_change(boring_environment)
            assert change < 0
    
    def test_velociraptor_threat_assessment(self, velociraptor_agent_model, openai_config, ag2_config):
        """Test Velociraptor intelligent threat assessment."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            raptor = VelociraptorAgent(velociraptor_agent_model, openai_config, ag2_config)
            
            # Should be concerned about large groups
            large_group = {"behavior": "observing", "distance": 10.0, "group_size": 8}
            threat_level = raptor._assess_threat_level(large_group)
            assert threat_level == "medium"  # Intelligent assessment


class TestDinosaurAgentFactory:
    """Test cases for DinosaurAgentFactory."""
    
    def test_factory_initialization(self, openai_config, ag2_config):
        """Test dinosaur agent factory initialization."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        assert factory.openai_config == openai_config
        assert factory.ag2_config == ag2_config
        assert factory.logger is not None
        assert factory._dinosaur_counter == 0
        assert len(factory.species_classes) == 6  # All species covered
    
    @patch('agents.dinosaur_agents.TyrannosaurusRexAgent')
    def test_create_dinosaur_agent_trex(self, mock_trex_class, trex_agent_model, openai_config, ag2_config):
        """Test T-Rex agent creation."""
        mock_trex_instance = Mock()
        mock_trex_class.return_value = mock_trex_instance
        
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        result = factory.create_dinosaur_agent(trex_agent_model)
        
        assert result == mock_trex_instance
        mock_trex_class.assert_called_once_with(trex_agent_model, openai_config, ag2_config)
    
    def test_create_dinosaur_agent_wrong_role(self, openai_config, ag2_config):
        """Test dinosaur agent creation with wrong role."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        wrong_role_agent = Agent(
            id="staff_001",
            name="Staff Member",
            role=AgentRole.PARK_RANGER,
            location=Location(0.0, 0.0, "entrance")
        )
        
        with pytest.raises(ValueError, match="Agent model must have DINOSAUR role"):
            factory.create_dinosaur_agent(wrong_role_agent)
    
    def test_create_dinosaur_agent_no_species(self, openai_config, ag2_config):
        """Test dinosaur agent creation without species."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        # The Agent model itself should prevent creation of dinosaur without species
        with pytest.raises(ValueError, match="Dinosaur agents must have a species"):
            Agent(
                id="dino_001",
                name="Unknown Dino",
                role=AgentRole.DINOSAUR,
                location=Location(0.0, 0.0, "habitat")
            )
    
    def test_create_tyrannosaurus_rex(self, openai_config, ag2_config):
        """Test T-Rex model creation."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        agent = factory.create_tyrannosaurus_rex("Rexy")
        
        assert agent.role == AgentRole.DINOSAUR
        assert agent.species == DinosaurSpecies.TYRANNOSAURUS_REX
        assert agent.name == "Rexy"
        assert agent.personality_traits[PersonalityTrait.BRAVE.value] >= 0.8
        assert "apex_predator" in agent.capabilities
        assert agent.location.zone == "carnivore_habitat"
    
    def test_create_triceratops(self, openai_config, ag2_config):
        """Test Triceratops model creation."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        agent = factory.create_triceratops("Trike")
        
        assert agent.role == AgentRole.DINOSAUR
        assert agent.species == DinosaurSpecies.TRICERATOPS
        assert agent.name == "Trike"
        assert agent.personality_traits[PersonalityTrait.EMPATHY.value] >= 0.7
        assert "herd_behavior" in agent.capabilities
        assert agent.location.zone == "herbivore_habitat"
    
    def test_create_velociraptor(self, openai_config, ag2_config):
        """Test Velociraptor model creation."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        agent = factory.create_velociraptor("Blue")
        
        assert agent.role == AgentRole.DINOSAUR
        assert agent.species == DinosaurSpecies.VELOCIRAPTOR
        assert agent.name == "Blue"
        assert agent.personality_traits[PersonalityTrait.ANALYTICAL.value] >= 0.8
        assert "pack_hunting" in agent.capabilities
        assert "problem_solving" in agent.capabilities
        assert agent.location.zone == "raptor_enclosure"
    
    def test_create_brachiosaurus(self, openai_config, ag2_config):
        """Test Brachiosaurus model creation."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        agent = factory.create_brachiosaurus("Brachy")
        
        assert agent.role == AgentRole.DINOSAUR
        assert agent.species == DinosaurSpecies.BRACHIOSAURUS
        assert agent.name == "Brachy"
        assert agent.personality_traits[PersonalityTrait.EMPATHY.value] >= 0.8
        assert "gentle_giant" in agent.capabilities
        assert agent.location.zone == "sauropod_habitat"
    
    def test_create_stegosaurus(self, openai_config, ag2_config):
        """Test Stegosaurus model creation."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        agent = factory.create_stegosaurus("Spike")
        
        assert agent.role == AgentRole.DINOSAUR
        assert agent.species == DinosaurSpecies.STEGOSAURUS
        assert agent.name == "Spike"
        assert agent.personality_traits[PersonalityTrait.CAUTIOUS.value] >= 0.7
        assert "defensive_behavior" in agent.capabilities
        assert agent.location.zone == "herbivore_habitat"
    
    def test_create_parasaurolophus(self, openai_config, ag2_config):
        """Test Parasaurolophus model creation."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        agent = factory.create_parasaurolophus("Para")
        
        assert agent.role == AgentRole.DINOSAUR
        assert agent.species == DinosaurSpecies.PARASAUROLOPHUS
        assert agent.name == "Para"
        assert agent.personality_traits[PersonalityTrait.FRIENDLY.value] >= 0.8
        assert "vocal_communication" in agent.capabilities
        assert "herd_behavior" in agent.capabilities
        assert agent.location.zone == "herbivore_habitat"
    
    def test_create_random_dinosaur(self, openai_config, ag2_config):
        """Test random dinosaur creation."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        # Create multiple random dinosaurs to test variety
        dinosaurs = [factory.create_random_dinosaur(f"Dino_{i}") for i in range(10)]
        
        # All should be dinosaurs
        assert all(dino.role == AgentRole.DINOSAUR for dino in dinosaurs)
        assert all(dino.species is not None for dino in dinosaurs)
        
        # Should have variety in species (at least some different)
        species = [dino.species for dino in dinosaurs]
        assert len(set(species)) > 1  # Should have some variety
    
    def test_dinosaur_counter_increment(self, openai_config, ag2_config):
        """Test that dinosaur counter increments properly."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        initial_counter = factory._dinosaur_counter
        
        factory.create_tyrannosaurus_rex("Rexy1")
        assert factory._dinosaur_counter == initial_counter + 1
        
        factory.create_triceratops("Trike1")
        assert factory._dinosaur_counter == initial_counter + 2
        
        factory.create_velociraptor("Blue1")
        assert factory._dinosaur_counter == initial_counter + 3
    
    def test_custom_location_assignment(self, openai_config, ag2_config):
        """Test that custom locations are used when provided."""
        factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        custom_location = Location(100.0, 200.0, "custom_habitat", "Custom dinosaur habitat")
        
        agent = factory.create_tyrannosaurus_rex("Custom Rex", location=custom_location)
        assert agent.location == custom_location
        assert agent.location.zone == "custom_habitat"


if __name__ == "__main__":
    pytest.main([__file__])