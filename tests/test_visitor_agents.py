"""Unit tests for visitor agent implementations."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from agents.visitor_agents import TouristAgent, VisitorAgentFactory
from models.core import Agent
from models.config import OpenAIConfig, AG2Config, Location
from models.enums import AgentRole, AgentState, PersonalityTrait


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
def adventure_tourist_model():
    """Create adventure-seeking tourist agent model for testing."""
    return Agent(
        id="tourist_001",
        name="Adventure Alice",
        role=AgentRole.TOURIST,
        personality_traits={
            PersonalityTrait.BRAVE.value: 0.8,
            PersonalityTrait.CAUTIOUS.value: 0.2,
            PersonalityTrait.FRIENDLY.value: 0.7,
            PersonalityTrait.CREATIVE.value: 0.6
        },
        capabilities=["observation", "feedback", "exploration", "risk_taking"],
        location=Location(10.0, 20.0, "adventure_zone", "High-thrill adventure area")
    )


@pytest.fixture
def family_tourist_model():
    """Create family tourist agent model for testing."""
    return Agent(
        id="tourist_002",
        name="Family Bob",
        role=AgentRole.TOURIST,
        personality_traits={
            PersonalityTrait.CAUTIOUS.value: 0.8,
            PersonalityTrait.EMPATHY.value: 0.9,
            PersonalityTrait.FRIENDLY.value: 0.7,
            PersonalityTrait.BRAVE.value: 0.3
        },
        capabilities=["observation", "feedback", "child_supervision", "safety_awareness"],
        location=Location(5.0, 15.0, "family_area", "Family-friendly zone")
    )


@pytest.fixture
def educational_tourist_model():
    """Create educational tourist agent model for testing."""
    return Agent(
        id="tourist_003",
        name="Scholar Carol",
        role=AgentRole.TOURIST,
        personality_traits={
            PersonalityTrait.ANALYTICAL.value: 0.9,
            PersonalityTrait.CAUTIOUS.value: 0.6,
            PersonalityTrait.FRIENDLY.value: 0.5,
            PersonalityTrait.CREATIVE.value: 0.4
        },
        capabilities=["observation", "feedback", "research", "documentation"],
        location=Location(15.0, 25.0, "education_center", "Educational exhibits area")
    )


class TestTouristAgent:
    """Test cases for TouristAgent."""
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    def test_tourist_initialization(self, mock_super_init, adventure_tourist_model, openai_config, ag2_config):
        """Test tourist agent initialization."""
        mock_super_init.return_value = None
        
        tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
        
        assert tourist.agent_model == adventure_tourist_model
        assert tourist.satisfaction_level == 0.7  # Default satisfaction
        assert 0.0 <= tourist.risk_tolerance <= 1.0
        assert isinstance(tourist.interests, list)
        assert tourist.visit_duration == 0
        assert tourist.logger.name.endswith("Tourist.Adventure Alice")
    
    def test_calculate_risk_tolerance_high_brave(self, adventure_tourist_model, openai_config, ag2_config):
        """Test risk tolerance calculation for brave tourist."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
            
            # Brave (0.8) and not cautious (0.2) should result in high risk tolerance
            expected_risk = (1.0 - 0.2 + 0.8) / 2.0  # (1.0 - cautious + brave) / 2.0
            assert abs(tourist.risk_tolerance - expected_risk) < 0.01
    
    def test_calculate_risk_tolerance_low_cautious(self, family_tourist_model, openai_config, ag2_config):
        """Test risk tolerance calculation for cautious tourist."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(family_tourist_model, openai_config, ag2_config)
            
            # Cautious (0.8) and not brave (0.3) should result in low risk tolerance
            expected_risk = (1.0 - 0.8 + 0.3) / 2.0
            assert abs(tourist.risk_tolerance - expected_risk) < 0.01
    
    def test_determine_interests_analytical(self, educational_tourist_model, openai_config, ag2_config):
        """Test interest determination for analytical tourist."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(educational_tourist_model, openai_config, ag2_config)
            
            # Analytical trait > 0.6 should include educational interests
            assert "educational_tours" in tourist.interests
            assert "scientific_exhibits" in tourist.interests
    
    def test_determine_interests_brave(self, adventure_tourist_model, openai_config, ag2_config):
        """Test interest determination for brave tourist."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
            
            # Brave trait > 0.7 should include adventure interests
            assert "adventure_tours" in tourist.interests
            assert "close_encounters" in tourist.interests
    
    def test_determine_interests_friendly(self, family_tourist_model, openai_config, ag2_config):
        """Test interest determination for friendly tourist."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(family_tourist_model, openai_config, ag2_config)
            
            # Friendly trait > 0.6 should include social interests
            assert "group_activities" in tourist.interests
            assert "interactive_experiences" in tourist.interests
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_react_to_event_high_severity(self, mock_generate_reply, mock_super_init,
                                        adventure_tourist_model, openai_config, ag2_config):
        """Test tourist reaction to high severity event."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "This is terrifying! I want to leave immediately!"
        
        tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
        tourist.update_state = Mock()
        
        result = tourist.react_to_event("Dinosaur escape", 9, "adventure_zone")
        
        assert "reaction" in result
        assert "reaction_intensity" in result
        assert "satisfaction_change" in result
        assert "new_satisfaction" in result
        assert "wants_to_leave" in result
        assert result["wants_to_leave"] is True  # High severity should trigger leave desire
        
        tourist.update_state.assert_any_call(AgentState.RESPONDING_TO_EVENT)
        tourist.update_state.assert_any_call(AgentState.IDLE)
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_react_to_event_low_severity(self, mock_generate_reply, mock_super_init,
                                       adventure_tourist_model, openai_config, ag2_config):
        """Test tourist reaction to low severity event."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "That's interesting! I'd like to see more."
        
        tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
        tourist.update_state = Mock()
        
        result = tourist.react_to_event("Minor equipment malfunction", 2, "maintenance_area")
        
        assert result["wants_to_leave"] is False  # Low severity shouldn't trigger leave desire
        assert result["reaction_intensity"] < 0.5  # Should be low intensity
    
    def test_calculate_reaction_intensity_brave_tourist(self, adventure_tourist_model, openai_config, ag2_config):
        """Test reaction intensity calculation for brave tourist."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
            
            # Brave tourist should have lower reaction intensity
            intensity = tourist._calculate_reaction_intensity(5)
            assert 0.0 <= intensity <= 1.0
            
            # Should be less reactive than a cautious tourist would be
            assert intensity < 0.8  # Reasonable threshold for brave tourist
    
    def test_calculate_reaction_intensity_cautious_tourist(self, family_tourist_model, openai_config, ag2_config):
        """Test reaction intensity calculation for cautious tourist."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(family_tourist_model, openai_config, ag2_config)
            
            # Cautious tourist should have higher reaction intensity
            intensity = tourist._calculate_reaction_intensity(5)
            assert 0.0 <= intensity <= 1.0
            
            # Should be more reactive than a brave tourist would be
            assert intensity > 0.3  # Reasonable threshold for cautious tourist
    
    def test_calculate_satisfaction_change_same_location(self, adventure_tourist_model, openai_config, ag2_config):
        """Test satisfaction change when event is in same location."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
            
            # Event in same location should have proximity penalty
            change = tourist._calculate_satisfaction_change(7, "adventure_zone")
            assert change < -0.1  # Should be negative due to proximity penalty
    
    def test_calculate_satisfaction_change_different_location(self, adventure_tourist_model, openai_config, ag2_config):
        """Test satisfaction change when event is in different location."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
            
            # Event in different location should have less impact
            change_same = tourist._calculate_satisfaction_change(7, "adventure_zone")
            change_different = tourist._calculate_satisfaction_change(7, "far_away_zone")
            
            assert change_different > change_same  # Different location should be less negative
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_provide_feedback(self, mock_generate_reply, mock_super_init,
                            adventure_tourist_model, openai_config, ag2_config):
        """Test feedback provision."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "The adventure tours were amazing! Very thrilling experience."
        
        tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
        
        result = tourist.provide_feedback("adventure_tours")
        
        assert "feedback" in result
        assert result["topic"] == "adventure_tours"
        assert "satisfaction_rating" in result
        assert "interests" in result
        assert "would_recommend" in result
        assert isinstance(result["would_recommend"], bool)
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_make_purchase_decision_interested_item(self, mock_generate_reply, mock_super_init,
                                                  adventure_tourist_model, openai_config, ag2_config):
        """Test purchase decision for item matching interests."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "Yes, I'll buy this adventure photo book!"
        
        tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
        tourist.satisfaction_level = 0.8  # High satisfaction
        
        # Use an item that should match adventure interests
        result = tourist.make_purchase_decision("adventure guidebook", 25.0)
        
        assert "decision" in result
        assert "will_buy" in result
        assert result["item"] == "adventure guidebook"
        assert result["price"] == 25.0
        assert "reasoning_factors" in result
        
        # Should likely buy due to high satisfaction and interest match
        factors = result["reasoning_factors"]
        assert factors["satisfaction"] == 0.8
        assert factors["interest_match"] is True
        assert factors["price_acceptable"] is True
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_make_purchase_decision_expensive_item(self, mock_generate_reply, mock_super_init,
                                                 adventure_tourist_model, openai_config, ag2_config):
        """Test purchase decision for expensive item."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "That's too expensive for me right now."
        
        tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
        
        result = tourist.make_purchase_decision("expensive souvenir", 75.0)
        
        assert result["will_buy"] is False  # Should not buy due to high price
        factors = result["reasoning_factors"]
        assert factors["price_acceptable"] is False
    
    def test_update_satisfaction(self, adventure_tourist_model, openai_config, ag2_config):
        """Test satisfaction level update."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
            
            initial_satisfaction = tourist.satisfaction_level
            tourist.update_satisfaction(0.2, "Great dinosaur show")
            
            assert tourist.satisfaction_level == initial_satisfaction + 0.2
            
            # Test bounds
            tourist.update_satisfaction(1.0, "Amazing experience")
            assert tourist.satisfaction_level == 1.0  # Should cap at 1.0
            
            tourist.update_satisfaction(-2.0, "Terrible incident")
            assert tourist.satisfaction_level == 0.0  # Should floor at 0.0
    
    def test_get_satisfaction_report(self, adventure_tourist_model, openai_config, ag2_config):
        """Test satisfaction report generation."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
            tourist.satisfaction_level = 0.8
            
            report = tourist.get_satisfaction_report()
            
            assert report["tourist_id"] == "tourist_001"
            assert report["name"] == "Adventure Alice"
            assert report["satisfaction_level"] == 0.8
            assert report["satisfaction_category"] == "very_satisfied"
            assert "risk_tolerance" in report
            assert "interests" in report
            assert "personality_traits" in report
            assert "current_location" in report
    
    def test_categorize_satisfaction(self, adventure_tourist_model, openai_config, ag2_config):
        """Test satisfaction categorization."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            tourist = TouristAgent(adventure_tourist_model, openai_config, ag2_config)
            
            # Test different satisfaction levels
            tourist.satisfaction_level = 0.9
            assert tourist._categorize_satisfaction() == "very_satisfied"
            
            tourist.satisfaction_level = 0.7
            assert tourist._categorize_satisfaction() == "satisfied"
            
            tourist.satisfaction_level = 0.5
            assert tourist._categorize_satisfaction() == "neutral"
            
            tourist.satisfaction_level = 0.3
            assert tourist._categorize_satisfaction() == "dissatisfied"
            
            tourist.satisfaction_level = 0.1
            assert tourist._categorize_satisfaction() == "very_dissatisfied"


class TestVisitorAgentFactory:
    """Test cases for VisitorAgentFactory."""
    
    def test_factory_initialization(self, openai_config, ag2_config):
        """Test visitor agent factory initialization."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        assert factory.openai_config == openai_config
        assert factory.ag2_config == ag2_config
        assert factory.logger is not None
        assert factory._visitor_counter == 0
    
    def test_create_adventure_seeker(self, openai_config, ag2_config):
        """Test adventure seeker creation."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        agent = factory.create_adventure_seeker("Test Adventurer")
        
        assert agent.role == AgentRole.TOURIST
        assert agent.name == "Test Adventurer"
        assert agent.personality_traits[PersonalityTrait.BRAVE.value] >= 0.7
        assert agent.personality_traits[PersonalityTrait.CAUTIOUS.value] <= 0.3
        assert "risk_taking" in agent.capabilities
    
    def test_create_family_visitor(self, openai_config, ag2_config):
        """Test family visitor creation."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        agent = factory.create_family_visitor("Test Family")
        
        assert agent.role == AgentRole.TOURIST
        assert agent.name == "Test Family"
        assert agent.personality_traits[PersonalityTrait.CAUTIOUS.value] >= 0.6
        assert agent.personality_traits[PersonalityTrait.EMPATHY.value] >= 0.7
        assert "child_supervision" in agent.capabilities
        assert "safety_awareness" in agent.capabilities
    
    def test_create_educational_visitor(self, openai_config, ag2_config):
        """Test educational visitor creation."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        agent = factory.create_educational_visitor("Test Scholar")
        
        assert agent.role == AgentRole.TOURIST
        assert agent.name == "Test Scholar"
        assert agent.personality_traits[PersonalityTrait.ANALYTICAL.value] >= 0.7
        assert "research" in agent.capabilities
        assert "documentation" in agent.capabilities
    
    def test_create_casual_visitor(self, openai_config, ag2_config):
        """Test casual visitor creation."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        agent = factory.create_casual_visitor("Test Visitor")
        
        assert agent.role == AgentRole.TOURIST
        assert agent.name == "Test Visitor"
        # Casual visitors should have balanced traits
        assert 0.3 <= agent.personality_traits[PersonalityTrait.FRIENDLY.value] <= 0.8
        assert 0.3 <= agent.personality_traits[PersonalityTrait.CAUTIOUS.value] <= 0.6
    
    def test_create_photographer_visitor(self, openai_config, ag2_config):
        """Test photographer visitor creation."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        agent = factory.create_photographer_visitor("Test Photographer")
        
        assert agent.role == AgentRole.TOURIST
        assert agent.name == "Test Photographer"
        assert agent.personality_traits[PersonalityTrait.CREATIVE.value] >= 0.7
        assert "photography" in agent.capabilities
        assert "artistic_appreciation" in agent.capabilities
    
    def test_create_tourist_agent(self, adventure_tourist_model, openai_config, ag2_config):
        """Test tourist agent instance creation."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        with patch('agents.visitor_agents.TouristAgent') as mock_tourist_class:
            mock_tourist_instance = Mock()
            mock_tourist_class.return_value = mock_tourist_instance
            
            result = factory.create_tourist_agent(adventure_tourist_model)
            
            assert result == mock_tourist_instance
            mock_tourist_class.assert_called_once_with(adventure_tourist_model, openai_config, ag2_config)
    
    def test_create_tourist_agent_wrong_role(self, openai_config, ag2_config):
        """Test tourist agent creation with wrong role."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        wrong_role_agent = Agent(
            id="staff_001",
            name="Staff Member",
            role=AgentRole.PARK_RANGER,
            location=Location(0.0, 0.0, "entrance")
        )
        
        with pytest.raises(ValueError, match="Agent model must have TOURIST role"):
            factory.create_tourist_agent(wrong_role_agent)
    
    def test_create_random_visitor(self, openai_config, ag2_config):
        """Test random visitor creation."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        # Create multiple random visitors to test variety
        visitors = [factory.create_random_visitor() for _ in range(10)]
        
        # All should be tourists
        assert all(visitor.role == AgentRole.TOURIST for visitor in visitors)
        
        # Should have variety in names (at least some different)
        names = [visitor.name for visitor in visitors]
        assert len(set(names)) > 1  # Should have some variety
    
    def test_visitor_counter_increment(self, openai_config, ag2_config):
        """Test that visitor counter increments properly."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        initial_counter = factory._visitor_counter
        
        factory.create_adventure_seeker()
        assert factory._visitor_counter == initial_counter + 1
        
        factory.create_family_visitor()
        assert factory._visitor_counter == initial_counter + 2
        
        factory.create_educational_visitor()
        assert factory._visitor_counter == initial_counter + 3
    
    def test_default_location_assignment(self, openai_config, ag2_config):
        """Test that default locations are assigned when not provided."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        adventure_agent = factory.create_adventure_seeker()
        assert adventure_agent.location.zone == "entrance"
        
        family_agent = factory.create_family_visitor()
        assert family_agent.location.zone == "family_area"
        
        educational_agent = factory.create_educational_visitor()
        assert educational_agent.location.zone == "education_center"
    
    def test_custom_location_assignment(self, openai_config, ag2_config):
        """Test that custom locations are used when provided."""
        factory = VisitorAgentFactory(openai_config, ag2_config)
        
        custom_location = Location(100.0, 200.0, "custom_zone", "Custom test zone")
        
        agent = factory.create_adventure_seeker(location=custom_location)
        assert agent.location == custom_location
        assert agent.location.zone == "custom_zone"


if __name__ == "__main__":
    pytest.main([__file__])