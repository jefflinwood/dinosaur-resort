"""Unit tests for staff agent implementations."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from agents.staff_agents import (
    ParkRangerAgent, VeterinarianAgent, SecurityAgent, MaintenanceAgent, StaffAgentFactory
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
def park_ranger_agent_model():
    """Create park ranger agent model for testing."""
    return Agent(
        id="ranger_001",
        name="Ranger Smith",
        role=AgentRole.PARK_RANGER,
        personality_traits={
            PersonalityTrait.CAUTIOUS.value: 0.8,
            PersonalityTrait.LEADERSHIP.value: 0.7,
            PersonalityTrait.EMPATHY.value: 0.6
        },
        capabilities=["wildlife_management", "visitor_safety", "emergency_response"],
        location=Location(10.0, 20.0, "ranger_station", "Main ranger station")
    )


@pytest.fixture
def veterinarian_agent_model():
    """Create veterinarian agent model for testing."""
    return Agent(
        id="vet_001",
        name="Dr. Johnson",
        role=AgentRole.VETERINARIAN,
        personality_traits={
            PersonalityTrait.ANALYTICAL.value: 0.9,
            PersonalityTrait.CAUTIOUS.value: 0.7,
            PersonalityTrait.EMPATHY.value: 0.8
        },
        capabilities=["medical_treatment", "health_assessment", "emergency_care"],
        location=Location(15.0, 25.0, "medical_center", "Veterinary medical center")
    )


@pytest.fixture
def security_agent_model():
    """Create security agent model for testing."""
    return Agent(
        id="security_001",
        name="Officer Brown",
        role=AgentRole.SECURITY,
        personality_traits={
            PersonalityTrait.BRAVE.value: 0.8,
            PersonalityTrait.LEADERSHIP.value: 0.7,
            PersonalityTrait.CAUTIOUS.value: 0.6
        },
        capabilities=["threat_assessment", "crowd_control", "emergency_response"],
        location=Location(5.0, 15.0, "security_office", "Main security office")
    )


@pytest.fixture
def maintenance_agent_model():
    """Create maintenance agent model for testing."""
    return Agent(
        id="maintenance_001",
        name="Tech Wilson",
        role=AgentRole.MAINTENANCE,
        personality_traits={
            PersonalityTrait.TECHNICAL.value: 0.9,
            PersonalityTrait.ANALYTICAL.value: 0.7,
            PersonalityTrait.CREATIVE.value: 0.6
        },
        capabilities=["equipment_repair", "facility_maintenance", "technical_support"],
        location=Location(20.0, 10.0, "maintenance_shop", "Main maintenance facility")
    )


class TestParkRangerAgent:
    """Test cases for ParkRangerAgent."""
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    def test_park_ranger_initialization(self, mock_super_init, park_ranger_agent_model, openai_config, ag2_config):
        """Test park ranger agent initialization."""
        mock_super_init.return_value = None
        
        ranger = ParkRangerAgent(park_ranger_agent_model, openai_config, ag2_config)
        
        assert ranger.agent_model == park_ranger_agent_model
        assert ranger.expertise_areas == ["wildlife_behavior", "safety_protocols", "emergency_response"]
        assert ranger.logger.name.endswith("ParkRanger.Ranger Smith")
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_assess_dinosaur_threat(self, mock_generate_reply, mock_super_init, 
                                  park_ranger_agent_model, openai_config, ag2_config):
        """Test dinosaur threat assessment."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "Threat level 7: Immediate evacuation required"
        
        ranger = ParkRangerAgent(park_ranger_agent_model, openai_config, ag2_config)
        ranger.update_state = Mock()
        
        dinosaur_info = {
            "species": "T-Rex",
            "behavior": "aggressive",
            "location": "visitor_area"
        }
        
        result = ranger.assess_dinosaur_threat(dinosaur_info)
        
        assert "assessment" in result
        assert result["ranger_id"] == "ranger_001"
        assert result["expertise"] == "wildlife_management"
        assert "timestamp" in result
        
        # Verify state changes
        ranger.update_state.assert_any_call(AgentState.ACTIVE)
        ranger.update_state.assert_any_call(AgentState.IDLE)
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_coordinate_evacuation(self, mock_generate_reply, mock_super_init,
                                 park_ranger_agent_model, openai_config, ag2_config):
        """Test evacuation coordination."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "Evacuating sector 7. All visitors to safe zone Alpha."
        
        ranger = ParkRangerAgent(park_ranger_agent_model, openai_config, ag2_config)
        ranger.update_state = Mock()
        
        result = ranger.coordinate_evacuation("sector_7", "dinosaur_escape")
        
        assert "Evacuating sector 7" in result
        ranger.update_state.assert_any_call(AgentState.ACTIVE)
        ranger.update_state.assert_any_call(AgentState.IDLE)
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_provide_safety_briefing(self, mock_generate_reply, mock_super_init,
                                   park_ranger_agent_model, openai_config, ag2_config):
        """Test safety briefing provision."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "Welcome to Dinosaur Park. Please follow these safety guidelines..."
        
        ranger = ParkRangerAgent(park_ranger_agent_model, openai_config, ag2_config)
        
        visitor_group = ["visitor_001", "visitor_002", "visitor_003"]
        result = ranger.provide_safety_briefing(visitor_group)
        
        assert "Welcome to Dinosaur Park" in result
        mock_generate_reply.assert_called_once()


class TestVeterinarianAgent:
    """Test cases for VeterinarianAgent."""
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    def test_veterinarian_initialization(self, mock_super_init, veterinarian_agent_model, openai_config, ag2_config):
        """Test veterinarian agent initialization."""
        mock_super_init.return_value = None
        
        vet = VeterinarianAgent(veterinarian_agent_model, openai_config, ag2_config)
        
        assert vet.agent_model == veterinarian_agent_model
        assert vet.medical_specialties == ["dinosaur_physiology", "emergency_medicine", "behavioral_health"]
        assert vet.logger.name.endswith("Veterinarian.Dr. Johnson")
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_diagnose_dinosaur_condition(self, mock_generate_reply, mock_super_init,
                                       veterinarian_agent_model, openai_config, ag2_config):
        """Test dinosaur condition diagnosis."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "Diagnosis: Respiratory infection. Recommend antibiotics and rest."
        
        vet = VeterinarianAgent(veterinarian_agent_model, openai_config, ag2_config)
        vet.update_state = Mock()
        
        symptoms = {
            "breathing": "labored",
            "appetite": "decreased",
            "activity": "lethargic"
        }
        
        result = vet.diagnose_dinosaur_condition(symptoms)
        
        assert "diagnosis" in result
        assert result["veterinarian_id"] == "vet_001"
        assert result["specialty"] == "dinosaur_medicine"
        assert result["urgency_level"] == "low"  # Based on symptoms
        
        vet.update_state.assert_any_call(AgentState.ACTIVE)
        vet.update_state.assert_any_call(AgentState.IDLE)
    
    def test_assess_urgency_critical(self, veterinarian_agent_model, openai_config, ag2_config):
        """Test urgency assessment for critical symptoms."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            vet = VeterinarianAgent(veterinarian_agent_model, openai_config, ag2_config)
            
            critical_symptoms = {"condition": "bleeding", "status": "unconscious"}
            urgency = vet._assess_urgency(critical_symptoms)
            
            assert urgency == "critical"
    
    def test_assess_urgency_high(self, veterinarian_agent_model, openai_config, ag2_config):
        """Test urgency assessment for high priority symptoms."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            vet = VeterinarianAgent(veterinarian_agent_model, openai_config, ag2_config)
            
            high_symptoms = {"behavior": "aggressive", "temperature": "fever"}
            urgency = vet._assess_urgency(high_symptoms)
            
            assert urgency == "high"
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_administer_treatment(self, mock_generate_reply, mock_super_init,
                                veterinarian_agent_model, openai_config, ag2_config):
        """Test treatment administration."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "Treatment administered successfully. Patient responding well."
        
        vet = VeterinarianAgent(veterinarian_agent_model, openai_config, ag2_config)
        vet.update_state = Mock()
        
        result = vet.administer_treatment("Antibiotic injection, 500mg", "dino_001")
        
        assert "Treatment administered successfully" in result
        vet.update_state.assert_any_call(AgentState.ACTIVE)
        vet.update_state.assert_any_call(AgentState.IDLE)


class TestSecurityAgent:
    """Test cases for SecurityAgent."""
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    def test_security_agent_initialization(self, mock_super_init, security_agent_model, openai_config, ag2_config):
        """Test security agent initialization."""
        mock_super_init.return_value = None
        
        security = SecurityAgent(security_agent_model, openai_config, ag2_config)
        
        assert security.agent_model == security_agent_model
        assert security.security_clearance == "high"
        assert security.response_protocols == ["crowd_control", "emergency_response", "threat_neutralization"]
        assert security.logger.name.endswith("Security.Officer Brown")
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_assess_security_threat(self, mock_generate_reply, mock_super_init,
                                  security_agent_model, openai_config, ag2_config):
        """Test security threat assessment."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "Threat level 8: Deploy additional units immediately"
        
        security = SecurityAgent(security_agent_model, openai_config, ag2_config)
        security.update_state = Mock()
        
        threat_info = {
            "type": "dinosaur_escape",
            "severity": 8,
            "location": "main_plaza"
        }
        
        result = security.assess_security_threat(threat_info)
        
        assert "threat_assessment" in result
        assert result["security_officer_id"] == "security_001"
        assert result["response_level"] == "critical"  # Based on severity 8
        
        security.update_state.assert_any_call(AgentState.ACTIVE)
        security.update_state.assert_any_call(AgentState.IDLE)
    
    def test_determine_response_level(self, security_agent_model, openai_config, ag2_config):
        """Test response level determination."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            security = SecurityAgent(security_agent_model, openai_config, ag2_config)
            
            # Test different severity levels
            assert security._determine_response_level({"severity": 9}) == "critical"
            assert security._determine_response_level({"severity": 7}) == "high"
            assert security._determine_response_level({"severity": 5}) == "medium"
            assert security._determine_response_level({"severity": 2}) == "low"
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_coordinate_emergency_response(self, mock_generate_reply, mock_super_init,
                                         security_agent_model, openai_config, ag2_config):
        """Test emergency response coordination."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "All units to positions. Establishing perimeter around incident."
        
        security = SecurityAgent(security_agent_model, openai_config, ag2_config)
        security.update_state = Mock()
        
        result = security.coordinate_emergency_response("dinosaur_escape", "sector_5")
        
        assert "All units to positions" in result
        security.update_state.assert_any_call(AgentState.ACTIVE)
        security.update_state.assert_any_call(AgentState.IDLE)


class TestMaintenanceAgent:
    """Test cases for MaintenanceAgent."""
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    def test_maintenance_agent_initialization(self, mock_super_init, maintenance_agent_model, openai_config, ag2_config):
        """Test maintenance agent initialization."""
        mock_super_init.return_value = None
        
        maintenance = MaintenanceAgent(maintenance_agent_model, openai_config, ag2_config)
        
        assert maintenance.agent_model == maintenance_agent_model
        assert maintenance.technical_skills == ["electrical", "mechanical", "structural", "systems"]
        assert maintenance.tools_available == ["diagnostic_equipment", "repair_tools", "safety_equipment"]
        assert maintenance.logger.name.endswith("Maintenance.Tech Wilson")
    
    @patch('agents.base_agent.ConversableAgent.__init__')
    @patch('agents.base_agent.DinosaurAgent.generate_reply')
    def test_diagnose_equipment_failure(self, mock_generate_reply, mock_super_init,
                                      maintenance_agent_model, openai_config, ag2_config):
        """Test equipment failure diagnosis."""
        mock_super_init.return_value = None
        mock_generate_reply.return_value = "Power supply failure. Replace main transformer unit."
        
        maintenance = MaintenanceAgent(maintenance_agent_model, openai_config, ag2_config)
        maintenance.update_state = Mock()
        
        equipment_info = {
            "type": "power_system",
            "symptoms": "no_power",
            "location": "main_building"
        }
        
        result = maintenance.diagnose_equipment_failure(equipment_info)
        
        assert "diagnosis" in result
        assert result["technician_id"] == "maintenance_001"
        assert result["repair_priority"] == "critical"  # Power system is critical
        assert result["estimated_duration"] == "4-6 hours"  # Default for medium complexity
        
        maintenance.update_state.assert_any_call(AgentState.ACTIVE)
        maintenance.update_state.assert_any_call(AgentState.IDLE)
    
    def test_assess_repair_priority(self, maintenance_agent_model, openai_config, ag2_config):
        """Test repair priority assessment."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            maintenance = MaintenanceAgent(maintenance_agent_model, openai_config, ag2_config)
            
            # Test different equipment types
            assert maintenance._assess_repair_priority({"type": "power_system"}) == "critical"
            assert maintenance._assess_repair_priority({"type": "visitor_facilities"}) == "high"
            assert maintenance._assess_repair_priority({"type": "operational_equipment"}) == "medium"
            assert maintenance._assess_repair_priority({"type": "office_equipment"}) == "low"
    
    def test_estimate_repair_time(self, maintenance_agent_model, openai_config, ag2_config):
        """Test repair time estimation."""
        with patch('agents.base_agent.ConversableAgent.__init__'):
            maintenance = MaintenanceAgent(maintenance_agent_model, openai_config, ag2_config)
            
            # Test different complexity levels
            assert maintenance._estimate_repair_time({"complexity": "simple"}) == "1-2 hours"
            assert maintenance._estimate_repair_time({"complexity": "medium"}) == "4-6 hours"
            assert maintenance._estimate_repair_time({"complexity": "complex"}) == "1-2 days"
            assert maintenance._estimate_repair_time({"complexity": "critical"}) == "Immediate attention required"


class TestStaffAgentFactory:
    """Test cases for StaffAgentFactory."""
    
    def test_factory_initialization(self, openai_config, ag2_config):
        """Test staff agent factory initialization."""
        factory = StaffAgentFactory(openai_config, ag2_config)
        
        assert factory.openai_config == openai_config
        assert factory.ag2_config == ag2_config
        assert factory.logger is not None
    
    @patch('agents.staff_agents.ParkRangerAgent')
    def test_create_park_ranger(self, mock_ranger_class, park_ranger_agent_model, openai_config, ag2_config):
        """Test park ranger creation."""
        mock_ranger_instance = Mock()
        mock_ranger_class.return_value = mock_ranger_instance
        
        factory = StaffAgentFactory(openai_config, ag2_config)
        result = factory.create_park_ranger(park_ranger_agent_model)
        
        assert result == mock_ranger_instance
        mock_ranger_class.assert_called_once_with(park_ranger_agent_model, openai_config, ag2_config)
    
    def test_create_park_ranger_wrong_role(self, veterinarian_agent_model, openai_config, ag2_config):
        """Test park ranger creation with wrong role."""
        factory = StaffAgentFactory(openai_config, ag2_config)
        
        with pytest.raises(ValueError, match="Agent model must have PARK_RANGER role"):
            factory.create_park_ranger(veterinarian_agent_model)
    
    @patch('agents.staff_agents.VeterinarianAgent')
    def test_create_veterinarian(self, mock_vet_class, veterinarian_agent_model, openai_config, ag2_config):
        """Test veterinarian creation."""
        mock_vet_instance = Mock()
        mock_vet_class.return_value = mock_vet_instance
        
        factory = StaffAgentFactory(openai_config, ag2_config)
        result = factory.create_veterinarian(veterinarian_agent_model)
        
        assert result == mock_vet_instance
        mock_vet_class.assert_called_once_with(veterinarian_agent_model, openai_config, ag2_config)
    
    @patch('agents.staff_agents.SecurityAgent')
    def test_create_security_agent(self, mock_security_class, security_agent_model, openai_config, ag2_config):
        """Test security agent creation."""
        mock_security_instance = Mock()
        mock_security_class.return_value = mock_security_instance
        
        factory = StaffAgentFactory(openai_config, ag2_config)
        result = factory.create_security_agent(security_agent_model)
        
        assert result == mock_security_instance
        mock_security_class.assert_called_once_with(security_agent_model, openai_config, ag2_config)
    
    @patch('agents.staff_agents.MaintenanceAgent')
    def test_create_maintenance_agent(self, mock_maintenance_class, maintenance_agent_model, openai_config, ag2_config):
        """Test maintenance agent creation."""
        mock_maintenance_instance = Mock()
        mock_maintenance_class.return_value = mock_maintenance_instance
        
        factory = StaffAgentFactory(openai_config, ag2_config)
        result = factory.create_maintenance_agent(maintenance_agent_model)
        
        assert result == mock_maintenance_instance
        mock_maintenance_class.assert_called_once_with(maintenance_agent_model, openai_config, ag2_config)
    
    @patch('agents.staff_agents.ParkRangerAgent')
    def test_create_staff_agent_park_ranger(self, mock_ranger_class, park_ranger_agent_model, openai_config, ag2_config):
        """Test generic staff agent creation for park ranger."""
        mock_ranger_instance = Mock()
        mock_ranger_class.return_value = mock_ranger_instance
        
        factory = StaffAgentFactory(openai_config, ag2_config)
        result = factory.create_staff_agent(park_ranger_agent_model)
        
        assert result == mock_ranger_instance
    
    def test_create_staff_agent_invalid_role(self, openai_config, ag2_config):
        """Test generic staff agent creation with invalid role."""
        tourist_agent = Agent(
            id="tourist_001",
            name="Tourist Joe",
            role=AgentRole.TOURIST,
            location=Location(0.0, 0.0, "entrance")
        )
        
        factory = StaffAgentFactory(openai_config, ag2_config)
        
        with pytest.raises(ValueError, match="No staff agent implementation for role"):
            factory.create_staff_agent(tourist_agent)


if __name__ == "__main__":
    pytest.main([__file__])