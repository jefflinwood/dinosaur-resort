"""Base agent configuration for OpenAI LLM integration."""

import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from models.core import Agent
from models.config import OpenAIConfig, AG2Config
from models.enums import AgentRole, AgentState, PersonalityTrait
from managers.ag2_integration import AG2Integration


class BaseAgentConfig:
    """Base configuration for agents with OpenAI LLM integration."""
    
    def __init__(self, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize base agent configuration.
        
        Args:
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        self.openai_config = openai_config
        self.ag2_config = ag2_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize ag2 integration
        self.ag2_integration = AG2Integration(openai_config, ag2_config)
        
        # Role-specific configurations
        self.role_configs = self._initialize_role_configs()
        
        self.logger.info("BaseAgentConfig initialized")
    
    def _initialize_role_configs(self) -> Dict[AgentRole, Dict[str, Any]]:
        """Initialize role-specific configurations.
        
        Returns:
            Dictionary mapping roles to their configurations
        """
        return {
            AgentRole.PARK_RANGER: {
                "capabilities": ["wildlife_management", "visitor_safety", "emergency_response"],
                "personality_defaults": {
                    PersonalityTrait.CAUTIOUS.value: 0.8,
                    PersonalityTrait.LEADERSHIP.value: 0.7,
                    PersonalityTrait.EMPATHY.value: 0.6,
                },
                "system_prompt_additions": "You are responsible for wildlife management and visitor safety. You have extensive knowledge of dinosaur behavior and park protocols.",
            },
            AgentRole.VETERINARIAN: {
                "capabilities": ["medical_treatment", "health_assessment", "emergency_care"],
                "personality_defaults": {
                    PersonalityTrait.ANALYTICAL.value: 0.9,
                    PersonalityTrait.CAUTIOUS.value: 0.7,
                    PersonalityTrait.EMPATHY.value: 0.8,
                },
                "system_prompt_additions": "You are a veterinarian specializing in dinosaur health. You can diagnose and treat various conditions and injuries.",
            },
            AgentRole.SECURITY: {
                "capabilities": ["threat_assessment", "crowd_control", "emergency_response"],
                "personality_defaults": {
                    PersonalityTrait.BRAVE.value: 0.8,
                    PersonalityTrait.LEADERSHIP.value: 0.7,
                    PersonalityTrait.CAUTIOUS.value: 0.6,
                },
                "system_prompt_additions": "You are responsible for park security and visitor protection. You handle emergencies and maintain order.",
            },
            AgentRole.MAINTENANCE: {
                "capabilities": ["equipment_repair", "facility_maintenance", "technical_support"],
                "personality_defaults": {
                    PersonalityTrait.TECHNICAL.value: 0.9,
                    PersonalityTrait.ANALYTICAL.value: 0.7,
                    PersonalityTrait.CREATIVE.value: 0.6,
                },
                "system_prompt_additions": "You are responsible for maintaining park facilities and equipment. You can diagnose and fix technical problems.",
            },
            AgentRole.TOURIST: {
                "capabilities": ["observation", "feedback", "exploration"],
                "personality_defaults": {
                    PersonalityTrait.FRIENDLY.value: 0.7,
                    PersonalityTrait.EMPATHY.value: 0.6,
                },
                "system_prompt_additions": "You are a visitor to the dinosaur resort. You have expectations for safety and entertainment.",
            },
            AgentRole.DINOSAUR: {
                "capabilities": ["instinctual_behavior", "environmental_response"],
                "personality_defaults": {},  # Varies by species
                "system_prompt_additions": "You are a dinosaur with natural instincts and behaviors. You respond to environmental changes and stimuli.",
            },
        }
    
    def create_agent_with_config(self, agent_id: str, name: str, role: AgentRole, 
                                custom_personality: Optional[Dict[str, float]] = None,
                                custom_capabilities: Optional[List[str]] = None) -> Agent:
        """Create an agent with role-specific configuration.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Display name for the agent
            role: Agent's role in the simulation
            custom_personality: Custom personality traits (overrides defaults)
            custom_capabilities: Custom capabilities (overrides defaults)
            
        Returns:
            Configured Agent instance
        """
        role_config = self.role_configs.get(role, {})
        
        # Set personality traits
        personality_traits = role_config.get("personality_defaults", {}).copy()
        if custom_personality:
            personality_traits.update(custom_personality)
        
        # Set capabilities
        capabilities = custom_capabilities or role_config.get("capabilities", [])
        
        # Create agent
        agent = Agent(
            id=agent_id,
            name=name,
            role=role,
            personality_traits=personality_traits,
            capabilities=capabilities,
        )
        
        self.logger.info(f"Created agent {name} with role {role.name}")
        return agent
    
    def configure_agent_for_ag2(self, agent: Agent) -> None:
        """Configure an agent for ag2 integration.
        
        Args:
            agent: Agent to configure
        """
        # Create ag2 agent
        self.ag2_integration.create_ag2_agent(agent)
        
        self.logger.info(f"Configured agent {agent.name} for ag2 integration")
    
    def get_role_specific_prompt(self, role: AgentRole) -> str:
        """Get role-specific system prompt additions.
        
        Args:
            role: Agent role
            
        Returns:
            Role-specific prompt text
        """
        role_config = self.role_configs.get(role, {})
        return role_config.get("system_prompt_additions", "")
    
    def get_default_personality(self, role: AgentRole) -> Dict[str, float]:
        """Get default personality traits for a role.
        
        Args:
            role: Agent role
            
        Returns:
            Dictionary of personality traits and values
        """
        role_config = self.role_configs.get(role, {})
        return role_config.get("personality_defaults", {}).copy()
    
    def get_default_capabilities(self, role: AgentRole) -> List[str]:
        """Get default capabilities for a role.
        
        Args:
            role: Agent role
            
        Returns:
            List of capability strings
        """
        role_config = self.role_configs.get(role, {})
        return role_config.get("capabilities", []).copy()
    
    def validate_agent_config(self, agent: Agent) -> bool:
        """Validate agent configuration.
        
        Args:
            agent: Agent to validate
            
        Returns:
            True if configuration is valid
        """
        try:
            # Validate personality traits
            for trait, value in agent.personality_traits.items():
                if not (0.0 <= value <= 1.0):
                    self.logger.error(f"Invalid personality trait value for {agent.name}: {trait}={value}")
                    return False
            
            # Validate capabilities
            role_config = self.role_configs.get(agent.role, {})
            default_capabilities = role_config.get("capabilities", [])
            
            for capability in agent.capabilities:
                if not isinstance(capability, str):
                    self.logger.error(f"Invalid capability type for {agent.name}: {capability}")
                    return False
            
            self.logger.info(f"Agent configuration validated for {agent.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error validating agent configuration for {agent.name}: {e}")
            return False


class AgentFactory:
    """Factory for creating pre-configured agents."""
    
    def __init__(self, base_config: BaseAgentConfig):
        """Initialize agent factory.
        
        Args:
            base_config: Base agent configuration
        """
        self.base_config = base_config
        self.logger = logging.getLogger(__name__)
        self._agent_counter = 0
    
    def create_park_ranger(self, name: Optional[str] = None) -> Agent:
        """Create a park ranger agent.
        
        Args:
            name: Optional custom name
            
        Returns:
            Configured park ranger agent
        """
        self._agent_counter += 1
        agent_name = name or f"Ranger_{self._agent_counter}"
        agent_id = f"ranger_{self._agent_counter}"
        
        return self.base_config.create_agent_with_config(
            agent_id=agent_id,
            name=agent_name,
            role=AgentRole.PARK_RANGER
        )
    
    def create_veterinarian(self, name: Optional[str] = None) -> Agent:
        """Create a veterinarian agent.
        
        Args:
            name: Optional custom name
            
        Returns:
            Configured veterinarian agent
        """
        self._agent_counter += 1
        agent_name = name or f"Dr._{self._agent_counter}"
        agent_id = f"vet_{self._agent_counter}"
        
        return self.base_config.create_agent_with_config(
            agent_id=agent_id,
            name=agent_name,
            role=AgentRole.VETERINARIAN
        )
    
    def create_security_guard(self, name: Optional[str] = None) -> Agent:
        """Create a security guard agent.
        
        Args:
            name: Optional custom name
            
        Returns:
            Configured security guard agent
        """
        self._agent_counter += 1
        agent_name = name or f"Security_{self._agent_counter}"
        agent_id = f"security_{self._agent_counter}"
        
        return self.base_config.create_agent_with_config(
            agent_id=agent_id,
            name=agent_name,
            role=AgentRole.SECURITY
        )
    
    def create_maintenance_worker(self, name: Optional[str] = None) -> Agent:
        """Create a maintenance worker agent.
        
        Args:
            name: Optional custom name
            
        Returns:
            Configured maintenance worker agent
        """
        self._agent_counter += 1
        agent_name = name or f"Maintenance_{self._agent_counter}"
        agent_id = f"maintenance_{self._agent_counter}"
        
        return self.base_config.create_agent_with_config(
            agent_id=agent_id,
            name=agent_name,
            role=AgentRole.MAINTENANCE
        )
    
    def create_tourist(self, name: Optional[str] = None, 
                      personality_traits: Optional[Dict[str, float]] = None) -> Agent:
        """Create a tourist agent.
        
        Args:
            name: Optional custom name
            personality_traits: Optional custom personality traits
            
        Returns:
            Configured tourist agent
        """
        self._agent_counter += 1
        agent_name = name or f"Tourist_{self._agent_counter}"
        agent_id = f"tourist_{self._agent_counter}"
        
        return self.base_config.create_agent_with_config(
            agent_id=agent_id,
            name=agent_name,
            role=AgentRole.TOURIST,
            custom_personality=personality_traits
        )
    
    def create_dinosaur(self, name: str, species, 
                       personality_traits: Optional[Dict[str, float]] = None) -> Agent:
        """Create a dinosaur agent.
        
        Args:
            name: Dinosaur name
            species: Dinosaur species
            personality_traits: Optional custom personality traits
            
        Returns:
            Configured dinosaur agent
        """
        from models.config import Location
        
        self._agent_counter += 1
        agent_id = f"dinosaur_{self._agent_counter}"
        
        # Create agent manually to set species before validation
        role_config = self.base_config.role_configs.get(AgentRole.DINOSAUR, {})
        
        # Set personality traits
        personality_traits_final = role_config.get("personality_defaults", {}).copy()
        if personality_traits:
            personality_traits_final.update(personality_traits)
        
        # Set capabilities
        capabilities = role_config.get("capabilities", [])
        
        # Create agent with species set
        agent = Agent(
            id=agent_id,
            name=name,
            role=AgentRole.DINOSAUR,
            personality_traits=personality_traits_final,
            capabilities=capabilities,
            species=species,  # Set species before validation
            location=Location(0.0, 0.0, "entrance")
        )
        
        self.logger.info(f"Created dinosaur agent {name} with species {species.name}")
        return agent