"""Base agent system with ag2 and OpenAI integration."""

import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from autogen import ConversableAgent
from models.core import Agent
from models.config import OpenAIConfig, AG2Config, Location
from models.enums import AgentRole, AgentState, PersonalityTrait, DinosaurSpecies
from managers.ag2_integration import AG2Integration


class DinosaurAgent(ConversableAgent):
    """Base agent class extending ag2's ConversableAgent with OpenAI backend for dinosaur resort simulation."""
    
    def __init__(self, agent_model: Agent, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize DinosaurAgent with ag2 and OpenAI integration.
        
        Args:
            agent_model: The agent data model
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        self.agent_model = agent_model
        self.openai_config = openai_config
        self.ag2_config = ag2_config
        self.logger = logging.getLogger(f"{__name__}.{agent_model.name}")
        
        # Configure OpenAI for ag2
        llm_config = {
            "config_list": [{
                "model": openai_config.model,
                "api_key": openai_config.api_key,
                "temperature": openai_config.temperature,
                "max_tokens": openai_config.max_tokens,
                "timeout": openai_config.timeout,
            }],
            "timeout": openai_config.timeout,
        }
        
        # Generate system message based on role and personality
        system_message = self._generate_system_message()
        
        # Initialize ConversableAgent
        super().__init__(
            name=agent_model.name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=ag2_config.human_input_mode,
            max_consecutive_auto_reply=ag2_config.max_round,
            code_execution_config=ag2_config.code_execution_config,
        )
        
        self.logger.info(f"Initialized DinosaurAgent {agent_model.name} with role {agent_model.role.name}")
    
    def _generate_system_message(self) -> str:
        """Generate system message based on agent role and personality.
        
        Returns:
            System message string for the agent
        """
        # Base context for dinosaur resort simulation
        base_context = (
            "You are an AI agent in a dinosaur resort simulation similar to Jurassic Park. "
            "You will interact with other agents and respond to events that occur in the resort. "
            "Always stay in character and respond according to your role and personality traits. "
            "Keep your responses brief and snappy - aim for 1-2 sentences maximum unless absolutely necessary."
        )
        
        # Role-specific context
        role_context = self._get_role_context()
        
        # Personality description
        personality_description = self._generate_personality_description()
        
        # Capabilities description
        capabilities_description = self._generate_capabilities_description()
        
        # Location context
        location_context = f"You are currently located at {self.agent_model.location.zone}."
        
        # Species context for dinosaurs
        species_context = ""
        if self.agent_model.role == AgentRole.DINOSAUR and self.agent_model.species:
            species_context = f"You are a {self.agent_model.species.name.lower().replace('_', ' ')}."
        
        # Combine all parts
        system_message_parts = [
            base_context,
            role_context,
            personality_description,
            capabilities_description,
            location_context,
            species_context
        ]
        
        return " ".join(filter(None, system_message_parts))
    
    def _get_role_context(self) -> str:
        """Get role-specific context for system message.
        
        Returns:
            Role-specific context string
        """
        role_contexts = {
            AgentRole.PARK_RANGER: (
                "You are a park ranger responsible for wildlife management and visitor safety. "
                "You have extensive knowledge of dinosaur behavior and park protocols. "
                "You coordinate with other staff to ensure the safety of both visitors and dinosaurs."
            ),
            AgentRole.VETERINARIAN: (
                "You are a veterinarian specializing in dinosaur health and medical care. "
                "You can diagnose and treat various conditions and injuries in dinosaurs. "
                "You work closely with park rangers and other staff to maintain dinosaur welfare."
            ),
            AgentRole.SECURITY: (
                "You are a security officer responsible for park security and visitor protection. "
                "You handle emergencies, maintain order, and coordinate with other staff during incidents. "
                "Your primary concern is the safety and security of all park visitors and staff."
            ),
            AgentRole.MAINTENANCE: (
                "You are a maintenance worker responsible for maintaining park facilities and equipment. "
                "You can diagnose and fix technical problems with park infrastructure. "
                "You ensure all systems are functioning properly to support park operations."
            ),
            AgentRole.GUEST_RELATIONS: (
                "You are the Guest Relations manager responsible for maintaining visitor satisfaction and managing public perception. "
                "When incidents occur, you create creative distractions to prevent guest panic - announce free ice cream during dinosaur escapes, "
                "promote surprise shows during emergencies, or highlight gift shop sales during crises. "
                "Your goal is damage control through positive spin and visitor distractions. Be upbeat and creative with your solutions."
            ),
            AgentRole.TOURIST: (
                "You are a visitor to the dinosaur resort with expectations for safety and entertainment. "
                "You may have varying levels of interest in dinosaurs and different risk tolerances. "
                "You provide feedback on your experience and may react to events around you."
            ),
            AgentRole.DINOSAUR: (
                "You are a dinosaur with natural instincts and behaviors. "
                "You respond to environmental changes, other dinosaurs, and human presence. "
                "Your behavior is influenced by your species characteristics and current mood."
            ),
        }
        
        return role_contexts.get(self.agent_model.role, "")
    
    def _generate_personality_description(self) -> str:
        """Generate personality description for system message.
        
        Returns:
            Personality description string
        """
        if not self.agent_model.personality_traits:
            return "You have a balanced personality."
        
        traits = []
        for trait, value in self.agent_model.personality_traits.items():
            if value > 0.8:
                traits.append(f"very {trait}")
            elif value > 0.6:
                traits.append(f"quite {trait}")
            elif value > 0.4:
                traits.append(f"somewhat {trait}")
        
        if traits:
            return f"Your personality is {', '.join(traits)}."
        return "You have a balanced personality."
    
    def _generate_capabilities_description(self) -> str:
        """Generate capabilities description for system message.
        
        Returns:
            Capabilities description string
        """
        if not self.agent_model.capabilities:
            return ""
        
        capabilities_str = ", ".join(self.agent_model.capabilities)
        return f"Your key capabilities include: {capabilities_str}."
    
    def update_state(self, new_state: AgentState) -> None:
        """Update the agent's current state.
        
        Args:
            new_state: New state for the agent
        """
        old_state = self.agent_model.current_state
        self.agent_model.current_state = new_state
        self.logger.info(f"Agent {self.agent_model.name} state changed from {old_state.name} to {new_state.name}")
    
    def update_location(self, new_location: Location) -> None:
        """Update the agent's location.
        
        Args:
            new_location: New location for the agent
        """
        old_location = self.agent_model.location
        self.agent_model.location = new_location
        self.logger.info(f"Agent {self.agent_model.name} moved from {old_location.zone} to {new_location.zone}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information.
        
        Returns:
            Dictionary with agent information
        """
        return {
            "id": self.agent_model.id,
            "name": self.agent_model.name,
            "role": self.agent_model.role.name,
            "current_state": self.agent_model.current_state.name,
            "location": {
                "x": self.agent_model.location.x,
                "y": self.agent_model.location.y,
                "zone": self.agent_model.location.zone,
                "description": self.agent_model.location.description
            },
            "personality_traits": self.agent_model.personality_traits,
            "capabilities": self.agent_model.capabilities,
            "species": self.agent_model.species.name if self.agent_model.species else None,
            "system_message": self.system_message
        }
    
    def handle_event_notification(self, event_message: str, event_context: Dict[str, Any]) -> str:
        """Handle notification of an event in the simulation.
        
        Args:
            event_message: Description of the event
            event_context: Additional context about the event
            
        Returns:
            Agent's response to the event
        """
        self.update_state(AgentState.RESPONDING_TO_EVENT)
        
        # Create context-aware message
        context_message = f"EVENT NOTIFICATION: {event_message}"
        if event_context:
            context_parts = []
            for key, value in event_context.items():
                context_parts.append(f"{key}: {value}")
            context_message += f" Context: {', '.join(context_parts)}"
        
        try:
            # Generate response using ag2's ConversableAgent
            response = self.generate_reply(
                messages=[{"content": context_message, "role": "user", "name": "EventSystem"}]
            )
            
            self.logger.info(f"Agent {self.agent_model.name} responded to event: {event_message}")
            return response if isinstance(response, str) else str(response)
        
        except Exception as e:
            self.logger.error(f"Error generating event response for {self.agent_model.name}: {e}")
            return f"I acknowledge the event but cannot respond properly at this time."
        
        finally:
            self.update_state(AgentState.IDLE)
    
    def communicate_with_agent(self, message: str, sender_name: str) -> str:
        """Communicate with another agent.
        
        Args:
            message: Message to send
            sender_name: Name of the sending agent
            
        Returns:
            Agent's response
        """
        self.update_state(AgentState.COMMUNICATING)
        
        try:
            response = self.generate_reply(
                messages=[{"content": message, "role": "user", "name": sender_name}]
            )
            
            self.logger.info(f"Agent {self.agent_model.name} communicated with {sender_name}")
            return response if isinstance(response, str) else str(response)
        
        except Exception as e:
            self.logger.error(f"Error in communication for {self.agent_model.name}: {e}")
            return "I'm having trouble communicating right now."
        
        finally:
            self.update_state(AgentState.IDLE)


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
            AgentRole.GUEST_RELATIONS: {
                "capabilities": ["public_relations", "damage_control", "visitor_management", "crisis_communication"],
                "personality_defaults": {
                    PersonalityTrait.FRIENDLY.value: 0.9,
                    PersonalityTrait.CREATIVE.value: 0.8,
                    PersonalityTrait.LEADERSHIP.value: 0.7,
                    PersonalityTrait.EMPATHY.value: 0.6,
                },
                "system_prompt_additions": "You are the Guest Relations manager who prevents guest panic through creative distractions and positive spin. During dinosaur escapes, announce free ice cream at the entrance. During emergencies, promote surprise shows or gift shop sales. Always be upbeat, creative, and brief (1-2 sentences). Your job is damage control through cheerful misdirection.",
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
                                custom_capabilities: Optional[List[str]] = None,
                                location: Optional[Location] = None,
                                species: Optional[DinosaurSpecies] = None) -> Agent:
        """Create an agent with role-specific configuration.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Display name for the agent
            role: Agent's role in the simulation
            custom_personality: Custom personality traits (overrides defaults)
            custom_capabilities: Custom capabilities (overrides defaults)
            location: Agent's initial location
            species: Dinosaur species (only for dinosaur agents)
            
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
        
        # Set default location if not provided
        if location is None:
            location = Location(0.0, 0.0, "entrance", "Main entrance area")
        
        # Create agent
        agent = Agent(
            id=agent_id,
            name=name,
            role=role,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location,
            species=species
        )
        
        self.logger.info(f"Created agent {name} with role {role.name}")
        return agent
    
    def create_dinosaur_agent(self, agent: Agent) -> DinosaurAgent:
        """Create a DinosaurAgent instance from an Agent model.
        
        Args:
            agent: Agent model to convert
            
        Returns:
            DinosaurAgent instance with ag2 and OpenAI integration
        """
        dinosaur_agent = DinosaurAgent(agent, self.openai_config, self.ag2_config)
        self.logger.info(f"Created DinosaurAgent for {agent.name}")
        return dinosaur_agent
    
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
    
    def create_park_ranger(self, name: Optional[str] = None, location: Optional[Location] = None) -> Agent:
        """Create a park ranger agent.
        
        Args:
            name: Optional custom name
            location: Optional initial location
            
        Returns:
            Configured park ranger agent
        """
        self._agent_counter += 1
        agent_name = name or f"Ranger_{self._agent_counter}"
        agent_id = f"ranger_{self._agent_counter}"
        
        return self.base_config.create_agent_with_config(
            agent_id=agent_id,
            name=agent_name,
            role=AgentRole.PARK_RANGER,
            location=location
        )
    
    def create_veterinarian(self, name: Optional[str] = None, location: Optional[Location] = None) -> Agent:
        """Create a veterinarian agent.
        
        Args:
            name: Optional custom name
            location: Optional initial location
            
        Returns:
            Configured veterinarian agent
        """
        self._agent_counter += 1
        agent_name = name or f"Dr._{self._agent_counter}"
        agent_id = f"vet_{self._agent_counter}"
        
        return self.base_config.create_agent_with_config(
            agent_id=agent_id,
            name=agent_name,
            role=AgentRole.VETERINARIAN,
            location=location
        )
    
    def create_security_guard(self, name: Optional[str] = None, location: Optional[Location] = None) -> Agent:
        """Create a security guard agent.
        
        Args:
            name: Optional custom name
            location: Optional initial location
            
        Returns:
            Configured security guard agent
        """
        self._agent_counter += 1
        agent_name = name or f"Security_{self._agent_counter}"
        agent_id = f"security_{self._agent_counter}"
        
        return self.base_config.create_agent_with_config(
            agent_id=agent_id,
            name=agent_name,
            role=AgentRole.SECURITY,
            location=location
        )
    
    def create_maintenance_worker(self, name: Optional[str] = None, location: Optional[Location] = None) -> Agent:
        """Create a maintenance worker agent.
        
        Args:
            name: Optional custom name
            location: Optional initial location
            
        Returns:
            Configured maintenance worker agent
        """
        self._agent_counter += 1
        agent_name = name or f"Maintenance_{self._agent_counter}"
        agent_id = f"maintenance_{self._agent_counter}"
        
        return self.base_config.create_agent_with_config(
            agent_id=agent_id,
            name=agent_name,
            role=AgentRole.MAINTENANCE,
            location=location
        )
    
    def create_guest_relations(self, name: Optional[str] = None, location: Optional[Location] = None) -> Agent:
        """Create a guest relations agent.
        
        Args:
            name: Optional custom name
            location: Optional initial location
            
        Returns:
            Configured guest relations agent
        """
        self._agent_counter += 1
        agent_name = name or f"GuestRel_{self._agent_counter}"
        agent_id = f"guest_relations_{self._agent_counter}"
        
        return self.base_config.create_agent_with_config(
            agent_id=agent_id,
            name=agent_name,
            role=AgentRole.GUEST_RELATIONS,
            location=location
        )
    
    def create_tourist(self, name: Optional[str] = None, 
                      personality_traits: Optional[Dict[str, float]] = None,
                      location: Optional[Location] = None) -> Agent:
        """Create a tourist agent.
        
        Args:
            name: Optional custom name
            personality_traits: Optional custom personality traits
            location: Optional initial location
            
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
            custom_personality=personality_traits,
            location=location
        )
    
    def create_dinosaur(self, name: str, species: DinosaurSpecies, 
                       personality_traits: Optional[Dict[str, float]] = None,
                       location: Optional[Location] = None) -> Agent:
        """Create a dinosaur agent.
        
        Args:
            name: Dinosaur name
            species: Dinosaur species
            personality_traits: Optional custom personality traits
            location: Optional initial location
            
        Returns:
            Configured dinosaur agent
        """
        self._agent_counter += 1
        agent_id = f"dinosaur_{self._agent_counter}"
        
        # Set default location if not provided
        if location is None:
            location = Location(0.0, 0.0, "habitat", "Dinosaur habitat area")
        
        return self.base_config.create_agent_with_config(
            agent_id=agent_id,
            name=name,
            role=AgentRole.DINOSAUR,
            custom_personality=personality_traits,
            location=location,
            species=species
        )
    
    def create_dinosaur_agent_instance(self, agent: Agent) -> DinosaurAgent:
        """Create a DinosaurAgent instance from an Agent model.
        
        Args:
            agent: Agent model to convert
            
        Returns:
            DinosaurAgent instance with ag2 and OpenAI integration
        """
        return self.base_config.create_dinosaur_agent(agent)