"""Dinosaur agent implementations for the AI Agent Dinosaur Simulator."""

import logging
import random
from typing import Dict, List, Optional, Any
from agents.base_agent import DinosaurAgent
from models.core import Agent
from models.config import OpenAIConfig, AG2Config, Location
from models.enums import AgentRole, AgentState, PersonalityTrait, DinosaurSpecies


class DinosaurSpeciesAgent(DinosaurAgent):
    """Base class for species-specific dinosaur agents."""
    
    def __init__(self, agent_model: Agent, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize dinosaur species agent.
        
        Args:
            agent_model: The agent data model
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        super().__init__(agent_model, openai_config, ag2_config)
        self.happiness_level = 0.7  # Start with neutral happiness
        self.health_status = "healthy"
        self.hunger_level = 0.3  # 0.0 = not hungry, 1.0 = very hungry
        self.stress_level = 0.2  # 0.0 = calm, 1.0 = very stressed
        self.social_needs = self._calculate_social_needs()
        self.territory_size = self._get_territory_requirements()
        self.logger = logging.getLogger(f"{__name__}.{agent_model.species.name}.{agent_model.name}")
    
    def _calculate_social_needs(self) -> float:
        """Calculate social interaction needs based on species.
        
        Returns:
            Social needs level (0.0 to 1.0)
        """
        # Default implementation - can be overridden by specific species
        return 0.5
    
    def _get_territory_requirements(self) -> float:
        """Get territory size requirements for the species.
        
        Returns:
            Territory size requirement (arbitrary units)
        """
        # Default implementation - can be overridden by specific species
        return 100.0
    
    def react_to_environment(self, environmental_factors: Dict[str, Any]) -> Dict[str, Any]:
        """React to environmental changes.
        
        Args:
            environmental_factors: Dictionary of environmental conditions
            
        Returns:
            Dinosaur's reaction and mood changes
        """
        self.update_state(AgentState.RESPONDING_TO_EVENT)
        
        try:
            # Calculate mood changes based on environmental factors
            happiness_change = self._calculate_environmental_happiness_change(environmental_factors)
            stress_change = self._calculate_environmental_stress_change(environmental_factors)
            
            # Generate contextual reaction
            reaction_prompt = (
                f"As a {self.agent_model.species.name.lower().replace('_', ' ')}, react to these environmental conditions: "
                f"{environmental_factors}. Express your instinctual response, comfort level, and any behavioral changes. "
                f"Your current happiness is {self.happiness_level:.1f} and stress level is {self.stress_level:.1f}."
            )
            
            response = self.generate_reply(
                messages=[{"content": reaction_prompt, "role": "user", "name": "Environment"}]
            )
            
            # Update mood levels
            self.happiness_level = max(0.0, min(1.0, self.happiness_level + happiness_change))
            self.stress_level = max(0.0, min(1.0, self.stress_level + stress_change))
            
            self.logger.info(f"Dinosaur {self.agent_model.name} reacted to environmental changes")
            
            return {
                "reaction": response,
                "happiness_change": happiness_change,
                "stress_change": stress_change,
                "new_happiness": self.happiness_level,
                "new_stress": self.stress_level,
                "behavioral_state": self._determine_behavioral_state(),
                "timestamp": self.agent_model.last_activity
            }
        
        except Exception as e:
            self.logger.error(f"Error reacting to environment: {e}")
            return {
                "reaction": "I sense changes in my environment but cannot respond properly.",
                "error": str(e)
            }
        finally:
            self.update_state(AgentState.IDLE)
    
    def _calculate_environmental_happiness_change(self, factors: Dict[str, Any]) -> float:
        """Calculate happiness change based on environmental factors.
        
        Args:
            factors: Environmental factors
            
        Returns:
            Happiness change (-1.0 to 1.0)
        """
        change = 0.0
        
        # Weather effects
        weather = factors.get("weather", "clear")
        if weather in ["sunny", "clear"]:
            change += 0.1
        elif weather in ["storm", "extreme_cold", "extreme_heat"]:
            change -= 0.2
        
        # Food availability
        food_availability = factors.get("food_availability", 0.5)
        if food_availability > 0.7:
            change += 0.15
        elif food_availability < 0.3:
            change -= 0.2
        
        # Crowding
        visitor_density = factors.get("visitor_density", 0.5)
        if visitor_density > 0.8:
            change -= 0.1  # Most dinosaurs don't like crowds
        
        # Territory space
        available_space = factors.get("available_space", 1.0)
        if available_space < 0.5:
            change -= 0.15
        
        return max(-0.5, min(0.5, change))
    
    def _calculate_environmental_stress_change(self, factors: Dict[str, Any]) -> float:
        """Calculate stress change based on environmental factors.
        
        Args:
            factors: Environmental factors
            
        Returns:
            Stress change (-1.0 to 1.0)
        """
        change = 0.0
        
        # Noise levels
        noise_level = factors.get("noise_level", 0.5)
        if noise_level > 0.7:
            change += 0.2
        
        # Visitor proximity
        visitor_proximity = factors.get("visitor_proximity", 0.5)
        if visitor_proximity > 0.8:
            change += 0.15
        
        # Disruptions
        disruptions = factors.get("disruptions", 0)
        change += disruptions * 0.1
        
        # Medical attention (reduces stress)
        medical_care = factors.get("medical_care", False)
        if medical_care:
            change -= 0.2
        
        return max(-0.3, min(0.4, change))
    
    def _determine_behavioral_state(self) -> str:
        """Determine current behavioral state based on mood.
        
        Returns:
            Behavioral state description
        """
        if self.stress_level > 0.8:
            return "highly_agitated"
        elif self.stress_level > 0.6:
            return "stressed"
        elif self.happiness_level > 0.8:
            return "content"
        elif self.happiness_level > 0.6:
            return "calm"
        elif self.happiness_level < 0.3:
            return "unhappy"
        else:
            return "neutral"
    
    def interact_with_humans(self, human_info: Dict[str, Any]) -> Dict[str, Any]:
        """Interact with humans (visitors or staff).
        
        Args:
            human_info: Information about the humans
            
        Returns:
            Interaction response and mood effects
        """
        self.update_state(AgentState.COMMUNICATING)
        
        try:
            # Calculate reaction based on current mood and human behavior
            interaction_type = human_info.get("type", "visitor")
            human_behavior = human_info.get("behavior", "observing")
            distance = human_info.get("distance", 10.0)
            
            reaction_prompt = (
                f"As a {self.agent_model.species.name.lower().replace('_', ' ')}, react to {interaction_type}s who are "
                f"{human_behavior} from {distance} meters away. Your current mood is {self._determine_behavioral_state()}. "
                f"Respond with natural dinosaur behavior and instincts."
            )
            
            response = self.generate_reply(
                messages=[{"content": reaction_prompt, "role": "user", "name": "HumanInteraction"}]
            )
            
            # Calculate mood changes from interaction
            happiness_change = self._calculate_interaction_happiness_change(human_info)
            stress_change = self._calculate_interaction_stress_change(human_info)
            
            self.happiness_level = max(0.0, min(1.0, self.happiness_level + happiness_change))
            self.stress_level = max(0.0, min(1.0, self.stress_level + stress_change))
            
            self.logger.info(f"Dinosaur {self.agent_model.name} interacted with {interaction_type}")
            
            return {
                "interaction": response,
                "happiness_change": happiness_change,
                "stress_change": stress_change,
                "new_happiness": self.happiness_level,
                "new_stress": self.stress_level,
                "threat_level": self._assess_threat_level(human_info),
                "timestamp": self.agent_model.last_activity
            }
        
        except Exception as e:
            self.logger.error(f"Error in human interaction: {e}")
            return {
                "interaction": "I notice humans nearby but cannot respond properly.",
                "error": str(e)
            }
        finally:
            self.update_state(AgentState.IDLE)
    
    def _calculate_interaction_happiness_change(self, human_info: Dict[str, Any]) -> float:
        """Calculate happiness change from human interaction.
        
        Args:
            human_info: Information about the humans
            
        Returns:
            Happiness change (-1.0 to 1.0)
        """
        change = 0.0
        
        interaction_type = human_info.get("type", "visitor")
        behavior = human_info.get("behavior", "observing")
        distance = human_info.get("distance", 10.0)
        
        # Staff interactions are generally more positive
        if interaction_type == "staff":
            if behavior in ["feeding", "medical_care"]:
                change += 0.2
            elif behavior == "maintenance":
                change += 0.1
        
        # Distance effects
        if distance < 2.0 and behavior != "feeding":
            change -= 0.1  # Too close can be stressful
        elif 5.0 <= distance <= 15.0:
            change += 0.05  # Good viewing distance
        
        # Behavior effects
        if behavior in ["loud_noises", "aggressive"]:
            change -= 0.3
        elif behavior in ["calm_observation", "photography"]:
            change += 0.05
        
        return max(-0.4, min(0.3, change))
    
    def _calculate_interaction_stress_change(self, human_info: Dict[str, Any]) -> float:
        """Calculate stress change from human interaction.
        
        Args:
            human_info: Information about the humans
            
        Returns:
            Stress change (-1.0 to 1.0)
        """
        change = 0.0
        
        behavior = human_info.get("behavior", "observing")
        distance = human_info.get("distance", 10.0)
        group_size = human_info.get("group_size", 1)
        
        # Behavior effects
        if behavior in ["loud_noises", "aggressive", "sudden_movements"]:
            change += 0.3
        elif behavior in ["calm_observation", "feeding"]:
            change -= 0.1
        
        # Distance effects
        if distance < 3.0:
            change += 0.2
        elif distance > 20.0:
            change -= 0.05
        
        # Group size effects
        if group_size > 10:
            change += 0.15
        
        return max(-0.2, min(0.4, change))
    
    def _assess_threat_level(self, human_info: Dict[str, Any]) -> str:
        """Assess threat level from human interaction.
        
        Args:
            human_info: Information about the humans
            
        Returns:
            Threat level (none, low, medium, high)
        """
        behavior = human_info.get("behavior", "observing")
        distance = human_info.get("distance", 10.0)
        
        if behavior in ["aggressive", "threatening"] or distance < 1.0:
            return "high"
        elif behavior in ["loud_noises", "sudden_movements"] or distance < 3.0:
            return "medium"
        elif distance < 8.0:
            return "low"
        else:
            return "none"
    
    def update_health_status(self, new_status: str, reason: str = "") -> None:
        """Update dinosaur health status.
        
        Args:
            new_status: New health status
            reason: Reason for the change
        """
        old_status = self.health_status
        self.health_status = new_status
        
        # Health affects happiness
        if new_status == "sick":
            self.happiness_level = max(0.0, self.happiness_level - 0.3)
            self.stress_level = min(1.0, self.stress_level + 0.2)
        elif new_status == "injured":
            self.happiness_level = max(0.0, self.happiness_level - 0.4)
            self.stress_level = min(1.0, self.stress_level + 0.3)
        elif new_status == "healthy" and old_status != "healthy":
            self.happiness_level = min(1.0, self.happiness_level + 0.2)
            self.stress_level = max(0.0, self.stress_level - 0.2)
        
        self.logger.info(f"Dinosaur {self.agent_model.name} health changed from {old_status} to {new_status}: {reason}")
    
    def get_mood_report(self) -> Dict[str, Any]:
        """Get comprehensive mood and status report.
        
        Returns:
            Detailed mood information
        """
        return {
            "dinosaur_id": self.agent_model.id,
            "name": self.agent_model.name,
            "species": self.agent_model.species.name,
            "happiness_level": self.happiness_level,
            "stress_level": self.stress_level,
            "health_status": self.health_status,
            "hunger_level": self.hunger_level,
            "behavioral_state": self._determine_behavioral_state(),
            "social_needs": self.social_needs,
            "territory_size": self.territory_size,
            "current_location": {
                "zone": self.agent_model.location.zone,
                "description": self.agent_model.location.description
            },
            "personality_traits": self.agent_model.personality_traits,
            "timestamp": self.agent_model.last_activity
        }


class TyrannosaurusRexAgent(DinosaurSpeciesAgent):
    """Tyrannosaurus Rex specific agent implementation."""
    
    def _calculate_social_needs(self) -> float:
        """T-Rex are generally solitary."""
        return 0.2
    
    def _get_territory_requirements(self) -> float:
        """T-Rex need large territories."""
        return 500.0
    
    def _calculate_environmental_happiness_change(self, factors: Dict[str, Any]) -> float:
        """T-Rex specific environmental reactions."""
        change = super()._calculate_environmental_happiness_change(factors)
        
        # T-Rex prefer warmer weather
        weather = factors.get("weather", "clear")
        if weather == "hot":
            change += 0.1
        elif weather in ["cold", "extreme_cold"]:
            change -= 0.15
        
        # T-Rex are apex predators - they like having space
        visitor_density = factors.get("visitor_density", 0.5)
        if visitor_density > 0.6:
            change -= 0.2  # Don't like crowds
        
        return max(-0.6, min(0.6, change))
    
    def _assess_threat_level(self, human_info: Dict[str, Any]) -> str:
        """T-Rex are naturally more aggressive."""
        base_threat = super()._assess_threat_level(human_info)
        
        # T-Rex are more likely to see things as threats
        if base_threat == "none" and human_info.get("distance", 10.0) < 15.0:
            return "low"
        elif base_threat == "low":
            return "medium"
        
        return base_threat


class TriceratopsAgent(DinosaurSpeciesAgent):
    """Triceratops specific agent implementation."""
    
    def _calculate_social_needs(self) -> float:
        """Triceratops are herd animals."""
        return 0.8
    
    def _get_territory_requirements(self) -> float:
        """Triceratops need moderate territory but prefer shared spaces."""
        return 200.0
    
    def _calculate_environmental_happiness_change(self, factors: Dict[str, Any]) -> float:
        """Triceratops specific environmental reactions."""
        change = super()._calculate_environmental_happiness_change(factors)
        
        # Triceratops are herbivores - vegetation is important
        vegetation = factors.get("vegetation_quality", 0.5)
        if vegetation > 0.7:
            change += 0.2
        elif vegetation < 0.3:
            change -= 0.25
        
        # They're more tolerant of visitors
        visitor_density = factors.get("visitor_density", 0.5)
        if 0.3 <= visitor_density <= 0.7:
            change += 0.05  # Moderate crowds are okay
        
        return max(-0.5, min(0.6, change))
    
    def _calculate_interaction_happiness_change(self, human_info: Dict[str, Any]) -> float:
        """Triceratops are generally more docile."""
        change = super()._calculate_interaction_happiness_change(human_info)
        
        # More positive interactions with calm humans
        behavior = human_info.get("behavior", "observing")
        if behavior in ["calm_observation", "feeding", "gentle_approach"]:
            change += 0.1
        
        return max(-0.3, min(0.4, change))


class VelociraptorAgent(DinosaurSpeciesAgent):
    """Velociraptor specific agent implementation."""
    
    def _calculate_social_needs(self) -> float:
        """Velociraptors are pack hunters."""
        return 0.9
    
    def _get_territory_requirements(self) -> float:
        """Velociraptors need moderate territory but hunt in groups."""
        return 150.0
    
    def _calculate_environmental_happiness_change(self, factors: Dict[str, Any]) -> float:
        """Velociraptor specific environmental reactions."""
        change = super()._calculate_environmental_happiness_change(factors)
        
        # Velociraptors are intelligent and curious
        enrichment = factors.get("environmental_enrichment", 0.5)
        if enrichment > 0.7:
            change += 0.2
        elif enrichment < 0.3:
            change -= 0.2
        
        # They're more active and need stimulation
        activity_level = factors.get("activity_opportunities", 0.5)
        if activity_level < 0.4:
            change -= 0.15
        
        return max(-0.5, min(0.5, change))
    
    def _assess_threat_level(self, human_info: Dict[str, Any]) -> str:
        """Velociraptors are intelligent and assess threats carefully."""
        base_threat = super()._assess_threat_level(human_info)
        
        # Velociraptors are smart - they assess group dynamics
        group_size = human_info.get("group_size", 1)
        if group_size > 5 and base_threat in ["none", "low"]:
            return "medium"  # Large groups are concerning
        
        return base_threat


class BrachiosaurusAgent(DinosaurSpeciesAgent):
    """Brachiosaurus specific agent implementation."""
    
    def _calculate_social_needs(self) -> float:
        """Brachiosaurus are gentle herd animals."""
        return 0.7
    
    def _get_territory_requirements(self) -> float:
        """Brachiosaurus need very large territories due to size."""
        return 800.0
    
    def _calculate_environmental_happiness_change(self, factors: Dict[str, Any]) -> float:
        """Brachiosaurus specific environmental reactions."""
        change = super()._calculate_environmental_happiness_change(factors)
        
        # Brachiosaurus need lots of vegetation
        vegetation = factors.get("vegetation_quality", 0.5)
        if vegetation > 0.8:
            change += 0.25
        elif vegetation < 0.4:
            change -= 0.3
        
        # They're gentle giants - less affected by visitors
        visitor_density = factors.get("visitor_density", 0.5)
        if visitor_density > 0.9:
            change -= 0.05  # Only very high crowds bother them
        
        return max(-0.6, min(0.6, change))
    
    def _calculate_interaction_stress_change(self, human_info: Dict[str, Any]) -> float:
        """Brachiosaurus are generally calm."""
        change = super()._calculate_interaction_stress_change(human_info)
        
        # They're less stressed by human presence due to size
        return change * 0.7  # Reduce stress by 30%


class StegosaurusAgent(DinosaurSpeciesAgent):
    """Stegosaurus specific agent implementation."""
    
    def _calculate_social_needs(self) -> float:
        """Stegosaurus are moderately social."""
        return 0.6
    
    def _get_territory_requirements(self) -> float:
        """Stegosaurus need moderate territory."""
        return 300.0
    
    def _calculate_environmental_happiness_change(self, factors: Dict[str, Any]) -> float:
        """Stegosaurus specific environmental reactions."""
        change = super()._calculate_environmental_happiness_change(factors)
        
        # Stegosaurus prefer temperate conditions
        weather = factors.get("weather", "clear")
        if weather in ["mild", "clear"]:
            change += 0.1
        elif weather in ["extreme_heat", "extreme_cold"]:
            change -= 0.2
        
        # They're defensive - don't like being approached from behind
        approach_angle = factors.get("approach_angle", "front")
        if approach_angle == "rear":
            change -= 0.15
        
        return max(-0.5, min(0.5, change))
    
    def _assess_threat_level(self, human_info: Dict[str, Any]) -> str:
        """Stegosaurus are defensive but not aggressive."""
        base_threat = super()._assess_threat_level(human_info)
        
        # They're more concerned about rear approaches
        approach_angle = human_info.get("approach_angle", "front")
        if approach_angle == "rear" and base_threat == "none":
            return "low"
        
        return base_threat


class ParasaurolophusAgent(DinosaurSpeciesAgent):
    """Parasaurolophus specific agent implementation."""
    
    def _calculate_social_needs(self) -> float:
        """Parasaurolophus are highly social herd animals."""
        return 0.9
    
    def _get_territory_requirements(self) -> float:
        """Parasaurolophus prefer shared territories."""
        return 250.0
    
    def _calculate_environmental_happiness_change(self, factors: Dict[str, Any]) -> float:
        """Parasaurolophus specific environmental reactions."""
        change = super()._calculate_environmental_happiness_change(factors)
        
        # Parasaurolophus are vocal - they like acoustic environments
        acoustic_quality = factors.get("acoustic_environment", 0.5)
        if acoustic_quality > 0.7:
            change += 0.15
        elif acoustic_quality < 0.3:
            change -= 0.1
        
        # They're social - benefit from other dinosaurs nearby
        social_interaction = factors.get("social_interaction", 0.5)
        if social_interaction > 0.7:
            change += 0.2
        elif social_interaction < 0.3:
            change -= 0.25
        
        return max(-0.5, min(0.6, change))
    
    def _calculate_interaction_happiness_change(self, human_info: Dict[str, Any]) -> float:
        """Parasaurolophus are curious about humans."""
        change = super()._calculate_interaction_happiness_change(human_info)
        
        # They're curious and social
        behavior = human_info.get("behavior", "observing")
        if behavior in ["calm_observation", "educational"]:
            change += 0.1
        
        return max(-0.3, min(0.4, change))


class DinosaurAgentFactory:
    """Factory for creating species-specific dinosaur agents."""
    
    def __init__(self, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize dinosaur agent factory.
        
        Args:
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        self.openai_config = openai_config
        self.ag2_config = ag2_config
        self.logger = logging.getLogger(__name__)
        self._dinosaur_counter = 0
        
        # Species-specific agent classes
        self.species_classes = {
            DinosaurSpecies.TYRANNOSAURUS_REX: TyrannosaurusRexAgent,
            DinosaurSpecies.TRICERATOPS: TriceratopsAgent,
            DinosaurSpecies.VELOCIRAPTOR: VelociraptorAgent,
            DinosaurSpecies.BRACHIOSAURUS: BrachiosaurusAgent,
            DinosaurSpecies.STEGOSAURUS: StegosaurusAgent,
            DinosaurSpecies.PARASAUROLOPHUS: ParasaurolophusAgent,
        }
    
    def create_dinosaur_agent(self, agent_model: Agent) -> DinosaurSpeciesAgent:
        """Create appropriate dinosaur agent based on species.
        
        Args:
            agent_model: Agent data model with dinosaur species
            
        Returns:
            Species-specific dinosaur agent instance
        """
        if agent_model.role != AgentRole.DINOSAUR:
            raise ValueError("Agent model must have DINOSAUR role")
        
        if not agent_model.species:
            raise ValueError("Dinosaur agent must have a species")
        
        species_class = self.species_classes.get(agent_model.species)
        if not species_class:
            raise ValueError(f"No implementation for species: {agent_model.species}")
        
        return species_class(agent_model, self.openai_config, self.ag2_config)
    
    def create_tyrannosaurus_rex(self, name: str, location: Optional[Location] = None) -> Agent:
        """Create a Tyrannosaurus Rex agent model.
        
        Args:
            name: Dinosaur name
            location: Optional initial location
            
        Returns:
            T-Rex agent model
        """
        self._dinosaur_counter += 1
        agent_id = f"trex_{self._dinosaur_counter}"
        
        personality_traits = {
            PersonalityTrait.BRAVE.value: random.uniform(0.8, 0.95),
            PersonalityTrait.CAUTIOUS.value: random.uniform(0.1, 0.3),
            PersonalityTrait.ANALYTICAL.value: random.uniform(0.6, 0.8),
        }
        
        capabilities = ["instinctual_behavior", "environmental_response", "apex_predator"]
        
        if location is None:
            location = Location(0.0, 0.0, "carnivore_habitat", "Large carnivore habitat")
        
        agent = Agent(
            id=agent_id,
            name=name,
            role=AgentRole.DINOSAUR,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location,
            species=DinosaurSpecies.TYRANNOSAURUS_REX
        )
        
        self.logger.info(f"Created T-Rex: {name}")
        return agent
    
    def create_triceratops(self, name: str, location: Optional[Location] = None) -> Agent:
        """Create a Triceratops agent model.
        
        Args:
            name: Dinosaur name
            location: Optional initial location
            
        Returns:
            Triceratops agent model
        """
        self._dinosaur_counter += 1
        agent_id = f"triceratops_{self._dinosaur_counter}"
        
        personality_traits = {
            PersonalityTrait.CAUTIOUS.value: random.uniform(0.6, 0.8),
            PersonalityTrait.EMPATHY.value: random.uniform(0.7, 0.9),
            PersonalityTrait.FRIENDLY.value: random.uniform(0.5, 0.7),
        }
        
        capabilities = ["instinctual_behavior", "environmental_response", "herd_behavior"]
        
        if location is None:
            location = Location(0.0, 0.0, "herbivore_habitat", "Herbivore grazing area")
        
        agent = Agent(
            id=agent_id,
            name=name,
            role=AgentRole.DINOSAUR,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location,
            species=DinosaurSpecies.TRICERATOPS
        )
        
        self.logger.info(f"Created Triceratops: {name}")
        return agent
    
    def create_velociraptor(self, name: str, location: Optional[Location] = None) -> Agent:
        """Create a Velociraptor agent model.
        
        Args:
            name: Dinosaur name
            location: Optional initial location
            
        Returns:
            Velociraptor agent model
        """
        self._dinosaur_counter += 1
        agent_id = f"velociraptor_{self._dinosaur_counter}"
        
        personality_traits = {
            PersonalityTrait.ANALYTICAL.value: random.uniform(0.8, 0.95),
            PersonalityTrait.CREATIVE.value: random.uniform(0.7, 0.9),
            PersonalityTrait.BRAVE.value: random.uniform(0.6, 0.8),
        }
        
        capabilities = ["instinctual_behavior", "environmental_response", "pack_hunting", "problem_solving"]
        
        if location is None:
            location = Location(0.0, 0.0, "raptor_enclosure", "Secure raptor habitat")
        
        agent = Agent(
            id=agent_id,
            name=name,
            role=AgentRole.DINOSAUR,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location,
            species=DinosaurSpecies.VELOCIRAPTOR
        )
        
        self.logger.info(f"Created Velociraptor: {name}")
        return agent
    
    def create_brachiosaurus(self, name: str, location: Optional[Location] = None) -> Agent:
        """Create a Brachiosaurus agent model.
        
        Args:
            name: Dinosaur name
            location: Optional initial location
            
        Returns:
            Brachiosaurus agent model
        """
        self._dinosaur_counter += 1
        agent_id = f"brachiosaurus_{self._dinosaur_counter}"
        
        personality_traits = {
            PersonalityTrait.EMPATHY.value: random.uniform(0.8, 0.95),
            PersonalityTrait.CAUTIOUS.value: random.uniform(0.3, 0.5),
            PersonalityTrait.FRIENDLY.value: random.uniform(0.7, 0.9),
        }
        
        capabilities = ["instinctual_behavior", "environmental_response", "gentle_giant"]
        
        if location is None:
            location = Location(0.0, 0.0, "sauropod_habitat", "Large herbivore habitat")
        
        agent = Agent(
            id=agent_id,
            name=name,
            role=AgentRole.DINOSAUR,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location,
            species=DinosaurSpecies.BRACHIOSAURUS
        )
        
        self.logger.info(f"Created Brachiosaurus: {name}")
        return agent
    
    def create_stegosaurus(self, name: str, location: Optional[Location] = None) -> Agent:
        """Create a Stegosaurus agent model.
        
        Args:
            name: Dinosaur name
            location: Optional initial location
            
        Returns:
            Stegosaurus agent model
        """
        self._dinosaur_counter += 1
        agent_id = f"stegosaurus_{self._dinosaur_counter}"
        
        personality_traits = {
            PersonalityTrait.CAUTIOUS.value: random.uniform(0.7, 0.9),
            PersonalityTrait.BRAVE.value: random.uniform(0.4, 0.6),
            PersonalityTrait.ANALYTICAL.value: random.uniform(0.3, 0.5),
        }
        
        capabilities = ["instinctual_behavior", "environmental_response", "defensive_behavior"]
        
        if location is None:
            location = Location(0.0, 0.0, "herbivore_habitat", "Herbivore grazing area")
        
        agent = Agent(
            id=agent_id,
            name=name,
            role=AgentRole.DINOSAUR,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location,
            species=DinosaurSpecies.STEGOSAURUS
        )
        
        self.logger.info(f"Created Stegosaurus: {name}")
        return agent
    
    def create_parasaurolophus(self, name: str, location: Optional[Location] = None) -> Agent:
        """Create a Parasaurolophus agent model.
        
        Args:
            name: Dinosaur name
            location: Optional initial location
            
        Returns:
            Parasaurolophus agent model
        """
        self._dinosaur_counter += 1
        agent_id = f"parasaurolophus_{self._dinosaur_counter}"
        
        personality_traits = {
            PersonalityTrait.FRIENDLY.value: random.uniform(0.8, 0.95),
            PersonalityTrait.EMPATHY.value: random.uniform(0.7, 0.9),
            PersonalityTrait.CREATIVE.value: random.uniform(0.6, 0.8),
        }
        
        capabilities = ["instinctual_behavior", "environmental_response", "vocal_communication", "herd_behavior"]
        
        if location is None:
            location = Location(0.0, 0.0, "herbivore_habitat", "Herbivore grazing area")
        
        agent = Agent(
            id=agent_id,
            name=name,
            role=AgentRole.DINOSAUR,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location,
            species=DinosaurSpecies.PARASAUROLOPHUS
        )
        
        self.logger.info(f"Created Parasaurolophus: {name}")
        return agent
    
    def create_random_dinosaur(self, name: str, location: Optional[Location] = None) -> Agent:
        """Create a random dinosaur species.
        
        Args:
            name: Dinosaur name
            location: Optional initial location
            
        Returns:
            Random dinosaur agent model
        """
        species_creators = [
            self.create_tyrannosaurus_rex,
            self.create_triceratops,
            self.create_velociraptor,
            self.create_brachiosaurus,
            self.create_stegosaurus,
            self.create_parasaurolophus
        ]
        
        # Randomly select a species
        selected_creator = random.choice(species_creators)
        return selected_creator(name, location)