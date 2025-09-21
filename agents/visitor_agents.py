"""Visitor agent implementations for the AI Agent Dinosaur Simulator."""

import logging
import random
from typing import Dict, List, Optional, Any
from agents.base_agent import DinosaurAgent
from models.core import Agent
from models.config import OpenAIConfig, AG2Config, Location
from models.enums import AgentRole, AgentState, PersonalityTrait


class TouristAgent(DinosaurAgent):
    """Tourist agent with varying personality traits and behaviors."""
    
    def __init__(self, agent_model: Agent, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize tourist agent.
        
        Args:
            agent_model: The agent data model
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        super().__init__(agent_model, openai_config, ag2_config)
        self.satisfaction_level = 0.7  # Start with neutral satisfaction
        self.risk_tolerance = self._calculate_risk_tolerance()
        self.interests = self._determine_interests()
        self.visit_duration = 0  # Hours spent in park
        self.logger = logging.getLogger(f"{__name__}.Tourist.{agent_model.name}")
    
    def _calculate_risk_tolerance(self) -> float:
        """Calculate risk tolerance based on personality traits.
        
        Returns:
            Risk tolerance value (0.0 to 1.0)
        """
        cautious = self.agent_model.personality_traits.get(PersonalityTrait.CAUTIOUS.value, 0.5)
        brave = self.agent_model.personality_traits.get(PersonalityTrait.BRAVE.value, 0.5)
        
        # Risk tolerance is inverse of caution plus bravery
        risk_tolerance = (1.0 - cautious + brave) / 2.0
        return max(0.0, min(1.0, risk_tolerance))
    
    def _determine_interests(self) -> List[str]:
        """Determine tourist interests based on personality.
        
        Returns:
            List of interest categories
        """
        interests = []
        traits = self.agent_model.personality_traits
        
        if traits.get(PersonalityTrait.ANALYTICAL.value, 0.0) > 0.6:
            interests.extend(["educational_tours", "scientific_exhibits"])
        
        if traits.get(PersonalityTrait.BRAVE.value, 0.0) > 0.7:
            interests.extend(["adventure_tours", "close_encounters"])
        
        if traits.get(PersonalityTrait.FRIENDLY.value, 0.0) > 0.6:
            interests.extend(["group_activities", "interactive_experiences"])
        
        if traits.get(PersonalityTrait.CREATIVE.value, 0.0) > 0.6:
            interests.extend(["photography", "art_workshops"])
        
        # Default interests if no specific traits
        if not interests:
            interests = ["general_sightseeing", "gift_shop"]
        
        return interests
    
    def react_to_event(self, event_description: str, event_severity: int, event_location: str) -> Dict[str, Any]:
        """React to an event in the park.
        
        Args:
            event_description: Description of the event
            event_severity: Severity level (1-10)
            event_location: Location where event occurred
            
        Returns:
            Tourist's reaction and satisfaction change
        """
        self.update_state(AgentState.RESPONDING_TO_EVENT)
        
        try:
            # Calculate reaction based on risk tolerance and event severity
            reaction_intensity = self._calculate_reaction_intensity(event_severity)
            satisfaction_change = self._calculate_satisfaction_change(event_severity, event_location)
            
            # Generate contextual reaction
            reaction_prompt = (
                f"As a tourist with risk tolerance {self.risk_tolerance:.1f}, react to this event: "
                f"{event_description} (severity {event_severity}) at {event_location}. "
                f"Express your feelings, concerns, and what you want to do next. "
                f"Your interests include: {', '.join(self.interests)}."
            )
            
            response = self.generate_reply(
                messages=[{"content": reaction_prompt, "role": "user", "name": "EventSystem"}]
            )
            
            # Update satisfaction
            self.satisfaction_level = max(0.0, min(1.0, self.satisfaction_level + satisfaction_change))
            
            self.logger.info(f"Tourist {self.agent_model.name} reacted to event with intensity {reaction_intensity}")
            
            return {
                "reaction": response,
                "reaction_intensity": reaction_intensity,
                "satisfaction_change": satisfaction_change,
                "new_satisfaction": self.satisfaction_level,
                "wants_to_leave": satisfaction_change < -0.3 or reaction_intensity > 0.8,
                "timestamp": self.agent_model.last_activity
            }
        
        except Exception as e:
            self.logger.error(f"Error generating event reaction: {e}")
            return {
                "reaction": "I'm not sure how to react to this situation.",
                "error": str(e)
            }
        finally:
            self.update_state(AgentState.IDLE)
    
    def _calculate_reaction_intensity(self, event_severity: int) -> float:
        """Calculate how intensely the tourist reacts to an event.
        
        Args:
            event_severity: Event severity (1-10)
            
        Returns:
            Reaction intensity (0.0 to 1.0)
        """
        # Base reaction from event severity
        base_reaction = event_severity / 10.0
        
        # Modify based on risk tolerance (lower tolerance = higher reaction)
        risk_modifier = (1.0 - self.risk_tolerance) * 0.5
        
        # Add personality modifiers
        cautious = self.agent_model.personality_traits.get(PersonalityTrait.CAUTIOUS.value, 0.5)
        brave = self.agent_model.personality_traits.get(PersonalityTrait.BRAVE.value, 0.5)
        
        personality_modifier = (cautious - brave) * 0.3
        
        reaction_intensity = base_reaction + risk_modifier + personality_modifier
        return max(0.0, min(1.0, reaction_intensity))
    
    def _calculate_satisfaction_change(self, event_severity: int, event_location: str) -> float:
        """Calculate change in satisfaction based on event.
        
        Args:
            event_severity: Event severity (1-10)
            event_location: Location of the event
            
        Returns:
            Satisfaction change (-1.0 to 1.0)
        """
        # Base satisfaction change (negative for bad events)
        base_change = -(event_severity - 5) / 10.0
        
        # Location proximity effect (worse if closer to tourist)
        current_zone = self.agent_model.location.zone
        if event_location == current_zone:
            proximity_penalty = -0.2
        elif "nearby" in event_location.lower() or current_zone in event_location:
            proximity_penalty = -0.1
        else:
            proximity_penalty = 0.0
        
        # Risk tolerance effect (high tolerance tourists less affected)
        risk_modifier = (self.risk_tolerance - 0.5) * 0.2
        
        total_change = base_change + proximity_penalty + risk_modifier
        return max(-1.0, min(1.0, total_change))
    
    def provide_feedback(self, topic: str) -> Dict[str, Any]:
        """Provide feedback on a specific topic.
        
        Args:
            topic: Topic to provide feedback on
            
        Returns:
            Feedback response with satisfaction rating
        """
        try:
            feedback_prompt = (
                f"As a tourist with satisfaction level {self.satisfaction_level:.1f}, "
                f"provide feedback on {topic}. Consider your interests: {', '.join(self.interests)} "
                f"and your experience so far. Be honest about what you liked and didn't like."
            )
            
            response = self.generate_reply(
                messages=[{"content": feedback_prompt, "role": "user", "name": "FeedbackSystem"}]
            )
            
            self.logger.info(f"Tourist {self.agent_model.name} provided feedback on {topic}")
            
            return {
                "feedback": response,
                "topic": topic,
                "satisfaction_rating": self.satisfaction_level,
                "interests": self.interests,
                "visit_duration": self.visit_duration,
                "would_recommend": self.satisfaction_level > 0.6,
                "timestamp": self.agent_model.last_activity
            }
        
        except Exception as e:
            self.logger.error(f"Error providing feedback: {e}")
            return {
                "feedback": "I'm having trouble expressing my thoughts right now.",
                "error": str(e)
            }
    
    def make_purchase_decision(self, item: str, price: float) -> Dict[str, Any]:
        """Make a decision about purchasing an item.
        
        Args:
            item: Item being considered for purchase
            price: Price of the item
            
        Returns:
            Purchase decision and reasoning
        """
        try:
            # Factor in satisfaction and interests - check both ways for matching
            interest_match = any(
                interest.lower() in item.lower() or 
                any(word in interest.lower() for word in item.lower().split())
                for interest in self.interests
            )
            satisfaction_factor = self.satisfaction_level
            
            decision_prompt = (
                f"As a tourist with satisfaction level {self.satisfaction_level:.1f}, "
                f"decide whether to buy {item} for ${price:.2f}. "
                f"Consider your interests ({', '.join(self.interests)}) and your experience so far. "
                f"Explain your decision."
            )
            
            response = self.generate_reply(
                messages=[{"content": decision_prompt, "role": "user", "name": "ShopSystem"}]
            )
            
            # Simple decision logic
            will_buy = (
                satisfaction_factor > 0.5 and 
                (interest_match or satisfaction_factor > 0.8) and 
                price < 50.0  # Arbitrary price threshold
            )
            
            self.logger.info(f"Tourist {self.agent_model.name} made purchase decision for {item}")
            
            return {
                "decision": response,
                "will_buy": will_buy,
                "item": item,
                "price": price,
                "reasoning_factors": {
                    "satisfaction": satisfaction_factor,
                    "interest_match": interest_match,
                    "price_acceptable": price < 50.0
                },
                "timestamp": self.agent_model.last_activity
            }
        
        except Exception as e:
            self.logger.error(f"Error making purchase decision: {e}")
            return {
                "decision": "I'm not sure about this purchase right now.",
                "will_buy": False,
                "error": str(e)
            }
    
    def update_satisfaction(self, change: float, reason: str) -> None:
        """Update tourist satisfaction level.
        
        Args:
            change: Change in satisfaction (-1.0 to 1.0)
            reason: Reason for the change
        """
        old_satisfaction = self.satisfaction_level
        self.satisfaction_level = max(0.0, min(1.0, self.satisfaction_level + change))
        
        self.logger.info(
            f"Tourist {self.agent_model.name} satisfaction changed from {old_satisfaction:.2f} "
            f"to {self.satisfaction_level:.2f} due to: {reason}"
        )
    
    def get_satisfaction_report(self) -> Dict[str, Any]:
        """Get detailed satisfaction report.
        
        Returns:
            Comprehensive satisfaction information
        """
        satisfaction_category = self._categorize_satisfaction()
        
        return {
            "tourist_id": self.agent_model.id,
            "name": self.agent_model.name,
            "satisfaction_level": self.satisfaction_level,
            "satisfaction_category": satisfaction_category,
            "risk_tolerance": self.risk_tolerance,
            "interests": self.interests,
            "visit_duration": self.visit_duration,
            "personality_traits": self.agent_model.personality_traits,
            "current_location": {
                "zone": self.agent_model.location.zone,
                "description": self.agent_model.location.description
            },
            "timestamp": self.agent_model.last_activity
        }
    
    def _categorize_satisfaction(self) -> str:
        """Categorize satisfaction level.
        
        Returns:
            Satisfaction category string
        """
        if self.satisfaction_level >= 0.8:
            return "very_satisfied"
        elif self.satisfaction_level >= 0.6:
            return "satisfied"
        elif self.satisfaction_level >= 0.4:
            return "neutral"
        elif self.satisfaction_level >= 0.2:
            return "dissatisfied"
        else:
            return "very_dissatisfied"


class VisitorAgentFactory:
    """Factory for creating different types of visitor agents."""
    
    def __init__(self, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize visitor agent factory.
        
        Args:
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        self.openai_config = openai_config
        self.ag2_config = ag2_config
        self.logger = logging.getLogger(__name__)
        self._visitor_counter = 0
    
    def create_adventure_seeker(self, name: Optional[str] = None, location: Optional[Location] = None) -> Agent:
        """Create an adventure-seeking tourist.
        
        Args:
            name: Optional custom name
            location: Optional initial location
            
        Returns:
            Adventure seeker agent model
        """
        self._visitor_counter += 1
        agent_name = name or f"Adventurer_{self._visitor_counter}"
        agent_id = f"adventure_tourist_{self._visitor_counter}"
        
        personality_traits = {
            PersonalityTrait.BRAVE.value: random.uniform(0.7, 0.9),
            PersonalityTrait.CAUTIOUS.value: random.uniform(0.1, 0.3),
            PersonalityTrait.FRIENDLY.value: random.uniform(0.5, 0.8),
            PersonalityTrait.CREATIVE.value: random.uniform(0.4, 0.7)
        }
        
        capabilities = ["observation", "feedback", "exploration", "risk_taking"]
        
        if location is None:
            location = Location(0.0, 0.0, "entrance", "Main park entrance")
        
        agent = Agent(
            id=agent_id,
            name=agent_name,
            role=AgentRole.TOURIST,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location
        )
        
        self.logger.info(f"Created adventure seeker tourist: {agent_name}")
        return agent
    
    def create_family_visitor(self, name: Optional[str] = None, location: Optional[Location] = None) -> Agent:
        """Create a family-oriented visitor.
        
        Args:
            name: Optional custom name
            location: Optional initial location
            
        Returns:
            Family visitor agent model
        """
        self._visitor_counter += 1
        agent_name = name or f"Family_{self._visitor_counter}"
        agent_id = f"family_tourist_{self._visitor_counter}"
        
        personality_traits = {
            PersonalityTrait.CAUTIOUS.value: random.uniform(0.6, 0.9),
            PersonalityTrait.EMPATHY.value: random.uniform(0.7, 0.9),
            PersonalityTrait.FRIENDLY.value: random.uniform(0.6, 0.8),
            PersonalityTrait.BRAVE.value: random.uniform(0.2, 0.5)
        }
        
        capabilities = ["observation", "feedback", "child_supervision", "safety_awareness"]
        
        if location is None:
            location = Location(0.0, 0.0, "family_area", "Family-friendly zone")
        
        agent = Agent(
            id=agent_id,
            name=agent_name,
            role=AgentRole.TOURIST,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location
        )
        
        self.logger.info(f"Created family visitor: {agent_name}")
        return agent
    
    def create_educational_visitor(self, name: Optional[str] = None, location: Optional[Location] = None) -> Agent:
        """Create an education-focused visitor.
        
        Args:
            name: Optional custom name
            location: Optional initial location
            
        Returns:
            Educational visitor agent model
        """
        self._visitor_counter += 1
        agent_name = name or f"Scholar_{self._visitor_counter}"
        agent_id = f"educational_tourist_{self._visitor_counter}"
        
        personality_traits = {
            PersonalityTrait.ANALYTICAL.value: random.uniform(0.7, 0.9),
            PersonalityTrait.CAUTIOUS.value: random.uniform(0.5, 0.7),
            PersonalityTrait.FRIENDLY.value: random.uniform(0.4, 0.7),
            PersonalityTrait.CREATIVE.value: random.uniform(0.3, 0.6)
        }
        
        capabilities = ["observation", "feedback", "research", "documentation"]
        
        if location is None:
            location = Location(0.0, 0.0, "education_center", "Educational exhibits area")
        
        agent = Agent(
            id=agent_id,
            name=agent_name,
            role=AgentRole.TOURIST,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location
        )
        
        self.logger.info(f"Created educational visitor: {agent_name}")
        return agent
    
    def create_casual_visitor(self, name: Optional[str] = None, location: Optional[Location] = None) -> Agent:
        """Create a casual visitor with balanced traits.
        
        Args:
            name: Optional custom name
            location: Optional initial location
            
        Returns:
            Casual visitor agent model
        """
        self._visitor_counter += 1
        agent_name = name or f"Visitor_{self._visitor_counter}"
        agent_id = f"casual_tourist_{self._visitor_counter}"
        
        # Balanced personality traits
        personality_traits = {
            PersonalityTrait.FRIENDLY.value: random.uniform(0.5, 0.8),
            PersonalityTrait.CAUTIOUS.value: random.uniform(0.4, 0.6),
            PersonalityTrait.BRAVE.value: random.uniform(0.3, 0.6),
            PersonalityTrait.EMPATHY.value: random.uniform(0.4, 0.7)
        }
        
        capabilities = ["observation", "feedback", "exploration"]
        
        if location is None:
            location = Location(0.0, 0.0, "main_plaza", "Central plaza area")
        
        agent = Agent(
            id=agent_id,
            name=agent_name,
            role=AgentRole.TOURIST,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location
        )
        
        self.logger.info(f"Created casual visitor: {agent_name}")
        return agent
    
    def create_photographer_visitor(self, name: Optional[str] = None, location: Optional[Location] = None) -> Agent:
        """Create a photography-focused visitor.
        
        Args:
            name: Optional custom name
            location: Optional initial location
            
        Returns:
            Photographer visitor agent model
        """
        self._visitor_counter += 1
        agent_name = name or f"Photographer_{self._visitor_counter}"
        agent_id = f"photo_tourist_{self._visitor_counter}"
        
        personality_traits = {
            PersonalityTrait.CREATIVE.value: random.uniform(0.7, 0.9),
            PersonalityTrait.ANALYTICAL.value: random.uniform(0.5, 0.8),
            PersonalityTrait.CAUTIOUS.value: random.uniform(0.3, 0.6),
            PersonalityTrait.BRAVE.value: random.uniform(0.4, 0.7)
        }
        
        capabilities = ["observation", "feedback", "photography", "artistic_appreciation"]
        
        if location is None:
            location = Location(0.0, 0.0, "scenic_overlook", "Scenic photography area")
        
        agent = Agent(
            id=agent_id,
            name=agent_name,
            role=AgentRole.TOURIST,
            personality_traits=personality_traits,
            capabilities=capabilities,
            location=location
        )
        
        self.logger.info(f"Created photographer visitor: {agent_name}")
        return agent
    
    def create_tourist_agent(self, agent_model: Agent) -> TouristAgent:
        """Create a TouristAgent instance from an Agent model.
        
        Args:
            agent_model: Agent data model
            
        Returns:
            TouristAgent instance
        """
        if agent_model.role != AgentRole.TOURIST:
            raise ValueError("Agent model must have TOURIST role")
        
        return TouristAgent(agent_model, self.openai_config, self.ag2_config)
    
    def create_random_visitor(self, name: Optional[str] = None, location: Optional[Location] = None) -> Agent:
        """Create a random visitor with varied characteristics.
        
        Args:
            name: Optional custom name
            location: Optional initial location
            
        Returns:
            Random visitor agent model
        """
        visitor_types = [
            self.create_adventure_seeker,
            self.create_family_visitor,
            self.create_educational_visitor,
            self.create_casual_visitor,
            self.create_photographer_visitor
        ]
        
        # Randomly select a visitor type
        selected_creator = random.choice(visitor_types)
        return selected_creator(name, location)