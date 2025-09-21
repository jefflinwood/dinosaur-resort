"""Agent communication and orchestration system using ag2's GroupChat functionality with real-time chat."""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from autogen import GroupChat, GroupChatManager
from models.core import Agent, Event
from models.config import OpenAIConfig, AG2Config, AgentConfig, Location
from models.enums import AgentRole, AgentState, DinosaurSpecies
from managers.ag2_integration import AG2Integration
from managers.real_time_agent_chat import RealTimeAgentChat
from agents.base_agent import DinosaurAgent, BaseAgentConfig, AgentFactory
from agents.staff_agents import StaffAgentFactory
from agents.visitor_agents import VisitorAgentFactory
from agents.dinosaur_agents import DinosaurAgentFactory


class AgentManager:
    """Manages all AI agents using ag2's GroupChat functionality for communication and orchestration."""
    
    def __init__(self, openai_config: OpenAIConfig, ag2_config: AG2Config, agent_config: AgentConfig):
        """Initialize agent manager with ag2 GroupChat functionality.
        
        Args:
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
            agent_config: Agent initialization configuration
        """
        self.openai_config = openai_config
        self.ag2_config = ag2_config
        self.agent_config = agent_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize ag2 integration
        self.ag2_integration = AG2Integration(openai_config, ag2_config)
        
        # Initialize real-time chat system
        self.real_time_chat = RealTimeAgentChat(openai_config)
        
        # System status manager will be set by simulation manager
        self.system_status_manager = None
        
        # Initialize agent factories
        self.base_config = BaseAgentConfig(openai_config, ag2_config)
        self.agent_factory = AgentFactory(self.base_config)
        self.staff_factory = StaffAgentFactory(openai_config, ag2_config)
        self.visitor_factory = VisitorAgentFactory(openai_config, ag2_config)
        self.dinosaur_factory = DinosaurAgentFactory(openai_config, ag2_config)
        
        # Agent storage
        self.agents: Dict[str, Agent] = {}  # agent_id -> Agent model
        self.agent_instances: Dict[str, DinosaurAgent] = {}  # agent_id -> DinosaurAgent instance
        self.group_chats: Dict[str, GroupChat] = {}  # group_name -> GroupChat
        self.group_chat_managers: Dict[str, GroupChatManager] = {}  # group_name -> GroupChatManager
        
        # Agent health monitoring
        self.agent_health: Dict[str, Dict[str, Any]] = {}  # agent_id -> health info
        self.last_health_check: datetime = datetime.now()
        
        # Communication history
        self.conversation_history: List[Dict[str, Any]] = []
        
        self.logger.info("AgentManager initialized with ag2 GroupChat functionality")
    
    def initialize_agents(self, config: Optional[AgentConfig] = None) -> List[Agent]:
        """Initialize agents with distinct personalities and roles.
        
        Args:
            config: Optional agent configuration (uses default if not provided)
            
        Returns:
            List of initialized agents
        """
        if config:
            self.agent_config = config
        
        agents = []
        
        try:
            # Create staff agents
            staff_agents = self._create_staff_agents()
            agents.extend(staff_agents)
            
            # Create visitor agents
            visitor_agents = self._create_visitor_agents()
            agents.extend(visitor_agents)
            
            # Create dinosaur agents
            dinosaur_agents = self._create_dinosaur_agents()
            agents.extend(dinosaur_agents)
            
            # Store agents and create instances
            for agent in agents:
                self.agents[agent.id] = agent
                self.agent_instances[agent.id] = self._create_agent_instance(agent)
                self.agent_health[agent.id] = self._initialize_agent_health(agent)
                
                # Register agent with real-time chat
                self.real_time_chat.register_agent(agent)
            
            # Create group chats for different scenarios
            self._initialize_group_chats()
            
            # Start real-time chat processing
            self.real_time_chat.start_real_time_processing()
            
            self.logger.info(f"Initialized {len(agents)} agents with ag2 integration and real-time chat")
            return agents
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {e}")
            raise
    
    def set_system_status_manager(self, system_status_manager) -> None:
        """Set the system status manager for agent interactions.
        
        Args:
            system_status_manager: SystemStatusManager instance
        """
        self.system_status_manager = system_status_manager
        if self.real_time_chat:
            self.real_time_chat.system_status_manager = system_status_manager
    
    def _create_staff_agents(self) -> List[Agent]:
        """Create staff agents based on configuration.
        
        Returns:
            List of staff agents
        """
        staff_agents = []
        
        for role, count in self.agent_config.staff_count.items():
            for i in range(count):
                if role == AgentRole.PARK_RANGER:
                    agent = self.agent_factory.create_park_ranger(f"Ranger_{i+1}")
                elif role == AgentRole.VETERINARIAN:
                    agent = self.agent_factory.create_veterinarian(f"Dr.Vet_{i+1}")
                elif role == AgentRole.SECURITY:
                    agent = self.agent_factory.create_security_guard(f"Security_{i+1}")
                elif role == AgentRole.MAINTENANCE:
                    agent = self.agent_factory.create_maintenance_worker(f"Maintenance_{i+1}")
                elif role == AgentRole.GUEST_RELATIONS:
                    agent = self.agent_factory.create_guest_relations(f"GuestRel_{i+1}")
                else:
                    continue
                
                staff_agents.append(agent)
        
        self.logger.info(f"Created {len(staff_agents)} staff agents")
        return staff_agents
    
    def _create_visitor_agents(self) -> List[Agent]:
        """Create visitor agents based on configuration.
        
        Returns:
            List of visitor agents
        """
        visitor_agents = []
        
        for i in range(self.agent_config.visitor_count):
            # Create different types of visitors
            agent = self.visitor_factory.create_random_visitor(f"Visitor_{i+1}")
            visitor_agents.append(agent)
        
        self.logger.info(f"Created {len(visitor_agents)} visitor agents")
        return visitor_agents
    
    def _create_dinosaur_agents(self) -> List[Agent]:
        """Create dinosaur agents based on configuration.
        
        Returns:
            List of dinosaur agents
        """
        dinosaur_agents = []
        
        for species, count in self.agent_config.dinosaur_config.items():
            for i in range(count):
                name = f"{species.name.title()}_{i+1}"
                
                if species == DinosaurSpecies.TYRANNOSAURUS_REX:
                    agent = self.dinosaur_factory.create_tyrannosaurus_rex(name)
                elif species == DinosaurSpecies.TRICERATOPS:
                    agent = self.dinosaur_factory.create_triceratops(name)
                elif species == DinosaurSpecies.VELOCIRAPTOR:
                    agent = self.dinosaur_factory.create_velociraptor(name)
                elif species == DinosaurSpecies.BRACHIOSAURUS:
                    agent = self.dinosaur_factory.create_brachiosaurus(name)
                elif species == DinosaurSpecies.STEGOSAURUS:
                    agent = self.dinosaur_factory.create_stegosaurus(name)
                elif species == DinosaurSpecies.PARASAUROLOPHUS:
                    agent = self.dinosaur_factory.create_parasaurolophus(name)
                else:
                    continue
                
                dinosaur_agents.append(agent)
        
        self.logger.info(f"Created {len(dinosaur_agents)} dinosaur agents")
        return dinosaur_agents
    
    def _create_agent_instance(self, agent: Agent) -> DinosaurAgent:
        """Create appropriate agent instance based on role.
        
        Args:
            agent: Agent model
            
        Returns:
            DinosaurAgent instance
        """
        if agent.role in [AgentRole.PARK_RANGER, AgentRole.VETERINARIAN, AgentRole.SECURITY, AgentRole.MAINTENANCE, AgentRole.GUEST_RELATIONS]:
            return self.staff_factory.create_staff_agent(agent)
        elif agent.role == AgentRole.TOURIST:
            return self.visitor_factory.create_tourist_agent(agent)
        elif agent.role == AgentRole.DINOSAUR:
            return self.dinosaur_factory.create_dinosaur_agent(agent)
        else:
            raise ValueError(f"Unknown agent role: {agent.role}")
    
    def _initialize_agent_health(self, agent: Agent) -> Dict[str, Any]:
        """Initialize health monitoring for an agent.
        
        Args:
            agent: Agent to initialize health for
            
        Returns:
            Initial health information
        """
        return {
            "status": "healthy",
            "last_response_time": datetime.now(),
            "response_count": 0,
            "error_count": 0,
            "last_error": None,
            "communication_failures": 0
        }
    
    def _initialize_group_chats(self) -> None:
        """Initialize group chats for different scenarios."""
        try:
            # All agents group chat
            all_agents = list(self.agents.values())
            self.group_chats["all_agents"] = self.ag2_integration.create_group_chat(all_agents)
            self.group_chat_managers["all_agents"] = self.ag2_integration.group_chat_manager
            
            # Staff only group chat
            staff_agents = [agent for agent in all_agents if agent.role in [
                AgentRole.PARK_RANGER, AgentRole.VETERINARIAN, AgentRole.SECURITY, AgentRole.MAINTENANCE, AgentRole.GUEST_RELATIONS
            ]]
            if staff_agents:
                self.ag2_integration.group_chat = None  # Reset for new group
                self.group_chats["staff"] = self.ag2_integration.create_group_chat(staff_agents)
                self.group_chat_managers["staff"] = self.ag2_integration.group_chat_manager
            
            # Emergency response team (rangers, security, veterinarians)
            emergency_agents = [agent for agent in all_agents if agent.role in [
                AgentRole.PARK_RANGER, AgentRole.VETERINARIAN, AgentRole.SECURITY, AgentRole.GUEST_RELATIONS
            ]]
            if emergency_agents:
                self.ag2_integration.group_chat = None  # Reset for new group
                self.group_chats["emergency"] = self.ag2_integration.create_group_chat(emergency_agents)
                self.group_chat_managers["emergency"] = self.ag2_integration.group_chat_manager
            
            self.logger.info(f"Initialized {len(self.group_chats)} group chats")
        
        except Exception as e:
            self.logger.error(f"Error initializing group chats: {e}")
            raise
    
    def get_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all agents.
        
        Returns:
            Dictionary mapping agent_id to agent state information
        """
        agent_states = {}
        
        for agent_id, agent in self.agents.items():
            agent_instance = self.agent_instances.get(agent_id)
            health_info = self.agent_health.get(agent_id, {})
            
            agent_states[agent_id] = {
                "id": agent.id,
                "name": agent.name,
                "role": agent.role.name,
                "current_state": agent.current_state.name,
                "location": {
                    "x": agent.location.x,
                    "y": agent.location.y,
                    "zone": agent.location.zone,
                    "description": agent.location.description
                },
                "personality_traits": agent.personality_traits,
                "capabilities": agent.capabilities,
                "species": agent.species.name if agent.species else None,
                "health": health_info,
                "last_activity": agent.last_activity.isoformat(),
                "is_responsive": health_info.get("status") == "healthy"
            }
            
            # Add agent-specific information
            if agent_instance:
                if hasattr(agent_instance, 'satisfaction_level'):
                    agent_states[agent_id]["satisfaction_level"] = agent_instance.satisfaction_level
                if hasattr(agent_instance, 'happiness_level'):
                    agent_states[agent_id]["happiness_level"] = agent_instance.happiness_level
                if hasattr(agent_instance, 'stress_level'):
                    agent_states[agent_id]["stress_level"] = agent_instance.stress_level
        
        return agent_states
    
    def broadcast_event(self, event: Event) -> Dict[str, Any]:
        """Broadcast an event to all relevant agents using real-time chat.
        
        Args:
            event: Event to broadcast
            
        Returns:
            Dictionary with broadcast results and agent responses
        """
        self.logger.info(f"Broadcasting event {event.id} ({event.type.name}) to agents via real-time chat")
        print(f"DEBUG: Broadcasting event {event.id} to agents via real-time chat")  # Debug print
        
        try:
            # Determine affected agents based on event type and location
            affected_agents = self._determine_affected_agents(event)
            
            # Use real-time chat system for immediate responses
            self.real_time_chat.trigger_event_response(event, affected_agents)
            
            # Get immediate responses from real-time chat
            event_messages = self.real_time_chat.get_messages_for_event(event.id)
            
            # Convert to legacy format for compatibility
            individual_responses = {}
            for msg in event_messages:
                if msg.message_type == "event_response":
                    individual_responses[msg.sender_id] = msg.content
            
            # Record communication for tracking
            self._record_event_communication(event, individual_responses, None)
            
            return {
                "event_id": event.id,
                "affected_agents": affected_agents,
                "individual_responses": individual_responses,
                "real_time_messages": len(event_messages),
                "broadcast_time": datetime.now().isoformat(),
                "method": "real_time_chat"
            }
        
        except Exception as e:
            self.logger.error(f"Error broadcasting event {event.id}: {e}")
            return {
                "event_id": event.id,
                "error": str(e),
                "broadcast_time": datetime.now().isoformat()
            }
    
    def _determine_affected_agents(self, event: Event) -> List[str]:
        """Determine which agents are affected by an event.
        
        Args:
            event: Event to analyze
            
        Returns:
            List of affected agent IDs
        """
        affected_agents = []
        
        # All agents are potentially affected by high-severity events
        if event.severity >= 8:
            return list(self.agents.keys())
        
        # Event type specific logic
        if event.type.name.startswith("DINOSAUR_"):
            # Dinosaur events affect staff and nearby visitors
            affected_agents.extend([
                agent_id for agent_id, agent in self.agents.items()
                if agent.role in [AgentRole.PARK_RANGER, AgentRole.VETERINARIAN, AgentRole.SECURITY, AgentRole.GUEST_RELATIONS]
            ])
            
            # Add nearby visitors and dinosaurs
            nearby_agents = self._get_agents_near_location(event.location, radius=50.0)
            affected_agents.extend(nearby_agents)
        
        elif event.type.name.startswith("VISITOR_"):
            # Visitor events affect staff and other visitors
            affected_agents.extend([
                agent_id for agent_id, agent in self.agents.items()
                if agent.role in [AgentRole.PARK_RANGER, AgentRole.SECURITY, AgentRole.GUEST_RELATIONS]
            ])
            
            # Add nearby agents
            nearby_agents = self._get_agents_near_location(event.location, radius=30.0)
            affected_agents.extend(nearby_agents)
        
        elif event.type.name.startswith("FACILITY_"):
            # Facility events affect maintenance and security
            affected_agents.extend([
                agent_id for agent_id, agent in self.agents.items()
                if agent.role in [AgentRole.MAINTENANCE, AgentRole.SECURITY]
            ])
        
        elif event.type.name.startswith("WEATHER_"):
            # Weather events affect all outdoor agents
            affected_agents = list(self.agents.keys())
        
        # Remove duplicates and ensure affected agents exist
        affected_agents = list(set(affected_agents))
        affected_agents = [agent_id for agent_id in affected_agents if agent_id in self.agents]
        
        return affected_agents
    
    def _get_agents_near_location(self, location: Location, radius: float) -> List[str]:
        """Get agents within a certain radius of a location.
        
        Args:
            location: Center location
            radius: Search radius
            
        Returns:
            List of agent IDs within radius
        """
        nearby_agents = []
        
        for agent_id, agent in self.agents.items():
            # Simple distance calculation
            distance = ((agent.location.x - location.x) ** 2 + (agent.location.y - location.y) ** 2) ** 0.5
            if distance <= radius:
                nearby_agents.append(agent_id)
        
        return nearby_agents
    
    def _create_event_message(self, event: Event) -> str:
        """Create a message describing the event for agents.
        
        Args:
            event: Event to describe
            
        Returns:
            Event description message
        """
        severity_desc = "minor" if event.severity <= 3 else "moderate" if event.severity <= 6 else "major" if event.severity <= 8 else "critical"
        
        message = f"ALERT: {severity_desc.upper()} {event.type.name.replace('_', ' ').title()} at {event.location.zone}"
        
        if event.description:
            message += f" - {event.description}"
        
        if event.parameters:
            details = ", ".join([f"{k}: {v}" for k, v in event.parameters.items()])
            message += f" Details: {details}"
        
        return message
    
    def _send_event_to_agent(self, agent_id: str, event_message: str, event: Event) -> Optional[str]:
        """Send event notification to a specific agent.
        
        Args:
            agent_id: ID of the agent
            event_message: Event message
            event: Event object
            
        Returns:
            Agent's response or None if failed
        """
        try:
            agent_instance = self.agent_instances.get(agent_id)
            if not agent_instance:
                self.logger.warning(f"Agent instance not found for {agent_id}")
                return None
            
            # Create event context
            event_context = {
                "event_id": event.id,
                "event_type": event.type.name,
                "severity": event.severity,
                "location": event.location.zone,
                "timestamp": event.timestamp.isoformat()
            }
            
            # Send event notification
            response = agent_instance.handle_event_notification(event_message, event_context)
            
            # Update agent health
            self._update_agent_health(agent_id, success=True)
            
            # Update agent's last activity
            self.agents[agent_id].last_activity = datetime.now()
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error sending event to agent {agent_id}: {e}")
            self._update_agent_health(agent_id, success=False, error=str(e))
            return None
    
    def _initiate_group_event_response(self, event: Event, affected_agents: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Initiate a group conversation for event response.
        
        Args:
            event: Event that triggered the group response
            affected_agents: List of affected agent IDs
            
        Returns:
            Group conversation messages or None if failed
        """
        try:
            # Determine appropriate group chat
            group_name = "emergency" if event.severity >= 7 else "staff"
            
            # Filter affected agents to those in the group
            if group_name == "emergency":
                group_agents = [
                    agent_id for agent_id in affected_agents
                    if self.agents[agent_id].role in [AgentRole.PARK_RANGER, AgentRole.VETERINARIAN, AgentRole.SECURITY, AgentRole.GUEST_RELATIONS]
                ]
            else:
                group_agents = [
                    agent_id for agent_id in affected_agents
                    if self.agents[agent_id].role in [AgentRole.PARK_RANGER, AgentRole.VETERINARIAN, AgentRole.SECURITY, AgentRole.MAINTENANCE, AgentRole.GUEST_RELATIONS]
                ]
            
            if not group_agents:
                return None
            
            # Create group chat if needed
            if group_name not in self.group_chats:
                agents_for_group = [self.agents[agent_id] for agent_id in group_agents]
                self.group_chats[group_name] = self.ag2_integration.create_group_chat(agents_for_group)
                self.group_chat_managers[group_name] = self.ag2_integration.group_chat_manager
            
            # Create group discussion prompt
            group_message = (
                f"EMERGENCY COORDINATION: {event.type.name.replace('_', ' ').title()} "
                f"(Severity {event.severity}) at {event.location.zone}. "
                f"Coordinate your response and determine action plan. "
                f"Each agent should state their role and intended actions."
            )
            
            # Initiate group conversation
            messages = self.ag2_integration.initiate_group_conversation(group_message, "EventSystem")
            
            self.logger.info(f"Initiated group response for event {event.id} with {len(group_agents)} agents")
            return messages
        
        except Exception as e:
            self.logger.error(f"Error initiating group event response: {e}")
            return None
    
    def _record_event_communication(self, event: Event, individual_responses: Dict[str, Any], 
                                   group_response: Optional[List[Dict[str, Any]]]) -> None:
        """Record event communication in history.
        
        Args:
            event: Event that was communicated
            individual_responses: Individual agent responses
            group_response: Group conversation messages
        """
        communication_record = {
            "timestamp": datetime.now().isoformat(),
            "event_id": event.id,
            "event_type": event.type.name,
            "individual_responses": individual_responses,
            "group_response": group_response,
            "participants": list(individual_responses.keys())
        }
        
        self.conversation_history.append(communication_record)
        print(f"DEBUG: Recorded conversation for event {event.id}, {len(individual_responses)} responses")  # Debug print
        
        # Keep only last 100 communications
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
    
    def _update_agent_health(self, agent_id: str, success: bool, error: Optional[str] = None) -> None:
        """Update agent health monitoring information.
        
        Args:
            agent_id: ID of the agent
            success: Whether the operation was successful
            error: Error message if operation failed
        """
        if agent_id not in self.agent_health:
            self.agent_health[agent_id] = self._initialize_agent_health(self.agents[agent_id])
        
        health_info = self.agent_health[agent_id]
        health_info["last_response_time"] = datetime.now()
        
        if success:
            health_info["response_count"] += 1
            health_info["communication_failures"] = 0  # Reset failure count
            if health_info["status"] != "healthy":
                health_info["status"] = "healthy"
                self.logger.info(f"Agent {agent_id} health restored to healthy")
        else:
            health_info["error_count"] += 1
            health_info["communication_failures"] += 1
            health_info["last_error"] = error
            
            # Update status based on failure count
            if health_info["communication_failures"] >= 3:
                health_info["status"] = "unhealthy"
                self.logger.warning(f"Agent {agent_id} marked as unhealthy after {health_info['communication_failures']} failures")
            elif health_info["communication_failures"] >= 1:
                health_info["status"] = "degraded"
    
    def get_agent_conversations(self) -> List[Dict[str, Any]]:
        """Get conversation history for all agents.
        
        Returns:
            List of conversation records
        """
        return self.conversation_history.copy()
    
    def get_agent_conversation_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of conversation messages for the agent
        """
        if agent_id not in self.agent_instances:
            return []
        
        return self.ag2_integration.get_agent_conversation_history(agent_id)
    
    def send_message_to_agent(self, agent_id: str, message: str, sender_name: str = "System") -> Optional[str]:
        """Send a direct message to a specific agent.
        
        Args:
            agent_id: ID of the target agent
            message: Message to send
            sender_name: Name of the message sender
            
        Returns:
            Agent's response or None if failed
        """
        try:
            response = self.ag2_integration.send_message_to_agent(agent_id, message, sender_name)
            
            if response:
                self._update_agent_health(agent_id, success=True)
                self.agents[agent_id].last_activity = datetime.now()
            else:
                self._update_agent_health(agent_id, success=False, error="No response received")
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error sending message to agent {agent_id}: {e}")
            self._update_agent_health(agent_id, success=False, error=str(e))
            return None
    
    def initiate_agent_conversation(self, agent_ids: List[str], initial_message: str, 
                                   sender_name: str = "System") -> List[Dict[str, Any]]:
        """Initiate a conversation between specific agents.
        
        Args:
            agent_ids: List of agent IDs to include in conversation
            initial_message: Message to start the conversation
            sender_name: Name of the message sender
            
        Returns:
            List of conversation messages
        """
        try:
            # Create temporary group chat with specified agents
            agents_for_chat = [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
            
            if len(agents_for_chat) < 2:
                self.logger.warning("Need at least 2 agents for conversation")
                return []
            
            # Create temporary group chat
            temp_group_chat = self.ag2_integration.create_group_chat(agents_for_chat)
            
            # Initiate conversation
            messages = self.ag2_integration.initiate_group_conversation(initial_message, sender_name)
            
            self.logger.info(f"Initiated conversation between {len(agent_ids)} agents")
            return messages
        
        except Exception as e:
            self.logger.error(f"Error initiating agent conversation: {e}")
            return []
    
    def check_agent_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health status of all agents.
        
        Returns:
            Dictionary with agent health information
        """
        current_time = datetime.now()
        health_report = {}
        
        for agent_id, health_info in self.agent_health.items():
            # Check if agent has been unresponsive for too long
            time_since_response = (current_time - health_info["last_response_time"]).total_seconds()
            
            if time_since_response > 300:  # 5 minutes
                health_info["status"] = "unresponsive"
            
            health_report[agent_id] = {
                "agent_name": self.agents[agent_id].name,
                "status": health_info["status"],
                "response_count": health_info["response_count"],
                "error_count": health_info["error_count"],
                "communication_failures": health_info["communication_failures"],
                "last_response_time": health_info["last_response_time"].isoformat(),
                "time_since_response": time_since_response,
                "last_error": health_info.get("last_error")
            }
        
        self.last_health_check = current_time
        return health_report
    
    def reset_agent_conversations(self) -> None:
        """Reset conversation history for all agents."""
        self.ag2_integration.reset_agent_conversations()
        self.conversation_history.clear()
        self.logger.info("Reset conversation history for all agents")
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent model or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_real_time_chat(self) -> RealTimeAgentChat:
        """Get the real-time chat system.
        
        Returns:
            RealTimeAgentChat instance
        """
        return self.real_time_chat
    
    def get_agents_by_role(self, role: AgentRole) -> List[Agent]:
        """Get all agents with a specific role.
        
        Args:
            role: Agent role to filter by
            
        Returns:
            List of agents with the specified role
        """
        return [agent for agent in self.agents.values() if agent.role == role]
    
    def check_event_resolution_by_agents(self, event: Event) -> Dict[str, Any]:
        """Check if agents have indicated that an event is resolved.
        
        Args:
            event: Event to check resolution for
            
        Returns:
            Dictionary with resolution status information
        """
        resolution_info = {
            "event_id": event.id,
            "agents_responding": [],
            "resolution_indicators": [],
            "escalation_indicators": [],
            "overall_status": "in_progress"
        }
        
        # Get recent conversations related to this event
        recent_conversations = self.get_agent_conversations()
        
        # Look for event-related conversations
        event_conversations = []
        for conv in recent_conversations:
            if conv.get("event_id") == event.id:
                event_conversations.append(conv)
        
        if not event_conversations:
            resolution_info["overall_status"] = "no_response"
            return resolution_info
        
        # Analyze agent responses for resolution indicators
        resolution_keywords = [
            "resolved", "handled", "completed", "fixed", "secured", "contained",
            "treated", "repaired", "evacuated", "safe", "under control", "situation normal"
        ]
        
        escalation_keywords = [
            "need help", "backup required", "escalate", "emergency", "critical",
            "cannot handle", "overwhelmed", "failed", "assistance needed", "need backup", "need immediate", "backup"
        ]
        
        for conv in event_conversations:
            individual_responses = conv.get("individual_responses", {})
            
            for agent_id, response in individual_responses.items():
                if response and isinstance(response, str):
                    response_lower = response.lower()
                    
                    resolution_info["agents_responding"].append(agent_id)
                    
                    # Check for resolution indicators
                    found_resolution = [kw for kw in resolution_keywords if kw in response_lower]
                    if found_resolution:
                        resolution_info["resolution_indicators"].extend(found_resolution)
                    
                    # Check for escalation indicators
                    found_escalation = [kw for kw in escalation_keywords if kw in response_lower]
                    if found_escalation:
                        resolution_info["escalation_indicators"].extend(found_escalation)
        
        # Determine overall status
        if resolution_info["escalation_indicators"]:
            resolution_info["overall_status"] = "needs_escalation"
        elif resolution_info["resolution_indicators"]:
            # Check if enough agents indicate resolution
            responding_agents = len(set(resolution_info["agents_responding"]))
            if responding_agents >= 2 or len(resolution_info["resolution_indicators"]) >= 3:
                resolution_info["overall_status"] = "resolved"
            else:
                resolution_info["overall_status"] = "partial_resolution"
        
        return resolution_info
    
    def get_agent_event_responses(self, event_id: str) -> Dict[str, str]:
        """Get agent responses for a specific event.
        
        Args:
            event_id: ID of the event
            
        Returns:
            Dictionary mapping agent IDs to their responses
        """
        responses = {}
        
        # Look through conversation history for event-related responses
        for conv in self.conversation_history:
            if conv.get("event_id") == event_id:
                individual_responses = conv.get("individual_responses", {})
                responses.update(individual_responses)
        
        return responses
    
    def notify_event_resolution(self, event: Event) -> None:
        """Notify agents that an event has been resolved.
        
        Args:
            event: Event that was resolved
        """
        resolution_message = (
            f"EVENT RESOLVED: {event.type.name.replace('_', ' ').title()} "
            f"at {event.location.zone} has been resolved. "
            f"Status: {event.resolution_status.name}. Thank you for your response."
        )
        
        # Notify all agents who were involved in the event
        for agent_id in event.affected_agents:
            try:
                self.send_message_to_agent(agent_id, resolution_message, "EventSystem")
            except Exception as e:
                self.logger.error(f"Error notifying agent {agent_id} of event resolution: {e}")
        
        self.logger.info(f"Notified {len(event.affected_agents)} agents of event {event.id} resolution")
    
    def escalate_event_to_agents(self, event: Event) -> Dict[str, Any]:
        """Escalate an event to additional agents or higher authority.
        
        Args:
            event: Event to escalate
            
        Returns:
            Dictionary with escalation results
        """
        escalation_message = (
            f"ESCALATED EVENT: {event.type.name.replace('_', ' ').title()} "
            f"(Severity {event.severity}) at {event.location.zone} requires immediate attention. "
            f"Previous response attempts have indicated need for escalation."
        )
        
        # Get all available agents for escalation
        all_agents = list(self.agents.keys())
        
        # Prioritize staff agents for escalation
        staff_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.role in [AgentRole.PARK_RANGER, AgentRole.VETERINARIAN, AgentRole.SECURITY]
        ]
        
        escalation_targets = staff_agents if staff_agents else all_agents
        
        escalation_responses = {}
        for agent_id in escalation_targets:
            try:
                response = self.send_message_to_agent(agent_id, escalation_message, "EventSystem")
                escalation_responses[agent_id] = response
            except Exception as e:
                self.logger.error(f"Error escalating to agent {agent_id}: {e}")
                escalation_responses[agent_id] = f"Error: {str(e)}"
        
        # Record escalation in conversation history
        escalation_record = {
            "timestamp": datetime.now().isoformat(),
            "event_id": event.id,
            "event_type": event.type.name,
            "escalation_responses": escalation_responses,
            "escalation_targets": escalation_targets,
            "escalation_reason": "Agent responses indicated need for escalation"
        }
        
        self.conversation_history.append(escalation_record)
        
        return {
            "event_id": event.id,
            "escalation_targets": escalation_targets,
            "escalation_responses": escalation_responses,
            "escalation_time": datetime.now().isoformat()
        }
    
    def get_agents_by_role(self, role: AgentRole) -> List[Agent]:
        """Get all agents with a specific role.
        
        Args:
            role: Agent role to filter by
            
        Returns:
            List of agents with the specified role
        """
        return [agent for agent in self.agents.values() if agent.role == role]
    
    def get_agents_by_location(self, zone: str) -> List[Agent]:
        """Get all agents in a specific location zone.
        
        Args:
            zone: Location zone to filter by
            
        Returns:
            List of agents in the specified zone
        """
        return [agent for agent in self.agents.values() if agent.location.zone == zone]
    
    def update_agent_location(self, agent_id: str, new_location: Location) -> bool:
        """Update an agent's location.
        
        Args:
            agent_id: ID of the agent
            new_location: New location for the agent
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            agent_instance = self.agent_instances.get(agent_id)
            
            # Update model
            agent.location = new_location
            agent.last_activity = datetime.now()
            
            # Update instance if available
            if agent_instance:
                agent_instance.update_location(new_location)
            
            self.logger.info(f"Updated location for agent {agent_id} to {new_location.zone}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error updating agent location: {e}")
            return False
    
    def update_agent_state(self, agent_id: str, new_state: AgentState) -> bool:
        """Update an agent's state.
        
        Args:
            agent_id: ID of the agent
            new_state: New state for the agent
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            agent_instance = self.agent_instances.get(agent_id)
            
            # Update model
            agent.current_state = new_state
            agent.last_activity = datetime.now()
            
            # Update instance if available
            if agent_instance:
                agent_instance.update_state(new_state)
            
            self.logger.info(f"Updated state for agent {agent_id} to {new_state.name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error updating agent state: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status.
        
        Returns:
            Dictionary with system status information
        """
        total_agents = len(self.agents)
        healthy_agents = sum(1 for health in self.agent_health.values() if health["status"] == "healthy")
        
        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "unhealthy_agents": total_agents - healthy_agents,
            "group_chats": len(self.group_chats),
            "conversation_records": len(self.conversation_history),
            "last_health_check": self.last_health_check.isoformat(),
            "ag2_integration_status": "active" if self.ag2_integration else "inactive"
        }