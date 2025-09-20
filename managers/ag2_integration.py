"""ag2 framework integration with OpenAI backend for the AI Agent Dinosaur Simulator."""

import logging
from typing import Dict, List, Optional, Any
from autogen import ConversableAgent, GroupChat, GroupChatManager
from models.config import OpenAIConfig, AG2Config
from models.core import Agent
from models.enums import AgentRole, AgentState


class AG2Integration:
    """Manages ag2 framework integration with OpenAI backend."""
    
    def __init__(self, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize ag2 integration with OpenAI configuration.
        
        Args:
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        self.openai_config = openai_config
        self.ag2_config = ag2_config
        self.logger = logging.getLogger(__name__)
        
        # Configure OpenAI for ag2
        self.llm_config = {
            "config_list": [{
                "model": self.openai_config.model,
                "api_key": self.openai_config.api_key,
                "temperature": self.openai_config.temperature,
                "max_tokens": self.openai_config.max_tokens,
                "timeout": self.openai_config.timeout,
            }],
            "timeout": self.openai_config.timeout,
        }
        
        # Store created ag2 agents
        self.ag2_agents: Dict[str, ConversableAgent] = {}
        self.group_chat: Optional[GroupChat] = None
        self.group_chat_manager: Optional[GroupChatManager] = None
        
        self.logger.info("AG2Integration initialized with OpenAI backend")
    
    def create_ag2_agent(self, agent: Agent) -> ConversableAgent:
        """Create an ag2 ConversableAgent from our Agent model.
        
        Args:
            agent: The agent model to convert
            
        Returns:
            ConversableAgent configured for the simulation
        """
        # Generate system message based on agent role and personality
        personality_description = self._generate_personality_description(agent)
        system_message = self.ag2_config.system_message_template.format(
            role=agent.role.name.lower().replace('_', ' '),
            personality_description=personality_description
        )
        
        # Create ag2 agent
        ag2_agent = ConversableAgent(
            name=agent.name,
            system_message=system_message,
            llm_config=self.llm_config,
            human_input_mode=self.ag2_config.human_input_mode,
            max_consecutive_auto_reply=self.ag2_config.max_round,
            code_execution_config=self.ag2_config.code_execution_config,
        )
        
        # Store reference
        self.ag2_agents[agent.id] = ag2_agent
        
        self.logger.info(f"Created ag2 agent for {agent.name} ({agent.role.name})")
        return ag2_agent
    
    def create_group_chat(self, agents: List[Agent]) -> GroupChat:
        """Create a group chat with the provided agents.
        
        Args:
            agents: List of agents to include in the group chat
            
        Returns:
            GroupChat instance for multi-agent conversations
        """
        # Create ag2 agents if they don't exist
        ag2_agents = []
        for agent in agents:
            if agent.id not in self.ag2_agents:
                self.create_ag2_agent(agent)
            ag2_agents.append(self.ag2_agents[agent.id])
        
        # Create group chat
        self.group_chat = GroupChat(
            agents=ag2_agents,
            messages=[],
            max_round=self.ag2_config.max_round,
        )
        
        # Create group chat manager
        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )
        
        self.logger.info(f"Created group chat with {len(ag2_agents)} agents")
        return self.group_chat
    
    def send_message_to_agent(self, agent_id: str, message: str, sender_name: str = "System") -> Optional[str]:
        """Send a message to a specific agent.
        
        Args:
            agent_id: ID of the target agent
            message: Message to send
            sender_name: Name of the message sender
            
        Returns:
            Agent's response or None if agent not found
        """
        if agent_id not in self.ag2_agents:
            self.logger.error(f"Agent {agent_id} not found in ag2_agents")
            return None
        
        try:
            ag2_agent = self.ag2_agents[agent_id]
            # Send message and get response
            response = ag2_agent.generate_reply(
                messages=[{"content": message, "role": "user", "name": sender_name}]
            )
            
            self.logger.info(f"Sent message to {agent_id}, received response")
            return response
        except Exception as e:
            self.logger.error(f"Error sending message to agent {agent_id}: {e}")
            return None
    
    def broadcast_message(self, message: str, sender_name: str = "System") -> Dict[str, Optional[str]]:
        """Broadcast a message to all agents.
        
        Args:
            message: Message to broadcast
            sender_name: Name of the message sender
            
        Returns:
            Dictionary mapping agent_id to their response
        """
        responses = {}
        for agent_id in self.ag2_agents:
            response = self.send_message_to_agent(agent_id, message, sender_name)
            responses[agent_id] = response
        
        self.logger.info(f"Broadcasted message to {len(self.ag2_agents)} agents")
        return responses
    
    def initiate_group_conversation(self, initial_message: str, sender_name: str = "System") -> List[Dict[str, Any]]:
        """Initiate a group conversation with all agents.
        
        Args:
            initial_message: Message to start the conversation
            sender_name: Name of the message sender
            
        Returns:
            List of conversation messages
        """
        if not self.group_chat_manager:
            self.logger.error("Group chat not initialized. Call create_group_chat first.")
            return []
        
        try:
            # Start the group conversation
            chat_result = self.group_chat_manager.initiate_chat(
                message=initial_message,
                recipient=self.group_chat_manager,
            )
            
            # Extract messages from chat result
            messages = []
            if hasattr(chat_result, 'chat_history'):
                messages = chat_result.chat_history
            elif isinstance(chat_result, list):
                messages = chat_result
            
            self.logger.info(f"Group conversation completed with {len(messages)} messages")
            return messages
        except Exception as e:
            self.logger.error(f"Error in group conversation: {e}")
            return []
    
    def get_agent_conversation_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of conversation messages for the agent
        """
        if agent_id not in self.ag2_agents:
            return []
        
        ag2_agent = self.ag2_agents[agent_id]
        # Get chat messages from the agent
        if hasattr(ag2_agent, 'chat_messages'):
            return ag2_agent.chat_messages
        return []
    
    def reset_agent_conversations(self):
        """Reset conversation history for all agents."""
        for ag2_agent in self.ag2_agents.values():
            if hasattr(ag2_agent, 'reset'):
                ag2_agent.reset()
        
        if self.group_chat:
            self.group_chat.messages = []
        
        self.logger.info("Reset conversation history for all agents")
    
    def _generate_personality_description(self, agent: Agent) -> str:
        """Generate personality description for system message.
        
        Args:
            agent: Agent to generate description for
            
        Returns:
            Personality description string
        """
        if not agent.personality_traits:
            return "You have a balanced personality."
        
        traits = []
        for trait, value in agent.personality_traits.items():
            if value > 0.7:
                traits.append(f"very {trait}")
            elif value > 0.5:
                traits.append(f"somewhat {trait}")
        
        if traits:
            return f"You are {', '.join(traits)}."
        return "You have a balanced personality."
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all ag2 agents.
        
        Returns:
            Dictionary with agent status information
        """
        status = {}
        for agent_id, ag2_agent in self.ag2_agents.items():
            status[agent_id] = {
                "name": ag2_agent.name,
                "system_message": ag2_agent.system_message,
                "human_input_mode": ag2_agent.human_input_mode,
                "max_consecutive_auto_reply": ag2_agent.max_consecutive_auto_reply,
                "message_count": len(getattr(ag2_agent, 'chat_messages', [])),
            }
        
        return status