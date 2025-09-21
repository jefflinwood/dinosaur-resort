"""Agent chat interface using Streamlit's chat components."""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional
from ui.session_state import SessionStateManager


class AgentChatInterface:
    """Manages the agent chat interface using Streamlit's chat components."""
    
    def __init__(self, session_manager: SessionStateManager):
        """Initialize the chat interface.
        
        Args:
            session_manager: Session state manager instance
        """
        self.session_manager = session_manager
        
        # Initialize chat messages in session state if not exists
        if 'agent_chat_messages' not in st.session_state:
            st.session_state.agent_chat_messages = []
    
    def add_agent_message(self, agent_id: str, agent_name: str, message: str, 
                         event_context: Optional[Dict[str, Any]] = None) -> None:
        """Add a message from an agent to the chat.
        
        Args:
            agent_id: ID of the agent
            agent_name: Display name of the agent
            message: Message content
            event_context: Optional event context
        """
        chat_message = {
            'timestamp': datetime.now(),
            'agent_id': agent_id,
            'agent_name': agent_name,
            'message': message,
            'event_context': event_context or {},
            'type': 'agent'
        }
        
        st.session_state.agent_chat_messages.append(chat_message)
        print(f"DEBUG CHAT: Added agent message from {agent_name}: {message[:50]}... (Total: {len(st.session_state.agent_chat_messages)})")
        
        # Keep only last 100 messages
        if len(st.session_state.agent_chat_messages) > 100:
            st.session_state.agent_chat_messages = st.session_state.agent_chat_messages[-100:]
    
    def add_system_message(self, message: str, event_id: Optional[str] = None) -> None:
        """Add a system message to the chat.
        
        Args:
            message: System message content
            event_id: Optional event ID
        """
        chat_message = {
            'timestamp': datetime.now(),
            'agent_id': 'system',
            'agent_name': 'System',
            'message': message,
            'event_context': {'event_id': event_id} if event_id else {},
            'type': 'system'
        }
        
        st.session_state.agent_chat_messages.append(chat_message)
        print(f"DEBUG CHAT: Added system message: {message[:50]}... (Total: {len(st.session_state.agent_chat_messages)})")
        
        # Keep only last 100 messages
        if len(st.session_state.agent_chat_messages) > 100:
            st.session_state.agent_chat_messages = st.session_state.agent_chat_messages[-100:]
    
    def render_chat(self) -> None:
        """Render the chat interface."""
        st.subheader("💬 Agent Communications")
        
        # Chat controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**Messages:** {len(st.session_state.agent_chat_messages)}")
        
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear Chat"):
                st.session_state.agent_chat_messages = []
                st.rerun()
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.agent_chat_messages:
                # Display messages in reverse chronological order (newest first)
                for msg in reversed(st.session_state.agent_chat_messages[-20:]):  # Show last 20 messages
                    self._render_message(msg)
            else:
                st.info("No agent communications yet. Start the simulation and trigger events to see agent responses.")
    
    def _render_message(self, message: Dict[str, Any]) -> None:
        """Render a single chat message.
        
        Args:
            message: Message dictionary
        """
        timestamp = message['timestamp'].strftime('%H:%M:%S')
        agent_name = message['agent_name']
        content = message['message']
        msg_type = message.get('type', 'agent')
        event_context = message.get('event_context', {})
        
        # Choose avatar and styling based on message type
        if msg_type == 'system':
            avatar = "🤖"
            with st.chat_message("assistant", avatar=avatar):
                st.write(f"**{timestamp}** - {content}")
                if event_context.get('event_id'):
                    st.caption(f"Event: {event_context['event_id'][:8]}...")
        else:
            # Agent message - choose avatar based on agent role/type
            avatar = self._get_agent_avatar(message['agent_id'], agent_name)
            with st.chat_message("user", avatar=avatar):
                st.write(f"**{agent_name}** ({timestamp})")
                st.write(content)
                if event_context.get('event_id'):
                    st.caption(f"Responding to event: {event_context['event_id'][:8]}...")
    
    def _get_agent_avatar(self, agent_id: str, agent_name: str) -> str:
        """Get appropriate avatar for an agent.
        
        Args:
            agent_id: Agent ID
            agent_name: Agent name
            
        Returns:
            Emoji avatar for the agent
        """
        # Try to determine agent type from ID or name
        agent_lower = (agent_id + agent_name).lower()
        
        if 'ranger' in agent_lower:
            return "🌲"
        elif 'vet' in agent_lower or 'doctor' in agent_lower:
            return "🩺"
        elif 'security' in agent_lower:
            return "🛡️"
        elif 'maintenance' in agent_lower:
            return "🔧"
        elif 'visitor' in agent_lower or 'tourist' in agent_lower:
            return "🧳"
        elif any(dino in agent_lower for dino in ['rex', 'raptor', 'ceratops', 'saurus']):
            return "🦕"
        else:
            return "👤"
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent chat messages.
        
        Args:
            count: Number of recent messages to return
            
        Returns:
            List of recent messages
        """
        return st.session_state.agent_chat_messages[-count:] if st.session_state.agent_chat_messages else []
    
    def clear_chat(self) -> None:
        """Clear all chat messages."""
        st.session_state.agent_chat_messages = []


def create_agent_chat_interface(session_manager: SessionStateManager) -> AgentChatInterface:
    """Create and return an agent chat interface.
    
    Args:
        session_manager: Session state manager instance
        
    Returns:
        AgentChatInterface instance
    """
    return AgentChatInterface(session_manager)