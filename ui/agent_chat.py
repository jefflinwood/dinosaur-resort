"""Agent chat interface using Streamlit's chat components with real-time updates."""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from ui.session_state import SessionStateManager
from managers.real_time_agent_chat import RealTimeAgentChat, QuickChatMessage


class AgentChatInterface:
    """Manages the agent chat interface using Streamlit's chat components with real-time updates."""
    
    def __init__(self, session_manager: SessionStateManager, real_time_chat: Optional[RealTimeAgentChat] = None):
        """Initialize the chat interface.
        
        Args:
            session_manager: Session state manager instance
            real_time_chat: Optional real-time chat system
        """
        self.session_manager = session_manager
        self.real_time_chat = real_time_chat
        
        # Initialize chat messages in session state if not exists
        if 'agent_chat_messages' not in st.session_state:
            st.session_state.agent_chat_messages = []
        
        # Track last sync time for real-time chat polling
        if 'last_chat_sync_time' not in st.session_state:
            st.session_state.last_chat_sync_time = datetime.now()
        
        # Sync messages from real-time chat system
        if self.real_time_chat:
            self._sync_real_time_messages()
    
    def _sync_real_time_messages(self) -> None:
        """Sync messages from real-time chat system to UI.
        
        This polls the real-time chat system for new messages instead of using callbacks
        to avoid threading issues with Streamlit's session state.
        """
        try:
            if not self.real_time_chat:
                return
            
            # Get all messages from real-time chat
            real_time_messages = self.real_time_chat.get_recent_messages(50)  # Get last 50 messages
            
            if not real_time_messages:
                return
            
            # Track which real-time messages we've already processed
            if 'processed_real_time_messages' not in st.session_state:
                st.session_state.processed_real_time_messages = set()
            
            # Convert and add new messages
            new_messages_added = 0
            for quick_message in real_time_messages:
                message_id = quick_message.id
                
                # Skip if we've already processed this message
                if message_id in st.session_state.processed_real_time_messages:
                    continue
                
                # Convert QuickChatMessage to UI format
                chat_message = {
                    'timestamp': quick_message.timestamp,
                    'agent_id': quick_message.sender_id,
                    'agent_name': quick_message.sender_name,
                    'message': quick_message.content,
                    'event_context': {'event_id': quick_message.event_id} if quick_message.event_id else {},
                    'type': 'agent',
                    'priority': quick_message.priority.name,
                    'message_type': quick_message.message_type
                }
                
                st.session_state.agent_chat_messages.append(chat_message)
                st.session_state.processed_real_time_messages.add(message_id)
                new_messages_added += 1
                
                print(f"DEBUG REAL-TIME: Synced message from {quick_message.sender_name}: {quick_message.content[:50]}...")
            
            if new_messages_added > 0:
                print(f"DEBUG REAL-TIME: Synced {new_messages_added} new real-time messages")
                
                # Keep only last 100 messages
                if len(st.session_state.agent_chat_messages) > 100:
                    st.session_state.agent_chat_messages = st.session_state.agent_chat_messages[-100:]
                
                # Clean up processed message tracking (keep only last 200 IDs)
                if len(st.session_state.processed_real_time_messages) > 200:
                    # Keep only the most recent message IDs
                    recent_ids = {msg.id for msg in real_time_messages[-100:]}
                    st.session_state.processed_real_time_messages = recent_ids
        
        except Exception as e:
            print(f"DEBUG REAL-TIME: Error syncing messages: {e}")
            # Don't let sync errors break the UI
    
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
        """Render the chat interface with real-time updates."""
        # Sync real-time messages before rendering
        if self.real_time_chat:
            self._sync_real_time_messages()
        
        st.subheader("💬 Agent Communications")
        
        # Chat controls and status
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.write(f"**Messages:** {len(st.session_state.agent_chat_messages)}")
            if self.real_time_chat:
                stats = self.real_time_chat.get_statistics()
                if stats['is_running']:
                    st.write("🟢 **Real-time:** Active")
                else:
                    st.write("🔴 **Real-time:** Inactive")
        
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear Chat"):
                st.session_state.agent_chat_messages = []
                if self.real_time_chat:
                    self.real_time_chat.clear_chat_history()
                st.rerun()
        
        with col4:
            # Auto-refresh toggle
            auto_refresh = st.checkbox("🔄 Auto", value=True, help="Auto-refresh for real-time updates")
            if auto_refresh:
                # Auto-refresh every 2 seconds when enabled
                time.sleep(0.1)  # Small delay to prevent too frequent updates
                st.rerun()
        
        # Real-time statistics
        if self.real_time_chat:
            stats = self.real_time_chat.get_statistics()
            with st.expander("📊 Real-time Stats", expanded=False):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Active Agents", stats['active_agents'])
                with col_b:
                    st.metric("Queue Size", stats['queue_size'])
                with col_c:
                    st.metric("Total Messages", stats['total_messages'])
        
        # Debug information
        if st.checkbox("🐛 Debug Chat", help="Show debug information about chat messages"):
            st.write(f"**Total messages in session state:** {len(st.session_state.agent_chat_messages)}")
            if st.session_state.agent_chat_messages:
                st.write("**Sample message structure:**")
                sample_msg = st.session_state.agent_chat_messages[-1]
                st.json(sample_msg)
                
                st.write("**Message types:**")
                msg_types = {}
                for msg in st.session_state.agent_chat_messages:
                    msg_type = msg.get('type', 'unknown')
                    msg_types[msg_type] = msg_types.get(msg_type, 0) + 1
                st.write(msg_types)
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.agent_chat_messages:
                messages_to_show = st.session_state.agent_chat_messages[-30:]  # Show last 30 messages
                st.write(f"**Displaying {len(messages_to_show)} of {len(st.session_state.agent_chat_messages)} messages:**")
                
                # Display messages in reverse chronological order (newest first)
                for i, msg in enumerate(reversed(messages_to_show)):
                    try:
                        self._render_message(msg)
                    except Exception as e:
                        st.error(f"Error rendering message {i}: {e}")
                        st.json(msg)  # Show the problematic message
            else:
                st.info("No agent communications yet. Start the simulation and trigger events to see real-time agent responses!")
    
    def _render_message(self, message: Dict[str, Any]) -> None:
        """Render a single chat message with enhanced real-time styling.
        
        Args:
            message: Message dictionary
        """
        try:
            timestamp = message['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif not isinstance(timestamp, datetime):
                # Handle other timestamp formats
                timestamp = datetime.now()
                print(f"DEBUG RENDER: Invalid timestamp format: {message['timestamp']}")
        except Exception as e:
            print(f"DEBUG RENDER: Error parsing timestamp: {e}")
            timestamp = datetime.now()
        
        try:
            time_str = timestamp.strftime('%H:%M:%S')
            agent_name = message.get('agent_name', 'Unknown Agent')
            content = message.get('message', 'No message content')
            msg_type = message.get('type', 'agent')
            event_context = message.get('event_context', {})
            priority = message.get('priority', 'NORMAL')
            message_type = message.get('message_type', 'agent_response')
            
            print(f"DEBUG RENDER: Rendering message from {agent_name}: {content[:30]}...")
            
            # Choose avatar and styling based on message type
            if msg_type == 'system':
                avatar = "🤖"
                with st.chat_message("assistant", avatar=avatar):
                    st.write(f"**{time_str}** - {content}")
                    if event_context.get('event_id'):
                        st.caption(f"Event: {event_context['event_id'][:8]}...")
            else:
                # Agent message - choose avatar based on agent role/type
                avatar = self._get_agent_avatar(message.get('agent_id', ''), agent_name)
                
                # Add priority indicators
                priority_indicator = ""
                if priority == "URGENT":
                    priority_indicator = "🚨 "
                elif priority == "HIGH":
                    priority_indicator = "⚠️ "
                
                # Add message type indicators
                type_indicator = ""
                if message_type == "follow_up":
                    type_indicator = "↳ "
                elif message_type == "event_response":
                    type_indicator = "📢 "
                
                with st.chat_message("user", avatar=avatar):
                    st.write(f"**{priority_indicator}{type_indicator}{agent_name}** ({time_str})")
                    st.write(content)
                    
                    # Show additional context
                    context_parts = []
                    if event_context.get('event_id'):
                        context_parts.append(f"Event: {event_context['event_id'][:8]}...")
                    if priority != 'NORMAL':
                        context_parts.append(f"Priority: {priority}")
                    if message_type != 'agent_response':
                        context_parts.append(f"Type: {message_type.replace('_', ' ').title()}")
                    
                    if context_parts:
                        st.caption(" | ".join(context_parts))
        
        except Exception as e:
            print(f"DEBUG RENDER: Error rendering message: {e}")
            # Fallback: simple message display
            with st.chat_message("user", avatar="💬"):
                st.write(f"**{message.get('agent_name', 'Unknown')}**: {message.get('message', 'Error displaying message')}")
                st.caption(f"Error: {e}")
    
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
        elif 'guest' in agent_lower or 'relations' in agent_lower or 'pr' in agent_lower:
            return "🎭"
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


def create_agent_chat_interface(session_manager: SessionStateManager, 
                               real_time_chat: Optional[RealTimeAgentChat] = None) -> AgentChatInterface:
    """Create and return an agent chat interface with optional real-time chat.
    
    Args:
        session_manager: Session state manager instance
        real_time_chat: Optional real-time chat system
        
    Returns:
        AgentChatInterface instance
    """
    return AgentChatInterface(session_manager, real_time_chat)