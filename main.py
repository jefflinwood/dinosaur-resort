"""Main entry point for the AI Agent Dinosaur Simulator."""

import streamlit as st
from datetime import datetime
from typing import Dict, Any
import logging

from ui.session_state import SessionStateManager
from models.core import SimulationState


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="AI Agent Dinosaur Simulator",
        page_icon="🦕",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_sidebar(session_manager: SessionStateManager) -> str:
    """Render the sidebar navigation and controls.
    
    Args:
        session_manager: Session state manager instance
        
    Returns:
        Selected page/section
    """
    st.sidebar.title("🦕 Dinosaur Resort")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Dashboard", "Control Panel", "Agent Monitor", "Metrics", "Event Log", "Settings"],
        index=0
    )
    
    st.sidebar.divider()
    
    # Simulation status
    sim_state = session_manager.get_simulation_state()
    status_color = "🟢" if sim_state.is_running else "🔴"
    st.sidebar.write(f"**Status:** {status_color} {'Running' if sim_state.is_running else 'Stopped'}")
    
    # Quick stats
    agents = session_manager.get_agents()
    events = session_manager.get_events()
    active_events = [e for e in events if e.resolution_status.name in ['PENDING', 'IN_PROGRESS']]
    
    st.sidebar.metric("Active Agents", len(agents))
    st.sidebar.metric("Active Events", len(active_events))
    st.sidebar.metric("Total Events", len(events))
    
    # Latest metrics
    latest_metrics = session_manager.get_latest_metrics()
    if latest_metrics:
        st.sidebar.write("**Latest Metrics:**")
        st.sidebar.write(f"Visitor Satisfaction: {latest_metrics.visitor_satisfaction:.1f}%")
        st.sidebar.write(f"Safety Rating: {latest_metrics.safety_rating:.1f}%")
        st.sidebar.write(f"Facility Efficiency: {latest_metrics.facility_efficiency:.1f}%")
    
    st.sidebar.divider()
    
    # Quick actions
    st.sidebar.write("**Quick Actions:**")
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear Session"):
        session_manager.clear_all()
        st.rerun()
    
    # Session info (collapsible)
    with st.sidebar.expander("Session Info"):
        session_info = session_manager.get_session_info()
        for key, value in session_info.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    return page


def render_dashboard_overview(session_manager: SessionStateManager):
    """Render the main dashboard overview page.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("🦕 AI Agent Dinosaur Simulator")
    st.write("Welcome to the Dinosaur Resort Simulation Dashboard!")
    
    # Current time and simulation status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Time", 
            datetime.now().strftime('%H:%M:%S'),
            help="Real-time clock"
        )
    
    with col2:
        sim_state = session_manager.get_simulation_state()
        status_text = "Running" if sim_state.is_running else "Stopped"
        st.metric("Simulation Status", status_text)
    
    with col3:
        uptime = "00:00:00"  # Placeholder for actual uptime calculation
        st.metric("Uptime", uptime)
    
    st.divider()
    
    # Main content areas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Agent Overview")
        agents = session_manager.get_agents()
        
        if agents:
            # Group agents by role
            agent_roles = {}
            for agent in agents.values():
                role = agent.role.value if hasattr(agent.role, 'value') else str(agent.role)
                if role not in agent_roles:
                    agent_roles[role] = []
                agent_roles[role].append(agent)
            
            for role, role_agents in agent_roles.items():
                st.write(f"**{role.title()}:** {len(role_agents)} agents")
        else:
            st.info("No agents currently active. Start the simulation to initialize agents.")
    
    with col2:
        st.subheader("📊 Current Metrics")
        latest_metrics = session_manager.get_latest_metrics()
        
        if latest_metrics:
            # Display metrics as progress bars
            st.write("**Visitor Satisfaction**")
            st.progress(latest_metrics.visitor_satisfaction / 100)
            
            st.write("**Safety Rating**")
            st.progress(latest_metrics.safety_rating / 100)
            
            st.write("**Facility Efficiency**")
            st.progress(latest_metrics.facility_efficiency / 100)
        else:
            st.info("No metrics data available. Start the simulation to begin tracking.")
    
    # Recent activity
    st.subheader("📝 Recent Activity")
    events = session_manager.get_events()
    
    if events:
        # Show last 5 events
        recent_events = sorted(events, key=lambda x: x.timestamp, reverse=True)[:5]
        
        for event in recent_events:
            status_icon = {
                'PENDING': '⏳',
                'IN_PROGRESS': '🔄',
                'RESOLVED': '✅',
                'FAILED': '❌'
            }.get(event.resolution_status.name, '❓')
            
            st.write(f"{status_icon} **{event.type.value}** - {event.timestamp.strftime('%H:%M:%S')}")
    else:
        st.info("No recent activity. Trigger events to see them here.")


def render_control_panel(session_manager: SessionStateManager):
    """Render the control panel page.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("🎮 Control Panel")
    st.write("Simulation controls and event triggering interface.")
    
    # Placeholder for future implementation
    st.info("Control panel functionality will be implemented in task 11.")
    
    # Basic simulation controls (placeholder)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start", disabled=True):
            pass
    
    with col2:
        if st.button("⏸️ Pause", disabled=True):
            pass
    
    with col3:
        if st.button("⏹️ Stop", disabled=True):
            pass
    
    with col4:
        if st.button("🔄 Reset", disabled=True):
            pass


def render_agent_monitor(session_manager: SessionStateManager):
    """Render the agent monitoring page.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("🤖 Agent Monitor")
    st.write("Real-time agent status and activity monitoring.")
    
    # Placeholder for future implementation
    st.info("Agent monitoring functionality will be implemented in task 12.")
    
    agents = session_manager.get_agents()
    if agents:
        st.write(f"**Total Agents:** {len(agents)}")
        
        # Simple agent list (placeholder)
        for agent_id, agent in agents.items():
            with st.expander(f"{agent.name} ({agent.role})"):
                st.write(f"**ID:** {agent_id}")
                st.write(f"**Role:** {agent.role}")
                st.write(f"**Location:** {agent.location}")
                st.write(f"**State:** {agent.current_state}")
    else:
        st.info("No agents to monitor. Start the simulation to see agent activity.")


def render_metrics_dashboard(session_manager: SessionStateManager):
    """Render the metrics dashboard page.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("📊 Metrics Dashboard")
    st.write("Resort performance metrics and historical trends.")
    
    # Placeholder for future implementation
    st.info("Metrics visualization will be implemented in task 13.")
    
    # Show current metrics if available
    latest_metrics = session_manager.get_latest_metrics()
    if latest_metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Visitor Satisfaction", f"{latest_metrics.visitor_satisfaction:.1f}%")
        
        with col2:
            st.metric("Safety Rating", f"{latest_metrics.safety_rating:.1f}%")
        
        with col3:
            st.metric("Facility Efficiency", f"{latest_metrics.facility_efficiency:.1f}%")
        
        # Metrics history count
        history = session_manager.get_metrics_history()
        st.write(f"**Historical Data Points:** {len(history)}")
    else:
        st.info("No metrics data available yet.")


def render_event_log(session_manager: SessionStateManager):
    """Render the event log page.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("📝 Event Log")
    st.write("Event history and resolution tracking.")
    
    # Placeholder for future implementation
    st.info("Event logging interface will be implemented in task 14.")
    
    events = session_manager.get_events()
    if events:
        st.write(f"**Total Events:** {len(events)}")
        
        # Simple event list (placeholder)
        for event in sorted(events, key=lambda x: x.timestamp, reverse=True):
            status_icon = {
                'PENDING': '⏳',
                'IN_PROGRESS': '🔄',
                'RESOLVED': '✅',
                'FAILED': '❌'
            }.get(event.resolution_status.name, '❓')
            
            with st.expander(f"{status_icon} {event.type.value} - {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
                st.write(f"**ID:** {event.id}")
                st.write(f"**Type:** {event.type.value}")
                st.write(f"**Severity:** {event.severity}")
                st.write(f"**Status:** {event.resolution_status.name}")
                st.write(f"**Location:** {event.location}")
                if event.affected_agents:
                    st.write(f"**Affected Agents:** {', '.join(event.affected_agents)}")
    else:
        st.info("No events logged yet. Trigger events to see them here.")


def render_settings(session_manager: SessionStateManager):
    """Render the settings page.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("⚙️ Settings")
    st.write("Configuration and system settings.")
    
    # Session management
    st.subheader("Session Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Session"):
            st.rerun()
            st.success("Session refreshed!")
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            session_manager.clear_all()
            st.success("All session data cleared!")
            st.rerun()
    
    # Configuration display
    st.subheader("Current Configuration")
    
    with st.expander("Simulation Config"):
        sim_config = session_manager.get_simulation_config()
        st.json(sim_config.__dict__ if hasattr(sim_config, '__dict__') else str(sim_config))
    
    with st.expander("Agent Config"):
        agent_config = session_manager.get_agent_config()
        st.json(agent_config.__dict__ if hasattr(agent_config, '__dict__') else str(agent_config))
    
    with st.expander("Session Info"):
        session_info = session_manager.get_session_info()
        st.json(session_info)


def main():
    """Main application entry point."""
    # Configure page
    configure_page()
    
    # Initialize session state manager
    session_manager = SessionStateManager()
    
    # Render sidebar and get selected page
    selected_page = render_sidebar(session_manager)
    
    # Render main content based on selected page
    if selected_page == "Dashboard":
        render_dashboard_overview(session_manager)
    elif selected_page == "Control Panel":
        render_control_panel(session_manager)
    elif selected_page == "Agent Monitor":
        render_agent_monitor(session_manager)
    elif selected_page == "Metrics":
        render_metrics_dashboard(session_manager)
    elif selected_page == "Event Log":
        render_event_log(session_manager)
    elif selected_page == "Settings":
        render_settings(session_manager)
    
    # Footer
    st.divider()
    st.caption(f"AI Agent Dinosaur Simulator | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()