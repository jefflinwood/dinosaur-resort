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
    
    # Import simulation manager here to avoid circular imports
    from managers.simulation_manager import SimulationManager
    
    # Initialize simulation manager if not in session state
    if 'simulation_manager' not in st.session_state:
        st.session_state.simulation_manager = SimulationManager(session_manager)
    
    sim_manager = st.session_state.simulation_manager
    sim_state = session_manager.get_simulation_state()
    
    # Real-time status display
    st.subheader("📊 Simulation Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if sim_state.is_running else "🔴"
        status_text = "Running" if sim_state.is_running else "Stopped"
        st.metric("Status", f"{status_color} {status_text}")
    
    with col2:
        uptime = "00:00:00"
        if sim_state.started_at and sim_state.is_running:
            uptime_delta = datetime.now() - sim_state.started_at
            uptime = str(uptime_delta).split('.')[0]  # Remove microseconds
        st.metric("Uptime", uptime)
    
    with col3:
        st.metric("Active Agents", sim_state.agent_count)
    
    with col4:
        active_events = len([e for e in session_manager.get_events() if e.resolution_status.name in ['PENDING', 'IN_PROGRESS']])
        st.metric("Active Events", active_events)
    
    st.divider()
    
    # Simulation control buttons
    st.subheader("🎛️ Simulation Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start", disabled=sim_state.is_running, use_container_width=True):
            try:
                with st.spinner("Starting simulation..."):
                    sim_manager.start_simulation()
                st.success("Simulation started successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start simulation: {str(e)}")
    
    with col2:
        if st.button("⏸️ Pause", disabled=not sim_state.is_running, use_container_width=True):
            try:
                sim_manager.pause_simulation()
                st.success("Simulation paused!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to pause simulation: {str(e)}")
    
    with col3:
        if st.button("⏹️ Stop", disabled=not sim_state.is_running, use_container_width=True):
            try:
                sim_manager.stop_simulation()
                st.success("Simulation stopped!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to stop simulation: {str(e)}")
    
    with col4:
        if st.button("🔄 Reset", use_container_width=True):
            try:
                with st.spinner("Resetting simulation..."):
                    sim_manager.reset_simulation()
                    # Reinitialize simulation manager
                    st.session_state.simulation_manager = SimulationManager(session_manager)
                st.success("Simulation reset successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to reset simulation: {str(e)}")
    
    st.divider()
    
    # Event triggering interface
    st.subheader("🚨 Event Triggering")
    
    if not sim_state.is_running:
        st.warning("⚠️ Start the simulation to trigger events")
        return
    
    # Get available events from event manager
    try:
        if hasattr(sim_manager, 'event_manager') and sim_manager.event_manager:
            available_events = sim_manager.event_manager.get_available_events()
        else:
            # Fallback: create a temporary event manager to get event definitions
            from managers.event_manager import EventManager
            temp_event_manager = EventManager()
            available_events = temp_event_manager.get_available_events()
    except Exception as e:
        st.error(f"Error loading available events: {str(e)}")
        return
    
    # Event selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        event_options = {f"{event['name']} - {event['description']}": event for event in available_events}
        selected_event_display = st.selectbox(
            "Select Event Type",
            options=list(event_options.keys()),
            help="Choose the type of event to trigger"
        )
        selected_event = event_options[selected_event_display]
    
    with col2:
        severity = st.slider(
            "Event Severity",
            min_value=1,
            max_value=10,
            value=selected_event['default_severity'],
            help="1 = Minor, 10 = Critical"
        )
    
    # Dynamic parameter inputs based on selected event
    st.write("**Event Parameters:**")
    
    parameters = {}
    
    # Required parameters
    if selected_event['required_parameters']:
        st.write("*Required Parameters:*")
        for param in selected_event['required_parameters']:
            if param == 'dinosaur_id':
                parameters[param] = st.selectbox(f"{param}", ["T-Rex-001", "Triceratops-001", "Velociraptor-001", "Brachiosaurus-001"])
            elif param == 'visitor_id':
                parameters[param] = st.selectbox(f"{param}", ["Visitor-001", "Visitor-002", "Visitor-003", "Tourist-001"])
            elif param == 'facility_id':
                parameters[param] = st.selectbox(f"{param}", ["Main-Gate", "Visitor-Center", "Enclosure-A", "Enclosure-B", "Power-Station"])
            elif param == 'enclosure_id':
                parameters[param] = st.selectbox(f"{param}", ["Enclosure-A", "Enclosure-B", "Enclosure-C", "Quarantine"])
            elif param == 'symptoms':
                parameters[param] = st.selectbox(f"{param}", ["Lethargy", "Loss of appetite", "Aggressive behavior", "Limping", "Fever"])
            elif param == 'injury_type':
                parameters[param] = st.selectbox(f"{param}", ["Minor cut", "Sprained ankle", "Bruising", "Bite wound", "Fall injury"])
            elif param == 'complaint_type':
                parameters[param] = st.selectbox(f"{param}", ["Poor service", "Dirty facilities", "Long wait times", "Rude staff", "Safety concerns"])
            elif param == 'emergency_type':
                parameters[param] = st.selectbox(f"{param}", ["Medical emergency", "Lost child", "Panic attack", "Allergic reaction", "Heart attack"])
            elif param == 'equipment_type':
                parameters[param] = st.selectbox(f"{param}", ["Security camera", "Electric fence", "Water pump", "Air conditioning", "Emergency alarm"])
            elif param == 'affected_systems':
                parameters[param] = st.multiselect(f"{param}", ["Lighting", "Security", "Climate control", "Water pumps", "Emergency systems"])
            elif param == 'storm_type':
                parameters[param] = st.selectbox(f"{param}", ["Thunderstorm", "Hurricane", "Tornado", "Hailstorm", "Lightning storm"])
            elif param == 'temperature_type':
                parameters[param] = st.selectbox(f"{param}", ["Extreme heat", "Extreme cold", "Sudden temperature drop", "Heat wave"])
            elif param == 'temperature':
                parameters[param] = st.number_input(f"{param} (°F)", min_value=-20, max_value=120, value=75)
            elif param == 'intensity':
                parameters[param] = st.slider(f"{param}", min_value=1, max_value=10, value=5)
            elif param == 'behavior_type':
                parameters[param] = st.selectbox(f"{param}", ["Territorial aggression", "Food aggression", "Mating behavior", "Stress response"])
            elif param == 'event_description':
                parameters[param] = st.text_area(f"{param}", placeholder="Describe the custom event...")
            else:
                parameters[param] = st.text_input(f"{param}")
    
    # Optional parameters
    if selected_event['optional_parameters']:
        with st.expander("Optional Parameters"):
            for param in selected_event['optional_parameters']:
                if param == 'escape_method':
                    parameters[param] = st.selectbox(f"{param}", ["Fence breach", "Gate malfunction", "Climbing", "Digging", "Unknown"], key=f"opt_{param}")
                elif param == 'last_seen_location':
                    parameters[param] = st.selectbox(f"{param}", ["Main path", "Visitor center", "Gift shop", "Parking lot", "Forest area"], key=f"opt_{param}")
                elif param == 'severity_level':
                    parameters[param] = st.selectbox(f"{param}", ["Mild", "Moderate", "Severe", "Critical"], key=f"opt_{param}")
                elif param == 'contagious':
                    parameters[param] = st.checkbox(f"{param}", key=f"opt_{param}")
                elif param == 'trigger_cause':
                    parameters[param] = st.text_input(f"{param}", key=f"opt_{param}")
                elif param == 'threat_level':
                    parameters[param] = st.selectbox(f"{param}", ["Low", "Medium", "High", "Critical"], key=f"opt_{param}")
                elif param == 'injury_severity':
                    parameters[param] = st.selectbox(f"{param}", ["Minor", "Moderate", "Severe", "Life-threatening"], key=f"opt_{param}")
                elif param == 'medical_attention_needed':
                    parameters[param] = st.checkbox(f"{param}", key=f"opt_{param}")
                elif param == 'complaint_details':
                    parameters[param] = st.text_area(f"{param}", key=f"opt_{param}")
                elif param == 'requested_resolution':
                    parameters[param] = st.text_input(f"{param}", key=f"opt_{param}")
                elif param == 'emergency_details':
                    parameters[param] = st.text_area(f"{param}", key=f"opt_{param}")
                elif param == 'immediate_danger':
                    parameters[param] = st.checkbox(f"{param}", key=f"opt_{param}")
                elif param == 'estimated_duration':
                    parameters[param] = st.number_input(f"{param} (minutes)", min_value=1, max_value=480, value=30, key=f"opt_{param}")
                elif param == 'backup_power_available':
                    parameters[param] = st.checkbox(f"{param}", key=f"opt_{param}")
                elif param == 'failure_cause':
                    parameters[param] = st.text_input(f"{param}", key=f"opt_{param}")
                elif param == 'repair_urgency':
                    parameters[param] = st.selectbox(f"{param}", ["Low", "Medium", "High", "Critical"], key=f"opt_{param}")
                elif param == 'duration':
                    parameters[param] = st.number_input(f"{param} (hours)", min_value=0.5, max_value=24.0, value=2.0, step=0.5, key=f"opt_{param}")
                elif param == 'wind_speed':
                    parameters[param] = st.number_input(f"{param} (mph)", min_value=0, max_value=200, value=30, key=f"opt_{param}")
                elif param == 'precipitation':
                    parameters[param] = st.selectbox(f"{param}", ["Light rain", "Heavy rain", "Snow", "Hail", "Sleet"], key=f"opt_{param}")
                elif param == 'affected_areas':
                    parameters[param] = st.multiselect(f"{param}", ["Visitor areas", "Enclosures", "Staff areas", "Parking", "Facilities"], key=f"opt_{param}")
                elif param == 'custom_parameters':
                    parameters[param] = st.text_area(f"{param}", placeholder="Additional custom parameters...", key=f"opt_{param}")
                else:
                    parameters[param] = st.text_input(f"{param}", key=f"opt_{param}")
    
    # Location selection
    st.write("**Event Location:**")
    col1, col2 = st.columns(2)
    
    with col1:
        zone = st.selectbox("Zone", ["main_area", "enclosure_a", "enclosure_b", "visitor_center", "parking", "staff_area"])
    
    with col2:
        location_description = st.text_input("Location Description", value="Event location in the resort")
    
    # Custom event description
    custom_description = st.text_input("Custom Event Description (optional)", 
                                     placeholder="Override default event description...")
    
    # Trigger event button
    if st.button("🚨 Trigger Event", type="primary", use_container_width=True):
        try:
            # Create location object
            from models.config import Location
            location = Location(x=0.0, y=0.0, zone=zone, description=location_description)
            
            # Filter out empty optional parameters
            filtered_parameters = {k: v for k, v in parameters.items() if v not in [None, "", []]}
            
            with st.spinner("Triggering event..."):
                event_id = sim_manager.trigger_event(
                    event_type=selected_event['type'].name,
                    parameters=filtered_parameters,
                    location=location,
                    severity=severity,
                    description=custom_description if custom_description else None
                )
            
            st.success(f"✅ Event triggered successfully! Event ID: {event_id}")
            
            # Show event details
            with st.expander("Event Details"):
                st.write(f"**Event ID:** {event_id}")
                st.write(f"**Type:** {selected_event['name']}")
                st.write(f"**Severity:** {severity}/10")
                st.write(f"**Location:** {zone} - {location_description}")
                st.write(f"**Parameters:** {filtered_parameters}")
            
            # Auto-refresh to show updated status
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to trigger event: {str(e)}")
    
    # Recent events display
    st.divider()
    st.subheader("📝 Recent Events")
    
    events = session_manager.get_events()
    if events:
        # Show last 5 events
        recent_events = sorted(events, key=lambda x: x.timestamp, reverse=True)[:5]
        
        for event in recent_events:
            status_icon = {
                'PENDING': '⏳',
                'IN_PROGRESS': '🔄',
                'RESOLVED': '✅',
                'ESCALATED': '⚠️',
                'FAILED': '❌'
            }.get(event.resolution_status.name, '❓')
            
            severity_color = "🔴" if event.severity >= 8 else "🟡" if event.severity >= 5 else "🟢"
            
            with st.expander(f"{status_icon} {event.type.value} - {event.timestamp.strftime('%H:%M:%S')} {severity_color}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**ID:** {event.id[:8]}...")
                    st.write(f"**Severity:** {event.severity}/10")
                    st.write(f"**Status:** {event.resolution_status.name}")
                
                with col2:
                    st.write(f"**Location:** {event.location.zone}")
                    st.write(f"**Affected Agents:** {len(event.affected_agents)}")
                    if event.resolution_time:
                        duration = event.resolution_time - event.timestamp
                        st.write(f"**Resolution Time:** {str(duration).split('.')[0]}")
                
                if event.parameters:
                    st.write("**Parameters:**")
                    for key, value in event.parameters.items():
                        st.write(f"- {key}: {value}")
    else:
        st.info("No events triggered yet. Use the controls above to trigger your first event!")


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