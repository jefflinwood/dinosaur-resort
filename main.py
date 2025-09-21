"""Main entry point for the AI Agent Dinosaur Simulator."""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
        ["Dashboard", "Control Panel", "Agent Monitor", "Agent Chat", "Metrics", "Event Log", "Settings"],
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
    """Render the main dashboard overview page with real-time updates.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("🦕 AI Agent Dinosaur Simulator")
    st.write("Welcome to the Dinosaur Resort Simulation Dashboard!")
    
    # Apply real-time updates
    try:
        from utils.real_time_sync import apply_real_time_updates, create_real_time_sync_context, display_sync_status_indicator
        
        # Check if component should refresh
        should_refresh = apply_real_time_updates("dashboard_overview", session_manager, refresh_interval=15)
        
        # Create sync context and display status
        sync_context = create_real_time_sync_context(session_manager)
        display_sync_status_indicator(sync_context)
        
        # Auto-refresh if needed
        if should_refresh and st.session_state.get('global_auto_refresh', False):
            st.rerun()
            
    except Exception as e:
        st.warning(f"Real-time sync unavailable: {str(e)}")
    
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
                role_display = role.replace('_', ' ').title() if isinstance(role, str) else str(role)
                st.write(f"**{role_display}:** {len(role_agents)} agents")
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
    
    # Recent activity with chat preview
    st.subheader("📝 Recent Activity")
    
    # Show recent events
    events = session_manager.get_events()
    if events:
        recent_events = sorted(events, key=lambda x: x.timestamp, reverse=True)[:3]
        for event in recent_events:
            status_icon = {
                'PENDING': '⏳',
                'IN_PROGRESS': '🔄',
                'RESOLVED': '✅',
                'FAILED': '❌'
            }.get(event.resolution_status.name, '❓')
            
            st.write(f"{status_icon} **{event.type.value}** - {event.timestamp.strftime('%H:%M:%S')}")
    
    # Show recent agent chat messages
    if 'agent_chat_messages' in st.session_state and st.session_state.agent_chat_messages:
        st.write("**Recent Agent Communications:**")
        recent_messages = st.session_state.agent_chat_messages[-3:]  # Last 3 messages
        
        for msg in reversed(recent_messages):
            if msg.get('type') == 'system':
                st.write(f"🤖 **System:** {msg['message']}")
            else:
                st.write(f"💬 **{msg['agent_name']}:** {msg['message'][:50]}...")
    
    if not events and not st.session_state.get('agent_chat_messages'):
        st.info("No recent activity. Trigger events to see them here.")


def render_control_panel(session_manager: SessionStateManager):
    """Render the control panel page with enhanced real-time integration.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("🎮 Control Panel")
    st.write("Simulation controls and event triggering interface.")
    
    # Import simulation manager here to avoid circular imports
    from managers.simulation_manager import SimulationManager
    
    # Initialize simulation manager with error handling
    try:
        if 'simulation_manager' not in st.session_state:
            st.session_state.simulation_manager = SimulationManager(session_manager)
        
        sim_manager = st.session_state.simulation_manager
        
        # Ensure simulation manager is properly connected to session state
        if sim_manager.session_manager != session_manager:
            sim_manager.session_manager = session_manager
        
    except Exception as e:
        st.error(f"Failed to initialize simulation manager: {str(e)}")
        st.write("**Troubleshooting Steps:**")
        st.write("1. Check that all required environment variables are set (OPENAI_API_KEY)")
        st.write("2. Verify that all dependencies are installed")
        st.write("3. Try refreshing the page or clearing session data")
        
        if st.button("🔄 Retry Initialization"):
            if 'simulation_manager' in st.session_state:
                del st.session_state['simulation_manager']
            st.rerun()
        
        return
    
    # Get current simulation state with error handling
    try:
        sim_state = session_manager.get_simulation_state()
        
        # Update simulation time if running
        if sim_state.is_running and hasattr(sim_manager, 'update_simulation_time'):
            sim_manager.update_simulation_time()
            
    except Exception as e:
        st.error(f"Error getting simulation state: {str(e)}")
        return
    
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
            
            # Add event to chat immediately
            try:
                from ui.agent_chat import create_agent_chat_interface
                
                # Get real-time chat system
                real_time_chat = None
                if hasattr(sim_manager, 'get_real_time_chat'):
                    real_time_chat = sim_manager.get_real_time_chat()
                
                chat_interface = create_agent_chat_interface(session_manager, real_time_chat)
                chat_interface.add_system_message(
                    f"🚨 {selected_event['name'].replace('_', ' ').title()} event triggered (Severity: {severity}/10)",
                    event_id
                )
            except Exception as chat_error:
                st.warning(f"Could not add event to chat: {str(chat_error)}")
            
            # Show event details
            with st.expander("Event Details"):
                st.write(f"**Event ID:** {event_id}")
                st.write(f"**Type:** {selected_event['name']}")
                st.write(f"**Severity:** {severity}/10")
                st.write(f"**Location:** {zone} - {location_description}")
                st.write(f"**Parameters:** {filtered_parameters}")
            
            st.info("🔄 Agent responses will appear in the Agent Monitor chat shortly...")
            
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
    """Render the agent monitoring page with enhanced real-time updates.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("🤖 Agent Monitor")
    st.write("Real-time agent status and activity monitoring.")
    
    # Apply real-time updates for agent monitoring
    try:
        from utils.real_time_sync import apply_real_time_updates, create_real_time_sync_context
        
        # Force sync to capture latest agent conversations
        if 'simulation_manager' in st.session_state:
            sim_manager = st.session_state['simulation_manager']
            if sim_manager and sim_manager.is_running():
                sync_context = create_real_time_sync_context(session_manager)
                sync_results = sync_context["synchronizer"].sync_simulation_data(sim_manager)
        
        # Check if component should refresh (more frequent for agent monitoring)
        should_refresh = apply_real_time_updates("agent_monitor", session_manager, refresh_interval=5)  # Even more frequent
        
        # Auto-refresh if needed
        if should_refresh and st.session_state.get('global_auto_refresh', False):
            st.rerun()
            
    except Exception as e:
        st.warning(f"Real-time sync unavailable: {str(e)}")
    
    # Get agents and simulation state with error handling
    try:
        agents = session_manager.get_agents()
        sim_state = session_manager.get_simulation_state()
    except Exception as e:
        st.error(f"Error retrieving agent data: {str(e)}")
        return
    
    if not agents:
        st.info("No agents to monitor. Start the simulation to see agent activity.")
        return
    
    # Auto-refresh controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.write(f"**Total Agents:** {len(agents)}")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False, help="Automatically refresh agent data every 5 seconds")
    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    with col4:
        # Test button to add a fake conversation for debugging
        if st.button("🧪 Test Chat"):
            try:
                from ui.agent_chat import create_agent_chat_interface
                
                # Get real-time chat system
                real_time_chat = None
                if 'simulation_manager' in st.session_state:
                    sim_manager = st.session_state['simulation_manager']
                    if hasattr(sim_manager, 'get_real_time_chat'):
                        real_time_chat = sim_manager.get_real_time_chat()
                
                chat_interface = create_agent_chat_interface(session_manager, real_time_chat)
                chat_interface.add_agent_message(
                    agent_id='test_ranger',
                    agent_name='Test Ranger',
                    message='This is a test message to verify the chat is working correctly!',
                    event_context={'event_id': 'test_event', 'event_type': 'TEST'}
                )
                st.success("Test message added to chat!")
                st.rerun()
            except Exception as e:
                st.error(f"Test failed: {str(e)}")
    
    # Auto-refresh functionality
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()
    
    st.divider()
    
    # Agent status overview
    st.subheader("📊 Agent Status Overview")
    
    # Group agents by role for overview
    agent_roles = {}
    agent_states = {}
    for agent in agents.values():
        role = agent.role.name
        state = agent.current_state.name
        
        if role not in agent_roles:
            agent_roles[role] = 0
        agent_roles[role] += 1
        
        if state not in agent_states:
            agent_states[state] = 0
        agent_states[state] += 1
    
    # Display role distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Agents by Role:**")
        for role, count in agent_roles.items():
            role_emoji = {
                'PARK_RANGER': '🌲',
                'VETERINARIAN': '🩺',
                'SECURITY': '🛡️',
                'MAINTENANCE': '🔧',
                'TOURIST': '🧳',
                'DINOSAUR': '🦕'
            }.get(role, '👤')
            st.write(f"{role_emoji} {role.replace('_', ' ').title()}: {count}")
    
    with col2:
        st.write("**Agents by State:**")
        for state, count in agent_states.items():
            state_emoji = {
                'IDLE': '😴',
                'ACTIVE': '⚡',
                'RESPONDING_TO_EVENT': '🔄',
                'COMMUNICATING': '💬',
                'UNAVAILABLE': '❌'
            }.get(state, '❓')
            st.write(f"{state_emoji} {state.replace('_', ' ').title()}: {count}")
    
    st.divider()
    
    # Filter and search controls
    st.subheader("🔍 Agent Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        role_filter = st.selectbox(
            "Filter by Role",
            options=["All"] + list(agent_roles.keys()),
            index=0
        )
    
    with col2:
        state_filter = st.selectbox(
            "Filter by State", 
            options=["All"] + list(agent_states.keys()),
            index=0
        )
    
    with col3:
        search_term = st.text_input("Search by Name", placeholder="Enter agent name...")
    
    # Filter agents based on selections
    filtered_agents = {}
    for agent_id, agent in agents.items():
        # Role filter
        if role_filter != "All" and agent.role.name != role_filter:
            continue
        
        # State filter
        if state_filter != "All" and agent.current_state.name != state_filter:
            continue
        
        # Search filter
        if search_term and search_term.lower() not in agent.name.lower():
            continue
        
        filtered_agents[agent_id] = agent
    
    st.write(f"**Showing {len(filtered_agents)} of {len(agents)} agents**")
    
    st.divider()
    
    # Real-time agent status display
    st.subheader("🔴 Real-time Agent Status")
    
    # Get agent manager for health information if available
    agent_health_info = {}
    if 'simulation_manager' in st.session_state:
        try:
            sim_manager = st.session_state['simulation_manager']
            if hasattr(sim_manager, 'agent_manager') and sim_manager.agent_manager:
                agent_health_info = sim_manager.agent_manager.check_agent_health()
        except Exception as e:
            st.warning(f"Could not retrieve agent health information: {str(e)}")
    
    # Display agents in a grid layout
    if filtered_agents:
        # Create columns for grid layout (3 agents per row)
        agents_per_row = 3
        agent_list = list(filtered_agents.items())
        
        for i in range(0, len(agent_list), agents_per_row):
            cols = st.columns(agents_per_row)
            
            for j, (agent_id, agent) in enumerate(agent_list[i:i+agents_per_row]):
                with cols[j]:
                    # Agent card
                    with st.container():
                        # Agent header with status indicator
                        health_info = agent_health_info.get(agent_id, {})
                        health_status = health_info.get('status', 'unknown')
                        
                        status_color = {
                            'healthy': '🟢',
                            'degraded': '🟡', 
                            'unhealthy': '🔴',
                            'unresponsive': '⚫',
                            'unknown': '⚪'
                        }.get(health_status, '⚪')
                        
                        role_emoji = {
                            'PARK_RANGER': '🌲',
                            'VETERINARIAN': '🩺',
                            'SECURITY': '🛡️',
                            'MAINTENANCE': '🔧',
                            'TOURIST': '🧳',
                            'DINOSAUR': '🦕'
                        }.get(agent.role.name, '👤')
                        
                        st.write(f"**{status_color} {role_emoji} {agent.name}**")
                        
                        # Basic info
                        st.write(f"*{agent.role.name.replace('_', ' ').title()}*")
                        st.write(f"**State:** {agent.current_state.name.replace('_', ' ').title()}")
                        st.write(f"**Location:** {agent.location.zone}")
                        
                        # Health info if available
                        if health_info:
                            st.write(f"**Health:** {health_status.title()}")
                            if health_info.get('response_count', 0) > 0:
                                st.write(f"**Responses:** {health_info['response_count']}")
                        
                        # Last activity
                        if hasattr(agent, 'last_activity') and agent.last_activity:
                            time_diff = datetime.now() - agent.last_activity
                            if time_diff.total_seconds() < 60:
                                st.write("**Last Active:** Just now")
                            elif time_diff.total_seconds() < 3600:
                                minutes = int(time_diff.total_seconds() / 60)
                                st.write(f"**Last Active:** {minutes}m ago")
                            else:
                                hours = int(time_diff.total_seconds() / 3600)
                                st.write(f"**Last Active:** {hours}h ago")
                        
                        # View details button
                        if st.button(f"View Details", key=f"details_{agent_id}"):
                            st.session_state[f"selected_agent_{agent_id}"] = True
    else:
        st.info("No agents match the current filters.")
    
    st.divider()
    
    # Agent location and activity tracking interface
    st.subheader("📍 Agent Location & Activity Tracking")
    
    # Location overview
    location_zones = {}
    for agent in filtered_agents.values():
        zone = agent.location.zone
        if zone not in location_zones:
            location_zones[zone] = []
        location_zones[zone].append(agent)
    
    if location_zones:
        st.write("**Agents by Location:**")
        
        for zone, zone_agents in location_zones.items():
            with st.expander(f"📍 {zone.replace('_', ' ').title()} ({len(zone_agents)} agents)"):
                for agent in zone_agents:
                    role_emoji = {
                        'PARK_RANGER': '🌲',
                        'VETERINARIAN': '🩺', 
                        'SECURITY': '🛡️',
                        'MAINTENANCE': '🔧',
                        'TOURIST': '🧳',
                        'DINOSAUR': '🦕'
                    }.get(agent.role.name, '👤')
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"{role_emoji} **{agent.name}**")
                    with col2:
                        st.write(f"*{agent.role.name.replace('_', ' ').title()}*")
                    with col3:
                        st.write(f"*{agent.current_state.name.replace('_', ' ').title()}*")
    
    st.divider()
    
    # Agent Chat Interface
    try:
        from ui.agent_chat import create_agent_chat_interface
        
        # Get real-time chat system
        real_time_chat = None
        if 'simulation_manager' in st.session_state:
            sim_manager = st.session_state['simulation_manager']
            if hasattr(sim_manager, 'get_real_time_chat'):
                real_time_chat = sim_manager.get_real_time_chat()
        
        # Create chat interface with real-time chat
        chat_interface = create_agent_chat_interface(session_manager, real_time_chat)
        
        # Debug info
        if st.checkbox("Show Chat Debug", key="chat_debug"):
            st.write(f"**Chat Messages:** {len(st.session_state.get('agent_chat_messages', []))}")
            if 'simulation_manager' in st.session_state:
                sim_manager = st.session_state['simulation_manager']
                if hasattr(sim_manager, 'agent_manager') and sim_manager.agent_manager:
                    agent_conversations = sim_manager.agent_manager.get_agent_conversations()
                    st.write(f"**Agent Manager Conversations:** {len(agent_conversations)}")
                    if agent_conversations:
                        with st.expander("Raw Conversation Data"):
                            st.json(agent_conversations[-2:])  # Show last 2
        
        # Sync agent conversations to chat
        if 'simulation_manager' in st.session_state:
            sim_manager = st.session_state['simulation_manager']
            if hasattr(sim_manager, 'agent_manager') and sim_manager.agent_manager:
                # Get recent conversations from agent manager
                agent_conversations = sim_manager.agent_manager.get_agent_conversations()
                
                # Track which conversations we've already added to chat
                if 'processed_conversations' not in st.session_state:
                    st.session_state.processed_conversations = set()
                
                # Add new conversations to chat
                for conv in agent_conversations:
                    conv_id = f"{conv.get('event_id', 'unknown')}_{conv.get('timestamp', 'unknown')}"
                    
                    if conv_id not in st.session_state.processed_conversations:
                        # Add system message for the event
                        event_type = conv.get('event_type', 'Unknown Event')
                        event_id = conv.get('event_id', 'unknown')
                        chat_interface.add_system_message(
                            f"🚨 {event_type.replace('_', ' ').title()} event triggered",
                            event_id
                        )
                        
                        # Add individual agent responses
                        individual_responses = conv.get('individual_responses', {})
                        for agent_id, response in individual_responses.items():
                            # Get agent name
                            agent_name = agent_id
                            if agent_id in filtered_agents:
                                agent_name = filtered_agents[agent_id].name
                            
                            chat_interface.add_agent_message(
                                agent_id=agent_id,
                                agent_name=agent_name,
                                message=response,
                                event_context={
                                    'event_id': event_id,
                                    'event_type': event_type
                                }
                            )
                        
                        st.session_state.processed_conversations.add(conv_id)
        
        # Manual sync button for debugging
        if st.button("🔄 Force Sync Chat", key="force_sync_agent_monitor"):
            if 'simulation_manager' in st.session_state:
                sim_manager = st.session_state['simulation_manager']
                if hasattr(sim_manager, 'agent_manager') and sim_manager.agent_manager:
                    agent_conversations = sim_manager.agent_manager.get_agent_conversations()
                    st.write(f"Found {len(agent_conversations)} conversations to sync")
                    
                    # Force sync all conversations (ignore processed check)
                    for i, conv in enumerate(agent_conversations):
                        st.write(f"Processing conversation {i+1}: {conv.get('event_type', 'Unknown')}")
                        
                        # Add system message for the event
                        event_type = conv.get('event_type', 'Unknown Event')
                        event_id = conv.get('event_id', 'unknown')
                        chat_interface.add_system_message(
                            f"🚨 {event_type.replace('_', ' ').title()} event triggered",
                            event_id
                        )
                        
                        # Add individual agent responses
                        individual_responses = conv.get('individual_responses', {})
                        for agent_id, response in individual_responses.items():
                            # Get agent name
                            agent_name = agent_id
                            if agent_id in filtered_agents:
                                agent_name = filtered_agents[agent_id].name
                            
                            chat_interface.add_agent_message(
                                agent_id=agent_id,
                                agent_name=agent_name,
                                message=response,
                                event_context={
                                    'event_id': event_id,
                                    'event_type': event_type
                                }
                            )
                            st.write(f"Added message from {agent_name}: {response[:50]}...")
                    
                    st.success(f"Synced {len(agent_conversations)} conversations!")
                    st.rerun()
        
        # Render the chat interface
        chat_interface.render_chat()
        
    except Exception as e:
        st.error(f"Error loading chat interface: {str(e)}")
        # Fallback to simple display
        st.subheader("💬 Agent Communications")
        st.info("Chat interface unavailable. Check logs for details.")
    
    st.divider()
    
    # Agent conversation history viewer
    st.subheader("📜 Agent Conversation History")
    
    # Agent selection for conversation history
    if filtered_agents:
        selected_agent_id = st.selectbox(
            "Select Agent for Detailed History",
            options=[""] + list(filtered_agents.keys()),
            format_func=lambda x: f"{filtered_agents[x].name} ({filtered_agents[x].role.name})" if x else "Select an agent...",
            key="conversation_agent_select"
        )
        
        if selected_agent_id:
            selected_agent = filtered_agents[selected_agent_id]
            st.write(f"**Conversation History for {selected_agent.name}**")
            
            # Get conversation history from session state
            conversation_history = session_manager.get_conversation_history()
            agent_conversations = conversation_history.get(selected_agent_id, [])
            
            # Also try to get from agent manager if available
            if 'simulation_manager' in st.session_state:
                try:
                    sim_manager = st.session_state['simulation_manager']
                    if hasattr(sim_manager, 'agent_manager') and sim_manager.agent_manager:
                        agent_manager_conversations = sim_manager.agent_manager.get_agent_conversation_history(selected_agent_id)
                        if agent_manager_conversations:
                            agent_conversations.extend(agent_manager_conversations)
                except Exception as e:
                    st.warning(f"Could not retrieve agent manager conversations: {str(e)}")
            
            if agent_conversations:
                # Display conversation messages
                st.write(f"**Total Messages:** {len(agent_conversations)}")
                
                # Show recent messages (last 10)
                recent_messages = agent_conversations[-10:] if len(agent_conversations) > 10 else agent_conversations
                
                for i, message in enumerate(reversed(recent_messages)):
                    with st.expander(f"Message {len(recent_messages) - i} - {message.get('timestamp', 'Unknown time')}"):
                        st.write(f"**From:** {message.get('sender', 'Unknown')}")
                        st.write(f"**To:** {message.get('recipient', 'Unknown')}")
                        st.write(f"**Content:** {message.get('content', message.get('message', 'No content'))}")
                        
                        if message.get('event_context'):
                            st.write(f"**Event Context:** {message['event_context']}")
                
                if len(agent_conversations) > 10:
                    st.info(f"Showing last 10 messages. Total: {len(agent_conversations)} messages.")
            else:
                st.info(f"No conversation history available for {selected_agent.name}.")
    
    # Detailed agent information modals
    for agent_id, agent in filtered_agents.items():
        if st.session_state.get(f"selected_agent_{agent_id}", False):
            with st.expander(f"🔍 Detailed Information - {agent.name}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"- **ID:** {agent_id}")
                    st.write(f"- **Name:** {agent.name}")
                    st.write(f"- **Role:** {agent.role.name.replace('_', ' ').title()}")
                    st.write(f"- **Current State:** {agent.current_state.name.replace('_', ' ').title()}")
                    
                    if agent.species:
                        st.write(f"- **Species:** {agent.species.name.replace('_', ' ').title()}")
                    
                    st.write("**Location:**")
                    st.write(f"- **Zone:** {agent.location.zone}")
                    st.write(f"- **Coordinates:** ({agent.location.x:.1f}, {agent.location.y:.1f})")
                    st.write(f"- **Description:** {agent.location.description}")
                
                with col2:
                    st.write("**Personality Traits:**")
                    if agent.personality_traits:
                        for trait, value in agent.personality_traits.items():
                            st.write(f"- **{trait.replace('_', ' ').title()}:** {value:.2f}")
                    else:
                        st.write("- No personality traits defined")
                    
                    st.write("**Capabilities:**")
                    if agent.capabilities:
                        for capability in agent.capabilities:
                            st.write(f"- {capability}")
                    else:
                        st.write("- No capabilities defined")
                
                # Health information if available
                health_info = agent_health_info.get(agent_id, {})
                if health_info:
                    st.write("**Health & Performance:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Health Status", health_info.get('status', 'Unknown').title())
                    with col2:
                        st.metric("Response Count", health_info.get('response_count', 0))
                    with col3:
                        st.metric("Error Count", health_info.get('error_count', 0))
                    
                    if health_info.get('last_error'):
                        st.write(f"**Last Error:** {health_info['last_error']}")
                
                # Close button
                if st.button(f"Close Details", key=f"close_{agent_id}"):
                    st.session_state[f"selected_agent_{agent_id}"] = False
                    st.rerun()


def render_metrics_dashboard(session_manager: SessionStateManager):
    """Render the metrics dashboard page with real-time data updates.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("📊 Metrics Dashboard")
    st.write("Resort performance metrics and historical trends.")
    
    # Apply real-time updates for metrics
    try:
        from utils.real_time_sync import apply_real_time_updates
        
        # Check if component should refresh
        should_refresh = apply_real_time_updates("metrics_dashboard", session_manager, refresh_interval=20)
        
        # Auto-refresh if needed
        if should_refresh and st.session_state.get('global_auto_refresh', False):
            st.rerun()
            
    except Exception as e:
        st.warning(f"Real-time sync unavailable: {str(e)}")
    
    # Get metrics manager for advanced functionality with error handling
    metrics_manager = None
    try:
        if 'simulation_manager' in st.session_state:
            sim_manager = st.session_state['simulation_manager']
            if hasattr(sim_manager, 'metrics_manager'):
                metrics_manager = sim_manager.metrics_manager
    except Exception as e:
        st.warning(f"Could not access metrics manager: {str(e)}")
    
    # Enhanced control panel
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.write("**Real-time Metrics Dashboard**")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False, help="Automatically refresh metrics every 5 seconds")
    with col3:
        refresh_rate = st.selectbox("Refresh Rate", [5, 10, 30, 60], index=0, help="Auto-refresh interval in seconds")
    with col4:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Auto-refresh functionality with configurable rate
    if auto_refresh:
        import time
        time.sleep(refresh_rate)
        st.rerun()
    
    # Get current metrics
    latest_metrics = session_manager.get_latest_metrics()
    history = session_manager.get_metrics_history()
    
    if not latest_metrics:
        st.info("No metrics data available yet. Start the simulation to begin tracking metrics.")
        return
    
    st.divider()
    
    # Real-time metrics display with current values
    st.subheader("🔴 Real-time Metrics")
    
    # Get metrics summary if metrics manager is available
    if metrics_manager:
        try:
            metrics_summary = metrics_manager.get_metrics_summary()
        except Exception as e:
            st.warning(f"Could not get detailed metrics summary: {str(e)}")
            metrics_summary = None
    else:
        metrics_summary = None
    
    # Alert system for critical metrics
    critical_alerts = []
    if latest_metrics.visitor_satisfaction < 0.3:
        critical_alerts.append("🚨 Critical: Visitor satisfaction is dangerously low!")
    if latest_metrics.safety_rating < 0.4:
        critical_alerts.append("🚨 Critical: Safety rating requires immediate attention!")
    if latest_metrics.facility_efficiency < 0.3:
        critical_alerts.append("🚨 Critical: Facility efficiency is critically low!")
    
    # Display alerts if any
    if critical_alerts:
        st.error("**CRITICAL ALERTS:**")
        for alert in critical_alerts:
            st.error(alert)
        st.divider()
    
    # Main metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        value = latest_metrics.visitor_satisfaction * 100
        status = metrics_summary['visitor_satisfaction']['status'] if metrics_summary else _get_metric_status(latest_metrics.visitor_satisfaction)
        delta = _calculate_metric_delta(history, 'visitor_satisfaction') if len(history) > 1 else None
        
        st.metric(
            "👥 Visitor Satisfaction", 
            f"{value:.1f}%",
            delta=f"{delta:+.1f}%" if delta else None,
            help=f"Status: {status}"
        )
        
        # Status indicator
        status_color = _get_status_color(latest_metrics.visitor_satisfaction)
        st.markdown(f"<div style='text-align: center; color: {status_color}; font-weight: bold;'>{status}</div>", unsafe_allow_html=True)
    
    with col2:
        value = latest_metrics.safety_rating * 100
        status = metrics_summary['safety_rating']['status'] if metrics_summary else _get_metric_status(latest_metrics.safety_rating)
        delta = _calculate_metric_delta(history, 'safety_rating') if len(history) > 1 else None
        
        st.metric(
            "🛡️ Safety Rating", 
            f"{value:.1f}%",
            delta=f"{delta:+.1f}%" if delta else None,
            help=f"Status: {status}"
        )
        
        status_color = _get_status_color(latest_metrics.safety_rating)
        st.markdown(f"<div style='text-align: center; color: {status_color}; font-weight: bold;'>{status}</div>", unsafe_allow_html=True)
    
    with col3:
        value = latest_metrics.facility_efficiency * 100
        status = metrics_summary['facility_efficiency']['status'] if metrics_summary else _get_metric_status(latest_metrics.facility_efficiency)
        delta = _calculate_metric_delta(history, 'facility_efficiency') if len(history) > 1 else None
        
        st.metric(
            "🏭 Facility Efficiency", 
            f"{value:.1f}%",
            delta=f"{delta:+.1f}%" if delta else None,
            help=f"Status: {status}"
        )
        
        status_color = _get_status_color(latest_metrics.facility_efficiency)
        st.markdown(f"<div style='text-align: center; color: {status_color}; font-weight: bold;'>{status}</div>", unsafe_allow_html=True)
    
    with col4:
        # Overall resort score
        if metrics_manager:
            try:
                overall_score = metrics_manager.calculate_overall_resort_score()
                value = overall_score * 100
                status = metrics_summary['overall_score']['status'] if metrics_summary else _get_metric_status(overall_score)
                
                st.metric(
                    "🏆 Overall Score", 
                    f"{value:.1f}%",
                    help=f"Weighted average of all metrics - Status: {status}"
                )
                
                status_color = _get_status_color(overall_score)
                st.markdown(f"<div style='text-align: center; color: {status_color}; font-weight: bold;'>{status}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.metric("🏆 Overall Score", "N/A", help="Could not calculate overall score")
        else:
            st.metric("🏆 Overall Score", "N/A", help="Metrics manager not available")
    
    # Dinosaur happiness overview
    if latest_metrics.dinosaur_happiness:
        st.subheader("🦕 Dinosaur Happiness")
        
        # Calculate average and show individual dinosaurs
        avg_happiness = sum(latest_metrics.dinosaur_happiness.values()) / len(latest_metrics.dinosaur_happiness)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                "Average Happiness", 
                f"{avg_happiness * 100:.1f}%",
                help=f"Average across {len(latest_metrics.dinosaur_happiness)} dinosaurs"
            )
            
            status = _get_metric_status(avg_happiness)
            status_color = _get_status_color(avg_happiness)
            st.markdown(f"<div style='text-align: center; color: {status_color}; font-weight: bold;'>{status}</div>", unsafe_allow_html=True)
        
        with col2:
            # Individual dinosaur happiness
            st.write("**Individual Dinosaur Status:**")
            
            # Create columns for dinosaur display
            dino_cols = st.columns(min(3, len(latest_metrics.dinosaur_happiness)))
            
            for i, (dino_id, happiness) in enumerate(latest_metrics.dinosaur_happiness.items()):
                col_idx = i % len(dino_cols)
                with dino_cols[col_idx]:
                    # Get dinosaur name (simplified from ID)
                    dino_name = dino_id.split('-')[0] if '-' in dino_id else dino_id
                    
                    happiness_pct = happiness * 100
                    status = _get_metric_status(happiness)
                    status_color = _get_status_color(happiness)
                    
                    st.write(f"**{dino_name}**")
                    st.progress(happiness)
                    st.write(f"{happiness_pct:.1f}% - {status}")
    
    st.divider()
    
    # Enhanced metric filtering and time range selection
    st.subheader("📈 Historical Trends & Analysis")
    
    if len(history) < 2:
        st.info("Not enough historical data for trend analysis. Metrics will appear here as the simulation runs.")
        return
    
    # Enhanced filter controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        time_range = st.selectbox(
            "Time Range",
            options=["15m", "1h", "6h", "24h", "7d", "all"],
            index=1,
            help="Select time range for historical data"
        )
    
    with col2:
        metrics_to_show = st.multiselect(
            "Metrics to Display",
            options=["Visitor Satisfaction", "Safety Rating", "Facility Efficiency", "Dinosaur Happiness"],
            default=["Visitor Satisfaction", "Safety Rating"],
            help="Select which metrics to show in charts"
        )
    
    with col3:
        chart_type = st.selectbox(
            "Chart Type",
            options=["Line Chart", "Area Chart", "Bar Chart", "Scatter Plot"],
            index=0,
            help="Select visualization type"
        )
    
    with col4:
        show_trend_lines = st.checkbox(
            "Show Trend Lines",
            value=True,
            help="Display trend lines on charts"
        )
    
    # Filter historical data by time range
    filtered_history = _filter_history_by_timerange(history, time_range)
    
    if not filtered_history:
        st.warning(f"No data available for the selected time range ({time_range})")
        return
    
    st.write(f"**Showing {len(filtered_history)} data points over {time_range}**")
    
    # Historical metrics charts and trend visualization
    chart_data = None
    if metrics_to_show:
        # Prepare data for visualization
        chart_data = _prepare_chart_data(filtered_history, metrics_to_show)
        
        if chart_data is not None and not chart_data.empty:
            # Display the selected chart type
            if chart_type == "Line Chart":
                st.line_chart(chart_data, height=400)
            elif chart_type == "Area Chart":
                st.area_chart(chart_data, height=400)
            elif chart_type == "Bar Chart":
                st.bar_chart(chart_data, height=400)
            elif chart_type == "Scatter Plot":
                st.scatter_chart(chart_data, height=400)
            
            # Enhanced trend analysis with correlation matrix
            if show_trend_lines and len(chart_data.columns) > 1:
                st.subheader("📊 Correlation Analysis")
                try:
                    correlation_matrix = chart_data.corr()
                    st.write("**Metric Correlations:**")
                    
                    # Display correlation insights
                    for i, metric1 in enumerate(correlation_matrix.columns):
                        for j, metric2 in enumerate(correlation_matrix.columns):
                            if i < j:  # Avoid duplicate pairs
                                corr_value = correlation_matrix.iloc[i, j]
                                if abs(corr_value) > 0.5:  # Only show significant correlations
                                    corr_strength = "Strong" if abs(corr_value) > 0.7 else "Moderate"
                                    corr_direction = "positive" if corr_value > 0 else "negative"
                                    st.write(f"• **{metric1}** and **{metric2}**: {corr_strength} {corr_direction} correlation ({corr_value:.2f})")
                except Exception as e:
                    st.info("Correlation analysis requires multiple metrics with sufficient data.")
            
            # Show trend analysis
            st.subheader("📊 Trend Analysis")
            
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                st.write("**Recent Trends:**")
                for metric in metrics_to_show:
                    metric_key = _get_metric_key(metric)
                    if metric_key in chart_data.columns:
                        trend = _calculate_trend(chart_data[metric_key])
                        trend_icon = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                        trend_text = "Improving" if trend > 0 else "Declining" if trend < 0 else "Stable"
                        st.write(f"{trend_icon} **{metric}:** {trend_text} ({trend:+.2f}%)")
            
            with trend_col2:
                st.write("**Statistics:**")
                for metric in metrics_to_show:
                    metric_key = _get_metric_key(metric)
                    if metric_key in chart_data.columns:
                        values = chart_data[metric_key] * 100  # Convert to percentage
                        st.write(f"**{metric}:**")
                        st.write(f"  • Min: {values.min():.1f}%")
                        st.write(f"  • Max: {values.max():.1f}%")
                        st.write(f"  • Avg: {values.mean():.1f}%")
                        st.write(f"  • Std Dev: {values.std():.1f}%")
        else:
            st.warning("No data available for the selected metrics and time range.")
    
    # Advanced metrics analysis section
    st.divider()
    st.subheader("🔬 Advanced Analysis")
    
    if chart_data is not None and not chart_data.empty:
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.write("**Performance Insights:**")
            
            # Calculate performance insights
            for metric in metrics_to_show:
                metric_key = _get_metric_key(metric)
                if metric_key in chart_data.columns:
                    values = chart_data[metric_key] * 100
                    
                    # Calculate volatility
                    volatility = values.std()
                    volatility_level = "High" if volatility > 10 else "Medium" if volatility > 5 else "Low"
                    
                    # Calculate recent performance (last 25% of data)
                    recent_data = values.tail(max(1, len(values) // 4))
                    recent_avg = recent_data.mean()
                    overall_avg = values.mean()
                    
                    performance_trend = "improving" if recent_avg > overall_avg else "declining" if recent_avg < overall_avg else "stable"
                    
                    st.write(f"**{metric}:**")
                    st.write(f"  • Volatility: {volatility_level} ({volatility:.1f}%)")
                    st.write(f"  • Recent trend: {performance_trend}")
                    st.write(f"  • Recent avg: {recent_avg:.1f}% vs Overall: {overall_avg:.1f}%")
        
        with analysis_col2:
            st.write("**Predictive Indicators:**")
            
            # Simple predictive analysis
            for metric in metrics_to_show:
                metric_key = _get_metric_key(metric)
                if metric_key in chart_data.columns:
                    values = chart_data[metric_key] * 100
                    
                    # Calculate moving average trend
                    if len(values) >= 5:
                        recent_ma = values.tail(5).mean()
                        earlier_ma = values.head(max(5, len(values) - 5)).mean()
                        
                        trend_direction = "📈 Upward" if recent_ma > earlier_ma else "📉 Downward" if recent_ma < earlier_ma else "➡️ Stable"
                        trend_strength = abs(recent_ma - earlier_ma)
                        
                        # Predict next value based on trend
                        if trend_strength > 1:
                            predicted_change = (recent_ma - earlier_ma) * 0.5  # Conservative prediction
                            predicted_value = values.iloc[-1] + predicted_change
                            predicted_value = max(0, min(100, predicted_value))  # Clamp to 0-100%
                            
                            st.write(f"**{metric}:**")
                            st.write(f"  • Trend: {trend_direction}")
                            st.write(f"  • Predicted next: {predicted_value:.1f}%")
                        else:
                            st.write(f"**{metric}:** Stable trend")
    
    # Performance benchmarking section
    st.divider()
    st.subheader("🎯 Performance Benchmarks")
    
    benchmark_col1, benchmark_col2 = st.columns(2)
    
    with benchmark_col1:
        st.write("**Industry Standards:**")
        benchmarks = {
            "Visitor Satisfaction": {"excellent": 85, "good": 70, "acceptable": 50},
            "Safety Rating": {"excellent": 95, "good": 85, "acceptable": 70},
            "Facility Efficiency": {"excellent": 90, "good": 75, "acceptable": 60},
            "Dinosaur Happiness": {"excellent": 80, "good": 65, "acceptable": 50}
        }
        
        for metric in metrics_to_show:
            if metric in benchmarks:
                current_value = getattr(latest_metrics, _get_metric_key(metric), 0) * 100
                bench = benchmarks[metric]
                
                if current_value >= bench["excellent"]:
                    status = "🏆 Excellent"
                elif current_value >= bench["good"]:
                    status = "✅ Good"
                elif current_value >= bench["acceptable"]:
                    status = "⚠️ Acceptable"
                else:
                    status = "❌ Below Standard"
                
                st.write(f"**{metric}:** {status}")
                st.write(f"  Current: {current_value:.1f}% | Target: {bench['excellent']}%")
    
    with benchmark_col2:
        st.write("**Goal Tracking:**")
        
        # Allow users to set custom goals
        if 'custom_goals' not in st.session_state:
            st.session_state['custom_goals'] = {}
        
        for metric in metrics_to_show:
            metric_key = _get_metric_key(metric)
            current_value = getattr(latest_metrics, metric_key, 0) * 100
            
            goal_key = f"goal_{metric_key}"
            if goal_key not in st.session_state.get('custom_goals', {}):
                st.session_state['custom_goals'][goal_key] = 80.0  # Default goal
            
            goal_value = st.number_input(
                f"{metric} Goal (%)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.get('custom_goals', {}).get(goal_key, 80.0),
                step=1.0,
                key=f"goal_input_{metric_key}"
            )
            st.session_state['custom_goals'][goal_key] = goal_value
            
            # Calculate progress towards goal
            progress = min(100, (current_value / goal_value) * 100) if goal_value > 0 else 0
            progress_color = "🟢" if progress >= 100 else "🟡" if progress >= 80 else "🔴"
            
            st.write(f"{progress_color} Progress: {progress:.1f}%")
            st.progress(progress / 100)
    
    # Individual dinosaur happiness trends
    if latest_metrics.dinosaur_happiness and "Dinosaur Happiness" in metrics_to_show:
        st.subheader("🦕 Individual Dinosaur Trends")
        
        # Let user select specific dinosaurs to view
        selected_dinosaurs = st.multiselect(
            "Select Dinosaurs",
            options=list(latest_metrics.dinosaur_happiness.keys()),
            default=list(latest_metrics.dinosaur_happiness.keys())[:3],  # Default to first 3
            help="Select specific dinosaurs to view happiness trends"
        )
        
        if selected_dinosaurs and metrics_manager:
            try:
                # Create dinosaur happiness chart
                dino_chart_data = _prepare_dinosaur_chart_data(filtered_history, selected_dinosaurs)
                
                if dino_chart_data is not None and not dino_chart_data.empty:
                    st.line_chart(dino_chart_data, height=300)
                    
                    # Show dinosaur-specific statistics
                    st.write("**Dinosaur Statistics:**")
                    dino_stats_cols = st.columns(len(selected_dinosaurs))
                    
                    for i, dino_id in enumerate(selected_dinosaurs):
                        with dino_stats_cols[i]:
                            if dino_id in dino_chart_data.columns:
                                values = dino_chart_data[dino_id] * 100
                                dino_name = dino_id.split('-')[0] if '-' in dino_id else dino_id
                                
                                st.write(f"**{dino_name}**")
                                st.write(f"Current: {latest_metrics.dinosaur_happiness[dino_id] * 100:.1f}%")
                                st.write(f"Min: {values.min():.1f}%")
                                st.write(f"Max: {values.max():.1f}%")
                                st.write(f"Avg: {values.mean():.1f}%")
                else:
                    st.info("No historical data available for selected dinosaurs.")
            except Exception as e:
                st.warning(f"Could not load dinosaur trend data: {str(e)}")
    
    # Data export option
    st.divider()
    st.subheader("💾 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Current Metrics"):
            try:
                import json
                export_data = {
                    'timestamp': latest_metrics.timestamp.isoformat(),
                    'visitor_satisfaction': latest_metrics.visitor_satisfaction,
                    'safety_rating': latest_metrics.safety_rating,
                    'facility_efficiency': latest_metrics.facility_efficiency,
                    'dinosaur_happiness': latest_metrics.dinosaur_happiness
                }
                
                st.download_button(
                    label="Download Current Metrics (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"metrics_current_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Failed to export current metrics: {str(e)}")
    
    with col2:
        if st.button("📈 Export Historical Data"):
            try:
                import json
                export_data = []
                for metrics in filtered_history:
                    export_data.append({
                        'timestamp': metrics.timestamp.isoformat(),
                        'visitor_satisfaction': metrics.visitor_satisfaction,
                        'safety_rating': metrics.safety_rating,
                        'facility_efficiency': metrics.facility_efficiency,
                        'dinosaur_happiness': metrics.dinosaur_happiness
                    })
                
                st.download_button(
                    label=f"Download Historical Data (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"metrics_history_{time_range}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Failed to export historical data: {str(e)}")


def _get_metric_status(value: float) -> str:
    """Get status description for a metric value.
    
    Args:
        value: Metric value (0.0 to 1.0)
        
    Returns:
        Status string
    """
    if value >= 0.8:
        return "Excellent"
    elif value >= 0.6:
        return "Good"
    elif value >= 0.4:
        return "Fair"
    elif value >= 0.2:
        return "Poor"
    else:
        return "Critical"


def _get_status_color(value: float) -> str:
    """Get color for status display.
    
    Args:
        value: Metric value (0.0 to 1.0)
        
    Returns:
        CSS color string
    """
    if value >= 0.8:
        return "#28a745"  # Green
    elif value >= 0.6:
        return "#17a2b8"  # Blue
    elif value >= 0.4:
        return "#ffc107"  # Yellow
    elif value >= 0.2:
        return "#fd7e14"  # Orange
    else:
        return "#dc3545"  # Red


def _calculate_metric_delta(history: list, metric_name: str) -> Optional[float]:
    """Calculate the change in a metric from the previous measurement.
    
    Args:
        history: List of MetricsSnapshot objects
        metric_name: Name of the metric to calculate delta for
        
    Returns:
        Percentage change from previous measurement, or None if not enough data
    """
    if len(history) < 2:
        return None
    
    try:
        # Get the two most recent measurements
        current = getattr(history[-1], metric_name)
        previous = getattr(history[-2], metric_name)
        
        # Calculate percentage change
        if previous != 0:
            delta = ((current - previous) / previous) * 100
            return delta
        else:
            return None
    except (AttributeError, IndexError):
        return None


def _filter_history_by_timerange(history: list, time_range: str) -> list:
    """Filter metrics history by time range.
    
    Args:
        history: List of MetricsSnapshot objects
        time_range: Time range string ('15m', '1h', '6h', '24h', '7d', 'all')
        
    Returns:
        Filtered list of MetricsSnapshot objects
    """
    if time_range == 'all' or not history:
        return history
    
    from datetime import timedelta
    now = datetime.now()
    
    if time_range == '15m':
        cutoff = now - timedelta(minutes=15)
    elif time_range == '1h':
        cutoff = now - timedelta(hours=1)
    elif time_range == '6h':
        cutoff = now - timedelta(hours=6)
    elif time_range == '24h':
        cutoff = now - timedelta(hours=24)
    elif time_range == '7d':
        cutoff = now - timedelta(days=7)
    else:
        return history
    
    return [m for m in history if m.timestamp >= cutoff]


def _get_metric_key(metric_display_name: str) -> str:
    """Convert display name to metric key.
    
    Args:
        metric_display_name: Display name of the metric
        
    Returns:
        Internal metric key name
    """
    mapping = {
        "Visitor Satisfaction": "visitor_satisfaction",
        "Safety Rating": "safety_rating", 
        "Facility Efficiency": "facility_efficiency",
        "Dinosaur Happiness": "dinosaur_happiness"
    }
    return mapping.get(metric_display_name, metric_display_name.lower().replace(' ', '_'))


def _prepare_chart_data(history: list, metrics_to_show: list):
    """Prepare data for chart visualization.
    
    Args:
        history: List of MetricsSnapshot objects
        metrics_to_show: List of metric display names to include
        
    Returns:
        pandas DataFrame with chart data, or None if pandas not available
    """
    try:
        import pandas as pd
        
        data = []
        for metrics in history:
            row = {'timestamp': metrics.timestamp}
            
            for metric_display in metrics_to_show:
                metric_key = _get_metric_key(metric_display)
                
                if metric_key == 'dinosaur_happiness':
                    # For dinosaur happiness, use average
                    if metrics.dinosaur_happiness:
                        avg_happiness = sum(metrics.dinosaur_happiness.values()) / len(metrics.dinosaur_happiness)
                        row[metric_display] = avg_happiness
                    else:
                        row[metric_display] = 0.0
                else:
                    # For other metrics, get the value directly
                    if hasattr(metrics, metric_key):
                        row[metric_display] = getattr(metrics, metric_key)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        
        return df
        
    except ImportError:
        # Fallback if pandas is not available
        st.warning("Pandas not available for advanced charting. Install pandas for better visualization.")
        return None
    except Exception as e:
        st.error(f"Error preparing chart data: {str(e)}")
        return None


def _prepare_dinosaur_chart_data(history: list, selected_dinosaurs: list):
    """Prepare dinosaur-specific chart data.
    
    Args:
        history: List of MetricsSnapshot objects
        selected_dinosaurs: List of dinosaur IDs to include
        
    Returns:
        pandas DataFrame with dinosaur happiness data, or None if not available
    """
    try:
        import pandas as pd
        
        data = []
        for metrics in history:
            row = {'timestamp': metrics.timestamp}
            
            for dino_id in selected_dinosaurs:
                if dino_id in metrics.dinosaur_happiness:
                    # Use simplified name for display
                    dino_name = dino_id.split('-')[0] if '-' in dino_id else dino_id
                    row[dino_name] = metrics.dinosaur_happiness[dino_id]
            
            # Only add row if it has dinosaur data
            if len(row) > 1:
                data.append(row)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        
        return df
        
    except ImportError:
        st.warning("Pandas not available for dinosaur trend charts.")
        return None
    except Exception as e:
        st.error(f"Error preparing dinosaur chart data: {str(e)}")
        return None


def _calculate_trend(values) -> float:
    """Calculate trend direction for a series of values.
    
    Args:
        values: Series of metric values
        
    Returns:
        Trend percentage (positive = improving, negative = declining)
    """
    try:
        if len(values) < 2:
            return 0.0
        
        # Simple trend calculation: compare first half to second half
        mid_point = len(values) // 2
        first_half_avg = values[:mid_point].mean()
        second_half_avg = values[mid_point:].mean()
        
        if first_half_avg != 0:
            trend = ((second_half_avg - first_half_avg) / first_half_avg) * 100
            return trend
        else:
            return 0.0
            
    except Exception:
        return 0.0


def render_event_log(session_manager: SessionStateManager):
    """Render the event log page with comprehensive event history and filtering.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("📝 Event Log")
    st.write("Comprehensive event history and resolution tracking.")
    
    # Get all events
    events = session_manager.get_events()
    
    if not events:
        st.info("No events logged yet. Trigger events from the Control Panel to see them here.")
        return
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"**Total Events:** {len(events)}")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False, help="Automatically refresh event data every 10 seconds")
    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Auto-refresh functionality
    if auto_refresh:
        import time
        time.sleep(10)
        st.rerun()
    
    st.divider()
    
    # Event statistics overview
    st.subheader("📊 Event Statistics")
    
    # Calculate statistics
    event_types = {}
    resolution_statuses = {}
    severity_distribution = {'Low (1-3)': 0, 'Medium (4-6)': 0, 'High (7-8)': 0, 'Critical (9-10)': 0}
    
    for event in events:
        # Event types
        event_type = event.type.name
        if event_type not in event_types:
            event_types[event_type] = 0
        event_types[event_type] += 1
        
        # Resolution statuses
        status = event.resolution_status.name
        if status not in resolution_statuses:
            resolution_statuses[status] = 0
        resolution_statuses[status] += 1
        
        # Severity distribution
        if 1 <= event.severity <= 3:
            severity_distribution['Low (1-3)'] += 1
        elif 4 <= event.severity <= 6:
            severity_distribution['Medium (4-6)'] += 1
        elif 7 <= event.severity <= 8:
            severity_distribution['High (7-8)'] += 1
        else:
            severity_distribution['Critical (9-10)'] += 1
    
    # Display statistics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Events by Type:**")
        for event_type, count in sorted(event_types.items()):
            type_emoji = {
                'DINOSAUR_ESCAPE': '🦕',
                'DINOSAUR_ILLNESS': '🤒',
                'DINOSAUR_AGGRESSIVE': '😡',
                'VISITOR_INJURY': '🩹',
                'VISITOR_COMPLAINT': '😤',
                'VISITOR_EMERGENCY': '🚨',
                'FACILITY_POWER_OUTAGE': '⚡',
                'FACILITY_EQUIPMENT_FAILURE': '🔧',
                'WEATHER_STORM': '⛈️',
                'WEATHER_EXTREME_TEMPERATURE': '🌡️',
                'CUSTOM': '⚙️'
            }.get(event_type, '📋')
            st.write(f"{type_emoji} {event_type.replace('_', ' ').title()}: {count}")
    
    with col2:
        st.write("**Resolution Status:**")
        for status, count in sorted(resolution_statuses.items()):
            status_emoji = {
                'PENDING': '⏳',
                'IN_PROGRESS': '🔄',
                'RESOLVED': '✅',
                'ESCALATED': '⚠️',
                'FAILED': '❌'
            }.get(status, '❓')
            st.write(f"{status_emoji} {status.replace('_', ' ').title()}: {count}")
    
    with col3:
        st.write("**Severity Distribution:**")
        for severity_range, count in severity_distribution.items():
            severity_emoji = {
                'Low (1-3)': '🟢',
                'Medium (4-6)': '🟡',
                'High (7-8)': '🟠',
                'Critical (9-10)': '🔴'
            }.get(severity_range, '⚪')
            st.write(f"{severity_emoji} {severity_range}: {count}")
    
    st.divider()
    
    # Event filtering and search functionality
    st.subheader("🔍 Event Filters & Search")
    
    # Filter controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Event type filter
        event_type_options = ["All"] + sorted(list(event_types.keys()))
        selected_event_type = st.selectbox(
            "Filter by Type",
            options=event_type_options,
            index=0
        )
    
    with col2:
        # Resolution status filter
        status_options = ["All"] + sorted(list(resolution_statuses.keys()))
        selected_status = st.selectbox(
            "Filter by Status",
            options=status_options,
            index=0
        )
    
    with col3:
        # Severity filter
        severity_options = ["All", "Low (1-3)", "Medium (4-6)", "High (7-8)", "Critical (9-10)"]
        selected_severity = st.selectbox(
            "Filter by Severity",
            options=severity_options,
            index=0
        )
    
    with col4:
        # Time range filter
        time_range_options = ["All Time", "Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"]
        selected_time_range = st.selectbox(
            "Time Range",
            options=time_range_options,
            index=0
        )
    
    # Search functionality
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input(
            "Search Events",
            placeholder="Search by event ID, description, location, or affected agents...",
            help="Search across event IDs, descriptions, locations, and affected agent names"
        )
    
    with col2:
        # Sort options
        sort_options = ["Newest First", "Oldest First", "Severity High to Low", "Severity Low to High"]
        selected_sort = st.selectbox("Sort By", options=sort_options, index=0)
    
    # Apply filters
    filtered_events = []
    current_time = datetime.now()
    
    for event in events:
        # Event type filter
        if selected_event_type != "All" and event.type.name != selected_event_type:
            continue
        
        # Status filter
        if selected_status != "All" and event.resolution_status.name != selected_status:
            continue
        
        # Severity filter
        if selected_severity != "All":
            if selected_severity == "Low (1-3)" and not (1 <= event.severity <= 3):
                continue
            elif selected_severity == "Medium (4-6)" and not (4 <= event.severity <= 6):
                continue
            elif selected_severity == "High (7-8)" and not (7 <= event.severity <= 8):
                continue
            elif selected_severity == "Critical (9-10)" and not (9 <= event.severity <= 10):
                continue
        
        # Time range filter
        if selected_time_range != "All Time":
            time_diff = current_time - event.timestamp
            if selected_time_range == "Last Hour" and time_diff.total_seconds() > 3600:
                continue
            elif selected_time_range == "Last 6 Hours" and time_diff.total_seconds() > 21600:
                continue
            elif selected_time_range == "Last 24 Hours" and time_diff.total_seconds() > 86400:
                continue
            elif selected_time_range == "Last Week" and time_diff.days > 7:
                continue
        
        # Search filter
        if search_term:
            search_lower = search_term.lower()
            searchable_text = f"{event.id} {event.description} {event.location.zone} {event.location.description} {' '.join(event.affected_agents)}".lower()
            if search_lower not in searchable_text:
                continue
        
        filtered_events.append(event)
    
    # Apply sorting
    if selected_sort == "Newest First":
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
    elif selected_sort == "Oldest First":
        filtered_events.sort(key=lambda x: x.timestamp)
    elif selected_sort == "Severity High to Low":
        filtered_events.sort(key=lambda x: x.severity, reverse=True)
    elif selected_sort == "Severity Low to High":
        filtered_events.sort(key=lambda x: x.severity)
    
    st.write(f"**Showing {len(filtered_events)} of {len(events)} events**")
    
    st.divider()
    
    # Chronological event history display
    st.subheader("📋 Event History")
    
    if not filtered_events:
        st.info("No events match the current filters. Try adjusting your search criteria.")
        return
    
    # Pagination for large event lists
    events_per_page = 10
    total_pages = (len(filtered_events) + events_per_page - 1) // events_per_page
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.selectbox(
                "Page",
                options=list(range(1, total_pages + 1)),
                index=0,
                format_func=lambda x: f"Page {x} of {total_pages}"
            )
        
        start_idx = (current_page - 1) * events_per_page
        end_idx = min(start_idx + events_per_page, len(filtered_events))
        page_events = filtered_events[start_idx:end_idx]
    else:
        page_events = filtered_events
    
    # Display events with detailed information and progress indicators
    for event in page_events:
        # Event status indicators
        status_icon = {
            'PENDING': '⏳',
            'IN_PROGRESS': '🔄',
            'RESOLVED': '✅',
            'ESCALATED': '⚠️',
            'FAILED': '❌'
        }.get(event.resolution_status.name, '❓')
        
        severity_color = {
            1: '🟢', 2: '🟢', 3: '🟢',  # Low
            4: '🟡', 5: '🟡', 6: '🟡',  # Medium
            7: '🟠', 8: '🟠',           # High
            9: '🔴', 10: '🔴'          # Critical
        }.get(event.severity, '⚪')
        
        event_type_emoji = {
            'DINOSAUR_ESCAPE': '🦕',
            'DINOSAUR_ILLNESS': '🤒',
            'DINOSAUR_AGGRESSIVE': '😡',
            'VISITOR_INJURY': '🩹',
            'VISITOR_COMPLAINT': '😤',
            'VISITOR_EMERGENCY': '🚨',
            'FACILITY_POWER_OUTAGE': '⚡',
            'FACILITY_EQUIPMENT_FAILURE': '🔧',
            'WEATHER_STORM': '⛈️',
            'WEATHER_EXTREME_TEMPERATURE': '🌡️',
            'CUSTOM': '⚙️'
        }.get(event.type.name, '📋')
        
        # Event header with status and progress
        event_title = f"{status_icon} {event_type_emoji} {event.type.name.replace('_', ' ').title()}"
        event_subtitle = f"{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Severity: {severity_color} {event.severity}/10"
        
        with st.expander(f"{event_title} - {event_subtitle}"):
            # Event details in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Event Information:**")
                st.write(f"**ID:** `{event.id}`")
                st.write(f"**Type:** {event.type.name.replace('_', ' ').title()}")
                st.write(f"**Severity:** {event.severity}/10 {severity_color}")
                st.write(f"**Status:** {event.resolution_status.name.replace('_', ' ').title()} {status_icon}")
                st.write(f"**Timestamp:** {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Resolution time and duration
                if event.resolution_time:
                    duration = event.resolution_time - event.timestamp
                    duration_str = str(duration).split('.')[0]  # Remove microseconds
                    st.write(f"**Resolution Time:** {event.resolution_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Duration:** {duration_str}")
                elif event.resolution_status.name in ['PENDING', 'IN_PROGRESS']:
                    ongoing_duration = current_time - event.timestamp
                    ongoing_str = str(ongoing_duration).split('.')[0]
                    st.write(f"**Ongoing Duration:** {ongoing_str}")
            
            with col2:
                st.write("**Location & Impact:**")
                st.write(f"**Zone:** {event.location.zone.replace('_', ' ').title()}")
                if event.location.description:
                    st.write(f"**Location Details:** {event.location.description}")
                st.write(f"**Coordinates:** ({event.location.x:.1f}, {event.location.y:.1f})")
                st.write(f"**Affected Agents:** {len(event.affected_agents)}")
                
                if event.affected_agents:
                    with st.expander("View Affected Agents"):
                        for agent_id in event.affected_agents:
                            st.write(f"• {agent_id}")
            
            # Event description
            if event.description:
                st.write("**Description:**")
                st.write(event.description)
            
            # Event parameters
            if event.parameters:
                st.write("**Event Parameters:**")
                param_cols = st.columns(2)
                param_items = list(event.parameters.items())
                
                for i, (key, value) in enumerate(param_items):
                    col_idx = i % 2
                    with param_cols[col_idx]:
                        if isinstance(value, list):
                            st.write(f"**{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}")
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            # Progress indicator for ongoing events
            if event.resolution_status.name in ['PENDING', 'IN_PROGRESS']:
                st.write("**Resolution Progress:**")
                
                # Simple progress estimation based on time elapsed and event type
                elapsed_time = (current_time - event.timestamp).total_seconds()
                
                # Estimated resolution times by event type (in seconds)
                estimated_times = {
                    'DINOSAUR_ESCAPE': 3600,      # 1 hour
                    'DINOSAUR_ILLNESS': 7200,     # 2 hours
                    'DINOSAUR_AGGRESSIVE': 1800,  # 30 minutes
                    'VISITOR_INJURY': 1800,       # 30 minutes
                    'VISITOR_COMPLAINT': 900,     # 15 minutes
                    'VISITOR_EMERGENCY': 600,     # 10 minutes
                    'FACILITY_POWER_OUTAGE': 2400, # 40 minutes
                    'FACILITY_EQUIPMENT_FAILURE': 3600, # 1 hour
                    'WEATHER_STORM': 14400,       # 4 hours
                    'WEATHER_EXTREME_TEMPERATURE': 7200, # 2 hours
                    'CUSTOM': 1800                 # 30 minutes
                }
                
                estimated_time = estimated_times.get(event.type.name, 1800)
                progress = min(elapsed_time / estimated_time, 1.0)
                
                # Adjust progress based on status
                if event.resolution_status.name == 'PENDING':
                    progress = min(progress * 0.3, 0.3)  # Max 30% for pending
                elif event.resolution_status.name == 'IN_PROGRESS':
                    progress = max(0.3, min(progress, 0.9))  # 30-90% for in progress
                
                st.progress(progress)
                
                if progress < 0.3:
                    st.write("🔍 **Status:** Assessing situation and mobilizing response team")
                elif progress < 0.6:
                    st.write("🚀 **Status:** Response team deployed, actively working on resolution")
                elif progress < 0.9:
                    st.write("🔧 **Status:** Implementing solution, monitoring progress")
                else:
                    st.write("✅ **Status:** Resolution nearly complete, finalizing details")
            
            # Action buttons for event management
            st.write("**Actions:**")
            action_cols = st.columns(4)
            
            with action_cols[0]:
                if st.button(f"📋 Copy ID", key=f"copy_{event.id}"):
                    st.write(f"Event ID: `{event.id}`")
            
            with action_cols[1]:
                if st.button(f"📍 Show Location", key=f"location_{event.id}"):
                    st.write(f"Location: {event.location.zone} ({event.location.x:.1f}, {event.location.y:.1f})")
            
            with action_cols[2]:
                if event.affected_agents and st.button(f"👥 View Agents", key=f"agents_{event.id}"):
                    st.write("Affected agents:")
                    for agent_id in event.affected_agents:
                        st.write(f"• {agent_id}")
            
            with action_cols[3]:
                if st.button(f"📊 Event Stats", key=f"stats_{event.id}"):
                    if event.resolution_time:
                        duration = event.resolution_time - event.timestamp
                        st.write(f"Resolution took: {str(duration).split('.')[0]}")
                    else:
                        ongoing = current_time - event.timestamp
                        st.write(f"Ongoing for: {str(ongoing).split('.')[0]}")
    
    # Summary footer
    if filtered_events:
        st.divider()
        st.subheader("📈 Summary")
        
        # Quick stats for filtered events
        resolved_count = len([e for e in filtered_events if e.resolution_status.name == 'RESOLVED'])
        pending_count = len([e for e in filtered_events if e.resolution_status.name in ['PENDING', 'IN_PROGRESS']])
        failed_count = len([e for e in filtered_events if e.resolution_status.name == 'FAILED'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Filtered", len(filtered_events))
        
        with col2:
            resolution_rate = (resolved_count / len(filtered_events)) * 100 if filtered_events else 0
            st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
        
        with col3:
            st.metric("Active Events", pending_count)
        
        with col4:
            st.metric("Failed Events", failed_count)
        
        # Average resolution time for resolved events
        resolved_events = [e for e in filtered_events if e.resolution_time]
        if resolved_events:
            total_duration = sum([(e.resolution_time - e.timestamp).total_seconds() for e in resolved_events])
            avg_duration = total_duration / len(resolved_events)
            avg_duration_str = str(datetime.fromtimestamp(avg_duration) - datetime.fromtimestamp(0)).split('.')[0]
            st.write(f"**Average Resolution Time:** {avg_duration_str}")
        
        # Export functionality
        if st.button("📥 Export Event Data", help="Export filtered events as JSON"):
            import json
            export_data = [event.to_dict() for event in filtered_events]
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"event_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def render_agent_chat(session_manager: SessionStateManager):
    """Render the dedicated agent chat page with real-time updates.
    
    Args:
        session_manager: Session state manager instance
    """
    st.title("💬 Agent Chat")
    st.write("Real-time communication feed from all agents in the simulation.")
    
    try:
        from ui.agent_chat import create_agent_chat_interface
        
        # Get real-time chat system from simulation manager
        real_time_chat = None
        if 'simulation_manager' in st.session_state:
            sim_manager = st.session_state['simulation_manager']
            if hasattr(sim_manager, 'get_real_time_chat'):
                real_time_chat = sim_manager.get_real_time_chat()
        
        # Create chat interface with real-time chat system
        chat_interface = create_agent_chat_interface(session_manager, real_time_chat)
        
        # Debug and status information
        if real_time_chat:
            stats = real_time_chat.get_statistics()
            if stats['is_running']:
                st.success(f"🟢 Real-time chat active: {stats['active_agents']} agents, {stats['total_messages']} messages")
            else:
                st.warning("🔴 Real-time chat system not running")
        else:
            st.warning("⚠️ Real-time chat system not available - using fallback mode")
        
        # Fallback: Auto-sync with agent manager for backward compatibility
        if 'simulation_manager' in st.session_state:
            sim_manager = st.session_state['simulation_manager']
            if hasattr(sim_manager, 'agent_manager') and sim_manager.agent_manager:
                # Get recent conversations from agent manager
                agent_conversations = sim_manager.agent_manager.get_agent_conversations()
                
                # Track which conversations we've already added to chat
                if 'processed_conversations' not in st.session_state:
                    st.session_state.processed_conversations = set()
                
                # Add new conversations to chat (fallback for when real-time chat isn't working)
                for conv in agent_conversations:
                    conv_id = f"{conv.get('event_id', 'unknown')}_{conv.get('timestamp', 'unknown')}"
                    
                    if conv_id not in st.session_state.processed_conversations:
                        # Add system message for the event
                        event_type = conv.get('event_type', 'Unknown Event')
                        event_id = conv.get('event_id', 'unknown')
                        chat_interface.add_system_message(
                            f"🚨 {event_type.replace('_', ' ').title()} event triggered",
                            event_id
                        )
                        
                        # Add individual agent responses (filter out boring ones)
                        individual_responses = conv.get('individual_responses', {})
                        for agent_id, response in individual_responses.items():
                            # Skip boring responses
                            if response and ("responding to" in response.lower() or 
                                           "acknowledged" in response.lower() or
                                           len(response) < 50):
                                continue
                            
                            # Get agent name from session state
                            agents = session_manager.get_agents()
                            agent_name = agent_id
                            if agent_id in agents:
                                agent_name = agents[agent_id].name
                            
                            chat_interface.add_agent_message(
                                agent_id=agent_id,
                                agent_name=agent_name,
                                message=response,
                                event_context={
                                    'event_id': event_id,
                                    'event_type': event_type
                                }
                            )
                        
                        st.session_state.processed_conversations.add(conv_id)
        
        # Render the chat interface
        chat_interface.render_chat()
        
        # Test and debug options
        st.divider()
        st.subheader("🧪 Chat Testing & Debug")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧪 Test Real-time Message"):
                if real_time_chat and real_time_chat.get_statistics()['is_running']:
                    # Create a test event to trigger real-time responses
                    from models.core import Event
                    from models.config import Location
                    from models.enums import EventType
                    
                    test_event = Event(
                        id="test_chat_001",
                        type=EventType.DINOSAUR_ESCAPE,
                        severity=7,
                        location=Location(0.0, 0.0, "test_area", "Test area"),
                        description="Test event for chat system"
                    )
                    
                    # Get all registered agents
                    active_agents = list(real_time_chat.active_agents.keys())
                    if active_agents:
                        real_time_chat.trigger_event_response(test_event, active_agents[:2])  # Test with first 2 agents
                        st.success(f"✅ Triggered test responses from {min(2, len(active_agents))} agents")
                    else:
                        st.warning("⚠️ No agents registered in real-time chat system")
                else:
                    st.error("❌ Real-time chat system not available or not running")
        
        with col2:
            if st.button("🔄 Force Refresh Chat"):
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear All Messages"):
                st.session_state.agent_chat_messages = []
                if real_time_chat:
                    real_time_chat.clear_chat_history()
                st.success("✅ Chat cleared")
                st.rerun()
        
        # Auto-refresh option
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            auto_refresh = st.checkbox("Auto-refresh chat", value=False, help="Automatically refresh every 10 seconds")
        
        with col2:
            if st.button("🧪 Add Test Message"):
                chat_interface.add_agent_message(
                    agent_id='test_agent',
                    agent_name='Test Agent',
                    message='This is a test message to verify the chat is working!',
                    event_context={'event_id': 'test', 'event_type': 'TEST'}
                )
                st.rerun()
        
        # Auto-refresh functionality
        if auto_refresh:
            import time
            time.sleep(10)
            st.rerun()
        
    except Exception as e:
        st.error(f"Error loading agent chat: {str(e)}")
        st.write("**Error Details:**")
        st.code(str(e))


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
            # Also clear chat messages
            if 'agent_chat_messages' in st.session_state:
                st.session_state.agent_chat_messages = []
            if 'processed_conversations' in st.session_state:
                st.session_state.processed_conversations = set()
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
    """Main application entry point with enhanced error handling and real-time updates."""
    try:
        # Configure page
        configure_page()
        
        # Initialize session state manager with error handling
        try:
            session_manager = SessionStateManager()
        except Exception as e:
            st.error(f"Failed to initialize session state: {str(e)}")
            st.stop()
        
        # Add real-time update controls in sidebar
        with st.sidebar:
            st.divider()
            st.write("**Real-time Updates:**")
            
            # Global auto-refresh setting
            auto_refresh_enabled = st.checkbox(
                "Enable Auto-refresh",
                value=st.session_state.get('global_auto_refresh', False),
                help="Automatically refresh all dashboard data"
            )
            st.session_state['global_auto_refresh'] = auto_refresh_enabled
            
            if auto_refresh_enabled:
                refresh_interval = st.selectbox(
                    "Refresh Interval",
                    options=[5, 10, 15, 30, 60],
                    index=1,
                    format_func=lambda x: f"{x} seconds",
                    help="How often to refresh data"
                )
                st.session_state['refresh_interval'] = refresh_interval
                
                # Auto-refresh logic
                import time
                if 'last_refresh' not in st.session_state:
                    st.session_state['last_refresh'] = time.time()
                
                current_time = time.time()
                if current_time - st.session_state['last_refresh'] >= refresh_interval:
                    st.session_state['last_refresh'] = current_time
                    st.rerun()
            
            # Manual refresh button
            if st.button("🔄 Refresh All Data", use_container_width=True):
                # Clear any cached data and force refresh
                if 'simulation_manager' in st.session_state:
                    try:
                        sim_manager = st.session_state['simulation_manager']
                        if hasattr(sim_manager, 'update_simulation_time'):
                            sim_manager.update_simulation_time()
                    except Exception as e:
                        st.warning(f"Could not update simulation time: {str(e)}")
                
                st.rerun()
        
        # Render sidebar and get selected page with error handling
        try:
            selected_page = render_sidebar(session_manager)
        except Exception as e:
            st.error(f"Error rendering sidebar: {str(e)}")
            selected_page = "Dashboard"  # Fallback to dashboard
        
        # Add error boundary for main content
        try:
            # Render main content based on selected page
            if selected_page == "Dashboard":
                render_dashboard_overview(session_manager)
            elif selected_page == "Control Panel":
                render_control_panel(session_manager)
            elif selected_page == "Agent Monitor":
                render_agent_monitor(session_manager)
            elif selected_page == "Agent Chat":
                render_agent_chat(session_manager)
            elif selected_page == "Metrics":
                render_metrics_dashboard(session_manager)
            elif selected_page == "Event Log":
                render_event_log(session_manager)
            elif selected_page == "Settings":
                render_settings(session_manager)
            else:
                st.error(f"Unknown page: {selected_page}")
                render_dashboard_overview(session_manager)
        
        except Exception as e:
            st.error(f"Error rendering {selected_page} page: {str(e)}")
            st.write("**Error Details:**")
            st.code(str(e))
            
            # Provide fallback options
            st.write("**Recovery Options:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🏠 Go to Dashboard"):
                    st.session_state['selected_page'] = 'Dashboard'
                    st.rerun()
            
            with col2:
                if st.button("🔄 Refresh Session"):
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Reset Session"):
                    session_manager.clear_all()
                    st.rerun()
        
        # Enhanced footer with system status
        st.divider()
        
        # System status footer
        footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
        
        with footer_col1:
            st.caption(f"AI Agent Dinosaur Simulator | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with footer_col2:
            # Connection status indicator
            try:
                sim_state = session_manager.get_simulation_state()
                if sim_state.is_running:
                    st.caption("🟢 System Online")
                else:
                    st.caption("🟡 System Ready")
            except Exception:
                st.caption("🔴 System Error")
        
        with footer_col3:
            # Performance indicator
            session_info = session_manager.get_session_info()
            total_objects = session_info.get('agent_count', 0) + session_info.get('event_count', 0)
            if total_objects > 100:
                st.caption("🟡 High Load")
            elif total_objects > 50:
                st.caption("🟢 Normal Load")
            else:
                st.caption("🟢 Light Load")
    
    except Exception as e:
        # Top-level error handler
        st.error("🚨 **Critical Application Error**")
        st.write("The application encountered a critical error and cannot continue.")
        st.code(str(e))
        
        st.write("**Recovery Actions:**")
        if st.button("🔄 Restart Application"):
            # Clear all session state and restart
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.write("If the problem persists, please check the application logs and configuration.")


if __name__ == "__main__":
    main()