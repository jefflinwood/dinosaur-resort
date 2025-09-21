"""Control room system diagram using Streamlit and Graphviz."""

import streamlit as st
from typing import Dict, List, Any, Optional
from managers.system_status_manager import SystemStatusManager, SystemStatus
from datetime import datetime


class ControlRoomDiagram:
    """Manages the control room system diagram display."""
    
    def __init__(self, system_status_manager: SystemStatusManager):
        """Initialize the control room diagram.
        
        Args:
            system_status_manager: System status manager instance
        """
        self.system_manager = system_status_manager
    
    def render_control_room(self) -> None:
        """Render the complete control room interface."""
        st.title("🎛️ Control Room - System Status")
        st.write("Live dependency graph showing park infrastructure status")
        
        # Control buttons
        self._render_control_buttons()
        
        st.divider()
        
        # System overview metrics
        self._render_system_overview()
        
        st.divider()
        
        # Main system diagram
        self._render_system_diagram()
        
        st.divider()
        
        # Status history and details
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_status_history()
        
        with col2:
            self._render_component_details()
    
    def _render_control_buttons(self) -> None:
        """Render control buttons for testing system failures."""
        st.subheader("🧪 System Test Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("⚡ Simulate Power Failure", type="secondary"):
                self.system_manager.simulate_power_failure()
                st.success("Power failure simulated!")
                st.rerun()
        
        with col2:
            if st.button("🚧 Simulate Fence Breach", type="secondary"):
                self.system_manager.simulate_fence_breach("enclosure_a")
                st.success("Fence breach simulated!")
                st.rerun()
        
        with col3:
            if st.button("🚪 Gate Malfunction", type="secondary"):
                self.system_manager.update_component_status(
                    "main_gate",
                    SystemStatus.WARNING,
                    "Gate control system malfunction",
                    "manual_test"
                )
                st.success("Gate malfunction simulated!")
                st.rerun()
        
        with col4:
            if st.button("🔧 Restore All Systems", type="primary"):
                self.system_manager.restore_all_systems()
                st.success("All systems restored!")
                st.rerun()
    
    def _render_system_overview(self) -> None:
        """Render system overview metrics."""
        overview = self.system_manager.get_system_overview()
        status_counts = overview["status_counts"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🟢 Operational",
                status_counts.get("green", 0),
                help="Systems running normally"
            )
        
        with col2:
            st.metric(
                "🟡 Warning",
                status_counts.get("amber", 0),
                help="Systems with minor issues"
            )
        
        with col3:
            st.metric(
                "🔴 Critical",
                status_counts.get("red", 0),
                help="Systems with major failures"
            )
        
        with col4:
            st.metric(
                "⚫ Offline",
                status_counts.get("gray", 0),
                help="Systems that are offline"
            )
    
    def _render_system_diagram(self) -> None:
        """Render the main system dependency diagram using Graphviz."""
        st.subheader("🔗 System Dependency Graph")
        
        # Generate Graphviz DOT notation
        dot_graph = self._generate_graphviz_diagram()
        
        # Display the graph
        try:
            st.graphviz_chart(dot_graph, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering diagram: {e}")
            st.code(dot_graph, language="dot")
    
    def _generate_graphviz_diagram(self) -> str:
        """Generate Graphviz DOT notation for the system diagram.
        
        Returns:
            DOT notation string for the system diagram
        """
        overview = self.system_manager.get_system_overview()
        components = overview["components"]
        
        # Start DOT graph
        dot_lines = [
            "digraph SystemDiagram {",
            "    rankdir=TB;",
            "    node [shape=box, style=filled, fontname=\"Arial\"];",
            "    edge [fontname=\"Arial\", fontsize=10];",
            ""
        ]
        
        # Define subgraphs for better layout
        dot_lines.extend([
            "    subgraph cluster_power {",
            "        label=\"Power Systems\";",
            "        style=filled;",
            "        color=lightgrey;",
        ])
        
        # Add power components
        for comp_id, comp_data in components.items():
            if comp_data["type"] == "power":
                color = comp_data["status"]
                label = comp_data["name"]
                dot_lines.append(f'        {comp_id} [label="{label}", fillcolor={color}];')
        
        dot_lines.extend([
            "    }",
            "",
            "    subgraph cluster_security {",
            "        label=\"Security Systems\";",
            "        style=filled;",
            "        color=lightblue;",
        ])
        
        # Add fence and gate components
        for comp_id, comp_data in components.items():
            if comp_data["type"] in ["fence", "gate"]:
                color = comp_data["status"]
                label = comp_data["name"]
                dot_lines.append(f'        {comp_id} [label="{label}", fillcolor={color}];')
        
        dot_lines.extend([
            "    }",
            "",
            "    subgraph cluster_zones {",
            "        label=\"Park Zones\";",
            "        style=filled;",
            "        color=lightyellow;",
        ])
        
        # Add zone components
        for comp_id, comp_data in components.items():
            if comp_data["type"] == "zone":
                color = comp_data["status"]
                label = comp_data["name"]
                dot_lines.append(f'        {comp_id} [label="{label}", fillcolor={color}];')
        
        dot_lines.extend([
            "    }",
            "",
            "    subgraph cluster_visitors {",
            "        label=\"Visitor Areas\";",
            "        style=filled;",
            "        color=lightpink;",
        ])
        
        # Add visitor area components
        for comp_id, comp_data in components.items():
            if comp_data["type"] == "visitor_area":
                color = comp_data["status"]
                label = comp_data["name"]
                dot_lines.append(f'        {comp_id} [label="{label}", fillcolor={color}];')
        
        dot_lines.extend([
            "    }",
            ""
        ])
        
        # Add dependency edges
        for comp_id, comp_data in components.items():
            for dep_id in comp_data["dependencies"]:
                if dep_id in components:
                    # Add edge with status-based styling
                    edge_color = "red" if comp_data["status"] == "red" else "black"
                    dot_lines.append(f'    {dep_id} -> {comp_id} [color={edge_color}];')
        
        # Add legend
        dot_lines.extend([
            "",
            "    subgraph cluster_legend {",
            "        label=\"Status Legend\";",
            "        style=filled;",
            "        color=white;",
            "        legend_green [label=\"Operational\", fillcolor=green, shape=ellipse];",
            "        legend_amber [label=\"Warning\", fillcolor=amber, shape=ellipse];",
            "        legend_red [label=\"Critical\", fillcolor=red, shape=ellipse];",
            "        legend_gray [label=\"Offline\", fillcolor=gray, shape=ellipse];",
            "    }",
            ""
        ])
        
        dot_lines.append("}")
        
        return "\n".join(dot_lines)
    
    def _render_status_history(self) -> None:
        """Render recent status change history."""
        st.subheader("📋 Recent Status Changes")
        
        history = self.system_manager.get_status_history(10)
        
        if history:
            for change in reversed(history):  # Most recent first
                timestamp = datetime.fromisoformat(change["timestamp"])
                time_str = timestamp.strftime("%H:%M:%S")
                
                # Status change indicator
                old_status = change["old_status"]
                new_status = change["new_status"]
                
                status_emoji = {
                    "green": "🟢",
                    "amber": "🟡", 
                    "red": "🔴",
                    "gray": "⚫"
                }
                
                old_emoji = status_emoji.get(old_status, "❓")
                new_emoji = status_emoji.get(new_status, "❓")
                
                with st.container():
                    st.write(f"**{time_str}** - {change['component_name']}")
                    st.write(f"{old_emoji} → {new_emoji} {change['details']}")
                    if change.get("agent_id"):
                        st.caption(f"Triggered by: {change['agent_id']}")
                    st.divider()
        else:
            st.info("No status changes recorded yet.")
    
    def _render_component_details(self) -> None:
        """Render detailed component information."""
        st.subheader("🔍 Component Details")
        
        # Component selector
        overview = self.system_manager.get_system_overview()
        components = overview["components"]
        
        component_options = {
            f"{comp_data['name']} ({comp_data['type']})": comp_id
            for comp_id, comp_data in components.items()
        }
        
        selected_display = st.selectbox(
            "Select Component",
            options=list(component_options.keys()),
            help="Choose a component to view detailed information"
        )
        
        if selected_display:
            selected_id = component_options[selected_display]
            comp_data = components[selected_id]
            
            # Display component details
            status_colors = {
                "green": "🟢 Operational",
                "amber": "🟡 Warning",
                "red": "🔴 Critical", 
                "gray": "⚫ Offline"
            }
            
            st.write(f"**Status:** {status_colors.get(comp_data['status'], comp_data['status'])}")
            st.write(f"**Type:** {comp_data['type'].replace('_', ' ').title()}")
            st.write(f"**Last Updated:** {datetime.fromisoformat(comp_data['last_updated']).strftime('%H:%M:%S')}")
            
            if comp_data["details"]:
                st.write(f"**Details:** {comp_data['details']}")
            
            if comp_data["dependencies"]:
                st.write("**Dependencies:**")
                for dep_id in comp_data["dependencies"]:
                    if dep_id in components:
                        dep_comp = components[dep_id]
                        dep_status = status_colors.get(dep_comp["status"], dep_comp["status"])
                        st.write(f"  - {dep_comp['name']}: {dep_status}")
            
            if comp_data["affects"]:
                st.write("**Affects:**")
                for affected_id in comp_data["affects"]:
                    if affected_id in components:
                        affected_comp = components[affected_id]
                        affected_status = status_colors.get(affected_comp["status"], affected_comp["status"])
                        st.write(f"  - {affected_comp['name']}: {affected_status}")
    
    def trigger_agent_action(self, agent_id: str, component_id: str, 
                           action: str, status: SystemStatus) -> bool:
        """Trigger a system status change from an agent action.
        
        Args:
            agent_id: ID of the agent triggering the action
            component_id: ID of the component being affected
            action: Description of the action taken
            status: New status for the component
            
        Returns:
            True if action was successful, False otherwise
        """
        return self.system_manager.update_component_status(
            component_id,
            status,
            f"Agent action: {action}",
            agent_id
        )


def create_control_room_diagram(system_status_manager: SystemStatusManager) -> ControlRoomDiagram:
    """Create and return a control room diagram interface.
    
    Args:
        system_status_manager: System status manager instance
        
    Returns:
        ControlRoomDiagram instance
    """
    return ControlRoomDiagram(system_status_manager)