"""System status manager for tracking park infrastructure and dependencies."""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field


class SystemStatus(Enum):
    """System component status levels."""
    OPERATIONAL = "green"      # All systems normal
    WARNING = "amber"          # Minor issues, degraded performance
    CRITICAL = "red"           # Major issues, system failure
    OFFLINE = "gray"           # System offline/disabled


@dataclass
class SystemComponent:
    """Represents a system component in the park infrastructure."""
    id: str
    name: str
    type: str  # power, fence, gate, zone, visitor_area
    status: SystemStatus = SystemStatus.OPERATIONAL
    dependencies: List[str] = field(default_factory=list)  # IDs of components this depends on
    affects: List[str] = field(default_factory=list)       # IDs of components this affects
    last_updated: datetime = field(default_factory=datetime.now)
    details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "affects": self.affects,
            "last_updated": self.last_updated.isoformat(),
            "details": self.details
        }


class SystemStatusManager:
    """Manages the status of all park infrastructure components."""
    
    def __init__(self):
        """Initialize the system status manager."""
        self.logger = logging.getLogger(__name__)
        self.components: Dict[str, SystemComponent] = {}
        self.status_history: List[Dict[str, Any]] = []
        
        # Initialize default park infrastructure
        self._initialize_park_systems()
        
        self.logger.info("SystemStatusManager initialized")
    
    def _initialize_park_systems(self) -> None:
        """Initialize the default park infrastructure components."""
        
        # Power Systems
        self.components["main_power"] = SystemComponent(
            id="main_power",
            name="Main Power Grid",
            type="power",
            details="Primary electrical supply for the entire park"
        )
        
        self.components["backup_power"] = SystemComponent(
            id="backup_power",
            name="Backup Generators",
            type="power",
            dependencies=["main_power"],  # Activates when main power fails
            details="Emergency backup power system"
        )
        
        # Fence Systems
        fence_zones = ["enclosure_a", "enclosure_b", "enclosure_c", "perimeter"]
        for zone in fence_zones:
            fence_id = f"fence_{zone}"
            self.components[fence_id] = SystemComponent(
                id=fence_id,
                name=f"{zone.replace('_', ' ').title()} Fence",
                type="fence",
                dependencies=["main_power", "backup_power"],
                details=f"Electrified containment fence for {zone}"
            )
        
        # Gate Systems
        gates = [
            ("main_gate", "Main Entrance Gate", ["fence_perimeter"]),
            ("enclosure_a_gate", "Enclosure A Service Gate", ["fence_enclosure_a"]),
            ("enclosure_b_gate", "Enclosure B Service Gate", ["fence_enclosure_b"]),
            ("enclosure_c_gate", "Enclosure C Service Gate", ["fence_enclosure_c"]),
            ("emergency_exit", "Emergency Exit Gate", ["fence_perimeter"])
        ]
        
        for gate_id, gate_name, fence_deps in gates:
            self.components[gate_id] = SystemComponent(
                id=gate_id,
                name=gate_name,
                type="gate",
                dependencies=["main_power"] + fence_deps,
                details=f"Automated security gate: {gate_name}"
            )
        
        # Zone Systems
        zones = [
            ("visitor_center", "Visitor Center", ["main_gate"]),
            ("enclosure_a_viewing", "Enclosure A Viewing Area", ["enclosure_a_gate", "fence_enclosure_a"]),
            ("enclosure_b_viewing", "Enclosure B Viewing Area", ["enclosure_b_gate", "fence_enclosure_b"]),
            ("enclosure_c_viewing", "Enclosure C Viewing Area", ["enclosure_c_gate", "fence_enclosure_c"]),
            ("gift_shop", "Gift Shop & Cafe", ["main_gate"]),
            ("parking_area", "Parking Area", ["main_gate"])
        ]
        
        for zone_id, zone_name, gate_deps in zones:
            self.components[zone_id] = SystemComponent(
                id=zone_id,
                name=zone_name,
                type="zone",
                dependencies=gate_deps,
                details=f"Park zone: {zone_name}"
            )
        
        # Visitor Areas (depend on zones)
        visitor_areas = [
            ("visitors_center", "Visitors in Center", ["visitor_center"]),
            ("visitors_enclosure_a", "Visitors at Enclosure A", ["enclosure_a_viewing"]),
            ("visitors_enclosure_b", "Visitors at Enclosure B", ["enclosure_b_viewing"]),
            ("visitors_enclosure_c", "Visitors at Enclosure C", ["enclosure_c_viewing"]),
            ("visitors_gift_shop", "Visitors in Gift Shop", ["gift_shop"]),
            ("visitors_parking", "Visitors in Parking", ["parking_area"])
        ]
        
        for visitor_id, visitor_name, zone_deps in visitor_areas:
            self.components[visitor_id] = SystemComponent(
                id=visitor_id,
                name=visitor_name,
                type="visitor_area",
                dependencies=zone_deps,
                details=f"Visitor safety depends on: {', '.join(zone_deps)}"
            )
        
        # Set up affects relationships (reverse dependencies)
        self._calculate_affects_relationships()
        
        self.logger.info(f"Initialized {len(self.components)} park system components")
    
    def _calculate_affects_relationships(self) -> None:
        """Calculate which components are affected by each component."""
        for component in self.components.values():
            component.affects.clear()
        
        for component in self.components.values():
            for dep_id in component.dependencies:
                if dep_id in self.components:
                    self.components[dep_id].affects.append(component.id)
    
    def update_component_status(self, component_id: str, status: SystemStatus, 
                              details: str = "", agent_id: Optional[str] = None) -> bool:
        """Update the status of a system component.
        
        Args:
            component_id: ID of the component to update
            status: New status for the component
            details: Additional details about the status change
            agent_id: ID of the agent that triggered this change (optional)
            
        Returns:
            True if update was successful, False otherwise
        """
        if component_id not in self.components:
            self.logger.warning(f"Component {component_id} not found")
            return False
        
        component = self.components[component_id]
        old_status = component.status
        
        component.status = status
        component.details = details
        component.last_updated = datetime.now()
        
        # Record status change
        status_change = {
            "timestamp": datetime.now().isoformat(),
            "component_id": component_id,
            "component_name": component.name,
            "old_status": old_status.value,
            "new_status": status.value,
            "details": details,
            "agent_id": agent_id
        }
        
        self.status_history.append(status_change)
        
        # Keep only last 100 status changes
        if len(self.status_history) > 100:
            self.status_history = self.status_history[-100:]
        
        self.logger.info(f"Component {component.name} status changed from {old_status.value} to {status.value}")
        
        # Propagate status changes to dependent components
        self._propagate_status_changes(component_id)
        
        return True
    
    def _propagate_status_changes(self, changed_component_id: str) -> None:
        """Propagate status changes to dependent components.
        
        Args:
            changed_component_id: ID of the component that changed
        """
        changed_component = self.components[changed_component_id]
        
        # If a component goes critical/offline, affect dependent components
        if changed_component.status in [SystemStatus.CRITICAL, SystemStatus.OFFLINE]:
            for affected_id in changed_component.affects:
                if affected_id in self.components:
                    affected_component = self.components[affected_id]
                    
                    # Check if all dependencies are still operational
                    dependency_status = self._check_dependency_status(affected_id)
                    
                    if dependency_status == SystemStatus.CRITICAL:
                        self.update_component_status(
                            affected_id, 
                            SystemStatus.CRITICAL,
                            f"Dependency failure: {changed_component.name}",
                            "system_cascade"
                        )
                    elif dependency_status == SystemStatus.WARNING:
                        self.update_component_status(
                            affected_id,
                            SystemStatus.WARNING,
                            f"Dependency degraded: {changed_component.name}",
                            "system_cascade"
                        )
    
    def _check_dependency_status(self, component_id: str) -> SystemStatus:
        """Check the overall status based on dependencies.
        
        Args:
            component_id: ID of the component to check
            
        Returns:
            Overall status based on dependencies
        """
        if component_id not in self.components:
            return SystemStatus.OFFLINE
        
        component = self.components[component_id]
        
        if not component.dependencies:
            return component.status
        
        # Check status of all dependencies
        critical_count = 0
        warning_count = 0
        
        for dep_id in component.dependencies:
            if dep_id in self.components:
                dep_status = self.components[dep_id].status
                if dep_status == SystemStatus.CRITICAL:
                    critical_count += 1
                elif dep_status == SystemStatus.WARNING:
                    warning_count += 1
        
        # Determine overall status
        if critical_count > 0:
            return SystemStatus.CRITICAL
        elif warning_count > 0:
            return SystemStatus.WARNING
        else:
            return SystemStatus.OPERATIONAL
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get an overview of all system components.
        
        Returns:
            Dictionary with system overview information
        """
        status_counts = {status.value: 0 for status in SystemStatus}
        
        for component in self.components.values():
            status_counts[component.status.value] += 1
        
        return {
            "total_components": len(self.components),
            "status_counts": status_counts,
            "components": {comp_id: comp.to_dict() for comp_id, comp in self.components.items()},
            "last_updated": datetime.now().isoformat()
        }
    
    def get_component_status(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a specific component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            Component status information or None if not found
        """
        if component_id not in self.components:
            return None
        
        return self.components[component_id].to_dict()
    
    def get_components_by_type(self, component_type: str) -> List[Dict[str, Any]]:
        """Get all components of a specific type.
        
        Args:
            component_type: Type of components to retrieve
            
        Returns:
            List of components of the specified type
        """
        return [
            comp.to_dict() for comp in self.components.values()
            if comp.type == component_type
        ]
    
    def get_critical_components(self) -> List[Dict[str, Any]]:
        """Get all components with critical status.
        
        Returns:
            List of components with critical status
        """
        return [
            comp.to_dict() for comp in self.components.values()
            if comp.status == SystemStatus.CRITICAL
        ]
    
    def get_status_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent status change history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of recent status changes
        """
        return self.status_history[-limit:] if self.status_history else []
    
    def simulate_power_failure(self) -> None:
        """Simulate a main power failure for testing."""
        self.update_component_status(
            "main_power",
            SystemStatus.CRITICAL,
            "Simulated power grid failure",
            "simulation"
        )
    
    def simulate_fence_breach(self, enclosure: str = "enclosure_a") -> None:
        """Simulate a fence breach for testing.
        
        Args:
            enclosure: Which enclosure fence to breach
        """
        fence_id = f"fence_{enclosure}"
        self.update_component_status(
            fence_id,
            SystemStatus.CRITICAL,
            f"Fence breach detected in {enclosure}",
            "simulation"
        )
    
    def restore_all_systems(self) -> None:
        """Restore all systems to operational status."""
        for component in self.components.values():
            component.status = SystemStatus.OPERATIONAL
            component.details = "System restored"
            component.last_updated = datetime.now()
        
        self.logger.info("All systems restored to operational status")