"""Metrics management system for the AI Agent Dinosaur Simulator."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from models.core import MetricsSnapshot, Agent
from models.enums import AgentRole, DinosaurSpecies
from ui.session_state import SessionStateManager


class MetricsManager:
    """Manages resort metrics including visitor satisfaction and dinosaur happiness."""
    
    def __init__(self, session_manager: SessionStateManager):
        """Initialize metrics manager.
        
        Args:
            session_manager: Session state manager for data persistence
        """
        self.session_manager = session_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize default metrics if none exist
        if not self.session_manager.get_latest_metrics():
            self._initialize_default_metrics()
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default metrics snapshot."""
        default_metrics = MetricsSnapshot(
            visitor_satisfaction=0.8,  # Start with good satisfaction
            dinosaur_happiness={},  # Will be populated as dinosaurs are added
            facility_efficiency=1.0,  # Start with perfect efficiency
            safety_rating=1.0,  # Start with perfect safety
            timestamp=datetime.now()
        )
        
        self.session_manager.add_metrics_snapshot(default_metrics)
        self.logger.info("Initialized default metrics")
    
    def update_visitor_satisfaction(self, visitor_id: str, change: float) -> None:
        """Update visitor satisfaction score.
        
        Args:
            visitor_id: ID of the visitor agent
            change: Change in satisfaction (-1.0 to 1.0)
        """
        if not (-1.0 <= change <= 1.0):
            raise ValueError("Satisfaction change must be between -1.0 and 1.0")
        
        current_metrics = self.get_current_metrics()
        
        # Apply change with bounds checking
        new_satisfaction = max(0.0, min(1.0, current_metrics.visitor_satisfaction + change))
        
        # Create new metrics snapshot
        new_metrics = MetricsSnapshot(
            visitor_satisfaction=new_satisfaction,
            dinosaur_happiness=current_metrics.dinosaur_happiness.copy(),
            facility_efficiency=current_metrics.facility_efficiency,
            safety_rating=current_metrics.safety_rating,
            timestamp=datetime.now()
        )
        
        self.session_manager.add_metrics_snapshot(new_metrics)
        
        self.logger.info(f"Updated visitor satisfaction for {visitor_id}: {change:+.2f} -> {new_satisfaction:.2f}")
    
    def update_dinosaur_happiness(self, dinosaur_id: str, change: float) -> None:
        """Update dinosaur happiness score.
        
        Args:
            dinosaur_id: ID of the dinosaur agent
            change: Change in happiness (-1.0 to 1.0)
        """
        if not (-1.0 <= change <= 1.0):
            raise ValueError("Happiness change must be between -1.0 and 1.0")
        
        current_metrics = self.get_current_metrics()
        dinosaur_happiness = current_metrics.dinosaur_happiness.copy()
        
        # Initialize dinosaur happiness if not exists
        if dinosaur_id not in dinosaur_happiness:
            dinosaur_happiness[dinosaur_id] = 0.8  # Default happiness
        
        # Apply change with bounds checking
        current_happiness = dinosaur_happiness[dinosaur_id]
        new_happiness = max(0.0, min(1.0, current_happiness + change))
        dinosaur_happiness[dinosaur_id] = new_happiness
        
        # Create new metrics snapshot
        new_metrics = MetricsSnapshot(
            visitor_satisfaction=current_metrics.visitor_satisfaction,
            dinosaur_happiness=dinosaur_happiness,
            facility_efficiency=current_metrics.facility_efficiency,
            safety_rating=current_metrics.safety_rating,
            timestamp=datetime.now()
        )
        
        self.session_manager.add_metrics_snapshot(new_metrics)
        
        self.logger.info(f"Updated dinosaur happiness for {dinosaur_id}: {change:+.2f} -> {new_happiness:.2f}")
    
    def update_facility_efficiency(self, change: float) -> None:
        """Update facility efficiency rating.
        
        Args:
            change: Change in efficiency (-1.0 to 1.0)
        """
        if not (-1.0 <= change <= 1.0):
            raise ValueError("Efficiency change must be between -1.0 and 1.0")
        
        current_metrics = self.get_current_metrics()
        
        # Apply change with bounds checking
        new_efficiency = max(0.0, min(1.0, current_metrics.facility_efficiency + change))
        
        # Create new metrics snapshot
        new_metrics = MetricsSnapshot(
            visitor_satisfaction=current_metrics.visitor_satisfaction,
            dinosaur_happiness=current_metrics.dinosaur_happiness.copy(),
            facility_efficiency=new_efficiency,
            safety_rating=current_metrics.safety_rating,
            timestamp=datetime.now()
        )
        
        self.session_manager.add_metrics_snapshot(new_metrics)
        
        self.logger.info(f"Updated facility efficiency: {change:+.2f} -> {new_efficiency:.2f}")
    
    def update_safety_rating(self, change: float) -> None:
        """Update safety rating.
        
        Args:
            change: Change in safety rating (-1.0 to 1.0)
        """
        if not (-1.0 <= change <= 1.0):
            raise ValueError("Safety rating change must be between -1.0 and 1.0")
        
        current_metrics = self.get_current_metrics()
        
        # Apply change with bounds checking
        new_safety = max(0.0, min(1.0, current_metrics.safety_rating + change))
        
        # Create new metrics snapshot
        new_metrics = MetricsSnapshot(
            visitor_satisfaction=current_metrics.visitor_satisfaction,
            dinosaur_happiness=current_metrics.dinosaur_happiness.copy(),
            facility_efficiency=current_metrics.facility_efficiency,
            safety_rating=new_safety,
            timestamp=datetime.now()
        )
        
        self.session_manager.add_metrics_snapshot(new_metrics)
        
        self.logger.info(f"Updated safety rating: {change:+.2f} -> {new_safety:.2f}")
    
    def get_current_metrics(self) -> MetricsSnapshot:
        """Get current metrics snapshot.
        
        Returns:
            Current metrics snapshot
        """
        latest = self.session_manager.get_latest_metrics()
        if latest is None:
            self._initialize_default_metrics()
            latest = self.session_manager.get_latest_metrics()
        
        return latest
    
    def get_metric_history(self, metric_name: str, timeframe: str = "1h") -> List[Tuple[datetime, float]]:
        """Get historical data for a specific metric.
        
        Args:
            metric_name: Name of the metric ('visitor_satisfaction', 'facility_efficiency', 'safety_rating')
            timeframe: Time range ('1h', '6h', '24h', 'all')
            
        Returns:
            List of (timestamp, value) tuples
        """
        if metric_name not in ['visitor_satisfaction', 'facility_efficiency', 'safety_rating']:
            raise ValueError(f"Invalid metric name: {metric_name}")
        
        history = self.session_manager.get_metrics_history()
        
        # Filter by timeframe
        if timeframe != 'all':
            cutoff_time = self._get_cutoff_time(timeframe)
            history = [m for m in history if m.timestamp >= cutoff_time]
        
        # Extract metric values
        return [(m.timestamp, getattr(m, metric_name)) for m in history]
    
    def get_dinosaur_happiness_history(self, dinosaur_id: str, timeframe: str = "1h") -> List[Tuple[datetime, float]]:
        """Get happiness history for a specific dinosaur.
        
        Args:
            dinosaur_id: ID of the dinosaur agent
            timeframe: Time range ('1h', '6h', '24h', 'all')
            
        Returns:
            List of (timestamp, happiness) tuples
        """
        history = self.session_manager.get_metrics_history()
        
        # Filter by timeframe
        if timeframe != 'all':
            cutoff_time = self._get_cutoff_time(timeframe)
            history = [m for m in history if m.timestamp >= cutoff_time]
        
        # Extract dinosaur happiness values
        result = []
        for metrics in history:
            if dinosaur_id in metrics.dinosaur_happiness:
                result.append((metrics.timestamp, metrics.dinosaur_happiness[dinosaur_id]))
        
        return result
    
    def get_all_dinosaur_happiness(self) -> Dict[str, float]:
        """Get current happiness for all dinosaurs.
        
        Returns:
            Dictionary mapping dinosaur_id to happiness score
        """
        current_metrics = self.get_current_metrics()
        return current_metrics.dinosaur_happiness.copy()
    
    def calculate_overall_resort_score(self) -> float:
        """Calculate overall resort performance score.
        
        Returns:
            Overall score (0.0 to 1.0) based on weighted metrics
        """
        current_metrics = self.get_current_metrics()
        
        # Calculate average dinosaur happiness
        if current_metrics.dinosaur_happiness:
            avg_dinosaur_happiness = sum(current_metrics.dinosaur_happiness.values()) / len(current_metrics.dinosaur_happiness)
        else:
            avg_dinosaur_happiness = 0.8  # Default if no dinosaurs
        
        # Weighted average of all metrics
        weights = {
            'visitor_satisfaction': 0.3,
            'dinosaur_happiness': 0.3,
            'facility_efficiency': 0.2,
            'safety_rating': 0.2
        }
        
        overall_score = (
            weights['visitor_satisfaction'] * current_metrics.visitor_satisfaction +
            weights['dinosaur_happiness'] * avg_dinosaur_happiness +
            weights['facility_efficiency'] * current_metrics.facility_efficiency +
            weights['safety_rating'] * current_metrics.safety_rating
        )
        
        return round(overall_score, 3)
    
    def initialize_agent_metrics(self, agent: Agent) -> None:
        """Initialize metrics for a new agent.
        
        Args:
            agent: Agent to initialize metrics for
        """
        if agent.role == AgentRole.DINOSAUR:
            # Initialize dinosaur happiness based on species
            initial_happiness = self._get_initial_dinosaur_happiness(agent.species)
            
            current_metrics = self.get_current_metrics()
            dinosaur_happiness = current_metrics.dinosaur_happiness.copy()
            dinosaur_happiness[agent.id] = initial_happiness
            
            # Create new metrics snapshot
            new_metrics = MetricsSnapshot(
                visitor_satisfaction=current_metrics.visitor_satisfaction,
                dinosaur_happiness=dinosaur_happiness,
                facility_efficiency=current_metrics.facility_efficiency,
                safety_rating=current_metrics.safety_rating,
                timestamp=datetime.now()
            )
            
            self.session_manager.add_metrics_snapshot(new_metrics)
            
            self.logger.info(f"Initialized metrics for dinosaur {agent.name} ({agent.species.name}): {initial_happiness:.2f}")
    
    def remove_agent_metrics(self, agent_id: str) -> None:
        """Remove metrics for an agent that's being removed.
        
        Args:
            agent_id: ID of agent being removed
        """
        current_metrics = self.get_current_metrics()
        
        if agent_id in current_metrics.dinosaur_happiness:
            dinosaur_happiness = current_metrics.dinosaur_happiness.copy()
            del dinosaur_happiness[agent_id]
            
            # Create new metrics snapshot
            new_metrics = MetricsSnapshot(
                visitor_satisfaction=current_metrics.visitor_satisfaction,
                dinosaur_happiness=dinosaur_happiness,
                facility_efficiency=current_metrics.facility_efficiency,
                safety_rating=current_metrics.safety_rating,
                timestamp=datetime.now()
            )
            
            self.session_manager.add_metrics_snapshot(new_metrics)
            
            self.logger.info(f"Removed metrics for agent {agent_id}")
    
    def apply_event_impact(self, event_type: str, severity: int, affected_agents: List[str]) -> None:
        """Apply metric changes based on an event.
        
        Args:
            event_type: Type of event that occurred
            severity: Event severity (1-10)
            affected_agents: List of agent IDs affected by the event
        """
        # Calculate impact based on event type and severity
        impact_multiplier = severity / 10.0  # Normalize severity to 0.0-1.0
        
        # Define event impact patterns
        event_impacts = {
            'DINOSAUR_ESCAPE': {
                'visitor_satisfaction': -0.3 * impact_multiplier,
                'safety_rating': -0.5 * impact_multiplier,
                'facility_efficiency': -0.2 * impact_multiplier
            },
            'VISITOR_INJURY': {
                'visitor_satisfaction': -0.4 * impact_multiplier,
                'safety_rating': -0.6 * impact_multiplier
            },
            'FACILITY_MALFUNCTION': {
                'facility_efficiency': -0.4 * impact_multiplier,
                'visitor_satisfaction': -0.2 * impact_multiplier
            },
            'WEATHER_STORM': {
                'visitor_satisfaction': -0.2 * impact_multiplier,
                'facility_efficiency': -0.3 * impact_multiplier
            }
        }
        
        # Apply impacts if event type is recognized
        if event_type in event_impacts:
            impacts = event_impacts[event_type]
            
            if 'visitor_satisfaction' in impacts:
                self.update_visitor_satisfaction('system', impacts['visitor_satisfaction'])
            
            if 'safety_rating' in impacts:
                self.update_safety_rating(impacts['safety_rating'])
            
            if 'facility_efficiency' in impacts:
                self.update_facility_efficiency(impacts['facility_efficiency'])
            
            # Apply dinosaur-specific impacts
            agents = self.session_manager.get_agents()
            for agent_id in affected_agents:
                if agent_id in agents and agents[agent_id].role == AgentRole.DINOSAUR:
                    # Dinosaurs get stressed by most events
                    stress_impact = -0.1 * impact_multiplier
                    self.update_dinosaur_happiness(agent_id, stress_impact)
            
            self.logger.info(f"Applied event impact for {event_type} (severity {severity})")
    
    def _get_initial_dinosaur_happiness(self, species: Optional[DinosaurSpecies]) -> float:
        """Get initial happiness for a dinosaur based on species.
        
        Args:
            species: Dinosaur species
            
        Returns:
            Initial happiness score (0.0 to 1.0)
        """
        if species is None:
            return 0.8
        
        # Different species have different baseline happiness
        species_happiness = {
            DinosaurSpecies.TRICERATOPS: 0.9,  # Generally calm and happy
            DinosaurSpecies.VELOCIRAPTOR: 0.7,  # More aggressive, harder to please
            DinosaurSpecies.TYRANNOSAURUS_REX: 0.6,  # Apex predator, often restless
            DinosaurSpecies.BRACHIOSAURUS: 0.85,  # Gentle giants, usually content
            DinosaurSpecies.STEGOSAURUS: 0.8,  # Peaceful herbivore
            DinosaurSpecies.PARASAUROLOPHUS: 0.85,  # Social and generally happy
        }
        
        return species_happiness.get(species, 0.8)
    
    def _get_cutoff_time(self, timeframe: str) -> datetime:
        """Get cutoff time for historical data filtering.
        
        Args:
            timeframe: Time range string
            
        Returns:
            Cutoff datetime
        """
        now = datetime.now()
        
        if timeframe == '1h':
            return now - timedelta(hours=1)
        elif timeframe == '6h':
            return now - timedelta(hours=6)
        elif timeframe == '24h':
            return now - timedelta(hours=24)
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
    
    def get_metrics_summary(self) -> Dict[str, any]:
        """Get a summary of current metrics for display.
        
        Returns:
            Dictionary with formatted metrics information
        """
        current_metrics = self.get_current_metrics()
        
        # Calculate average dinosaur happiness
        if current_metrics.dinosaur_happiness:
            avg_dinosaur_happiness = sum(current_metrics.dinosaur_happiness.values()) / len(current_metrics.dinosaur_happiness)
            dinosaur_count = len(current_metrics.dinosaur_happiness)
        else:
            avg_dinosaur_happiness = 0.0
            dinosaur_count = 0
        
        return {
            'visitor_satisfaction': {
                'value': current_metrics.visitor_satisfaction,
                'percentage': f"{current_metrics.visitor_satisfaction * 100:.1f}%",
                'status': self._get_metric_status(current_metrics.visitor_satisfaction)
            },
            'dinosaur_happiness': {
                'value': avg_dinosaur_happiness,
                'percentage': f"{avg_dinosaur_happiness * 100:.1f}%",
                'count': dinosaur_count,
                'status': self._get_metric_status(avg_dinosaur_happiness)
            },
            'facility_efficiency': {
                'value': current_metrics.facility_efficiency,
                'percentage': f"{current_metrics.facility_efficiency * 100:.1f}%",
                'status': self._get_metric_status(current_metrics.facility_efficiency)
            },
            'safety_rating': {
                'value': current_metrics.safety_rating,
                'percentage': f"{current_metrics.safety_rating * 100:.1f}%",
                'status': self._get_metric_status(current_metrics.safety_rating)
            },
            'overall_score': {
                'value': self.calculate_overall_resort_score(),
                'percentage': f"{self.calculate_overall_resort_score() * 100:.1f}%",
                'status': self._get_metric_status(self.calculate_overall_resort_score())
            },
            'last_updated': current_metrics.timestamp
        }
    
    def _get_metric_status(self, value: float) -> str:
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