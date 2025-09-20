"""Unit tests for MetricsManager."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from models.core import MetricsSnapshot, Agent
from models.enums import AgentRole, AgentState, DinosaurSpecies
from models.config import Location
from managers.metrics_manager import MetricsManager
from ui.session_state import SessionStateManager


class TestMetricsManager:
    """Test cases for MetricsManager class."""
    
    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager for testing."""
        mock_manager = Mock(spec=SessionStateManager)
        mock_manager.get_latest_metrics.return_value = None
        mock_manager.get_metrics_history.return_value = []
        mock_manager.get_agents.return_value = {}
        return mock_manager
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing."""
        return MetricsSnapshot(
            visitor_satisfaction=0.8,
            dinosaur_happiness={'dino1': 0.7, 'dino2': 0.9},
            facility_efficiency=0.9,
            safety_rating=0.95,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def sample_agent(self):
        """Create sample agent for testing."""
        return Agent(
            id='test_agent',
            name='Test Dinosaur',
            role=AgentRole.DINOSAUR,
            species=DinosaurSpecies.TRICERATOPS,
            location=Location(0.0, 0.0, 'paddock')
        )
    
    def test_initialization_with_no_existing_metrics(self, mock_session_manager):
        """Test MetricsManager initialization when no metrics exist."""
        manager = MetricsManager(mock_session_manager)
        
        # Should call add_metrics_snapshot to initialize default metrics
        mock_session_manager.add_metrics_snapshot.assert_called_once()
        
        # Check that default metrics were created
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert isinstance(call_args, MetricsSnapshot)
        assert call_args.visitor_satisfaction == 0.8
        assert call_args.facility_efficiency == 1.0
        assert call_args.safety_rating == 1.0
        assert call_args.dinosaur_happiness == {}
    
    def test_initialization_with_existing_metrics(self, mock_session_manager, sample_metrics):
        """Test MetricsManager initialization when metrics already exist."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        
        manager = MetricsManager(mock_session_manager)
        
        # Should not call add_metrics_snapshot since metrics exist
        mock_session_manager.add_metrics_snapshot.assert_not_called()
    
    def test_update_visitor_satisfaction_valid_change(self, mock_session_manager, sample_metrics):
        """Test updating visitor satisfaction with valid change."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        manager.update_visitor_satisfaction('visitor1', 0.1)
        
        # Should call add_metrics_snapshot with updated metrics
        mock_session_manager.add_metrics_snapshot.assert_called()
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert call_args.visitor_satisfaction == 0.9  # 0.8 + 0.1
    
    def test_update_visitor_satisfaction_bounds_checking(self, mock_session_manager, sample_metrics):
        """Test visitor satisfaction bounds checking."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        # Test upper bound
        manager.update_visitor_satisfaction('visitor1', 0.5)  # Should cap at 1.0
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert call_args.visitor_satisfaction == 1.0
        
        # Test lower bound
        sample_metrics.visitor_satisfaction = 0.1
        manager.update_visitor_satisfaction('visitor1', -0.5)  # Should cap at 0.0
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert call_args.visitor_satisfaction == 0.0
    
    def test_update_visitor_satisfaction_invalid_change(self, mock_session_manager, sample_metrics):
        """Test visitor satisfaction update with invalid change value."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        with pytest.raises(ValueError, match="Satisfaction change must be between -1.0 and 1.0"):
            manager.update_visitor_satisfaction('visitor1', 1.5)
        
        with pytest.raises(ValueError, match="Satisfaction change must be between -1.0 and 1.0"):
            manager.update_visitor_satisfaction('visitor1', -1.5)
    
    def test_update_dinosaur_happiness_existing_dinosaur(self, mock_session_manager, sample_metrics):
        """Test updating happiness for existing dinosaur."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        manager.update_dinosaur_happiness('dino1', 0.2)
        
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert abs(call_args.dinosaur_happiness['dino1'] - 0.9) < 0.001  # 0.7 + 0.2
        assert call_args.dinosaur_happiness['dino2'] == 0.9  # Unchanged
    
    def test_update_dinosaur_happiness_new_dinosaur(self, mock_session_manager, sample_metrics):
        """Test updating happiness for new dinosaur."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        manager.update_dinosaur_happiness('dino3', 0.1)
        
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert call_args.dinosaur_happiness['dino3'] == 0.9  # 0.8 default + 0.1
        assert 'dino1' in call_args.dinosaur_happiness  # Existing ones preserved
    
    def test_update_dinosaur_happiness_bounds_checking(self, mock_session_manager, sample_metrics):
        """Test dinosaur happiness bounds checking."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        # Test upper bound
        manager.update_dinosaur_happiness('dino2', 0.5)  # 0.9 + 0.5 should cap at 1.0
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert call_args.dinosaur_happiness['dino2'] == 1.0
        
        # Test lower bound
        sample_metrics.dinosaur_happiness['dino1'] = 0.1
        manager.update_dinosaur_happiness('dino1', -0.5)  # Should cap at 0.0
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert call_args.dinosaur_happiness['dino1'] == 0.0
    
    def test_update_dinosaur_happiness_invalid_change(self, mock_session_manager, sample_metrics):
        """Test dinosaur happiness update with invalid change value."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        with pytest.raises(ValueError, match="Happiness change must be between -1.0 and 1.0"):
            manager.update_dinosaur_happiness('dino1', 1.5)
    
    def test_update_facility_efficiency(self, mock_session_manager, sample_metrics):
        """Test updating facility efficiency."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        manager.update_facility_efficiency(-0.1)
        
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert call_args.facility_efficiency == 0.8  # 0.9 - 0.1
    
    def test_update_facility_efficiency_bounds_checking(self, mock_session_manager, sample_metrics):
        """Test facility efficiency bounds checking."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        # Test upper bound
        manager.update_facility_efficiency(0.5)  # Should cap at 1.0
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert call_args.facility_efficiency == 1.0
        
        # Test lower bound
        sample_metrics.facility_efficiency = 0.1
        manager.update_facility_efficiency(-0.5)  # Should cap at 0.0
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert call_args.facility_efficiency == 0.0
    
    def test_update_safety_rating(self, mock_session_manager, sample_metrics):
        """Test updating safety rating."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        manager.update_safety_rating(-0.05)
        
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert abs(call_args.safety_rating - 0.9) < 0.001  # 0.95 - 0.05
    
    def test_get_current_metrics_with_existing(self, mock_session_manager, sample_metrics):
        """Test getting current metrics when they exist."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        result = manager.get_current_metrics()
        
        assert result == sample_metrics
    
    def test_get_current_metrics_without_existing(self, mock_session_manager):
        """Test getting current metrics when none exist."""
        # Mock the behavior where first call returns None, second call returns the initialized metrics
        default_metrics = MetricsSnapshot(
            visitor_satisfaction=0.8,
            dinosaur_happiness={},
            facility_efficiency=1.0,
            safety_rating=1.0
        )
        mock_session_manager.get_latest_metrics.side_effect = [None, default_metrics]
        manager = MetricsManager(mock_session_manager)
        
        # Should initialize and return default metrics
        result = manager.get_current_metrics()
        
        assert isinstance(result, MetricsSnapshot)
        mock_session_manager.add_metrics_snapshot.assert_called()
    
    def test_get_metric_history_visitor_satisfaction(self, mock_session_manager):
        """Test getting visitor satisfaction history."""
        # Create sample history
        now = datetime.now()
        history = [
            MetricsSnapshot(visitor_satisfaction=0.8, timestamp=now - timedelta(minutes=30)),
            MetricsSnapshot(visitor_satisfaction=0.7, timestamp=now - timedelta(minutes=15)),
            MetricsSnapshot(visitor_satisfaction=0.9, timestamp=now)
        ]
        mock_session_manager.get_metrics_history.return_value = history
        
        manager = MetricsManager(mock_session_manager)
        result = manager.get_metric_history('visitor_satisfaction', '1h')
        
        assert len(result) == 3
        assert result[0] == (history[0].timestamp, 0.8)
        assert result[1] == (history[1].timestamp, 0.7)
        assert result[2] == (history[2].timestamp, 0.9)
    
    def test_get_metric_history_invalid_metric(self, mock_session_manager):
        """Test getting history for invalid metric name."""
        manager = MetricsManager(mock_session_manager)
        
        with pytest.raises(ValueError, match="Invalid metric name"):
            manager.get_metric_history('invalid_metric', '1h')
    
    def test_get_metric_history_timeframe_filtering(self, mock_session_manager):
        """Test metric history timeframe filtering."""
        now = datetime.now()
        history = [
            MetricsSnapshot(visitor_satisfaction=0.8, timestamp=now - timedelta(hours=2)),  # Outside 1h
            MetricsSnapshot(visitor_satisfaction=0.7, timestamp=now - timedelta(minutes=30)),  # Within 1h
            MetricsSnapshot(visitor_satisfaction=0.9, timestamp=now)  # Within 1h
        ]
        mock_session_manager.get_metrics_history.return_value = history
        
        manager = MetricsManager(mock_session_manager)
        result = manager.get_metric_history('visitor_satisfaction', '1h')
        
        # Should only return the last 2 entries (within 1 hour)
        assert len(result) == 2
        assert result[0][1] == 0.7
        assert result[1][1] == 0.9
    
    def test_get_dinosaur_happiness_history(self, mock_session_manager):
        """Test getting dinosaur happiness history."""
        now = datetime.now()
        history = [
            MetricsSnapshot(visitor_satisfaction=0.8, dinosaur_happiness={'dino1': 0.8}, timestamp=now - timedelta(minutes=30)),
            MetricsSnapshot(visitor_satisfaction=0.8, dinosaur_happiness={'dino1': 0.7}, timestamp=now - timedelta(minutes=15)),
            MetricsSnapshot(visitor_satisfaction=0.8, dinosaur_happiness={'dino1': 0.9}, timestamp=now)
        ]
        mock_session_manager.get_metrics_history.return_value = history
        
        manager = MetricsManager(mock_session_manager)
        result = manager.get_dinosaur_happiness_history('dino1', '1h')
        
        assert len(result) == 3
        assert result[0] == (history[0].timestamp, 0.8)
        assert result[1] == (history[1].timestamp, 0.7)
        assert result[2] == (history[2].timestamp, 0.9)
    
    def test_get_all_dinosaur_happiness(self, mock_session_manager, sample_metrics):
        """Test getting all dinosaur happiness scores."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        result = manager.get_all_dinosaur_happiness()
        
        assert result == {'dino1': 0.7, 'dino2': 0.9}
        # Ensure it's a copy, not the original
        result['dino1'] = 0.5
        assert sample_metrics.dinosaur_happiness['dino1'] == 0.7
    
    def test_calculate_overall_resort_score(self, mock_session_manager, sample_metrics):
        """Test calculating overall resort score."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        result = manager.calculate_overall_resort_score()
        
        # Expected calculation:
        # visitor_satisfaction: 0.8 * 0.3 = 0.24
        # avg_dinosaur_happiness: 0.8 * 0.3 = 0.24  # (0.7 + 0.9) / 2 = 0.8
        # facility_efficiency: 0.9 * 0.2 = 0.18
        # safety_rating: 0.95 * 0.2 = 0.19
        # Total: 0.85
        expected = 0.85
        assert abs(result - expected) < 0.01  # Allow for floating point precision
    
    def test_calculate_overall_resort_score_no_dinosaurs(self, mock_session_manager):
        """Test calculating overall resort score with no dinosaurs."""
        metrics = MetricsSnapshot(
            visitor_satisfaction=0.8,
            dinosaur_happiness={},  # No dinosaurs
            facility_efficiency=0.9,
            safety_rating=0.95
        )
        mock_session_manager.get_latest_metrics.return_value = metrics
        manager = MetricsManager(mock_session_manager)
        
        result = manager.calculate_overall_resort_score()
        
        # Should use default 0.8 for dinosaur happiness
        expected = 0.8 * 0.3 + 0.8 * 0.3 + 0.9 * 0.2 + 0.95 * 0.2
        assert abs(result - expected) < 0.01
    
    def test_initialize_agent_metrics_dinosaur(self, mock_session_manager, sample_metrics, sample_agent):
        """Test initializing metrics for a dinosaur agent."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        manager.initialize_agent_metrics(sample_agent)
        
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert 'test_agent' in call_args.dinosaur_happiness
        # Triceratops should have initial happiness of 0.9
        assert call_args.dinosaur_happiness['test_agent'] == 0.9
    
    def test_initialize_agent_metrics_non_dinosaur(self, mock_session_manager, sample_metrics):
        """Test initializing metrics for a non-dinosaur agent."""
        agent = Agent(
            id='staff1',
            name='Park Ranger',
            role=AgentRole.PARK_RANGER,
            location=Location(0.0, 0.0, 'entrance')
        )
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        manager.initialize_agent_metrics(agent)
        
        # Should not call add_metrics_snapshot for non-dinosaur agents
        mock_session_manager.add_metrics_snapshot.assert_not_called()
    
    def test_remove_agent_metrics(self, mock_session_manager, sample_metrics):
        """Test removing agent metrics."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        manager.remove_agent_metrics('dino1')
        
        call_args = mock_session_manager.add_metrics_snapshot.call_args[0][0]
        assert 'dino1' not in call_args.dinosaur_happiness
        assert 'dino2' in call_args.dinosaur_happiness  # Other dinosaurs preserved
    
    def test_remove_agent_metrics_nonexistent(self, mock_session_manager, sample_metrics):
        """Test removing metrics for non-existent agent."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        manager.remove_agent_metrics('nonexistent')
        
        # Should not call add_metrics_snapshot since agent doesn't exist
        mock_session_manager.add_metrics_snapshot.assert_not_called()
    
    def test_apply_event_impact_dinosaur_escape(self, mock_session_manager, sample_metrics):
        """Test applying event impact for dinosaur escape."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        mock_session_manager.get_agents.return_value = {
            'dino1': Agent(id='dino1', name='Rex', role=AgentRole.DINOSAUR, species=DinosaurSpecies.TYRANNOSAURUS_REX)
        }
        manager = MetricsManager(mock_session_manager)
        
        manager.apply_event_impact('DINOSAUR_ESCAPE', 8, ['dino1'])
        
        # Should be called multiple times for different metric updates
        assert mock_session_manager.add_metrics_snapshot.call_count >= 3
    
    def test_apply_event_impact_unknown_event(self, mock_session_manager, sample_metrics):
        """Test applying event impact for unknown event type."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        manager.apply_event_impact('UNKNOWN_EVENT', 5, [])
        
        # Should not call add_metrics_snapshot for unknown events
        mock_session_manager.add_metrics_snapshot.assert_not_called()
    
    def test_get_initial_dinosaur_happiness_by_species(self, mock_session_manager):
        """Test getting initial happiness for different dinosaur species."""
        manager = MetricsManager(mock_session_manager)
        
        # Test different species
        assert manager._get_initial_dinosaur_happiness(DinosaurSpecies.TRICERATOPS) == 0.9
        assert manager._get_initial_dinosaur_happiness(DinosaurSpecies.VELOCIRAPTOR) == 0.7
        assert manager._get_initial_dinosaur_happiness(DinosaurSpecies.TYRANNOSAURUS_REX) == 0.6
        assert manager._get_initial_dinosaur_happiness(DinosaurSpecies.BRACHIOSAURUS) == 0.85
        assert manager._get_initial_dinosaur_happiness(None) == 0.8
    
    def test_get_metrics_summary(self, mock_session_manager, sample_metrics):
        """Test getting metrics summary for display."""
        mock_session_manager.get_latest_metrics.return_value = sample_metrics
        manager = MetricsManager(mock_session_manager)
        
        result = manager.get_metrics_summary()
        
        assert 'visitor_satisfaction' in result
        assert 'dinosaur_happiness' in result
        assert 'facility_efficiency' in result
        assert 'safety_rating' in result
        assert 'overall_score' in result
        
        # Check visitor satisfaction
        assert result['visitor_satisfaction']['value'] == 0.8
        assert result['visitor_satisfaction']['percentage'] == "80.0%"
        assert result['visitor_satisfaction']['status'] == "Excellent"  # 0.8 is >= 0.8, so it's Excellent
        
        # Check dinosaur happiness
        assert result['dinosaur_happiness']['count'] == 2
        assert result['dinosaur_happiness']['value'] == 0.8  # (0.7 + 0.9) / 2
    
    def test_get_metric_status(self, mock_session_manager):
        """Test getting metric status descriptions."""
        manager = MetricsManager(mock_session_manager)
        
        assert manager._get_metric_status(0.9) == "Excellent"
        assert manager._get_metric_status(0.7) == "Good"
        assert manager._get_metric_status(0.5) == "Fair"
        assert manager._get_metric_status(0.3) == "Poor"
        assert manager._get_metric_status(0.1) == "Critical"
    
    def test_get_cutoff_time(self, mock_session_manager):
        """Test getting cutoff time for different timeframes."""
        manager = MetricsManager(mock_session_manager)
        
        now = datetime.now()
        with patch('managers.metrics_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            
            cutoff_1h = manager._get_cutoff_time('1h')
            cutoff_6h = manager._get_cutoff_time('6h')
            cutoff_24h = manager._get_cutoff_time('24h')
            
            assert cutoff_1h == now - timedelta(hours=1)
            assert cutoff_6h == now - timedelta(hours=6)
            assert cutoff_24h == now - timedelta(hours=24)
        
        with pytest.raises(ValueError, match="Invalid timeframe"):
            manager._get_cutoff_time('invalid')


if __name__ == '__main__':
    pytest.main([__file__])