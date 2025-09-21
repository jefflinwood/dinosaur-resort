"""Tests for metrics dashboard UI functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from models.core import MetricsSnapshot
from ui.session_state import SessionStateManager
from typing import Optional


class TestMetricsDashboard:
    """Test suite for metrics dashboard functionality."""
    
    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session state manager with test data."""
        manager = Mock(spec=SessionStateManager)
        
        # Create test metrics data
        now = datetime.now()
        test_metrics = [
            MetricsSnapshot(
                visitor_satisfaction=0.8,
                dinosaur_happiness={'T-Rex-001': 0.7, 'Triceratops-001': 0.9},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now - timedelta(hours=2)
            ),
            MetricsSnapshot(
                visitor_satisfaction=0.75,
                dinosaur_happiness={'T-Rex-001': 0.65, 'Triceratops-001': 0.85},
                facility_efficiency=0.85,
                safety_rating=0.8,
                timestamp=now - timedelta(hours=1)
            ),
            MetricsSnapshot(
                visitor_satisfaction=0.82,
                dinosaur_happiness={'T-Rex-001': 0.72, 'Triceratops-001': 0.92},
                facility_efficiency=0.92,
                safety_rating=0.88,
                timestamp=now
            )
        ]
        
        manager.get_latest_metrics.return_value = test_metrics[-1]
        manager.get_metrics_history.return_value = test_metrics
        
        return manager
    
    @pytest.fixture
    def mock_metrics_manager(self):
        """Create a mock metrics manager."""
        manager = Mock()
        manager.get_metrics_summary.return_value = {
            'visitor_satisfaction': {
                'value': 0.82,
                'percentage': '82.0%',
                'status': 'Excellent'
            },
            'dinosaur_happiness': {
                'value': 0.82,
                'percentage': '82.0%',
                'count': 2,
                'status': 'Excellent'
            },
            'facility_efficiency': {
                'value': 0.92,
                'percentage': '92.0%',
                'status': 'Excellent'
            },
            'safety_rating': {
                'value': 0.88,
                'percentage': '88.0%',
                'status': 'Excellent'
            },
            'overall_score': {
                'value': 0.86,
                'percentage': '86.0%',
                'status': 'Excellent'
            },
            'last_updated': datetime.now()
        }
        manager.calculate_overall_resort_score.return_value = 0.86
        return manager
    
    def test_metrics_dashboard_no_data(self, mock_session_manager):
        """Test metrics dashboard when no data is available."""
        # Mock no metrics data
        mock_session_manager.get_latest_metrics.return_value = None
        mock_session_manager.get_metrics_history.return_value = []
        
        with patch('streamlit.title') as mock_title, \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.info') as mock_info:
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            mock_title.assert_called_with("📊 Metrics Dashboard")
            mock_info.assert_called_with("No metrics data available yet. Start the simulation to begin tracking metrics.")
    
    def test_real_time_metrics_display(self, mock_session_manager):
        """Test real-time metrics display with current values."""
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'):
            
            # Mock columns return with context manager support - flexible return based on call
            mock_col = Mock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            
            def mock_columns_side_effect(cols):
                if isinstance(cols, list):
                    return [mock_col] * len(cols)
                else:
                    return [mock_col] * cols
            
            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify metrics are displayed
            assert mock_metric.call_count >= 4  # At least 4 main metrics
            
            # Check that visitor satisfaction metric is called
            metric_calls = [call[0] for call in mock_metric.call_args_list]
            assert any("👥 Visitor Satisfaction" in str(call) for call in metric_calls)
            assert any("🛡️ Safety Rating" in str(call) for call in metric_calls)
            assert any("🏭 Facility Efficiency" in str(call) for call in metric_calls)
    
    def test_dinosaur_happiness_display(self, mock_session_manager):
        """Test dinosaur happiness overview display."""
        with patch('streamlit.title'), \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress') as mock_progress:
            
            mock_col = Mock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify dinosaur happiness section is displayed
            write_calls = [str(call) for call in mock_write.call_args_list]
            assert any("Individual Dinosaur Status" in call for call in write_calls)
            
            # Verify progress bars are created for dinosaurs
            assert mock_progress.call_count >= 2  # At least 2 dinosaurs
    
    def test_historical_trends_insufficient_data(self, mock_session_manager):
        """Test historical trends section when insufficient data is available."""
        # Mock insufficient historical data
        mock_session_manager.get_metrics_history.return_value = [mock_session_manager.get_latest_metrics.return_value]
        
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.info') as mock_info:
            
            mock_col = Mock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            
            mock_checkbox.return_value = False
            mock_button.return_value = False
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify insufficient data message is shown
            info_calls = [str(call) for call in mock_info.call_args_list]
            assert any("Not enough historical data" in call for call in info_calls)
    
    def test_time_range_filtering(self, mock_session_manager):
        """Test time range selection and filtering functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.number_input') as mock_number_input:
            
            mock_col = Mock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]  # Refresh rate, time range, chart type
            mock_multiselect.return_value = ["Visitor Satisfaction", "Safety Rating"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify time range selector is called
            selectbox_calls = [call[0] for call in mock_selectbox.call_args_list]
            assert any("Time Range" in str(call) for call in selectbox_calls)
            
            # Verify metrics selector is called
            multiselect_calls = [call[0] for call in mock_multiselect.call_args_list]
            assert any("Metrics to Display" in str(call) for call in multiselect_calls)
    
    @patch('pandas.DataFrame')
    def test_chart_visualization(self, mock_dataframe, mock_session_manager):
        """Test chart visualization functionality."""
        # Mock pandas DataFrame
        mock_df = Mock()
        mock_df.empty = False
        mock_df.columns = ['Visitor Satisfaction', 'Safety Rating']
        mock_dataframe.return_value = mock_df
        
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.line_chart') as mock_line_chart:
            
            mock_col = Mock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]  # refresh_rate, time_range, chart_type
            mock_multiselect.return_value = ["Visitor Satisfaction", "Safety Rating"]
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify line chart is created
            mock_line_chart.assert_called_once()
    
    def test_trend_analysis(self, mock_session_manager):
        """Test trend analysis functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.line_chart') as mock_line_chart, \
             patch('pandas.DataFrame') as mock_dataframe:
            
            # Mock pandas DataFrame with trend data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.columns = ['Visitor Satisfaction']
            mock_series = Mock()
            mock_series.mean.return_value = 0.8
            mock_series.min.return_value = 0.7
            mock_series.max.return_value = 0.9
            mock_df.__getitem__.return_value = mock_series
            mock_dataframe.return_value = mock_df
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]  # refresh_rate, time_range, chart_type
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify trend analysis section is displayed
            write_calls = [str(call) for call in mock_write.call_args_list]
            assert any("Recent Trends" in call for call in write_calls)
            assert any("Statistics" in call for call in write_calls)
            
            # Verify line chart is called
            mock_line_chart.assert_called()
    
    def test_data_export_functionality(self, mock_session_manager):
        """Test data export functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.download_button') as mock_download:
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.side_effect = [False, False, True, False]  # Export current metrics button
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]  # refresh_rate, time_range, chart_type
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify download button is created
            mock_download.assert_called()
    
    def test_auto_refresh_functionality(self, mock_session_manager):
        """Test auto-refresh functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.rerun') as mock_rerun, \
             patch('time.sleep') as mock_sleep:
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = True  # Auto-refresh enabled
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]  # refresh_rate, time_range, chart_type
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify auto-refresh triggers sleep and rerun
            mock_sleep.assert_called_with(5)  # Should use the refresh_rate value
            mock_rerun.assert_called()
    
    def test_metric_status_helper_functions(self):
        """Test helper functions for metric status and colors."""
        from main import _get_metric_status, _get_status_color
        
        # Test status function
        assert _get_metric_status(0.9) == "Excellent"
        assert _get_metric_status(0.7) == "Good"
        assert _get_metric_status(0.5) == "Fair"
        assert _get_metric_status(0.3) == "Poor"
        assert _get_metric_status(0.1) == "Critical"
        
        # Test color function
        assert _get_status_color(0.9) == "#28a745"  # Green
        assert _get_status_color(0.7) == "#17a2b8"  # Blue
        assert _get_status_color(0.5) == "#ffc107"  # Yellow
        assert _get_status_color(0.3) == "#fd7e14"  # Orange
        assert _get_status_color(0.1) == "#dc3545"  # Red
    
    def test_metric_delta_calculation(self):
        """Test metric delta calculation helper function."""
        from main import _calculate_metric_delta
        from models.core import MetricsSnapshot
        
        now = datetime.now()
        history = [
            MetricsSnapshot(
                visitor_satisfaction=0.8,
                dinosaur_happiness={},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now - timedelta(hours=1)
            ),
            MetricsSnapshot(
                visitor_satisfaction=0.84,  # 5% increase
                dinosaur_happiness={},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now
            )
        ]
        
        delta = _calculate_metric_delta(history, 'visitor_satisfaction')
        assert abs(delta - 5.0) < 0.1  # Should be approximately 5% increase
        
        # Test with insufficient data
        delta = _calculate_metric_delta(history[:1], 'visitor_satisfaction')
        assert delta is None
    
    def test_timerange_filtering_helper(self):
        """Test time range filtering helper function."""
        from main import _filter_history_by_timerange
        from models.core import MetricsSnapshot
        
        now = datetime.now()
        history = [
            MetricsSnapshot(
                visitor_satisfaction=0.8,
                dinosaur_happiness={},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now - timedelta(hours=25)  # Outside 24h range
            ),
            MetricsSnapshot(
                visitor_satisfaction=0.82,
                dinosaur_happiness={},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now - timedelta(hours=2)  # Within all ranges
            ),
            MetricsSnapshot(
                visitor_satisfaction=0.84,
                dinosaur_happiness={},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now  # Current
            )
        ]
        
        # Test 24h filter
        filtered = _filter_history_by_timerange(history, '24h')
        assert len(filtered) == 2  # Should exclude the 25h old entry
        
        # Test 1h filter
        filtered = _filter_history_by_timerange(history, '1h')
        assert len(filtered) == 1  # Should only include current entry
        
        # Test 'all' filter
        filtered = _filter_history_by_timerange(history, 'all')
        assert len(filtered) == 3  # Should include all entries
    
    def test_chart_data_preparation(self):
        """Test chart data preparation helper function."""
        from main import _prepare_chart_data
        from models.core import MetricsSnapshot
        
        now = datetime.now()
        history = [
            MetricsSnapshot(
                visitor_satisfaction=0.8,
                dinosaur_happiness={'T-Rex-001': 0.7},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now - timedelta(hours=1)
            ),
            MetricsSnapshot(
                visitor_satisfaction=0.82,
                dinosaur_happiness={'T-Rex-001': 0.72},
                facility_efficiency=0.92,
                safety_rating=0.87,
                timestamp=now
            )
        ]
        
        metrics_to_show = ["Visitor Satisfaction", "Safety Rating"]
        
        with patch('pandas.DataFrame') as mock_dataframe:
            mock_df = Mock()
            mock_dataframe.return_value = mock_df
            
            result = _prepare_chart_data(history, metrics_to_show)
            
            # Verify DataFrame was created
            mock_dataframe.assert_called_once()
            assert result == mock_df
    
    def test_trend_calculation(self):
        """Test trend calculation helper function."""
        from main import _calculate_trend
        
        # Mock pandas Series
        with patch('pandas.Series') as mock_series:
            # Test improving trend
            mock_values = MagicMock()
            mock_values.__len__.return_value = 4
            
            # Mock the slicing behavior
            first_half = Mock()
            first_half.mean.return_value = 0.7
            second_half = Mock()
            second_half.mean.return_value = 0.8
            
            mock_values.__getitem__.side_effect = [first_half, second_half]
            
            trend = _calculate_trend(mock_values)
            expected_trend = ((0.8 - 0.7) / 0.7) * 100  # ~14.3% improvement
            assert abs(trend - expected_trend) < 0.1
    
    def test_critical_alerts_display(self, mock_session_manager):
        """Test critical alerts display for low metrics."""
        # Create metrics with critical values
        now = datetime.now()
        critical_metrics = MetricsSnapshot(
            visitor_satisfaction=0.2,  # Critical
            dinosaur_happiness={'T-Rex-001': 0.3},
            facility_efficiency=0.25,  # Critical
            safety_rating=0.35,  # Critical
            timestamp=now
        )
        
        mock_session_manager.get_latest_metrics.return_value = critical_metrics
        mock_session_manager.get_metrics_history.return_value = [critical_metrics]
        
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.number_input') as mock_number_input:
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify critical alerts are displayed
            error_calls = [str(call) for call in mock_error.call_args_list]
            assert any("CRITICAL ALERTS" in call for call in error_calls)
            assert any("visitor satisfaction" in call.lower() for call in error_calls)
            assert any("safety rating" in call.lower() for call in error_calls)
            assert any("facility efficiency" in call.lower() for call in error_calls)
    
    def test_enhanced_time_range_options(self):
        """Test enhanced time range filtering with new options."""
        from main import _filter_history_by_timerange
        from models.core import MetricsSnapshot
        
        now = datetime.now()
        history = [
            MetricsSnapshot(
                visitor_satisfaction=0.8,
                dinosaur_happiness={},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now - timedelta(days=8)  # Outside 7d range
            ),
            MetricsSnapshot(
                visitor_satisfaction=0.82,
                dinosaur_happiness={},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now - timedelta(minutes=30)  # Within all ranges
            ),
            MetricsSnapshot(
                visitor_satisfaction=0.84,
                dinosaur_happiness={},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now - timedelta(minutes=5)  # Within 15m range
            )
        ]
        
        # Test 15m filter
        filtered = _filter_history_by_timerange(history, '15m')
        assert len(filtered) == 1  # Should only include 5m old entry
        
        # Test 7d filter
        filtered = _filter_history_by_timerange(history, '7d')
        assert len(filtered) == 2  # Should exclude the 8d old entry
    
    def test_scatter_plot_visualization(self, mock_session_manager):
        """Test scatter plot chart type functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.scatter_chart') as mock_scatter_chart, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('pandas.DataFrame') as mock_dataframe:
            
            # Mock pandas DataFrame
            mock_df = Mock()
            mock_df.empty = False
            mock_df.columns = ['Visitor Satisfaction', 'Safety Rating']
            mock_dataframe.return_value = mock_df
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.side_effect = [False, True]  # Auto-refresh off, trend lines on
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Scatter Plot"]  # Refresh rate, time range, chart type
            mock_multiselect.return_value = ["Visitor Satisfaction", "Safety Rating"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify scatter chart is created
            mock_scatter_chart.assert_called_once()
    
    def test_correlation_analysis(self, mock_session_manager):
        """Test correlation analysis functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.line_chart'), \
             patch('streamlit.number_input') as mock_number_input, \
             patch('pandas.DataFrame') as mock_dataframe:
            
            # Mock pandas DataFrame with correlation
            mock_df = Mock()
            mock_df.empty = False
            mock_df.columns = ['Visitor Satisfaction', 'Safety Rating']
            
            # Mock correlation matrix
            mock_corr = Mock()
            mock_corr.columns = ['Visitor Satisfaction', 'Safety Rating']
            mock_corr.iloc = Mock()
            mock_corr.iloc.__getitem__ = Mock(return_value=0.8)  # Strong positive correlation
            mock_df.corr.return_value = mock_corr
            
            mock_dataframe.return_value = mock_df
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.side_effect = [False, True]  # Auto-refresh off, trend lines on
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]
            mock_multiselect.return_value = ["Visitor Satisfaction", "Safety Rating"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify correlation analysis is displayed
            write_calls = [str(call) for call in mock_write.call_args_list]
            assert any("Correlation Analysis" in call for call in write_calls)
            assert any("Strong positive correlation" in call for call in write_calls)
    
    def test_performance_benchmarking(self, mock_session_manager):
        """Test performance benchmarking functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress') as mock_progress, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('streamlit.info'):
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            mock_number_input.return_value = 90.0  # Custom goal
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify benchmarking sections are displayed
            write_calls = [str(call) for call in mock_write.call_args_list]
            assert any("Industry Standards" in call for call in write_calls)
            assert any("Goal Tracking" in call for call in write_calls)
            assert any("Progress:" in call for call in write_calls)
            
            # Verify progress bars are created for goals
            assert mock_progress.call_count > 0
    
    def test_advanced_analysis_section(self, mock_session_manager):
        """Test advanced analysis section functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('streamlit.info'), \
             patch('pandas.DataFrame') as mock_dataframe:
            
            # Mock pandas DataFrame with analysis data
            mock_df = Mock()
            mock_df.empty = False
            mock_df.columns = ['Visitor Satisfaction']
            
            # Mock series for analysis
            mock_series = MagicMock()
            mock_series.std.return_value = 8.5  # Medium volatility
            mock_series.tail.return_value = Mock(mean=lambda: 82.0)
            mock_series.mean.return_value = 80.0
            mock_series.head.return_value = Mock(mean=lambda: 78.0)
            mock_series.iloc = [-1]  # Last value
            mock_series.__getitem__.return_value = 81.0
            mock_series.__len__.return_value = 10
            
            mock_df.__getitem__.return_value = mock_series
            mock_dataframe.return_value = mock_df
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify advanced analysis sections are displayed
            write_calls = [str(call) for call in mock_write.call_args_list]
            assert any("Advanced Analysis" in call for call in write_calls)
            assert any("Performance Insights" in call for call in write_calls)
            assert any("Predictive Indicators" in call for call in write_calls)
            assert any("Volatility:" in call for call in write_calls)


class TestMetricsDashboardIntegration:
    """Integration tests for metrics dashboard with simulation manager."""
    
    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session state manager with test data."""
        manager = Mock(spec=SessionStateManager)
        
        # Create test metrics data
        now = datetime.now()
        test_metrics = [
            MetricsSnapshot(
                visitor_satisfaction=0.8,
                dinosaur_happiness={'T-Rex-001': 0.7, 'Triceratops-001': 0.9},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now - timedelta(hours=2)
            ),
            MetricsSnapshot(
                visitor_satisfaction=0.75,
                dinosaur_happiness={'T-Rex-001': 0.65, 'Triceratops-001': 0.85},
                facility_efficiency=0.85,
                safety_rating=0.8,
                timestamp=now - timedelta(hours=1)
            ),
            MetricsSnapshot(
                visitor_satisfaction=0.82,
                dinosaur_happiness={'T-Rex-001': 0.72, 'Triceratops-001': 0.92},
                facility_efficiency=0.92,
                safety_rating=0.88,
                timestamp=now
            )
        ]
        
        manager.get_latest_metrics.return_value = test_metrics[-1]
        manager.get_metrics_history.return_value = test_metrics
        
        return manager
    
    @pytest.fixture
    def mock_metrics_manager(self):
        """Create a mock metrics manager."""
        manager = Mock()
        manager.get_metrics_summary.return_value = {
            'visitor_satisfaction': {
                'value': 0.82,
                'percentage': '82.0%',
                'status': 'Excellent'
            },
            'dinosaur_happiness': {
                'value': 0.82,
                'percentage': '82.0%',
                'count': 2,
                'status': 'Excellent'
            },
            'facility_efficiency': {
                'value': 0.92,
                'percentage': '92.0%',
                'status': 'Excellent'
            },
            'safety_rating': {
                'value': 0.88,
                'percentage': '88.0%',
                'status': 'Excellent'
            },
            'overall_score': {
                'value': 0.86,
                'percentage': '86.0%',
                'status': 'Excellent'
            },
            'last_updated': datetime.now()
        }
        manager.calculate_overall_resort_score.return_value = 0.86
        return manager
    
    def test_metrics_dashboard_with_simulation_manager(self, mock_session_manager, mock_metrics_manager):
        """Test metrics dashboard integration with simulation manager."""
        # Mock simulation manager in session state
        mock_sim_manager = Mock()
        mock_sim_manager.metrics_manager = mock_metrics_manager
        
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch.dict('streamlit.session_state', {'simulation_manager': mock_sim_manager}):
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify metrics manager methods are called
            mock_metrics_manager.get_metrics_summary.assert_called_once()
            mock_metrics_manager.calculate_overall_resort_score.assert_called_once()
    
    def test_error_handling_in_metrics_dashboard(self, mock_session_manager):
        """Test error handling in metrics dashboard."""
        # Mock simulation manager that raises exceptions
        mock_sim_manager = Mock()
        mock_metrics_manager = Mock()
        mock_metrics_manager.get_metrics_summary.side_effect = Exception("Test error")
        mock_sim_manager.metrics_manager = mock_metrics_manager
        
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.warning') as mock_warning, \
             patch.dict('streamlit.session_state', {'simulation_manager': mock_sim_manager}):
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify warning is displayed for error
            mock_warning.assert_called()
            warning_calls = [str(call) for call in mock_warning.call_args_list]
            assert any("Could not get detailed metrics summary" in call for call in warning_calls)
    
    def test_overall_score_calculation_error(self, mock_session_manager):
        """Test error handling when overall score calculation fails."""
        mock_sim_manager = Mock()
        mock_metrics_manager = Mock()
        mock_metrics_manager.get_metrics_summary.return_value = {}
        mock_metrics_manager.calculate_overall_resort_score.side_effect = Exception("Calculation error")
        mock_sim_manager.metrics_manager = mock_metrics_manager
        
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch.dict('streamlit.session_state', {'simulation_manager': mock_sim_manager}):
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify N/A is displayed for overall score when calculation fails
            metric_calls = [call[1] for call in mock_metric.call_args_list]
            overall_score_calls = [call for call in metric_calls if "Overall Score" in str(call)]
            assert any("N/A" in str(call) for call in overall_score_calls)
    
    def test_area_chart_visualization(self, mock_session_manager):
        """Test area chart visualization functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.area_chart') as mock_area_chart, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('pandas.DataFrame') as mock_dataframe:
            
            # Mock pandas DataFrame
            mock_df = Mock()
            mock_df.empty = False
            mock_df.columns = ['Visitor Satisfaction', 'Safety Rating']
            mock_dataframe.return_value = mock_df
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Area Chart"]  # Refresh rate, time range, chart type
            mock_multiselect.return_value = ["Visitor Satisfaction", "Safety Rating"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify area chart is created
            mock_area_chart.assert_called_once()
    
    def test_bar_chart_visualization(self, mock_session_manager):
        """Test bar chart visualization functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.bar_chart') as mock_bar_chart, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('pandas.DataFrame') as mock_dataframe:
            
            # Mock pandas DataFrame
            mock_df = Mock()
            mock_df.empty = False
            mock_df.columns = ['Visitor Satisfaction', 'Safety Rating']
            mock_dataframe.return_value = mock_df
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Bar Chart"]  # Refresh rate, time range, chart type
            mock_multiselect.return_value = ["Visitor Satisfaction", "Safety Rating"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify bar chart is created
            mock_bar_chart.assert_called_once()
    
    def test_performance_benchmarking_display(self, mock_session_manager):
        """Test performance benchmarking section display."""
        with patch('streamlit.title'), \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.number_input') as mock_number_input:
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]
            mock_multiselect.return_value = ["Visitor Satisfaction", "Safety Rating"]
            mock_number_input.return_value = 85.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify benchmarking section is displayed
            write_calls = [str(call) for call in mock_write.call_args_list]
            assert any("Industry Standards" in call for call in write_calls)
            assert any("Goal Tracking" in call for call in write_calls)
            assert any("Performance Benchmarks" in call for call in write_calls)
    
    def test_individual_dinosaur_trends_display(self, mock_session_manager):
        """Test individual dinosaur trends section."""
        with patch('streamlit.title'), \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.line_chart') as mock_line_chart, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('pandas.DataFrame') as mock_dataframe:
            
            # Mock pandas DataFrame for dinosaur data
            mock_df = Mock()
            mock_df.empty = False
            mock_df.columns = ['T-Rex', 'Triceratops']
            mock_dataframe.return_value = mock_df
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]
            mock_multiselect.side_effect = [
                ["Visitor Satisfaction", "Dinosaur Happiness"],  # Main metrics
                ["T-Rex-001", "Triceratops-001"]  # Selected dinosaurs
            ]
            mock_number_input.return_value = 80.0
            
            # Mock metrics manager
            mock_sim_manager = Mock()
            mock_metrics_manager = Mock()
            mock_sim_manager.metrics_manager = mock_metrics_manager
            
            with patch.dict('streamlit.session_state', {'simulation_manager': mock_sim_manager}):
                from main import render_metrics_dashboard
                render_metrics_dashboard(mock_session_manager)
                
                # Verify dinosaur trends section is displayed
                write_calls = [str(call) for call in mock_write.call_args_list]
                assert any("Individual Dinosaur Trends" in call for call in write_calls)
    
    def test_advanced_analysis_section(self, mock_session_manager):
        """Test advanced analysis section functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.line_chart'), \
             patch('streamlit.number_input') as mock_number_input, \
             patch('pandas.DataFrame') as mock_dataframe:
            
            # Mock pandas DataFrame with statistical methods
            mock_df = Mock()
            mock_df.empty = False
            mock_df.columns = ['Visitor Satisfaction']
            
            # Mock series for statistical calculations
            mock_series = Mock()
            mock_series.std.return_value = 8.5  # Medium volatility
            mock_series.tail.return_value = Mock(mean=lambda: 75.0)  # Recent average
            mock_series.mean.return_value = 70.0  # Overall average
            mock_series.head.return_value = Mock(mean=lambda: 65.0)  # Earlier average
            mock_series.__len__.return_value = 10
            mock_series.iloc = Mock()
            mock_series.iloc.__getitem__ = Mock(return_value=72.0)  # Last value
            
            mock_df.__getitem__.return_value = mock_series
            mock_dataframe.return_value = mock_df
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify advanced analysis section is displayed
            write_calls = [str(call) for call in mock_write.call_args_list]
            assert any("Advanced Analysis" in call for call in write_calls)
            assert any("Performance Insights" in call for call in write_calls)
            assert any("Predictive Indicators" in call for call in write_calls)
    
    def test_custom_goal_tracking(self, mock_session_manager):
        """Test custom goal tracking functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress') as mock_progress, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.number_input') as mock_number_input, \
             patch.dict('streamlit.session_state', {'custom_goals': {}}):
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            mock_number_input.return_value = 85.0  # Custom goal
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify goal tracking elements are displayed
            write_calls = [str(call) for call in mock_write.call_args_list]
            assert any("Goal Tracking" in call for call in write_calls)
            
            # Verify progress bars are created for goal tracking
            assert mock_progress.call_count >= 1
    
    def test_export_functionality_error_handling(self, mock_session_manager):
        """Test error handling in data export functionality."""
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('streamlit.error') as mock_error, \
             patch('json.dumps', side_effect=Exception("JSON error")):
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.side_effect = [False, False, True, False]  # Export current metrics button
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify error handling for export failure
            mock_error.assert_called()
            error_calls = [str(call) for call in mock_error.call_args_list]
            assert any("Failed to export current metrics" in call for call in error_calls)
    
    def test_historical_data_export_error(self, mock_session_manager):
        """Test error handling in historical data export."""
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('streamlit.error') as mock_error, \
             patch('json.dumps', side_effect=Exception("JSON error")):
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.side_effect = [False, False, False, True]  # Export historical data button
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify error handling for historical export failure
            mock_error.assert_called()
            error_calls = [str(call) for call in mock_error.call_args_list]
            assert any("Failed to export historical data" in call for call in error_calls)
    
    def test_dinosaur_chart_data_preparation_error(self):
        """Test error handling in dinosaur chart data preparation."""
        from main import _prepare_dinosaur_chart_data
        from models.core import MetricsSnapshot
        
        now = datetime.now()
        history = [
            MetricsSnapshot(
                visitor_satisfaction=0.8,
                dinosaur_happiness={'T-Rex-001': 0.7},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now
            )
        ]
        
        selected_dinosaurs = ['T-Rex-001']
        
        with patch('pandas.DataFrame', side_effect=Exception("DataFrame error")), \
             patch('streamlit.error') as mock_error:
            
            result = _prepare_dinosaur_chart_data(history, selected_dinosaurs)
            
            # Should return None on error
            assert result is None
            mock_error.assert_called()
    
    def test_chart_data_preparation_without_pandas(self):
        """Test chart data preparation when pandas is not available."""
        from main import _prepare_chart_data
        from models.core import MetricsSnapshot
        
        now = datetime.now()
        history = [
            MetricsSnapshot(
                visitor_satisfaction=0.8,
                dinosaur_happiness={},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now
            )
        ]
        
        metrics_to_show = ["Visitor Satisfaction"]
        
        with patch('pandas.DataFrame', side_effect=ImportError("No pandas")), \
             patch('streamlit.warning') as mock_warning:
            
            result = _prepare_chart_data(history, metrics_to_show)
            
            # Should return None and show warning
            assert result is None
            mock_warning.assert_called()
            warning_calls = [str(call) for call in mock_warning.call_args_list]
            assert any("Pandas not available" in call for call in warning_calls)
    
    def test_dinosaur_chart_data_without_pandas(self):
        """Test dinosaur chart data preparation when pandas is not available."""
        from main import _prepare_dinosaur_chart_data
        from models.core import MetricsSnapshot
        
        now = datetime.now()
        history = [
            MetricsSnapshot(
                visitor_satisfaction=0.8,
                dinosaur_happiness={'T-Rex-001': 0.7},
                facility_efficiency=0.9,
                safety_rating=0.85,
                timestamp=now
            )
        ]
        
        selected_dinosaurs = ['T-Rex-001']
        
        with patch('pandas.DataFrame', side_effect=ImportError("No pandas")), \
             patch('streamlit.warning') as mock_warning:
            
            result = _prepare_dinosaur_chart_data(history, selected_dinosaurs)
            
            # Should return None and show warning
            assert result is None
            mock_warning.assert_called()
            warning_calls = [str(call) for call in mock_warning.call_args_list]
            assert any("Pandas not available" in call for call in warning_calls)
    
    def test_empty_chart_data_handling(self, mock_session_manager):
        """Test handling of empty chart data."""
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('streamlit.warning') as mock_warning, \
             patch('pandas.DataFrame') as mock_dataframe:
            
            # Mock empty DataFrame
            mock_df = Mock()
            mock_df.empty = True
            mock_dataframe.return_value = mock_df
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "1h", "Line Chart"]
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify warning is displayed for empty data
            mock_warning.assert_called()
            warning_calls = [str(call) for call in mock_warning.call_args_list]
            assert any("No data available for the selected metrics" in call for call in warning_calls)
    
    def test_filtered_history_empty_warning(self, mock_session_manager):
        """Test warning display when filtered history is empty."""
        # Mock empty filtered history
        mock_session_manager.get_metrics_history.return_value = []
        
        with patch('streamlit.title'), \
             patch('streamlit.write'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric'), \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.subheader'), \
             patch('streamlit.divider'), \
             patch('streamlit.markdown'), \
             patch('streamlit.progress'), \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('streamlit.warning') as mock_warning:
            
            mock_col = Mock()

            
            mock_col.__enter__ = Mock(return_value=mock_col)

            
            mock_col.__exit__ = Mock(return_value=None)
            def mock_columns_side_effect(cols):

                if isinstance(cols, list):

                    return [mock_col] * len(cols)

                else:

                    return [mock_col] * cols

            

            mock_columns.side_effect = mock_columns_side_effect
            mock_checkbox.return_value = False
            mock_button.return_value = False
            mock_selectbox.side_effect = [5, "15m", "Line Chart"]  # Short time range
            mock_multiselect.return_value = ["Visitor Satisfaction"]
            mock_number_input.return_value = 80.0
            
            from main import render_metrics_dashboard
            render_metrics_dashboard(mock_session_manager)
            
            # Verify warning is displayed for empty filtered data
            mock_warning.assert_called()
            warning_calls = [str(call) for call in mock_warning.call_args_list]
            assert any("No data available for the selected time range" in call for call in warning_calls)


if __name__ == "__main__":
    pytest.main([__file__])