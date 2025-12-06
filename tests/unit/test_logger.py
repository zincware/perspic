"""Unit tests for the logger module."""

import pytest

from perspic.logger import LogarithmicWindowSchedule, logarithmic_windows


class TestLogarithmicWindowSchedule:
    """Test the LogarithmicWindowSchedule dataclass."""

    @pytest.fixture
    def sample_schedule(self):
        """Create a sample schedule for testing."""
        return LogarithmicWindowSchedule(
            windows={0: [0, 1, 2], 1: [10, 11, 12]},
            window_centers={0: 0, 1: 10},
            step_to_window={0: 0, 1: 0, 2: 0, 10: 1, 11: 1, 12: 1},
        )

    def test_steps_property_returns_set(self, sample_schedule):
        """Test steps property returns all scheduled steps."""
        assert sample_schedule.steps == {0, 1, 2, 10, 11, 12}

    def test_steps_property_empty_schedule(self):
        """Test steps property on empty schedule."""
        schedule = LogarithmicWindowSchedule()
        assert schedule.steps == set()

    def test_should_analyze_true_for_scheduled_step(self, sample_schedule):
        """Test should_analyze returns True for scheduled steps."""
        assert sample_schedule.should_analyze(0) is True
        assert sample_schedule.should_analyze(10) is True

    def test_should_analyze_false_for_unscheduled_step(self, sample_schedule):
        """Test should_analyze returns False for unscheduled steps."""
        assert sample_schedule.should_analyze(5) is False
        assert sample_schedule.should_analyze(100) is False

    def test_get_window_info_valid_step(self, sample_schedule):
        """Test get_window_info returns correct dict for valid step."""
        info = sample_schedule.get_window_info(10)
        assert info == {"window_id": 1, "window_center": 10, "window_width": 3}

    def test_get_window_info_invalid_step(self, sample_schedule):
        """Test get_window_info returns None for invalid step."""
        assert sample_schedule.get_window_info(5) is None

    def test_repr(self, sample_schedule):
        """Test __repr__ format."""
        r = repr(sample_schedule)
        assert "num_windows=2" in r
        assert "total_steps=6" in r


class TestLogarithmicWindowsFunction:
    """Test the logarithmic_windows factory function."""

    def test_zero_or_negative_max_steps(self):
        """Test non-positive max_steps returns single window at zero."""
        for val in [0, -10]:
            schedule = logarithmic_windows(max_steps=val)
            assert 0 in schedule.steps
            assert len(schedule.windows) == 1

    def test_max_steps_one(self):
        """Test max_steps=1 with base_window=1 includes step 0."""
        schedule = logarithmic_windows(max_steps=1, base_window=1)
        assert 0 in schedule.steps
        assert all(s <= 1 for s in schedule.steps)

    def test_large_max_steps(self):
        """Test large max_steps doesn't cause issues."""
        schedule = logarithmic_windows(max_steps=1_000_000)
        assert 0 in schedule.steps
        assert len(schedule.windows) > 0

    def test_all_steps_within_max_steps(self):
        """Test all steps are within max_steps."""
        schedule = logarithmic_windows(max_steps=100)
        assert all(s <= 100 for s in schedule.steps)

    def test_windows_are_non_empty(self):
        """Test all windows contain at least one step."""
        schedule = logarithmic_windows(max_steps=1000)
        for steps in schedule.windows.values():
            assert len(steps) > 0

    def test_base_window_determines_size(self):
        """Test base_window sets window size when adaptive_scale=0."""
        schedule = logarithmic_windows(max_steps=1000, base_window=3, adaptive_scale=0)
        for steps in schedule.windows.values():
            assert len(steps) == 3

    def test_adaptive_scale_positive_grows_windows(self):
        """Test adaptive_scale > 0 grows window sizes."""
        schedule = logarithmic_windows(
            max_steps=10000, base_window=3, adaptive_scale=2.0
        )
        # Later windows should be larger than base_window
        sizes = [len(steps) for window_id, steps in schedule.windows.items()]
        assert max(sizes) > 3

    def test_higher_points_per_decade_denser(self):
        """Test higher points_per_decade gives more windows."""
        sparse = logarithmic_windows(max_steps=1000, points_per_decade=3)
        dense = logarithmic_windows(max_steps=1000, points_per_decade=10)
        assert len(dense.windows) > len(sparse.windows)

    def test_step_to_window_consistency(self):
        """Test step_to_window contains all steps from windows."""
        schedule = logarithmic_windows(max_steps=500)
        all_steps = set()
        for steps in schedule.windows.values():
            all_steps.update(steps)
        assert all_steps == set(schedule.step_to_window.keys())

    def test_window_centers_match_ids(self):
        """Test window_centers keys match windows keys."""
        schedule = logarithmic_windows(max_steps=500)
        assert set(schedule.windows.keys()) == set(schedule.window_centers.keys())

    def test_windows_roughly_log_spaced(self):
        """Test window centers are approximately logarithmically spaced."""

        schedule = logarithmic_windows(max_steps=10000, base_window=1)
        centers = sorted(schedule.window_centers.values())
        # Check that spacing grows (log-spaced means larger gaps at higher values)
        if len(centers) > 3:
            early_gap = centers[2] - centers[1]
            late_gap = centers[-1] - centers[-2]
            assert late_gap > early_gap
