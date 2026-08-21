"""Analysis scheduling for perspic."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class LogarithmicWindowSchedule:
    """Schedule for logarithmically spaced measurement windows.

    Attributes:
        windows: Mapping from window_id to list of steps in that window.
        window_centers: Mapping from window_id to the center (log point) step.
        step_to_window: Mapping from step to its window_id.
    """

    windows: Dict[int, List[int]] = field(default_factory=dict)
    window_centers: Dict[int, int] = field(default_factory=dict)
    step_to_window: Dict[int, int] = field(default_factory=dict)

    @property
    def steps(self) -> Set[int]:
        """Flat set of all steps where analysis should run."""
        return set(self.step_to_window.keys())

    def should_analyze(self, step: int) -> bool:
        """Check if analysis should run at this step."""
        return step in self.step_to_window

    def get_window_info(self, step: int) -> Optional[Dict[str, int]]:
        """Get window metadata for a step.

        Returns:
            Dict with 'window_id', 'window_center', and 'window_width' if step
            is in a window, None otherwise.
        """
        window_id = self.step_to_window.get(step)
        if window_id is None:
            return None
        return {
            "window_id": window_id,
            "window_center": self.window_centers[window_id],
            "window_width": len(self.windows[window_id]),
        }

    def __repr__(self) -> str:
        return (
            f"LogarithmicWindowSchedule("
            f"num_windows={len(self.windows)}, "
            f"total_steps={len(self.steps)})"
        )


def logarithmic_windows(
    max_steps: int,
    points_per_decade: int = 10,
    base_window: int = 5,
    adaptive_scale: float = 0.0,
    target_coverage: Optional[float] = None,
) -> LogarithmicWindowSchedule:
    """Generate logarithmically spaced measurement windows.

    Creates a schedule where measurements are taken at logarithmically spaced
    intervals, with each measurement point followed by a window of consecutive
    measurements for statistical robustness.

    Args:
        max_steps: Maximum training step to generate windows for.
        points_per_decade: Number of log points per 10x increase in steps.
            Higher values give denser coverage. Default is 10.
        base_window: Minimum window size (number of consecutive measurements
            per log point). Default is 5. Ignored if target_coverage is set.
        adaptive_scale: Controls how window size grows with step number.
            - 0.0: Fixed window size (always base_window)
            - 1.0: Window grows by ~1 step per decade
            - 2.0: Window grows by ~2 steps per decade
            The formula is: window_size = base_window + adaptive_scale * log10(step)
            Default is 0.0 (fixed windows).
        target_coverage: If set, overrides base_window with a value derived
            from max_steps so that num_points * base_window / max_steps
            approximates this fraction (clamped to at least 1; the *actual*
            coverage -- len(schedule.steps) / max_steps -- will still differ
            somewhat, since overlapping/truncated windows collapse below this
            nominal estimate). Use this instead of a hand-picked base_window
            when comparing schedules across runs whose max_steps varies for
            reasons unrelated to how much you want measured -- e.g. a
            batch-size sweep at a fixed token budget, where max_steps shrinks
            linearly with batch size. With a fixed base_window, window
            *count* (num_points above) only grows with log10(max_steps), so
            realized coverage balloons as max_steps shrinks; deriving
            base_window from target_coverage instead keeps it roughly
            constant across such a sweep. None (default) preserves the
            original fixed-base_window behavior exactly.

    Returns:
        LogarithmicWindowSchedule containing:
            - steps: Set of all steps to measure
            - windows: Dict mapping window_id -> list of steps
            - window_centers: Dict mapping window_id -> center step
            - step_to_window: Dict mapping step -> window_id

    Examples:
        # Fixed window size of 5, ~10 windows per decade
        schedule = logarithmic_windows(max_steps=10000)

        # Adaptive windows that grow with step number
        schedule = logarithmic_windows(
            max_steps=100000,
            points_per_decade=5,
            base_window=3,
            adaptive_scale=2.0
        )

        # Batch-size sweep: hold ~11% coverage constant across runs whose
        # max_steps varies with batch size, instead of a fixed base_window
        # that would make coverage balloon as batch size grows and max_steps
        # shrinks.
        schedule = logarithmic_windows(max_steps=max_steps, target_coverage=0.11)

        # Use with analyzer
        model = analyzer(MyModule, analysis_schedule=schedule, ...)
    """
    if max_steps <= 0:
        return LogarithmicWindowSchedule(
            windows={0: [0]},
            window_centers={0: 0},
            step_to_window={0: 0},
        )

    # Generate log-spaced center points in the range [10^0, 10^log10(max_steps)]
    # Step 0 will be added separately below
    num_points = int(points_per_decade * math.log10(max_steps)) + 1
    if target_coverage is not None:
        base_window = max(1, round(target_coverage * max_steps / num_points))
    # Generate logspace without numpy: 10^(start + i * step) for i in range(num_points)
    log_start, log_end = 0, math.log10(max_steps)
    log_step = (log_end - log_start) / (num_points - 1) if num_points > 1 else 0
    log_centers = [10 ** (log_start + i * log_step) for i in range(num_points)]
    log_centers = sorted(set(int(c) for c in log_centers))  # Unique integers

    # Always include step 0
    if 0 not in log_centers:
        log_centers = [0] + log_centers

    windows: Dict[int, List[int]] = {}
    window_centers: Dict[int, int] = {}
    step_to_window: Dict[int, int] = {}

    for window_id, center in enumerate(log_centers):
        # Calculate window size (adaptive or fixed)
        if adaptive_scale > 0 and center > 0:
            # Window grows logarithmically: base + scale * log10(center)
            window_size = base_window + int(adaptive_scale * math.log10(center))
        else:
            window_size = base_window

        # Skip windows that would be truncated (can't fit full window)
        if center + window_size - 1 > max_steps:
            continue

        # Generate steps in this window (starting from the center point)
        window_steps = [center + offset for offset in range(window_size)]

        if window_steps:  # Only add non-empty windows
            windows[window_id] = window_steps
            window_centers[window_id] = center

            # Map steps to window (later windows overwrite earlier for overlaps)
            for step in window_steps:
                step_to_window[step] = window_id

    return LogarithmicWindowSchedule(
        windows=windows,
        window_centers=window_centers,
        step_to_window=step_to_window,
    )
