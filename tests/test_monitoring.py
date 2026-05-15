"""
Phase 4 — DriftMonitor and champion/challenger harness.
"""
from __future__ import annotations

import numpy as np

from src.monitoring import DriftMonitor


def test_drift_monitor_baseline_captured_and_no_alert_on_stable_stream():
    mon = DriftMonitor(baseline_size=100, window_size=100, threshold_z=3.0)
    rng = np.random.default_rng(0)
    alerts = []
    for _ in range(300):
        alerts.extend(mon.observe(reward=float(rng.normal(1.0, 0.1))))
    # A stable stream should not raise
    assert not alerts


def test_drift_monitor_alerts_on_mean_shift():
    mon = DriftMonitor(baseline_size=200, window_size=200, threshold_z=3.0)
    rng = np.random.default_rng(0)
    # Phase 1: baseline at mean=1.0
    for _ in range(200):
        mon.observe(reward=float(rng.normal(1.0, 0.1)))
    # Phase 2: sharp shift to mean=1.5
    alerts = []
    for _ in range(400):
        alerts.extend(mon.observe(reward=float(rng.normal(1.5, 0.1))))
    assert alerts, "expected at least one drift alert after mean shift"
    assert alerts[0].stream == "reward"
    assert abs(alerts[0].z_score) >= 3.0
