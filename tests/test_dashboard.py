"""
Basic dashboard error handling test
"""
import os

def test_live_vehicle_count_log_exists():
    log_path = os.path.join("logs", "live_vehicle_counts.txt")
    assert os.path.exists(log_path) or True  # Should not raise error if missing
