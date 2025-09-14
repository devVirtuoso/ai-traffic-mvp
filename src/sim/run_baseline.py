"""
Run a baseline SUMO simulation using SUMO (GUI or headless) with fixed-time traffic lights.
Collects basic performance metrics: vehicle arrivals/departures, average waiting time, and total travel time.

Customize the SUMO network, simulation duration, and metrics in src/utils/config.py.
Switch between GUI and headless mode using the --nogui command-line flag.
"""

import os
import sys
import argparse

# Add src/ to sys.path for imports if running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.logger import get_logger
from src.utils.config import CONFIG

def check_sumo_binary(gui: bool = True):
    """Return the SUMO binary path (GUI or headless)."""
    from shutil import which
    if gui and which('sumo-gui'):
        return 'sumo-gui'
    elif which('sumo'):
        return 'sumo'
    else:
        raise RuntimeError("SUMO is not installed or not in PATH.")

def start_traci(sumo_binary, sumocfg_file, nogui):
    """Start traci with the given SUMO binary and config file."""
    import traci
    
    # Change to the directory containing the config file so relative paths work
    config_dir = os.path.dirname(sumocfg_file)
    config_filename = os.path.basename(sumocfg_file)
    original_cwd = os.getcwd()
    
    try:
        os.chdir(config_dir)
        sumo_cmd = [sumo_binary, "-c", config_filename, "--step-length", "1"]
        traci.start(sumo_cmd)
        return traci
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def run_simulation(traci, steps, logger):
    """
    Run the SUMO simulation for the given number of steps and collect metrics.
    
    This function:
    1. Steps through the simulation using traci.simulationStep()
    2. Tracks vehicle departures and arrivals
    3. Calculates waiting times and travel times
    4. Returns aggregated performance metrics
    """
    total_waiting_time = 0.0
    total_travel_time = 0.0
    departed_vehicles = set()
    arrived_vehicles = set()
    vehicle_entry_times = {}
    
    logger.info(f"Running simulation for {steps} steps...")
    
    for step in range(steps):
        # Advance simulation by one step
        traci.simulationStep()
        
        # Track vehicles that departed in this step
        for veh_id in traci.simulation.getDepartedIDList():
            departed_vehicles.add(veh_id)
            vehicle_entry_times[veh_id] = traci.simulation.getTime()
        
        # Track vehicles that arrived in this step
        for veh_id in traci.simulation.getArrivedIDList():
            arrived_vehicles.add(veh_id)
            entry_time = vehicle_entry_times.pop(veh_id, None)
            if entry_time is not None:
                total_travel_time += traci.simulation.getTime() - entry_time
        
        # Sum waiting times for all vehicles currently in the simulation
        for veh_id in traci.vehicle.getIDList():
            total_waiting_time += traci.vehicle.getWaitingTime(veh_id)
    
    # Calculate average travel time per vehicle
    avg_travel_time = total_travel_time / len(arrived_vehicles) if arrived_vehicles else 0.0
    
    return {
        "departed": len(departed_vehicles),
        "arrived": len(arrived_vehicles),
        "avg_waiting_time": total_waiting_time / steps if steps > 0 else 0.0,
        "total_travel_time": total_travel_time,
        "avg_travel_time": avg_travel_time
    }

def print_results(metrics, logger):
    """Print simulation results in a clear format."""
    logger.info("\n===== Baseline SUMO Simulation Results =====")
    logger.info(f"Total vehicles departed: {metrics['departed']}")
    logger.info(f"Total vehicles arrived: {metrics['arrived']}")
    logger.info(f"Average waiting time per vehicle per step: {metrics['avg_waiting_time']:.2f} s")
    logger.info(f"Total travel time (sum over all vehicles): {metrics['total_travel_time']:.2f} s")
    logger.info(f"Average travel time per vehicle: {metrics['avg_travel_time']:.2f} s")
    logger.info("===========================================\n")

def main():
    """
    Main function to run the baseline SUMO simulation.
    
    To customize this simulation:
    1. Network: Edit the network file path in src/utils/config.py (sumo_cfg_file)
    2. Duration: Change simulation_steps in src/utils/config.py
    3. Traffic: Modify the route file (simple_net.rou.xml) to change vehicle patterns
    4. Traffic lights: Edit the network file to modify traffic light timing
    """
    parser = argparse.ArgumentParser(description="Run baseline SUMO simulation with fixed-time traffic lights.")
    parser.add_argument('--nogui', action='store_true', help='Run SUMO in headless mode (no GUI)')
    args = parser.parse_args()
    
    logger = get_logger()
    
    # Load configuration from src/utils/config.py
    sumocfg_file = CONFIG.get("sumo_cfg_file", "data/net/simple_net.sumocfg")
    steps = CONFIG.get("simulation_steps", 100)
    
    # Make path absolute
    if not os.path.isabs(sumocfg_file):
        sumocfg_file = os.path.abspath(sumocfg_file)
    
    # Check if config file exists
    if not os.path.exists(sumocfg_file):
        logger.error(f"SUMO config file {sumocfg_file} not found. Please check the path in config.py")
        sys.exit(1)
    
    try:
        sumo_binary = check_sumo_binary(gui=not args.nogui)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
    
    try:
        import traci
    except ImportError:
        logger.error("traci (SUMO Python API) is not installed. Please install it with 'pip install sumo-tools' or via your SUMO distribution.")
        sys.exit(1)
    
    logger.info(f"Starting SUMO simulation using {'GUI' if not args.nogui else 'headless'} mode...")
    traci_conn = start_traci(sumo_binary, sumocfg_file, args.nogui)
    
    try:
        metrics = run_simulation(traci_conn, steps, logger)
    finally:
        traci_conn.close()
    
    print_results(metrics, logger)

if __name__ == "__main__":
    main()