import os
import glob
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

# Exact bounds discovered from your master dataset characteristics
# Place near your imports to share across functions
NORM_BOUNDS = {
    "speed": {"min": 0.0, "max": 120.0},
    "distance": {"min": 0.0, "max": 500.0},
    "lane": {"min": -4.0, "max": 16.0}
}

def get_normalized_vector(speed, distance, lane):
    s = (speed - NORM_BOUNDS["speed"]["min"]) / (NORM_BOUNDS["speed"]["max"] - NORM_BOUNDS["speed"]["min"])
    d = (distance - NORM_BOUNDS["distance"]["min"]) / (NORM_BOUNDS["distance"]["max"] - NORM_BOUNDS["distance"]["min"])
    l = (lane - NORM_BOUNDS["lane"]["min"]) / (NORM_BOUNDS["lane"]["max"] - NORM_BOUNDS["lane"]["min"])
    return np.array([np.clip(s, 0.0, 1.0), np.clip(d, 0.0, 1.0), np.clip(l, 0.0, 1.0)], dtype=np.float32)

# ==============================================================================
# 1. FUNCTIONAL CORE (Pure, Testable, Deterministic Functions)
# ==============================================================================

def read_raw_csv(filepath: str) -> pd.DataFrame:
    """Reads a single raw participant telemetry CSV into a DataFrame."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"[Error] Failed to read {filepath}: {e}")
        return pd.DataFrame()


def isolate_ego_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the human player's row(s) per timestamp based on unique behavior profile.
    
    Rule 1: Player can exceed the enemy car speed limits (vx < 20 or vx > 40).
    Rule 2: Player changes lanes (vy != 0).
    """
    if df.empty:
        return df

    # Group by timestamp to analyze the full snapshot of the road at each millisecond
    grouped = df.groupby('time')
    
    # Vectorized check: flag entire rows matching player behavior anomalies
    is_player_speed = (df['vx'] < 20.0) | (df['vx'] > 40.0)
    is_player_steering = df['vy'].abs() > 1e-5
    
    df = df.assign(is_anomaly=is_player_speed | is_player_steering)
    
    # Identify which unique 'x' positions at a given timestamp belong to the anomaly creator
    player_times = df[df['is_anomaly']]['time'].unique()
    
    # Filter down to just the rows matching the human player profile
    ego_df = df[df['time'].isin(player_times) & df['is_anomaly']]
    
    # Fall back to matching rows using structural indicators if smooth tracking lacked anomalies
    if ego_df.empty:
        ego_df = df[df['control'] == 1.0] if 'control' in df.columns else df.head(1)
        
    # Drop duplicates to ensure exactly one ego row per unique timestamp
    return ego_df.drop_duplicates(subset=['time'])


def calculate_distance_to_lead(ego_row: pd.Series, full_snapshot: pd.DataFrame) -> float:
    """
    Calculates the distance from the ego vehicle to the closest lead vehicle 
    moving ahead of it in the exact same lane.
    """
    ego_x = ego_row['x']
    ego_y = ego_row['y']
    
    # Filter snapshot to find surrounding cars in the same lane that are *ahead* of the ego car
    same_lane_cars = full_snapshot[
        (full_snapshot['y'] - ego_y).abs() < 1.0
    ]
    
    lead_cars = same_lane_cars[same_lane_cars['x'] > ego_x]
    
    if lead_cars.empty:
        return 500.0  # Return a default maximum distance sensor reading if highway is clear
        
    return float((lead_cars['x'] - ego_x).min())


def map_snapshot_to_features(time_group: Tuple[float, pd.DataFrame]) -> Optional[dict]:
    """Processes a single timestamp group to build a structured feature vector."""
    timestamp, frame_df = time_group
    
    # Isolate player row within this timestamp block
    ego_rows = isolate_ego_vehicle(frame_df)
    if ego_rows.empty:
        return None
        
    ego_car = ego_rows.iloc[0]
    
    # Execute distance calculation logic
    lead_distance = calculate_distance_to_lead(ego_car, frame_df)
    
    # Extract control data feature (falling back to 0.0 if column formatting slips)
    control_value = float(ego_car['control']) if 'control' in ego_car.index else 0.0
    
    # Construct unified state space vector containing the control marker
    return {
        "time": timestamp,
        "speed": float(ego_car['vx']),
        "dist_to_lead_car": lead_distance,
        "lane": float(ego_car['y']),
        "control": control_value  # <-- Successfully integrated action telemetry element
    }

def process_telemetry_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw multi-car CSV matrices into a clean, single-player normalized state space timeline."""
    if df.empty:
        return pd.DataFrame()
        
    grouped_frames = df.groupby('time')
    processed_records = [
        map_snapshot_to_features((time, frame)) 
        for time, frame in grouped_frames
    ]
    
    valid_records = [r for r in processed_records if r is not None]
    if not valid_records:
        return pd.DataFrame()
        
    flat_df = pd.DataFrame(valid_records)
    flat_df = flat_df.rename(columns={"dist_to_lead_car": "distance"})
    
    # --- APPLY DETERMINISTIC MIN-MAX NORMALIZATION MATCHING YOUR ENVIRONMENT ---
    flat_df["speed"] = (flat_df["speed"] - 0.0) / (120.0 - 0.0)
    flat_df["distance"] = (flat_df["distance"] - 0.0) / (500.0 - 0.0)
    flat_df["lane"] = (flat_df["lane"] - (-4.0)) / (16.0 - (-4.0))
    
    # Clip limits to handle outliers cleanly
    flat_df["speed"] = flat_df["speed"].clip(0.0, 1.0)
    flat_df["distance"] = flat_df["distance"].clip(0.0, 1.0)
    flat_df["lane"] = flat_df["lane"].clip(0.0, 1.0)
    
    # Re-order and arrange columns logically
    column_order = ["time", "speed", "distance", "lane", "control"]
    remaining_cols = [c for c in flat_df.columns if c not in column_order]
    
    return flat_df[column_order + remaining_cols].sort_values(by='time').reset_index(drop=True)

def merge_episodes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Combines multiple processed episode dataframes into a single collection."""
    valid_dfs = [df for df in dataframes if not df.empty]
    if not valid_dfs:
        return pd.DataFrame()
    return pd.concat(valid_dfs, ignore_index=True)


# ==============================================================================
# 2. IMPERATIVE SHELL (Stateful, I/O Operations, OS File Walkers)
# ==============================================================================

def process_single_csv(csv_path: str, output_dir: str, episode_idx: int) -> pd.DataFrame:
    """Handles the I/O wrapper for reading, transforming, and saving an individual CSV file."""
    raw_df = read_raw_csv(csv_path)
    processed_df = process_telemetry_dataframe(raw_df)
    
    if processed_df.empty:
        return pd.DataFrame()
        
    processed_df = processed_df.assign(episode=episode_idx)
    
    filename = os.path.basename(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    
    processed_df.to_csv(out_path, index=False)
    return processed_df


def process_participant_folder(folder_path: str, global_output_dir: str) -> pd.DataFrame:
    """Processes all CSVs inside a specific participant's folder, ignoring practice runs."""
    participant_name = os.path.basename(folder_path.rstrip(os.sep))
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    if not csv_files:
        return pd.DataFrame()
        
    print(f"[Starting] Processing participant: '{participant_name}' ({len(csv_files)} files found)")
    
    participant_output_dir = os.path.join(global_output_dir, participant_name)
    episode_dfs = []
    
    running_episode_idx = 1
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        
        if filename.lower().startswith("prac"):
            print(f"  [Skipped] Ignoring practice file: {filename}")
            continue
            
        ep_df = process_single_csv(csv_path, participant_output_dir, episode_idx=running_episode_idx)
        if not ep_df.empty:
            episode_dfs.append(ep_df)
            running_episode_idx += 1 
            
    participant_combined = merge_episodes(episode_dfs)
    if not participant_combined.empty:
        combined_filename = f"{participant_name}_all_episodes_combined.csv"
        participant_combined.to_csv(os.path.join(participant_output_dir, combined_filename), index=False)
        
        participant_combined = participant_combined.assign(participant=participant_name)
        print(f"[Finished] Completed parsing data for participant: '{participant_name}'")
        
    return participant_combined


def main_pipeline_execution(data_directory: str, output_directory: str):
    """Orchestrates the entire multi-threaded extraction engine pipeline."""
    print("Initializing Data Ingestion Engine Pipeline...\n" + "="*60)
    
    participant_folders = [
        os.path.join(data_directory, f) 
        for f in os.listdir(data_directory) 
        if os.path.isdir(os.path.join(data_directory, f))
    ]
    
    if not participant_folders:
        print(f"Error: No target sub-directories discovered in paths matching: {data_directory}")
        return

    all_participants_data = []
    
    max_workers = min(os.cpu_count() or 4, len(participant_folders))
    print(f"Spawning ThreadPoolExecutor managing [ {max_workers} ] structural threads.")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_participant_folder, folder, output_directory): folder 
            for folder in participant_folders
        }
        
        for future in futures:
            try:
                result_df = future.result()
                if not result_df.empty:
                    all_participants_data.append(result_df)
            except Exception as e:
                bad_folder = futures[future]
                print(f"[Pipeline Exception Hit] Folder processing collapsed on: {bad_folder}. Trace: {e}")

    print("\n" + "="*60 + "\nAggregating final master matrix...")
    master_df = merge_episodes(all_participants_data)
    
    if not master_df.empty:
        os.makedirs(output_directory, exist_ok=True)
        master_output_path = os.path.join(output_directory, "master_airl_expert_dataset.csv")
        master_df.to_csv(master_output_path, index=False)
        print(f"Success! Master baseline file built with {len(master_df)} rows.")
        print(f"File saved directly to: {master_output_path}")
    else:
        print("Pipeline execution completed, but zero valid structured data rows were generated.")


if __name__ == "__main__":
    DATA_DIRECTORY = r"C:\Users\Danie\Desktop\Project Highway\behavior_v2\data_v3"
    OUTPUT_DIRECTORY = r"C:\Users\Danie\Desktop\PracticumRepository\src\PythonAIRL\Original_Data"
    
    main_pipeline_execution(DATA_DIRECTORY, OUTPUT_DIRECTORY)