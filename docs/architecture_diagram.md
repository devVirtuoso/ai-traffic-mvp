# 📊 System Architecture Diagram & Data Flow

## High-Level Architecture

```mermaid
graph TD
    A[Camera & IoT Sensors] -->|RTSP/IoT Data| B[Data Pipeline]
    B --> C[CV Model (YOLOv8, OpenCV)]
    C --> D[RL Agent (PPO/MA-PPO)]
    D --> E[Decision Engine]
    E --> F[SUMO Simulation]
    E --> G[Manual Override]
    F --> H[Dashboard]
    G --> H
    H --> I[Traffic Authority]
```

## Data Flow
1. **Camera & IoT Sensors**: Capture real-time traffic data.
2. **Data Pipeline**: Preprocesses feeds and sensor data.
3. **CV Model**: Detects vehicles, estimates density.
4. **RL Agent**: Suggests optimal signal timings.
5. **Decision Engine**: Applies timing plan, allows manual override.
6. **SUMO Simulation**: Runs traffic scenario, outputs metrics.
7. **Dashboard**: Visualizes live feed, heatmaps, metrics, and controls.

---
This diagram and flow can be used directly in your hackathon pitch deck.