# AI-Based Traffic Management System (MVP)

This is a prototype of an **AI-powered traffic signal optimization system** designed to reduce urban traffic congestion by analyzing real-time data and using reinforcement learning.

## 🎯 Project Goal

Develop a software prototype that reduces average commute time by **10%** in a simulated urban environment, with a dashboard for traffic authorities to monitor and control signals.

## 🚀 MVP Features

- **Traffic Simulation**: Using SUMO (Simulation of Urban Mobility) for realistic urban traffic scenarios
- **Reinforcement Learning**: MA-PPO agents to optimize signal timings based on real-time traffic data
- **Computer Vision**: YOLOv8 + OpenCV for vehicle detection and counting from traffic cameras
- **React Dashboard**: Modern UI built with React and Tailwind CSS for live views and manual control
- **Multi-Service Orchestration**: Docker Compose setup for backend, dashboard, Redis, and SUMO simulation
- **Performance Metrics**: Comprehensive evaluation of traffic flow improvements

## 📁 Project Structure

```
ai-traffic-mvp/
├── src/
│   ├── sim/              # SUMO simulation and RL environment
│   ├── cv/               # Computer vision for vehicle detection
│   ├── dashboard/        # React + Tailwind dashboard (live views, manual control)
│   └── utils/            # Utility functions and helpers
├── data/                 # SUMO configs, traffic videos, datasets
├── notebooks/            # Jupyter notebooks for experiments
├── models/               # Trained RL models and CV models
├── logs/                 # Training logs and evaluation results
├── docs/                 # Architecture diagram and documentation
└── tests/                # Unit tests
```

## 🛠️ Technical Stack

- **Simulation**: SUMO (Simulation of Urban Mobility)
- **Reinforcement Learning**: Stable-Baselines3 (PPO)
- **Computer Vision**: YOLOv8, OpenCV
- **Dashboard**: React + Tailwind CSS
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Plotly

## 📋 Prerequisites

- Python 3.10+
- SUMO (Simulation of Urban Mobility)
- CUDA-capable GPU (recommended for CV and RL training)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ai-traffic-mvp
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup React Dashboard**
   ```bash
   cd dashboard
   npm install
   npm start
   ```

4. **Docker Compose (Recommended for full stack)**
   ```bash
   docker-compose up --build
   ```

5. **Install SUMO**
   - **Ubuntu/Debian**: `sudo apt-get install sumo sumo-tools`
   - **macOS**: `brew install sumo`
   - **Windows**: Download from [SUMO website](https://sumo.dlr.de/docs/Downloads.php)

6. **Verify installation**
   ```bash
   python -c "import traci; print('SUMO Python bindings installed successfully')"
   ```

## 🎮 Quick Start

1. **Run the simulation baseline**
   ```bash
   python src/sim/run_baseline.py
   ```

2. **Train the RL agent**
   ```bash
   python src/sim/train_agent.py
   ```

3. **Launch the dashboard**
   ```bash
   cd dashboard
   npm start
   ```

Or run all services together:
   ```bash
   docker-compose up --build
   ```

## 📊 Expected Outcomes

- **Primary Metric**: ≥10% reduction in average travel time
- **Secondary Metrics**: 
  - Reduced queue lengths
  - Fewer vehicle stops
  - Improved throughput
  - Better signal coordination

## 📐 Architecture & References
- See `docs/architecture_diagram.md` for system architecture and data flow
- RL training notebook: `notebooks/rl_training_notebook.ipynb` for baseline experiments

## 🔬 Evaluation

The system will be evaluated using:
- **Simulation Environment**: SUMO with realistic urban scenarios
- **Baseline Comparison**: Fixed-time signal control vs. AI-optimized control
- **Statistical Significance**: Multiple runs with different random seeds
- **Real-world Validation**: Testing on actual traffic camera data

## 🛡️ Safety & Constraints

- **Safety Layer**: Hard-coded constraints for minimum green times, pedestrian phases
- **Fail-safe**: Automatic fallback to pre-configured signal plans
- **Human-in-the-loop**: Operator override capabilities
- **Privacy**: No storage of personal data, only traffic metadata

## 📈 Development Roadmap

- **Phase 1 (Weeks 1-4)**: Basic simulation and RL agent
- **Phase 2 (Weeks 5-8)**: Computer vision integration
- **Phase 3 (Weeks 9-12)**: Dashboard development (React + Tailwind)
- **Phase 4 (Weeks 13-16)**: Real-world testing and optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact the development team.

---

**Note**: This is an MVP prototype for research and demonstration purposes. Production deployment would require additional safety certifications and regulatory compliance.
