# AI-Based Traffic Management System (MVP)

This is a prototype of an **AI-powered traffic signal optimization system** designed to reduce urban traffic congestion by analyzing real-time data and using reinforcement learning.

## 🎯 Project Goal

Develop a software prototype that reduces average commute time by **10%** in a simulated urban environment, with a dashboard for traffic authorities to monitor and control signals.

## 🚀 MVP Features

- **Traffic Simulation**: Using SUMO (Simulation of Urban Mobility) for realistic urban traffic scenarios
- **Reinforcement Learning**: PPO agent to optimize signal timings based on real-time traffic data
- **Computer Vision**: YOLOv8 + OpenCV for vehicle detection and counting from traffic cameras
- **Dashboard**: Real-time monitoring interface for traffic authorities
- **Performance Metrics**: Comprehensive evaluation of traffic flow improvements

## 📁 Project Structure

```
ai-traffic-mvp/
├── src/
│   ├── sim/              # SUMO simulation and RL environment
│   ├── cv/               # Computer vision for vehicle detection
│   ├── dashboard/        # Web dashboard for monitoring
│   └── utils/            # Utility functions and helpers
├── data/                 # SUMO configs, traffic videos, datasets
├── notebooks/            # Jupyter notebooks for experiments
├── models/               # Trained RL models and CV models
├── logs/                 # Training logs and evaluation results
└── tests/                # Unit tests
```

## 🛠️ Technical Stack

- **Simulation**: SUMO (Simulation of Urban Mobility)
- **Reinforcement Learning**: Stable-Baselines3 (PPO)
- **Computer Vision**: YOLOv8, OpenCV
- **Dashboard**: Streamlit
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

3. **Install SUMO**
   - **Ubuntu/Debian**: `sudo apt-get install sumo sumo-tools`
   - **macOS**: `brew install sumo`
   - **Windows**: Download from [SUMO website](https://sumo.dlr.de/docs/Downloads.php)

4. **Verify installation**
   ```bash
   python -c "import sumo; print('SUMO Python bindings installed successfully')"
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
   streamlit run src/dashboard/app.py
   ```

## 📊 Expected Outcomes

- **Primary Metric**: ≥10% reduction in average travel time
- **Secondary Metrics**: 
  - Reduced queue lengths
  - Fewer vehicle stops
  - Improved throughput
  - Better signal coordination

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
- **Phase 3 (Weeks 9-12)**: Dashboard development
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