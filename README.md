# Hallucination Curvature Experiment
Proof-of-concept implementation for "From Error to Estimation: The Geometry of Hallucination"

## Quick Start
```bash
# 1. Setup environment
python setup.py

# 2. Run complete experiment
python run\_all.py
```

## Requirements
- Python 3.8+
- Windows 10 (tested)
- 8GB+ RAM
- Optional: CUDA-capable GPU

## Project Structure
```
hallucination\_curvature\_experiment
├── setup.py                    # Environment setup
├── run\_all.py                 # Master execution script
├── requirements.txt            # Package dependencies
├── data/
│   └── test\_inputs.json       # Test cases
├── src/
│   ├── 01\_compute\_rho.py     # Information density
│   ├── 02\_compute\_kappa.py   # Semantic curvature
│   ├── 03\_compute\_h.py       # Hallucination rate
│   └── 04\_analyze\_results.py # Analysis \& visualization
└── results/                    # Output directory (auto-generated)
```

## License
MIT License


