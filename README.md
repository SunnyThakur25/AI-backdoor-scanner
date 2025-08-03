# AI-backdoor-scanner

# BackdoorScanner 🔍🛡️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Advanced AI Model Backdoor Detection Tool for security professionals and ML engineers.



## Features ✨

- **Multi-Framework Support**: PyTorch, TensorFlow, ONNX, HuggingFace, ModelScope
- **Comprehensive Detection Methods**:
  - Activation clustering
  - Neuron inspection
  - Adversarial analysis
  - Spectral analysis
  - Input-output pattern testing
  - Weight distribution analysis
- **Professional Reporting**:
  - JSON and HTML report formats
  - Interactive visualizations
  - Actionable recommendations
- **Enterprise Ready**:
  - Detailed logging
  - System information tracking
  - CI/CD integration support

## Installation 🛠️

### Prerequisites

- Python 3.7+
- pip

### Recommended Installation (with all dependencies)

```bash
git clone https://github.com/SunnyThakur25/backdoor-scanner.git
cd backdoor-scanner
pip install -r requirements.txt
```

Minimal Installation (core only)

`pip install numpy scipy scikit-learn matplotlib`

Usage 🚀
Basic Scan
`
python backdoor_scanner.py path/to/your/model
`

Full Options
```bash
python backdoor_scanner.py path/to/model \
  --source huggingface \
  --type text \
  --methods activation_clustering neuron_inspection \
  --data path/to/clean_dataset \
  --output results \
  --report-format html \
  --visualization-format pdf
```

Command Line Options


Option	,Description	,Default

```
model_path	Path to model or model identifier	Required
-s, --source	Model source (huggingface, modelscope, local)	local
-t, --type	Model type (text, vision, multimodal, auto)	auto
-f, --framework	Framework (pytorch, tensorflow, onnx, auto)	auto
-m, --methods	Detection methods to use	All methods
-d, --data	Path to clean dataset for comparison	None
-p, --patterns	Custom trigger patterns to test	None
-o, --output	Output directory for reports	Current directory
-r, --report-format	Report format (json, html)	json
-v, --visualization-format	Visualization format (png, pdf, svg)	png
-V, --verbose	Enable verbose output	False
```

Detection Methods 🕵️‍♂️
1. Activation Clustering

    Purpose: Identify unusual activation patterns

    Technique: PCA + K-means clustering

    Output: Cluster analysis and outlier detection

2. Neuron Inspection

    Purpose: Find suspicious individual neurons

    Technique: Statistical analysis of activations

    Output: Anomalous neuron characteristics

3. Adversarial Analysis

    Purpose: Test model robustness

    Technique: Small input perturbations

    Output: Sensitivity metrics

4. Spectral Analysis

    Purpose: Detect weight matrix anomalies

    Technique: Singular value decomposition

    Output: Spectral properties and rank analysis

5. Input-Output Analysis

    Purpose: Find trigger patterns

    Technique: Targeted input testing

    Output: Suspicious input-output mappings

6. Weight Analysis

    Purpose: Identify unusual weight distributions

    Technique: Statistical distribution analysis

    Output: Weight distribution metrics

Report Interpretation 📊
Score Range	Confidence	Recommendation
0.0-0.4	Low	Standard monitoring recommended
0.4-0.7	Medium	Further investigation needed
0.7-1.0	High	Do not use in production
Development 🧑‍💻
Contributing

    Fork the repository

    Create your feature branch (git checkout -b feature/AmazingFeature)

    Commit your changes (git commit -m 'Add some AmazingFeature')

    Push to the branch (git push origin feature/AmazingFeature)

    Open a Pull Request

Testing
```bash

pytest tests/
```
Code Style
```bash

black .
flake8
mypy .
```
License 📜

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments 🙏

    Inspired by academic research on ML backdoor detection

    Built with the open source community in mind

    
