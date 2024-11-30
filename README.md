# scalable-bgmm
## Project Overview : 
Implement a 1B records scalable and production ready code of VGM.

## Folder Structure
```
scalable-bgmm/
├── models/                     # Contains the model implementations
│   └── dpgmm.py                # Custom Implementation of the Dirichlet Process GMM
├── transformers/               # Transformation utilities for data preprocessing
│   ├── __init__.py             # Initialization file for the transformers module
│   ├── continuous_data_transformer.py # Handles transformations for continuous data.
├── evaluation.py               # Script for evaluating the model's performance
├── processor.py                # Script for processing data
├── synthetic_data_generator.py # Script for generating synthetic datasets
├── code_runner.ipynb           # Jupyter notebook for running and testing the code
├── README.md                   # Project documentation
```

## Features
- **Scalable Bayesian Gaussian Mixture Model (BGMM):**
  - Implements a scalable variation of the BGMM algorithm to handle large datasets efficiently.
- **Model Implementation:**
  - Includes an advanced implementation of the Dirichlet Process Gaussian Mixture Model (DPGMM) in the `models` directory.
- **Data Transformation:**
  - Comprehensive utilities for transforming and preprocessing continuous data in the `transformers` module.
  - Modular design for easy integration into larger pipelines.
- **Synthetic Data Generation:**
  - Provides a script to generate synthetic datasets for testing and validation purposes (`synthetic_data_generator.py`).
- **Performance Evaluation:**
  - Evaluation scripts to test the model's performance across multiple metrics (`evaluation.py`).
- **Interactive Experimentation:**
  - Includes a Jupyter notebook (`code_runner.ipynb`) for running experiments
- **Extensible Framework:**
  - Designed with modular components to allow seamless extension for additional functionality or new features.
