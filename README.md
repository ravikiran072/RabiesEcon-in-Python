# Rabies Economic Analysis

A comprehensive data science project for analyzing the economic impact of rabies prevention and control measures. This project includes a fully integrated Streamlit application that performs complete economic analysis internally, from mathematical modeling to visualization generation.

## 🚀 Quick Start - Comprehensive App

To run the complete analysis application:

```bash
python run_app.py
```

This launches a comprehensive Streamlit app that:
- Performs all mathematical modeling internally
- Calculates economic metrics and cost-effectiveness
- Generates the program summary table
- Creates 2x2 visualization plots
- Provides interactive results exploration

See `app/COMPREHENSIVE_APP_README.md` for detailed information about the comprehensive application.

## Project Structure

```
├── README.md          <- The top-level README for developers using this project
├── data/              <- Data directory
│   ├── external/      <- Data from third party sources
│   ├── processed/     <- The final, canonical data sets for modeling
│   └── raw/           <- The original, immutable data dump
├── docs/              <- Project documentation
├── notebooks/         <- Jupyter notebooks for exploration and communication
├── reports/           <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/       <- Generated graphics and figures for reporting
├── requirements.txt   <- Package dependencies
├── setup.py          <- Setup script for installing the project as a package
├── src/               <- Source code for use in this project
│   ├── __init__.py    <- Makes src a Python module
│   ├── data/          <- Scripts to download or generate data
│   ├── features/      <- Scripts to turn raw data into features for modeling
│   ├── models/        <- Scripts to train models and make predictions
│   └── visualization/ <- Scripts to create exploratory and results visualizations
├── tests/             <- Unit tests and integration tests
└── config/            <- Configuration files for reproducible environments
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Place your raw data files in `data/raw/`
2. Use Jupyter notebooks in `notebooks/` for exploratory data analysis
3. Develop reusable code modules in `src/`
4. Generate reports and figures in `reports/`

## Project Organization

This project follows a standard data science project template for reproducibility and collaboration.

## Contributing

1. Create feature branches for new work
2. Write tests for your code in the `tests/` directory
3. Update documentation as needed
4. Submit pull requests for review

## License

This project is licensed under the MIT License.