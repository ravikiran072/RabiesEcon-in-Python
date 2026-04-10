# 🐕 Rabies Economic Analysis Model

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rabiesecon.streamlit.app)

A comprehensive economic impact assessment tool for rabies vaccination programs using epidemiological modeling and cost-benefit analysis.

## Quick Start

### **Live Demo**
**[Try the App](https://rabiesecon.streamlit.app)** 

### **Local Installation**
```bash
git clone https://github.com/ravikiran072/RabiesEcon-in-Python.git
cd RabiesEcon-in-Python
pip install -r requirements.txt
streamlit run app/comprehensive_rabies_app.py
```

## Features

### **Model Parameters**
- **Geographic & Population Variables**: Program area, human/dog populations
- **Dog Density Adjustment Factor**: Control carrying capacity (0.95, 1.0, 1.05)
- **Economic Parameters**: Vaccination costs, PEP costs, YLL values
- **Real-time Validation**: Parameter consistency checks

### **Coverage Data Configuration**
- **Phased Vaccination Programs**:
  - Phase I (Years 1-3): Initial mass vaccination
  - Phase II (Years 4-6): Intensified efforts  
  - Phase III (Years 7-13): High coverage maintenance
  - Phase IV (Years 14-30): Post-elimination maintenance
- **Custom Scenarios**: Interactive coverage rate configuration
- **Visual Preview**: Real-time charts and data tables

### **Analysis Outputs**
- **Executive Summary**: Key metrics and ROI analysis
- **Program Comparison**: No vaccination vs. annual vaccination
- **Detailed Results**: Year-by-year breakdowns
- **Interactive Visualizations**: Population dynamics, costs, benefits

## Technical Details

### **Core Files**
- `app/comprehensive_rabies_app.py` - Main Streamlit application
- `notebooks_initial_run.py` - Core epidemiological simulation
- `data/model_parameters.xlsx` - Default parameter values
- `data/coverage_data.csv` - Vaccination coverage data

### **Key Components**
- **Epidemiological Model**: SEIR-based dog population dynamics
- **Economic Model**: Cost-benefit analysis with DALYs
- **Coverage Model**: Time-varying vaccination and PEP coverage
- **Sensitivity Analysis**: Parameter adjustment capabilities

### **Dependencies**
- Python 3.11+
- Streamlit, NumPy, Pandas, Matplotlib, Seaborn
- See `requirements.txt` for complete list

## Model Background

This model implements a compartmental epidemiological framework for rabies transmission in dog populations, coupled with economic analysis of vaccination interventions. The model considers:

- **Dog Population Dynamics**: Birth, death, vaccination, disease progression
- **Human Exposure Risk**: Dog-human transmission, PEP effectiveness
- **Economic Costs**: Vaccination programs, PEP treatment, productivity losses
- **Policy Scenarios**: Comparison of intervention strategies

## Use Cases

- **Public Health Planning**: Design rabies elimination programs
- **Economic Evaluation**: Cost-effectiveness of vaccination strategies  
- **Policy Analysis**: Compare intervention scenarios
- **Research Applications**: Epidemiological modeling studies

## Repository Structure

```
├── app/                           # Main application
│   ├── comprehensive_rabies_app.py   # Streamlit app
│   └── archive/                       # Legacy app versions
├── data/                          # Model data
│   ├── model_parameters.xlsx         # Default parameters
│   └── coverage_data.csv             # Coverage scenarios
├── config/                        # Configuration files
├── docs/                          # Documentation
├── archive/                       # Archived components
│   ├── extra_scripts_archive/         # Analysis scripts
│   ├── notebooks_archive/             # Jupyter notebooks
│   └── reports_archive/               # Report outputs
├── notebooks_initial_run.py       # Core simulation logic
├── requirements.txt               # Dependencies
└── run_app.py                     # App launcher

```

## Contributing

This model is designed for epidemiological research and public health applications. For questions or contributions, please open an issue or submit a pull request.

## License

This project is available for academic and public health use. Please cite appropriately in any publications or applications.

## Links

- **Live Application**: [rabiesecon.streamlit.app](https://rabiesecon.streamlit.app)
- **Documentation**: See `README_DEPLOYMENT.md` for deployment details
- **Support**: Open an issue for questions or bug reports