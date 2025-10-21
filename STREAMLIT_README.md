# Rabies Economic Analysis Dashboard

A comprehensive Streamlit web application for analyzing the economic impact of rabies vaccination programs.

## Features

- **Interactive Parameter Adjustment**: Modify key epidemiological and economic parameters
- **Scenario Comparison**: Compare vaccination vs. no vaccination programs
- **Comprehensive Visualization**: 
  - Program summary tables matching Excel format
  - 2x2 epidemiological impact plots
  - Economic cost breakdown and cost-effectiveness analysis
- **Real-time Analysis**: Instantly see results as you adjust parameters

## Quick Start

### Option 1: Using the Launcher Script
```bash
python run_app.py
```

### Option 2: Direct Streamlit Command
```bash
streamlit run app/main_app.py
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the Excel parameter file:
   - `model_parameters.xlsx` should be in the project root

## Usage

1. **Launch the app** using one of the methods above
2. **Adjust parameters** in the left sidebar:
   - Geographic parameters (area, population)
   - Disease parameters (R0, transmission rates)
   - Economic parameters (vaccination costs, treatment costs)
3. **Click "Run Analysis"** to execute the model
4. **View results** in three tabs:
   - **Summary Table**: Comprehensive program comparison
   - **Epidemiological Charts**: Disease impact visualization
   - **Economic Analysis**: Cost breakdown and effectiveness

## Key Outputs

### Program Summary Table
- Rabid dogs (annual and cumulative)
- Human deaths (annual and cumulative) 
- Program costs (annual and cumulative)
- Cost per death averted
- Cost per DALY averted

### Visualization
- 2x2 grid comparing vaccination vs no vaccination scenarios
- Cost breakdown over time
- Cost-effectiveness trends

## Model Scenarios

**No Vaccination Program:**
- 5% baseline vaccination coverage
- 25% PEP treatment rate
- Higher disease transmission

**Annual Vaccination Program:**
- 70% vaccination coverage
- 50% PEP treatment rate
- Reduced disease transmission

## Technical Details

- Built with Streamlit and Plotly for interactive visualization
- Uses simplified SEIRD compartmental disease model
- Integrates with comprehensive parameter management system
- Maintains fidelity to Excel reference model calculations

## Parameter Categories

- **Variable Parameters**: Vary by program area (population, area, costs)
- **Constant Parameters**: Fixed biological/epidemiological values
- **Calculated Parameters**: Automatically computed from other parameters

## Files Structure

```
app/
├── main_app.py          # Main Streamlit application
├── model_parameters.py  # Parameter management system
├── streamlit_utils.py   # Streamlit-specific utilities
└── test_parameters.py   # Parameter validation tests
```

## Troubleshooting

If you encounter import errors:
```bash
pip install streamlit plotly pandas numpy matplotlib openpyxl
```

If the Excel file is not found, ensure `model_parameters.xlsx` is in the project root directory.

For parameter validation issues, run the test suite:
```bash
python -m pytest app/test_parameters.py -v
```