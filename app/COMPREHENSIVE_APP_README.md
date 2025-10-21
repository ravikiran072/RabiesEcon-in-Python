# Comprehensive Rabies Economic Analysis App

## Overview

This is a fully integrated Streamlit application that performs comprehensive rabies economic analysis internally. The app includes all the mathematical modeling, economic calculations, and visualization generation without requiring external wrapper functions.

## Features

### 🔬 Complete Internal Analysis
- **Mathematical Modeling**: Implements the full rabies transmission model with SEIR compartments for both dog and human populations
- **Economic Calculations**: Calculates vaccination costs, PEP costs, suspect exposure costs, and cost-effectiveness metrics
- **Scenario Comparison**: Compares "No Annual Vaccination" vs "Annual Vaccination" programs over 30 years

### 📊 Comprehensive Results Display

#### Executive Summary Tab
- Key metrics for 10-year analysis period
- Deaths averted, additional costs, cost per death averted, cost per DALY averted
- Key findings and economic benefits overview

#### Program Summary Tab
- Program definitions for both scenarios
- Suspect exposure rates per 100,000 persons
- Fixed timeframe analysis (Years 5, 10, 30)
- Detailed metrics including rabid dogs, human deaths, program costs, and cost-effectiveness

#### Detailed Results Tab
- Year-by-year comparison tables
- Selectable years for detailed analysis
- Cumulative and annual metrics
- Cost breakdowns and deaths averted calculations

#### Visualizations Tab
- 2x2 subplot grid showing:
  - Rabid dogs (annual)
  - Canine rabies cases (cumulative)
  - Human deaths due to rabies (annual)
  - Human deaths (cumulative)
- Interactive plots comparing both scenarios over 30 years

### 📈 Analysis Components

The app internally performs:

1. **Initial Equilibrium Simulation**: 10,000-week simulation to establish baseline conditions
2. **No Annual Vaccination Scenario**: 2,300-week simulation with minimal vaccination
3. **Annual Vaccination Scenario**: 2,300-week simulation with time-varying vaccination coverage
4. **Economic Analysis**: Complete cost-benefit analysis including:
   - Vaccination costs
   - PEP (Post-Exposure Prophylaxis) costs
   - Suspect exposure investigation costs
   - Cost per death averted
   - Cost per DALY (Disability-Adjusted Life Year) averted

### 🎯 Key Outputs

- **Program Summary Table**: Excel-equivalent summary matching the original model
- **2x2 Visualization Plots**: Four coordinated plots showing disease and death trends
- **Cost-Effectiveness Metrics**: WHO-standard metrics for health economic evaluation
- **Interactive Results**: Streamlit interface for exploring different scenarios

## Technical Implementation

### Data Sources
- **Model Parameters**: Loaded from `data/model_parameters.xlsx`
- **Coverage Data**: Time-varying vaccination and PEP coverage from `data/coverage_data.csv`

### Caching
- Uses `@st.cache_data` for efficient data loading and parameter extraction
- Optimized for fast re-runs and parameter changes

### Mathematical Model
- **Dog Population**: SEIR model with vaccination, density-dependent mortality
- **Human Population**: Exposure-based model with PEP intervention
- **Economic Model**: Multi-component cost structure with time-varying parameters

### Visualization
- **Matplotlib Integration**: High-quality plots with consistent styling
- **Interactive Elements**: Streamlit widgets for parameter exploration
- **Responsive Design**: Adapts to different screen sizes and orientations

## Usage

1. **Launch the App**:
   ```bash
   python run_app.py
   ```

2. **Review Parameters**: Check the sidebar for key model parameters loaded from your data files

3. **Run Analysis**: Click the "🚀 Run Analysis" button to perform the complete calculation

4. **Explore Results**: Navigate through the four tabs to examine different aspects of the analysis

5. **Interpret Findings**: Use the executive summary and visualizations to understand the economic impact

## Results Interpretation

### Cost-Effectiveness Thresholds
- **WHO Guidelines**: Cost per DALY averted < 3× GDP per capita is considered cost-effective
- **Highly Cost-Effective**: Cost per DALY averted < 1× GDP per capita

### Key Metrics to Monitor
- **Deaths Averted**: Direct health impact of vaccination program
- **Cost per Death Averted**: Economic efficiency measure
- **Cost per DALY Averted**: International standard for health economics
- **Program ROI**: Long-term cost savings vs. upfront investment

### Visualization Insights
- **Canine Trends**: Shows how vaccination reduces disease transmission
- **Human Impact**: Demonstrates lives saved through prevention
- **Time Horizon**: Reveals when benefits outweigh costs
- **Cumulative Effects**: Shows long-term population health impact

## Data Requirements

The app requires two data files in the `data/` directory:

1. **`model_parameters.xlsx`**: Contains all model parameters including population sizes, transmission rates, costs, and epidemiological parameters

2. **`coverage_data.csv`**: Contains time-varying vaccination coverage and PEP access rates for both scenarios by year

## Technical Notes

- **Performance**: Complete analysis runs in under 60 seconds on standard hardware
- **Memory Usage**: Optimized for datasets with 30+ years of projections
- **Accuracy**: Results match Excel-based calculations within rounding precision
- **Extensibility**: Modular design allows easy addition of new scenarios or metrics

## Troubleshooting

### Common Issues

1. **Data File Not Found**: Ensure `model_parameters.xlsx` and `coverage_data.csv` are in the `data/` directory
2. **Parameter Missing**: Check that all required parameters are present in the Excel file
3. **Memory Issues**: For very large datasets, consider reducing the simulation time horizon

### Performance Optimization

- Results are cached for faster subsequent runs
- Use the parameter sidebar to verify data loading before running analysis
- Close other browser tabs if experiencing slow performance

## Dependencies

```python
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
openpyxl>=3.0.0  # For Excel file reading
```

## Output Files

The app generates in-memory results that can be:
- Viewed interactively in the browser
- Downloaded as images (plots)
- Copied as data tables for further analysis

This comprehensive app provides a complete, self-contained solution for rabies economic analysis without requiring external calculation scripts or wrapper functions.