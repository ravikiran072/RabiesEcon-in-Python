#!/usr/bin/env python3
"""
Quick test script to validate mortality rate plots functionality
"""

import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import required functions
from comprehensive_rabies_app import load_data, extract_model_parameters, run_scenario_simulation, extract_summary_values, create_mortality_rate_plots

def test_mortality_plots():
    """Test the mortality rate plots functionality"""
    print("Loading data...")
    coverage_data, model_parameters = load_data()
    params = extract_model_parameters(model_parameters)
    
    print("Running simulations...")
    # Get initial data (just use the first row of model parameters as initial conditions)
    initial_run = model_parameters.iloc[:1].copy()
    
    # Run both scenarios
    no_annual_vaccination = run_scenario_simulation(initial_run, params, coverage_data, "no_annual_vaccination")
    annual_vaccination = run_scenario_simulation(initial_run, params, coverage_data, "annual_vaccination")
    
    print("Extracting summary values...")
    # Extract summary values
    no_annual_summary = extract_summary_values(no_annual_vaccination, "No Annual Vaccination", params, coverage_data)
    annual_summary = extract_summary_values(annual_vaccination, "Annual Vaccination", params, coverage_data)
    
    print(f"No annual summary shape: {no_annual_summary.shape}")
    print(f"Annual summary shape: {annual_summary.shape}")
    print(f"No annual summary columns: {list(no_annual_summary.columns)}")
    
    # Check a few sample values
    print("\nSample data from year 1:")
    if len(no_annual_summary) > 1:
        year1_no = no_annual_summary.iloc[1]
        year1_annual = annual_summary.iloc[1]
        
        print(f"No vaccination - Dog deaths: {year1_no['Dog_deaths_annual']}, Dog population: {year1_no['Canine_population']}")
        print(f"Annual vaccination - Dog deaths: {year1_annual['Dog_deaths_annual']}, Dog population: {year1_annual['Canine_population']}")
        print(f"No vaccination - Human deaths: {year1_no['Human_rabies_annual']}, Human population: {year1_no['Human_population']}")
        print(f"Annual vaccination - Human deaths: {year1_annual['Human_rabies_annual']}, Human population: {year1_annual['Human_population']}")
    
    print("\nTesting mortality rate plots...")
    try:
        fig = create_mortality_rate_plots(no_annual_summary, annual_summary)
        print("✅ Mortality rate plots created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating mortality rate plots: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mortality_plots()
    if success:
        print("\n🎉 Test passed! Mortality rate plots are working correctly.")
    else:
        print("\n💥 Test failed! Check the errors above.")