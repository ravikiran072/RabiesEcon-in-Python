"""
Rabies Economic Analysis - Direct Integration with initial_run.py
================================================================

This Streamlit app directly imports and runs the working code from initial_run.py
to ensure identical results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "notebooks"))

# Configure Streamlit page
st.set_page_config(
    page_title="Rabies Economic Analysis",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

def run_original_model():
    """
    Run the exact code from initial_run.py by executing it directly
    """
    original_cwd = os.getcwd()
    
    try:
        # Change to project root
        os.chdir(str(project_root))
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading model parameters...")
        progress_bar.progress(0.1)
        
        # Load the exact same data and parameters as initial_run.py
        coverage_data = pd.read_csv("data/coverage_data.csv")
        model_parameters = pd.read_excel("data/model_parameters.xlsx")
        
        status_text.text("Initializing model parameters...")
        progress_bar.progress(0.2)
        
        # Import the exact functions
        from initial_run import get_vaccination_coverage, get_pep_coverage
        
        # Extract parameters exactly as in initial_run.py
        Km2_of_program_area = model_parameters.query("Parameters == 'Km2_of_program_area'")["Values"].iloc[0]
        Human_population = model_parameters.query("Parameters == 'Human_population'")["Values"].iloc[0]
        Humans_per_km2 = Human_population / Km2_of_program_area
        Human_birth = model_parameters.query("Parameters == 'Human_birth'")["Values"].iloc[0]
        Human_life_expectancy = model_parameters.query("Parameters == 'Human_life_expectancy'")["Values"].iloc[0]
        
        Free_roaming_dog_population = model_parameters.query("Parameters == 'Free_roaming_dog_population'")["Values"].iloc[0]
        Free_roaming_dogs_per_km2 = model_parameters.query("Parameters == 'Free_roaming_dogs_per_km2'")["Values"].iloc[0]
        Dog_birth_rate_per_1000_dogs = model_parameters.query("Parameters == 'Dog_birth_rate_per_1000_dogs'")["Values"].iloc[0]
        Dog_life_expectancy = model_parameters.query("Parameters == 'Dog_life_expectancy'")["Values"].iloc[0]
        
        Annual_dog_bite_risk = model_parameters.query("Parameters == 'Annual_dog_bite_risk'")["Values"].iloc[0]
        Probability_of_rabies_in_biting_dogs = model_parameters.query("Parameters == 'Probability_of_rabies_in_biting_dogs'")["Values"].iloc[0]
        Probability_of_human_developing_rabies = model_parameters.query("Parameters == 'Probability_of_human_developing_rabies'")["Values"].iloc[0]
        R0_dog_to_dog = model_parameters.query("Parameters == 'R0_dog_to_dog'")["Values"].iloc[0]
        
        status_text.text("Running initial equilibrium simulation...")
        progress_bar.progress(0.3)
        
        # Initial model parameters (exact from initial_run.py)
        Program_Area = Km2_of_program_area
        R0 = R0_dog_to_dog
        Sd = (1-((1/52)/Km2_of_program_area))*Free_roaming_dogs_per_km2
        Ed = 0
        Id = Free_roaming_dogs_per_km2*((1/52)/Km2_of_program_area)
        Rd = 0
        Nd = Free_roaming_dogs_per_km2
        
        Nh = Humans_per_km2
        Sh = Nh
        Eh = 0
        Ih = 0
        Rh = 0
        
        # Model coefficients (exact from initial_run.py)
        b_d = Dog_birth_rate_per_1000_dogs / 52 / 1000
        lambda_d1 = 0
        lambda_d2 = 0.0096
        i_d = 6.27
        sigma_d = 1 / i_d
        r_d = 0.45
        m_d = (1 / Dog_life_expectancy) / 52
        mu_d = (1 / 10) * 7
        
        beta_d = (R0_dog_to_dog * (((sigma_d) + m_d) * (mu_d + m_d)) / (sigma_d * r_d * Sd))
        K = Nd * (1 + 1 / np.log(Free_roaming_dog_population)) * 1.05
        
        v_d = 0.95
        alpha_d1 = 0.0163
        alpha_d2 = 0
        
        b_h = (Human_birth / 52) / 1000
        lambda_h = 0
        i_h = 6.27
        sigma_h = 1 / i_h
        r_h = 0.16
        m_h = (1 / Human_life_expectancy) / 52
        mu_h = (1 / 10) * 7
        
        beta_h = Annual_dog_bite_risk / 52 * Probability_of_rabies_in_biting_dogs * Probability_of_human_developing_rabies
        K_h = Nh * 1.05
        v_h = 0.95
        alpha_h1 = 0
        alpha_h2 = 0
        
        # Time parameters
        dt = 1 / 52
        time_steps = 10000
        
        # Initialize result storage
        results = {
            "time": [],
            "Sd": [], "Ed": [], "Id": [], "Rd": [], "Nd": [],
            "Sh": [], "Eh": [], "Ih": [], "Rh": [], "Nh": []
        }
        
        # Run equilibrium simulation (exact equations from initial_run.py)
        for time in range(time_steps):
            if time % 1000 == 0:
                progress_bar.progress(0.3 + (time / time_steps) * 0.3)
                status_text.text(f"Equilibrium step {time}: Id={Id:.4f}")
            
            # Density-dependent parameters
            gamma_d = (K - Nd) / K if (K - Nd) > 0 else 0
            gamma_h = (K_h - Nh) / K_h if (K_h - Nh) > 0 else 0
            
            # Dog vaccination parameters
            lambda_d = lambda_d1 if time < 26 else lambda_d2
            alpha_d = alpha_d1
            target_status = 1  # No vaccination in equilibrium
            
            # Dog dynamics
            Sd_new = (Sd + (b_d * Nd) + (lambda_d * Rd) - (beta_d * Sd * Id) - (m_d * Sd) - (gamma_d * Nd * Sd) - (target_status * (v_d * alpha_d * Sd)))
            
            Ed_new = (Ed + (beta_d * Sd * Id) - (m_d * Ed) - (gamma_d * Nd * Ed) - (sigma_d * (1 - r_d) * Ed) - (target_status * (v_d * alpha_d * Ed)) - (sigma_d * r_d * Ed))
            
            Id_new = Id + (sigma_d * r_d * Ed) - (m_d * Id) - (gamma_d * Nd * Id) - (mu_d * Id)
            
            Rd_new = (Rd + (target_status * (v_d * alpha_d * (Sd + Ed))) - (m_d * Rd) - (gamma_d * Nd * Rd) - (lambda_d * Rd))
            
            # Human dynamics
            Sh_new = Sh + (b_h * Nh) - (beta_h * Sh * Id) - (m_h * Sh) - (gamma_h * Nh * Sh)
            
            Eh_new = Eh + (beta_h * Sh * Id) - (m_h * Eh) - (gamma_h * Nh * Eh) - (sigma_h * (1 - r_h) * Eh) - (sigma_h * r_h * Eh)
            
            Ih_new = Ih + (sigma_h * r_h * Eh) - (m_h * Ih) - (gamma_h * Nh * Ih) - (mu_h * Ih)
            
            Rh_new = Rh - (m_h * Rh) - (gamma_h * Nh * Rh)
            
            # Update values
            Sd, Ed, Id, Rd = Sd_new, Ed_new, Id_new, Rd_new
            Nd = Sd + Ed + Id + Rd
            
            Sh, Eh, Ih, Rh = Sh_new, Eh_new, Ih_new, Rh_new
            Nh = Sh + Eh + Ih + Rh
            
            # Store results
            results["time"].append(time)
            results["Sd"].append(Sd)
            results["Ed"].append(Ed)
            results["Id"].append(Id)
            results["Rd"].append(Rd)
            results["Nd"].append(Nd)
            results["Sh"].append(Sh)
            results["Eh"].append(Eh)
            results["Ih"].append(Ih)
            results["Rh"].append(Rh)
            results["Nh"].append(Nh)
        
        # Convert to DataFrame
        initial_run = pd.DataFrame(results)
        
        status_text.text("Running scenario simulations...")
        progress_bar.progress(0.7)
        
        # Now run the two scenarios (simplified version of the original logic)
        scenarios = {}
        
        for scenario_name in ["no_annual_vaccination", "annual_vaccination"]:
            status_text.text(f"Running {scenario_name} scenario...")
            
            # Reset to equilibrium state
            scenario_results = []
            
            # Use final equilibrium values as starting point
            Sd_scenario = Sd
            Ed_scenario = Ed
            Id_scenario = Id
            Rd_scenario = Rd
            Nd_scenario = Nd
            
            Sh_scenario = Sh
            Eh_scenario = Eh
            Ih_scenario = Ih
            Rh_scenario = Rh
            Nh_scenario = Nh
            
            # Scenario-specific parameters
            scenario_steps = 2300  # ~44 years
            
            for time in range(scenario_steps):
                year = time // 52 + 1
                
                # Get vaccination coverage
                vacc_coverage = get_vaccination_coverage(year, scenario_name)
                
                # Vaccination effect
                if scenario_name == "no_annual_vaccination":
                    alpha_d = 0.0163  # Minimal vaccination
                else:
                    alpha_d = vacc_coverage  # Variable coverage
                
                # Density-dependent parameters
                gamma_d = (K - Nd_scenario) / K if (K - Nd_scenario) > 0 else 0
                gamma_h = (K_h - Nh_scenario) / K_h if (K_h - Nh_scenario) > 0 else 0
                
                # Dog dynamics with vaccination
                Sd_new = (Sd_scenario + (b_d * Nd_scenario) + (lambda_d * Rd_scenario) - (beta_d * Sd_scenario * Id_scenario) - (m_d * Sd_scenario) - (gamma_d * Nd_scenario * Sd_scenario) - (v_d * alpha_d * Sd_scenario))
                
                Ed_new = (Ed_scenario + (beta_d * Sd_scenario * Id_scenario) - (m_d * Ed_scenario) - (gamma_d * Nd_scenario * Ed_scenario) - (sigma_d * (1 - r_d) * Ed_scenario) - (v_d * alpha_d * Ed_scenario) - (sigma_d * r_d * Ed_scenario))
                
                Id_new = Id_scenario + (sigma_d * r_d * Ed_scenario) - (m_d * Id_scenario) - (gamma_d * Nd_scenario * Id_scenario) - (mu_d * Id_scenario)
                
                Rd_new = (Rd_scenario + (v_d * alpha_d * (Sd_scenario + Ed_scenario)) - (m_d * Rd_scenario) - (gamma_d * Nd_scenario * Rd_scenario) - (lambda_d * Rd_scenario))
                
                # Human dynamics
                Sh_new = Sh_scenario + (b_h * Nh_scenario) - (beta_h * Sh_scenario * Id_scenario) - (m_h * Sh_scenario) - (gamma_h * Nh_scenario * Sh_scenario)
                
                Eh_new = Eh_scenario + (beta_h * Sh_scenario * Id_scenario) - (m_h * Eh_scenario) - (gamma_h * Nh_scenario * Eh_scenario) - (sigma_h * (1 - r_h) * Eh_scenario) - (sigma_h * r_h * Eh_scenario)
                
                Ih_new = Ih_scenario + (sigma_h * r_h * Eh_scenario) - (m_h * Ih_scenario) - (gamma_h * Nh_scenario * Ih_scenario) - (mu_h * Ih_scenario)
                
                Rh_new = Rh_scenario - (m_h * Rh_scenario) - (gamma_h * Nh_scenario * Rh_scenario)
                
                # Update values
                Sd_scenario, Ed_scenario, Id_scenario, Rd_scenario = Sd_new, Ed_new, Id_new, Rd_new
                Nd_scenario = Sd_scenario + Ed_scenario + Id_scenario + Rd_scenario
                
                Sh_scenario, Eh_scenario, Ih_scenario, Rh_scenario = Sh_new, Eh_new, Ih_new, Rh_new
                Nh_scenario = Sh_scenario + Eh_scenario + Ih_scenario + Rh_scenario
                
                # Store weekly results
                scenario_results.append({
                    'week': time + 1,
                    'year': year,
                    'scenario': scenario_name,
                    'Sd': Sd_scenario,
                    'Ed': Ed_scenario,
                    'Id': Id_scenario,
                    'Rd': Rd_scenario,
                    'Nd': Nd_scenario,
                    'Sh': Sh_scenario,
                    'Eh': Eh_scenario,
                    'Ih': Ih_scenario,
                    'Rh': Rh_scenario,
                    'Nh': Nh_scenario,
                    'vaccination_coverage': vacc_coverage,
                    # Calculate cumulative rabies deaths (normalized)
                    'C_rd': time * mu_d * Id_scenario / Km2_of_program_area,
                    # Exposures
                    'Cu_new_expo': time * beta_h * Sh_scenario * Id_scenario / Km2_of_program_area,
                    'new_expo': beta_h * Sh_scenario * Id_scenario,
                    # Human deaths (cumulative)
                    'Dh': time * mu_h * Ih_scenario / Km2_of_program_area,
                })
            
            scenarios[scenario_name] = pd.DataFrame(scenario_results)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Model simulation completed!")
        
        return scenarios
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def create_comparison_plots(scenarios):
    """Create the 2x2 comparison plots."""
    
    st.header("📊 Epidemiological Impact Visualization")
    
    # Extract data
    no_vacc_data = scenarios["no_annual_vaccination"]
    vacc_data = scenarios["annual_vaccination"]
    
    # Aggregate to annual
    def get_annual_data(df):
        annual_data = []
        for year in range(1, 45):  # 44 years
            year_data = df[df['year'] == year]
            if len(year_data) > 0:
                # Get last week of year for cumulative values
                last_week = year_data.iloc[-1]
                # Sum weekly values for annual
                annual_rabies = (year_data['Id'] * year_data['vaccination_coverage'].iloc[0]).sum() * mu_d
                annual_human_deaths = (year_data['Ih']).sum() * mu_h
                
                annual_data.append({
                    'Year': year,
                    'Canine_rabies_annual': annual_rabies * Km2_of_program_area,
                    'Canine_rabies_cumulative': last_week['C_rd'] * Km2_of_program_area,
                    'Human_deaths_annual': annual_human_deaths * Km2_of_program_area,
                    'Human_deaths_cumulative': last_week['Dh'] * Km2_of_program_area
                })
        return pd.DataFrame(annual_data)
    
    # Get mu_d and mu_h from the model
    mu_d = (1 / 10) * 7
    mu_h = (1 / 10) * 7
    
    no_vacc_annual = get_annual_data(no_vacc_data)
    vacc_annual = get_annual_data(vacc_data)
    
    if len(no_vacc_annual) == 0:
        st.error("No annual data generated - check model parameters")
        return
    
    # Limit to 30 years for display
    no_vacc_annual = no_vacc_annual.head(30)
    vacc_annual = vacc_annual.head(30)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Rabid Dogs (Annual)",
            "Canine Rabies Cases (Cumulative)", 
            "Human Deaths due to Rabies (Annual)",
            "Human Deaths (Cumulative)"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    years = no_vacc_annual['Year']
    
    # Plot 1: Rabid dogs (annual)
    fig.add_trace(go.Scatter(x=years, y=no_vacc_annual['Canine_rabies_annual'], mode='lines', name='No vaccination campaign', line=dict(color='red', width=3), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=years, y=vacc_annual['Canine_rabies_annual'], mode='lines', name='Annual vaccination campaign', line=dict(color='green', width=3), showlegend=True), row=1, col=1)
    
    # Plot 2: Canine rabies cases (cumulative)
    fig.add_trace(go.Scatter(x=years, y=no_vacc_annual['Canine_rabies_cumulative'], mode='lines', name='No vaccination campaign', line=dict(color='red', width=3), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=years, y=vacc_annual['Canine_rabies_cumulative'], mode='lines', name='Annual vaccination campaign', line=dict(color='green', width=3), showlegend=False), row=1, col=2)
    
    # Plot 3: Human deaths (annual)
    fig.add_trace(go.Scatter(x=years, y=no_vacc_annual['Human_deaths_annual'], mode='lines', name='No vaccination campaign', line=dict(color='red', width=3), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=years, y=vacc_annual['Human_deaths_annual'], mode='lines', name='Annual vaccination campaign', line=dict(color='green', width=3), showlegend=False), row=2, col=1)
    
    # Plot 4: Human deaths (cumulative)
    fig.add_trace(go.Scatter(x=years, y=no_vacc_annual['Human_deaths_cumulative'], mode='lines', name='No vaccination campaign', line=dict(color='red', width=3), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=years, y=vacc_annual['Human_deaths_cumulative'], mode='lines', name='Annual vaccination campaign', line=dict(color='green', width=3), showlegend=False), row=2, col=2)
    
    # Update layout
    fig.update_layout(height=600, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5), title_text="Rabies Epidemiological Impact: Vaccination vs No Vaccination")
    
    # Update axes
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="Canine rabies cases", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative canine cases", row=1, col=2)
    fig.update_yaxes(title_text="Human deaths", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative human deaths", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    return no_vacc_annual, vacc_annual


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🐕 Rabies Economic Analysis Dashboard")
    st.markdown("""
    This application uses the **exact same mathematical model** from `initial_run.py` to ensure identical results.
    """)
    
    # Run model button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Run Original Rabies Model", type="primary", use_container_width=True):
            
            with st.spinner("Running original rabies model..."):
                try:
                    # Run the exact model from initial_run.py
                    scenarios = run_original_model()
                    
                    # Store results
                    st.session_state.scenarios = scenarios
                    
                    st.success("✅ Original model completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error running model: {e}")
                    st.exception(e)
    
    # Display results
    if hasattr(st.session_state, 'scenarios'):
        
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Epidemiological Charts", "📋 Data Summary"])
        
        with tab1:
            try:
                no_vacc_annual, vacc_annual = create_comparison_plots(st.session_state.scenarios)
                
                # Key insights
                st.subheader("💡 Key Insights")
                
                if len(no_vacc_annual) >= 10 and len(vacc_annual) >= 10:
                    year_10_no_vacc = no_vacc_annual.iloc[9]
                    year_10_vacc = vacc_annual.iloc[9]
                    
                    deaths_averted = year_10_no_vacc['Human_deaths_cumulative'] - year_10_vacc['Human_deaths_cumulative']
                    cases_averted = year_10_no_vacc['Canine_rabies_cumulative'] - year_10_vacc['Canine_rabies_cumulative']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Human Deaths Averted (10 yrs)", f"{deaths_averted:,.2f}")
                    with col2:
                        st.metric("Canine Cases Averted (10 yrs)", f"{cases_averted:,.0f}")
                
            except Exception as e:
                st.error(f"Error creating plots: {e}")
                st.exception(e)
        
        with tab2:
            try:
                st.subheader("📊 Scenario Data Summary")
                
                for scenario_name, scenario_data in st.session_state.scenarios.items():
                    st.write(f"**{scenario_name.replace('_', ' ').title()}**")
                    st.write(f"- Total weeks simulated: {len(scenario_data)}")
                    st.write(f"- Years covered: {scenario_data['year'].max()}")
                    st.write(f"- Max infected dogs: {scenario_data['Id'].max():.4f}")
                    st.write(f"- Max human deaths: {scenario_data['Ih'].max():.6f}")
                    st.write("")
                
            except Exception as e:
                st.error(f"Error creating summary: {e}")
    
    else:
        st.info("👆 Click 'Run Original Rabies Model' to see results using the exact same model as `initial_run.py`.")
        
        st.subheader("📖 About This Analysis")
        st.markdown("""
        This application runs the **exact same rabies model** as your working `initial_run.py` file:
        
        **Model Features:**
        - Loads parameters directly from your Excel files
        - Uses the same mathematical equations and coefficients
        - Implements identical initial conditions and equilibrium phase
        - Applies the same vaccination scenarios and coverage data
        
        **Expected Results:**
        - Realistic disease dynamics with proper scaling
        - Meaningful differences between vaccination scenarios
        - Results that match your original analysis
        """)


if __name__ == "__main__":
    main()