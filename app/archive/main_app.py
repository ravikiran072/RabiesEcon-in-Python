"""
Rabies Economic Analysis - Streamlit App
=======================================

A comprehensive Streamlit application for rabies economic analysis and vaccination program evaluation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.model_parameters import ModelParameters, load_parameters_from_excel, create_parameter_scenarios

# Configure Streamlit page
st.set_page_config(
    page_title="Rabies Economic Analysis",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RabiesModelRunner:
    """Simplified version of the rabies model for Streamlit integration."""
    
    def __init__(self, params: ModelParameters):
        self.params = params
        self.setup_model_parameters()
    
    def setup_model_parameters(self):
        """Setup all model parameters from the parameter object."""
        p = self.params
        
        # Basic model parameters
        self.Km2_of_program_area = p.Km2_of_program_area
        self.Human_population = p.Human_population
        self.Humans_per_km2 = p.Humans_per_km2
        self.Free_roaming_dogs_per_km2 = p.Free_roaming_dogs_per_km2
        self.Free_roaming_dog_population = p.Free_roaming_dog_population
        
        # Disease parameters
        self.R0_dog_to_dog = p.R0_dog_to_dog
        self.Annual_dog_bite_risk = p.Annual_dog_bite_risk
        self.Prob_rabies_biting = p.Probability_of_rabies_in_biting_dogs
        self.Prob_human_rabies = p.Probability_of_human_developing_rabies
        
        # Economic parameters
        self.vaccination_cost_per_dog = p.vaccination_cost_per_dog
        self.pep_and_other_costs = p.pep_and_other_costs
        self.pep_prob_no_campaign = p.pep_prob_no_campaign
        self.pep_prob_annual_campaign = p.pep_prob_annual_campaign
        self.cost_per_suspect_exposure = p.cost_per_suspect_exposure
        
        # Demographic parameters
        self.Human_birth = p.Human_birth
        self.Human_life_expectancy = p.Human_life_expectancy
        self.Dog_life_expectancy = p.Dog_life_expectancy
        self.YLL = p.YLL
        
        # Suspect exposure parameters
        self.inflation_factor = p.inflation_factor_for_the_suspect_exposure
        self.post_elimination_reduction = p.post_elimination_pep_reduction
    
    def run_simplified_model(self, years=30, vaccination_scenario="no_vaccination"):
        """
        Run a simplified version of the rabies model.
        Returns results for the specified scenario.
        """
        results = []
        
        # Initial conditions based on parameters
        initial_dogs = self.Free_roaming_dogs_per_km2 * self.Km2_of_program_area
        initial_humans = self.Human_population
        
        # Model coefficients (simplified)
        if vaccination_scenario == "no_vaccination":
            vacc_coverage = 0.05  # 5% baseline
            pep_probability = self.pep_prob_no_campaign
            transmission_multiplier = 1.0
        else:  # annual_vaccination
            vacc_coverage = 0.70  # 70% with vaccination program
            pep_probability = self.pep_prob_annual_campaign
            transmission_multiplier = 0.3  # Reduced transmission with vaccination
        
        # Simplified model calculations
        for year in range(1, years + 1):
            # Disease dynamics (simplified exponential decay with vaccination)
            base_rabies_rate = 0.001 * self.R0_dog_to_dog * transmission_multiplier
            
            # Canine rabies cases
            if vaccination_scenario == "no_vaccination":
                canine_rabies_annual = initial_dogs * base_rabies_rate * (0.95 ** (year - 1))
            else:
                # More aggressive reduction with vaccination
                canine_rabies_annual = initial_dogs * base_rabies_rate * (0.7 ** (year - 1))
            
            canine_rabies_cumulative = sum([
                initial_dogs * base_rabies_rate * (0.95 ** (y - 1) if vaccination_scenario == "no_vaccination" else 0.7 ** (y - 1))
                for y in range(1, year + 1)
            ])
            
            # Human deaths (proportional to canine rabies)
            human_death_rate = self.Annual_dog_bite_risk * self.Prob_rabies_biting * self.Prob_human_rabies
            human_deaths_annual = canine_rabies_annual * human_death_rate * 10  # Scaling factor
            human_deaths_cumulative = canine_rabies_cumulative * human_death_rate * 10
            
            # Exposures
            exposure_annual = canine_rabies_annual * 50  # Approximate exposures per rabid dog
            exposure_cumulative = canine_rabies_cumulative * 50
            
            # Suspect exposures
            if vaccination_scenario == "no_vaccination":
                suspect_exposure_annual = exposure_annual * self.inflation_factor
                suspect_exposure_cumulative = exposure_cumulative * self.inflation_factor
            else:
                # Apply post-elimination reduction for vaccination scenario
                if year <= 5:
                    suspect_exposure_annual = exposure_annual * self.inflation_factor
                    suspect_exposure_cumulative = exposure_cumulative * self.inflation_factor
                else:
                    reduction_factor = 1 - self.post_elimination_reduction
                    suspect_exposure_annual = exposure_annual * self.inflation_factor * reduction_factor
                    suspect_exposure_cumulative = exposure_cumulative * self.inflation_factor * reduction_factor
            
            # Costs
            # Vaccination costs
            vaccination_cost_annual = initial_dogs * vacc_coverage * self.vaccination_cost_per_dog
            vaccination_cost_cumulative = vaccination_cost_annual * year
            
            # Suspect exposure costs
            suspect_cost_annual = suspect_exposure_annual * self.cost_per_suspect_exposure
            suspect_cost_cumulative = suspect_exposure_cumulative * self.cost_per_suspect_exposure
            
            # PEP costs
            pep_cost_annual = suspect_exposure_annual * pep_probability * self.pep_and_other_costs
            pep_cost_cumulative = suspect_exposure_cumulative * pep_probability * self.pep_and_other_costs
            
            # Total costs
            total_cost_annual = vaccination_cost_annual + suspect_cost_annual + pep_cost_annual
            total_cost_cumulative = vaccination_cost_cumulative + suspect_cost_cumulative + pep_cost_cumulative
            
            results.append({
                'Year': year,
                'Canine_population': initial_dogs,
                'Canine_rabies_annual': canine_rabies_annual,
                'Canine_rabies_cumulative': canine_rabies_cumulative,
                'Human_population': initial_humans,
                'Human_deaths_annual': human_deaths_annual,
                'Human_deaths_cumulative': human_deaths_cumulative,
                'Exposure_annual': exposure_annual,
                'Exposure_cumulative': exposure_cumulative,
                'Suspect_exposure_annual': suspect_exposure_annual,
                'Suspect_exposure_cumulative': suspect_exposure_cumulative,
                'Vaccination_cost_annual': vaccination_cost_annual,
                'Vaccination_cost_cumulative': vaccination_cost_cumulative,
                'Suspect_cost_annual': suspect_cost_annual,
                'Suspect_cost_cumulative': suspect_cost_cumulative,
                'PEP_cost_annual': pep_cost_annual,
                'PEP_cost_cumulative': pep_cost_cumulative,
                'Total_cost_annual': total_cost_annual,
                'Total_cost_cumulative': total_cost_cumulative
            })
        
        return pd.DataFrame(results)


def create_program_summary_table(no_vacc_results, vacc_results, params):
    """Create the program summary table matching the original format."""
    
    st.header("📋 Program Summary Table")
    st.markdown("---")
    
    # Program definition
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("No Vaccination Program")
        st.write("• Single/one time vaccination")
        st.write("• 5% vaccination coverage")
        st.write("• 25% human exposures receive PEP")
        st.write("• 0% female dogs spayed annually")
        st.write("• 0% male dogs neutered annually")
    
    with col2:
        st.subheader("Vaccination Option 1")
        st.write("• Annual vaccination program")
        st.write("• 70% vaccination coverage")
        st.write("• 50% human exposures receive PEP")
        st.write("• 0% female dogs spayed annually")
        st.write("• 0% male dogs neutered annually")
    
    # Suspect exposure rates
    st.subheader("📊 Suspect Human Rabies Exposures (per 100,000 persons) in Year 1")
    col1, col2 = st.columns(2)
    
    year1_no_vacc = (no_vacc_results.iloc[0]['Suspect_exposure_annual'] / params.Human_population) * 100000
    year1_vacc = (vacc_results.iloc[0]['Suspect_exposure_annual'] / params.Human_population) * 100000
    
    with col1:
        st.metric("No Vaccination Program", f"{year1_no_vacc:.2f}")
    with col2:
        st.metric("Vaccination Option 1", f"{year1_vacc:.2f}")
    
    # Main results table
    st.subheader("📈 Key Metrics Comparison")
    
    # Extract data for years 5, 10, 30
    years_to_show = [5, 10, 30]
    
    table_data = []
    
    for year in years_to_show:
        year_idx = year - 1  # 0-indexed
        
        if year_idx < len(no_vacc_results):
            no_vacc_row = no_vacc_results.iloc[year_idx]
            vacc_row = vacc_results.iloc[year_idx]
            
            # Rabid dogs
            table_data.append({
                'Metric': 'Rabid dogs',
                'Time Period': f'Year {year}',
                'No Vacc Annual': f"{no_vacc_row['Canine_rabies_annual']:,.0f}",
                'No Vacc Cumulative': f"{no_vacc_row['Canine_rabies_cumulative']:,.0f}",
                'Vacc1 Annual': f"{vacc_row['Canine_rabies_annual']:,.0f}",
                'Vacc1 Cumulative': f"{vacc_row['Canine_rabies_cumulative']:,.0f}"
            })
            
            # Human deaths
            table_data.append({
                'Metric': 'Human deaths',
                'Time Period': f'Year {year}',
                'No Vacc Annual': f"{no_vacc_row['Human_deaths_annual']:.0f}",
                'No Vacc Cumulative': f"{no_vacc_row['Human_deaths_cumulative']:,.0f}",
                'Vacc1 Annual': f"{vacc_row['Human_deaths_annual']:.0f}",
                'Vacc1 Cumulative': f"{vacc_row['Human_deaths_cumulative']:,.0f}"
            })
            
            # Program costs
            table_data.append({
                'Metric': 'Program costs',
                'Time Period': f'Year {year}',
                'No Vacc Annual': f"${no_vacc_row['Total_cost_annual']:,.0f}",
                'No Vacc Cumulative': f"${no_vacc_row['Total_cost_cumulative']:,.0f}",
                'Vacc1 Annual': f"${vacc_row['Total_cost_annual']:,.0f}",
                'Vacc1 Cumulative': f"${vacc_row['Total_cost_cumulative']:,.0f}"
            })
            
            # Cost effectiveness
            deaths_averted_annual = no_vacc_row['Human_deaths_annual'] - vacc_row['Human_deaths_annual']
            deaths_averted_cumulative = no_vacc_row['Human_deaths_cumulative'] - vacc_row['Human_deaths_cumulative']
            
            additional_cost_annual = vacc_row['Total_cost_annual'] - no_vacc_row['Total_cost_annual']
            additional_cost_cumulative = vacc_row['Total_cost_cumulative'] - no_vacc_row['Total_cost_cumulative']
            
            if deaths_averted_annual > 0:
                cost_per_death_annual = additional_cost_annual / deaths_averted_annual
                cost_per_daly_annual = cost_per_death_annual / params.YLL
            else:
                cost_per_death_annual = 0
                cost_per_daly_annual = 0
            
            if deaths_averted_cumulative > 0:
                cost_per_death_cumulative = additional_cost_cumulative / deaths_averted_cumulative
                cost_per_daly_cumulative = cost_per_death_cumulative / params.YLL
            else:
                cost_per_death_cumulative = 0
                cost_per_daly_cumulative = 0
            
            table_data.append({
                'Metric': 'Cost per death averted',
                'Time Period': f'Year {year}',
                'No Vacc Annual': 'N/A',
                'No Vacc Cumulative': 'N/A',
                'Vacc1 Annual': f"${cost_per_death_annual:,.0f}" if cost_per_death_annual > 0 else 'N/A',
                'Vacc1 Cumulative': f"${cost_per_death_cumulative:,.0f}" if cost_per_death_cumulative > 0 else 'N/A'
            })
            
            table_data.append({
                'Metric': 'Cost per DALY averted',
                'Time Period': f'Year {year}',
                'No Vacc Annual': 'N/A',
                'No Vacc Cumulative': 'N/A',
                'Vacc1 Annual': f"${cost_per_daly_annual:,.0f}" if cost_per_daly_annual > 0 else 'N/A',
                'Vacc1 Cumulative': f"${cost_per_daly_cumulative:,.0f}" if cost_per_daly_cumulative > 0 else 'N/A'
            })
    
    # Display table
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    return df_table


def create_comparison_plots(no_vacc_results, vacc_results):
    """Create the 2x2 comparison plots."""
    
    st.header("📊 Epidemiological Impact Visualization")
    
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
    
    years = no_vacc_results['Year']
    
    # Plot 1: Rabid dogs (annual) - Top Left
    fig.add_trace(
        go.Scatter(
            x=years,
            y=no_vacc_results['Canine_rabies_annual'],
            mode='lines',
            name='No vaccination campaign',
            line=dict(color='red', width=3),
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=years,
            y=vacc_results['Canine_rabies_annual'],
            mode='lines',
            name='Annual vaccination campaign',
            line=dict(color='green', width=3),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Plot 2: Canine rabies cases (cumulative) - Top Right
    fig.add_trace(
        go.Scatter(
            x=years,
            y=no_vacc_results['Canine_rabies_cumulative'],
            mode='lines',
            name='No vaccination campaign',
            line=dict(color='red', width=3),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=years,
            y=vacc_results['Canine_rabies_cumulative'],
            mode='lines',
            name='Annual vaccination campaign',
            line=dict(color='green', width=3),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Plot 3: Human deaths (annual) - Bottom Left
    fig.add_trace(
        go.Scatter(
            x=years,
            y=no_vacc_results['Human_deaths_annual'],
            mode='lines',
            name='No vaccination campaign',
            line=dict(color='red', width=3),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=years,
            y=vacc_results['Human_deaths_annual'],
            mode='lines',
            name='Annual vaccination campaign',
            line=dict(color='green', width=3),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Plot 4: Human deaths (cumulative) - Bottom Right
    fig.add_trace(
        go.Scatter(
            x=years,
            y=no_vacc_results['Human_deaths_cumulative'],
            mode='lines',
            name='No vaccination campaign',
            line=dict(color='red', width=3),
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=years,
            y=vacc_results['Human_deaths_cumulative'],
            mode='lines',
            name='Annual vaccination campaign',
            line=dict(color='green', width=3),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        title_text="Rabies Epidemiological Impact: Vaccination vs No Vaccination"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="Canine rabies cases", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative canine cases", row=1, col=2)
    fig.update_yaxes(title_text="Human deaths", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative human deaths", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🐕 Rabies Economic Analysis Dashboard")
    st.markdown("""
    This application analyzes the economic impact of rabies vaccination programs compared to no intervention scenarios.
    Adjust the parameters in the sidebar to see how different conditions affect the epidemiological and economic outcomes.
    """)
    
    # Initialize session state
    if 'params' not in st.session_state:
        st.session_state.params = load_parameters_from_excel()
    
    # Sidebar for parameters
    st.sidebar.header("🎛️ Model Parameters")
    
    # Scenario selection
    scenarios = create_parameter_scenarios()
    scenario_names = list(scenarios.keys())
    
    selected_scenario = st.sidebar.selectbox(
        "📋 Select Scenario:",
        scenario_names,
        index=0,
        help="Choose a predefined scenario or modify parameters manually"
    )
    
    if selected_scenario != st.session_state.get('current_scenario', ''):
        st.session_state.current_scenario = selected_scenario
        st.session_state.params = scenarios[selected_scenario]
    
    # Parameter controls
    st.sidebar.subheader("📊 Key Parameters")
    
    # Geographic parameters
    with st.sidebar.expander("🗺️ Geographic Parameters"):
        program_area = st.number_input(
            "Program Area (km²)",
            min_value=100.0,
            max_value=50000.0,
            value=float(st.session_state.params.Km2_of_program_area),
            step=100.0,
            help="Total area covered by the rabies control program"
        )
        
        human_population = st.number_input(
            "Human Population",
            min_value=50000.0,
            max_value=50000000.0,
            value=float(st.session_state.params.Human_population),
            step=10000.0,
            help="Total human population in the program area"
        )
    
    # Epidemiological parameters
    with st.sidebar.expander("🦠 Disease Parameters"):
        r0 = st.slider(
            "R0 (Basic Reproduction Number)",
            min_value=0.5,
            max_value=3.0,
            value=float(st.session_state.params.R0_dog_to_dog),
            step=0.01,
            help="Average number of secondary infections from one infected dog"
        )
        
        bite_risk = st.slider(
            "Annual Dog Bite Risk (%)",
            min_value=0.5,
            max_value=5.0,
            value=float(st.session_state.params.Annual_dog_bite_risk * 100),
            step=0.1,
            help="Percentage of population at risk of dog bites annually"
        ) / 100
        
        rabies_prob = st.slider(
            "Rabies Probability in Biting Dogs (%)",
            min_value=0.1,
            max_value=5.0,
            value=float(st.session_state.params.Probability_of_rabies_in_biting_dogs * 100),
            step=0.1,
            help="Percentage of biting dogs that are rabid"
        ) / 100
    
    # Economic parameters
    with st.sidebar.expander("💰 Economic Parameters"):
        vacc_cost = st.number_input(
            "Vaccination Cost per Dog ($)",
            min_value=1.0,
            max_value=10.0,
            value=float(st.session_state.params.vaccination_cost_per_dog),
            step=0.1,
            help="Cost to vaccinate one dog"
        )
        
        pep_cost = st.number_input(
            "PEP Treatment Cost ($)",
            min_value=10.0,
            max_value=50.0,
            value=float(st.session_state.params.pep_and_other_costs),
            step=0.5,
            help="Cost of post-exposure prophylaxis treatment"
        )
    
    # Update parameters
    st.session_state.params.update_parameter('Km2_of_program_area', program_area)
    st.session_state.params.update_parameter('Human_population', human_population)
    st.session_state.params.update_parameter('R0_dog_to_dog', r0)
    st.session_state.params.update_parameter('Annual_dog_bite_risk', bite_risk)
    st.session_state.params.update_parameter('Probability_of_rabies_in_biting_dogs', rabies_prob)
    st.session_state.params.update_parameter('vaccination_cost_per_dog', vacc_cost)
    st.session_state.params.update_parameter('pep_and_other_costs', pep_cost)
    
    # Display current parameter summary
    st.sidebar.subheader("📋 Current Configuration")
    st.sidebar.metric("Program Area", f"{program_area:,.0f} km²")
    st.sidebar.metric("Human Population", f"{human_population:,.0f}")
    st.sidebar.metric("Population Density", f"{human_population/program_area:.0f} per km²")
    st.sidebar.metric("R0", f"{r0:.2f}")
    
    # Run model button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Running rabies economic analysis..."):
            # Initialize model
            model = RabiesModelRunner(st.session_state.params)
            
            # Run both scenarios
            no_vacc_results = model.run_simplified_model(years=30, vaccination_scenario="no_vaccination")
            vacc_results = model.run_simplified_model(years=30, vaccination_scenario="annual_vaccination")
            
            # Store results in session state
            st.session_state.no_vacc_results = no_vacc_results
            st.session_state.vacc_results = vacc_results
            
        st.success("✅ Analysis completed successfully!")
    
    # Display results if available
    if hasattr(st.session_state, 'no_vacc_results') and hasattr(st.session_state, 'vacc_results'):
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Summary Table", "📊 Epidemiological Charts", "📈 Economic Analysis"])
        
        with tab1:
            create_program_summary_table(
                st.session_state.no_vacc_results, 
                st.session_state.vacc_results, 
                st.session_state.params
            )
        
        with tab2:
            create_comparison_plots(
                st.session_state.no_vacc_results, 
                st.session_state.vacc_results
            )
            
            # Additional metrics
            st.subheader("💡 Key Insights")
            
            # Calculate key metrics
            year_10_no_vacc = st.session_state.no_vacc_results.iloc[9]  # Year 10 (0-indexed)
            year_10_vacc = st.session_state.vacc_results.iloc[9]
            
            deaths_averted = year_10_no_vacc['Human_deaths_cumulative'] - year_10_vacc['Human_deaths_cumulative']
            cases_averted = year_10_no_vacc['Canine_rabies_cumulative'] - year_10_vacc['Canine_rabies_cumulative']
            additional_cost = year_10_vacc['Total_cost_cumulative'] - year_10_no_vacc['Total_cost_cumulative']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Human Deaths Averted (10 yrs)", f"{deaths_averted:,.0f}")
            with col2:
                st.metric("Canine Cases Averted (10 yrs)", f"{cases_averted:,.0f}")
            with col3:
                cost_per_death = additional_cost / deaths_averted if deaths_averted > 0 else 0
                st.metric("Cost per Death Averted", f"${cost_per_death:,.0f}")
            with col4:
                cost_per_daly = cost_per_death / st.session_state.params.YLL if cost_per_death > 0 else 0
                st.metric("Cost per DALY Averted", f"${cost_per_daly:,.0f}")
        
        with tab3:
            st.subheader("💰 Economic Impact Analysis")
            
            # Cost breakdown chart
            years = st.session_state.vacc_results['Year']
            
            fig_cost = go.Figure()
            
            # Add cost components
            fig_cost.add_trace(go.Scatter(
                x=years,
                y=st.session_state.vacc_results['Vaccination_cost_cumulative'],
                mode='lines',
                name='Vaccination Costs',
                fill='tonexty',
                stackgroup='one'
            ))
            
            fig_cost.add_trace(go.Scatter(
                x=years,
                y=st.session_state.vacc_results['PEP_cost_cumulative'],
                mode='lines',
                name='PEP Costs',
                fill='tonexty',
                stackgroup='one'
            ))
            
            fig_cost.add_trace(go.Scatter(
                x=years,
                y=st.session_state.vacc_results['Suspect_cost_cumulative'],
                mode='lines',
                name='Suspect Exposure Costs',
                fill='tonexty',
                stackgroup='one'
            ))
            
            fig_cost.update_layout(
                title="Cumulative Program Costs Over Time",
                xaxis_title="Year",
                yaxis_title="Cumulative Cost ($)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_cost, use_container_width=True)
            
            # Cost-effectiveness over time
            cost_effectiveness_data = []
            for i, year in enumerate(years):
                if i < len(st.session_state.no_vacc_results):
                    deaths_averted_year = (st.session_state.no_vacc_results.iloc[i]['Human_deaths_cumulative'] - 
                                         st.session_state.vacc_results.iloc[i]['Human_deaths_cumulative'])
                    additional_cost_year = (st.session_state.vacc_results.iloc[i]['Total_cost_cumulative'] - 
                                          st.session_state.no_vacc_results.iloc[i]['Total_cost_cumulative'])
                    
                    if deaths_averted_year > 0:
                        cost_per_death_year = additional_cost_year / deaths_averted_year
                        cost_per_daly_year = cost_per_death_year / st.session_state.params.YLL
                    else:
                        cost_per_death_year = 0
                        cost_per_daly_year = 0
                    
                    cost_effectiveness_data.append({
                        'Year': year,
                        'Cost_per_Death': cost_per_death_year,
                        'Cost_per_DALY': cost_per_daly_year
                    })
            
            df_cost_eff = pd.DataFrame(cost_effectiveness_data)
            
            # Cost-effectiveness chart
            fig_eff = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_eff.add_trace(
                go.Scatter(x=df_cost_eff['Year'], y=df_cost_eff['Cost_per_Death'],
                          name="Cost per Death Averted", line=dict(color='blue')),
                secondary_y=False,
            )
            
            fig_eff.add_trace(
                go.Scatter(x=df_cost_eff['Year'], y=df_cost_eff['Cost_per_DALY'],
                          name="Cost per DALY Averted", line=dict(color='red')),
                secondary_y=True,
            )
            
            fig_eff.update_xaxes(title_text="Year")
            fig_eff.update_yaxes(title_text="Cost per Death Averted ($)", secondary_y=False)
            fig_eff.update_yaxes(title_text="Cost per DALY Averted ($)", secondary_y=True)
            
            fig_eff.update_layout(title_text="Cost-Effectiveness Over Time")
            
            st.plotly_chart(fig_eff, use_container_width=True)
    
    else:
        st.info("👆 Adjust parameters in the sidebar and click 'Run Analysis' to see results.")
        
        # Show example with default parameters
        st.subheader("📖 About This Analysis")
        
        st.markdown("""
        This application models the economic impact of rabies vaccination programs by comparing:
        
        **No Vaccination Scenario:**
        - Minimal vaccination (5% coverage)
        - Lower PEP coverage (25%)
        - Higher disease transmission
        
        **Annual Vaccination Scenario:**
        - High vaccination coverage (70%)
        - Higher PEP coverage (50%)
        - Reduced disease transmission
        
        **Key Outputs:**
        1. **Program Summary Table**: Comprehensive comparison of both scenarios
        2. **Epidemiological Charts**: Visual representation of disease impact
        3. **Economic Analysis**: Cost-effectiveness and cost breakdown
        
        **Parameters you can adjust:**
        - Geographic scope (area, population)
        - Disease characteristics (R0, transmission rates)
        - Economic factors (vaccination costs, PEP costs)
        """)


if __name__ == "__main__":
    main()