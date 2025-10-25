import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time as tm
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Rabies Economic Analysis", 
    page_icon="🐕", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #1e3a8a;
        color: white;
        border: 1px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .stMetric label {
        color: #e0e7ff !important;
        font-weight: 600;
    }
    .stMetric [data-testid="metric-container"] > div {
        color: white !important;
    }
    .stMetric [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: bold;
    }
    .stMetric [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #86efac !important;
    }
    .summary-table {
        font-size: 0.9rem;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the model parameters and coverage data"""
    try:
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        coverage_data = pd.read_csv(os.path.join(project_root, "data/coverage_data.csv"))
        model_parameters = pd.read_excel(os.path.join(project_root, "data/model_parameters.xlsx"))
        
        return coverage_data, model_parameters
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        st.stop()

def create_coverage_data_editor(coverage_data):
    """Create interactive coverage data editor with phased approach"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Coverage Data Configuration")
    
    # Option to use default data or create custom
    use_custom_data = st.sidebar.checkbox("Use Custom Coverage Data", value=False)
    
    if use_custom_data:
        st.sidebar.info("💡 Configure coverage rates by vaccination program phase")
        
        # Create tabs for different coverage types
        cov_tab1, cov_tab2 = st.sidebar.tabs(["🐕 Vaccination", "💉 PEP"])
        
        with cov_tab1:
            st.markdown("**Vaccination Coverage by Phase**")
            
            # Mass Vaccination Phases
            st.markdown("*Mass Vax: Phase I (Years 1-3):*")
            vacc_no_annual_phase1 = st.slider("No Annual Vaccination (Phase I)", 0.0, 1.0, 0.10, 0.01, key="vacc_no_p1")
            vacc_annual_phase1 = st.slider("Annual Vaccination (Phase I)", 0.0, 1.0, 0.30, 0.01, key="vacc_ann_p1")
            
            st.markdown("*Mass Vax: Phase II (Years 4-6):*")
            vacc_no_annual_phase2 = st.slider("No Annual Vaccination (Phase II)", 0.0, 1.0, 0.10, 0.01, key="vacc_no_p2")
            vacc_annual_phase2 = st.slider("Annual Vaccination (Phase II)", 0.0, 1.0, 0.50, 0.01, key="vacc_ann_p2")
            
            st.markdown("*Mass Vax: Phase III (Years 7-13):*")
            vacc_no_annual_phase3 = st.slider("No Annual Vaccination (Phase III)", 0.0, 1.0, 0.10, 0.01, key="vacc_no_p3")
            vacc_annual_phase3 = st.slider("Annual Vaccination (Phase III)", 0.0, 1.0, 0.70, 0.01, key="vacc_ann_p3")
            
            st.markdown("*Mass Vax: Phase IV (Years 14-30):*")
            vacc_no_annual_phase4 = st.slider("No Annual Vaccination (Phase IV)", 0.0, 1.0, 0.10, 0.01, key="vacc_no_p4")
            vacc_annual_phase4 = st.slider("Annual Vaccination (Phase IV - Maintenance)", 0.0, 1.0, 0.80, 0.01, key="vacc_ann_p4")
        
        with cov_tab2:
            st.markdown("**PEP Coverage by Phase**")
            
            st.markdown("*Phase I (Years 1-3):*")
            pep_no_annual_phase1 = st.slider("PEP No Annual (Phase I)", 0.0, 1.0, 0.25, 0.01, key="pep_no_p1")
            pep_annual_phase1 = st.slider("PEP Annual (Phase I)", 0.0, 1.0, 0.40, 0.01, key="pep_ann_p1")
            
            st.markdown("*Phase II (Years 4-6):*")
            pep_no_annual_phase2 = st.slider("PEP No Annual (Phase II)", 0.0, 1.0, 0.25, 0.01, key="pep_no_p2")
            pep_annual_phase2 = st.slider("PEP Annual (Phase II)", 0.0, 1.0, 0.50, 0.01, key="pep_ann_p2")
            
            st.markdown("*Phase III (Years 7-13):*")
            pep_no_annual_phase3 = st.slider("PEP No Annual (Phase III)", 0.0, 1.0, 0.25, 0.01, key="pep_no_p3")
            pep_annual_phase3 = st.slider("PEP Annual (Phase III)", 0.0, 1.0, 0.65, 0.01, key="pep_ann_p3")
            
            st.markdown("*Phase IV (Years 14-30):*")
            pep_no_annual_phase4 = st.slider("PEP No Annual (Phase IV)", 0.0, 1.0, 0.25, 0.01, key="pep_no_p4")
            pep_annual_phase4 = st.slider("PEP Annual (Phase IV - Maintenance)", 0.0, 1.0, 0.75, 0.01, key="pep_ann_p4")
        
        # Generate custom coverage data based on phases
        custom_coverage_data = []
        
        for year in range(1, 31):
            # Determine phase based on year
            if 1 <= year <= 3:  # Phase I
                no_annual_vacc = vacc_no_annual_phase1
                annual_vacc = vacc_annual_phase1
                no_annual_pep = pep_no_annual_phase1
                annual_pep = pep_annual_phase1
            elif 4 <= year <= 6:  # Phase II
                no_annual_vacc = vacc_no_annual_phase2
                annual_vacc = vacc_annual_phase2
                no_annual_pep = pep_no_annual_phase2
                annual_pep = pep_annual_phase2
            elif 7 <= year <= 13:  # Phase III
                no_annual_vacc = vacc_no_annual_phase3
                annual_vacc = vacc_annual_phase3
                no_annual_pep = pep_no_annual_phase3
                annual_pep = pep_annual_phase3
            else:  # Phase IV (Years 14-30)
                no_annual_vacc = vacc_no_annual_phase4
                annual_vacc = vacc_annual_phase4
                no_annual_pep = pep_no_annual_phase4
                annual_pep = pep_annual_phase4
            
            custom_coverage_data.append({
                'year': year,
                'no_annual_vaccination_coverage': no_annual_vacc,
                'annual_vaccination_coverage': annual_vacc,
                'no_annual_p_PEP_Exposed': no_annual_pep,
                'annual_p_PEP_Exposed': annual_pep
            })
        
        # Show preview of generated data
        st.sidebar.markdown("---")
        if st.sidebar.button("📊 Preview Coverage Data"):
            custom_df = pd.DataFrame(custom_coverage_data)
            
            # Create preview in main area
            st.header("📊 Coverage Data Preview")
            
            # Phase information
            st.info("""
            **Vaccination Program Phases:**
            - **Phase I (Years 1-3):** Initial mass vaccination campaign
            - **Phase II (Years 4-6):** Intensified vaccination efforts  
            - **Phase III (Years 7-13):** High coverage maintenance
            - **Phase IV (Years 14-30):** Post-elimination maintenance
            """)
            
            # Create preview chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Vaccination coverage plot
            ax1.plot(custom_df['year'], custom_df['no_annual_vaccination_coverage'], 
                     label='No Annual Vaccination', color='red', linewidth=2.5, marker='o', markersize=4)
            ax1.plot(custom_df['year'], custom_df['annual_vaccination_coverage'], 
                     label='Annual Vaccination', color='green', linewidth=2.5, marker='s', markersize=4)
            
            # Add phase boundaries
            ax1.axvline(x=3.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax1.axvline(x=6.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax1.axvline(x=13.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add phase labels
            ax1.text(2, 0.9, 'Phase I', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax1.text(5, 0.9, 'Phase II', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            ax1.text(10, 0.9, 'Phase III', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
            ax1.text(22, 0.9, 'Phase IV', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            
            ax1.set_title('Vaccination Coverage by Program Phase', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Coverage Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(1, 30)
            ax1.set_ylim(0, 1)
            
            # PEP coverage plot
            ax2.plot(custom_df['year'], custom_df['no_annual_p_PEP_Exposed'], 
                     label='No Annual Vaccination', color='red', linewidth=2.5, marker='o', markersize=4)
            ax2.plot(custom_df['year'], custom_df['annual_p_PEP_Exposed'], 
                     label='Annual Vaccination', color='green', linewidth=2.5, marker='s', markersize=4)
            
            # Add phase boundaries
            ax2.axvline(x=3.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax2.axvline(x=6.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax2.axvline(x=13.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add phase labels
            ax2.text(2, 0.9, 'Phase I', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax2.text(5, 0.9, 'Phase II', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            ax2.text(10, 0.9, 'Phase III', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
            ax2.text(22, 0.9, 'Phase IV', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            
            ax2.set_title('PEP Coverage by Program Phase', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Coverage Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(1, 30)
            ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show data table with phase indicators
            st.subheader("📋 Generated Coverage Data by Phase")
            
            # Add phase column for better visualization
            display_df = custom_df.copy()
            display_df['Phase'] = display_df['year'].apply(lambda x: 
                'Phase I' if 1 <= x <= 3 else
                'Phase II' if 4 <= x <= 6 else
                'Phase III' if 7 <= x <= 13 else
                'Phase IV'
            )
            
            # Reorder columns
            display_df = display_df[['year', 'Phase', 'no_annual_vaccination_coverage', 'annual_vaccination_coverage', 'no_annual_p_PEP_Exposed', 'annual_p_PEP_Exposed']]
            
            st.dataframe(
                display_df,
                column_config={
                    "year": "Year",
                    "Phase": "Program Phase",
                    "no_annual_vaccination_coverage": st.column_config.NumberColumn("No Annual Vacc", format="%.2f"),
                    "annual_vaccination_coverage": st.column_config.NumberColumn("Annual Vacc", format="%.2f"),
                    "no_annual_p_PEP_Exposed": st.column_config.NumberColumn("No Annual PEP", format="%.2f"),
                    "annual_p_PEP_Exposed": st.column_config.NumberColumn("Annual PEP", format="%.2f")
                },
                use_container_width=True,
                hide_index=True
            )
        
        return pd.DataFrame(custom_coverage_data)
    
    else:
        # Use original data from CSV
        st.sidebar.info("""
        📁 **Using default coverage data**
        
        
        • **Status Quo Vaccination**: 5% (minimal/opportunistic)
        
        • **Annual Vaccination Program**: 20% (basic program)
        
        • **PEP Coverage**: 25% (Status Quo) / 50% (Annual Program)
        
        """)
        return coverage_data

def get_vaccination_coverage(year, scenario="annual_vaccination", coverage_data=None):
    """Get vaccination coverage for a specific year and scenario from CSV data"""
    if coverage_data is None:
        return 0.10 if scenario == "no_annual_vaccination" else 0.50
    
    # Cap year at maximum available in data
    year = min(year, coverage_data["year"].max())
    year = max(year, coverage_data["year"].min())

    # Get the row for this year
    coverage_row = coverage_data[coverage_data["year"] == year]

    if len(coverage_row) > 0:
        if scenario == "no_annual_vaccination":
            return coverage_row["no_annual_vaccination_coverage"].iloc[0]
        else:  # annual_vaccination scenario
            return coverage_row["annual_vaccination_coverage"].iloc[0]
    else:
        # Fallback values if year not found
        return 0.10 if scenario == "no_annual_vaccination" else 0.50

def get_pep_coverage(year, scenario="annual_vaccination", coverage_data=None):
    """Get PEP coverage for a specific year and scenario from CSV data"""
    if coverage_data is None:
        return 0.25
    
    # Cap year at maximum available in data
    year = min(year, coverage_data["year"].max())
    year = max(year, coverage_data["year"].min())

    # Get the row for this year
    coverage_row = coverage_data[coverage_data["year"] == year]

    if len(coverage_row) > 0:
        if scenario == "no_annual_vaccination":
            return coverage_row["no_annual_p_PEP_Exposed"].iloc[0]
        else:  # annual_vaccination scenario
            return coverage_row["annual_p_PEP_Exposed"].iloc[0]
    else:
        # Fallback values if year not found
        return 0.25

@st.cache_data
def extract_model_parameters(model_parameters):
    """Extract all model parameters from the Excel file"""
    params = {}
    
    # Extract all parameters with error handling
    param_list = [
        'Km2_of_program_area', 'Human_population', 'Human_birth', 'Human_life_expectancy',
        'Humans_per_free_roaming_dog', 'Free_roaming_dog_population', 'Free_roaming_dogs_per_km2',
        'Dog_birth_rate_per_1000_dogs', 'Dog_life_expectancy', 'Annual_dog_bite_risk',
        'Probability_of_rabies_in_biting_dogs', 'Probability_of_human_developing_rabies',
        'Dog_Human_transmission_rate', 'R0_dog_to_dog', 'inflation_factor_for_the_suspect_exposure',
        'post_elimination_pep_reduction'
    ]
    
    for param in param_list:
        try:
            params[param] = model_parameters.query(f"Parameters == '{param}'")["Values"].iloc[0]
        except IndexError:
            st.warning(f"Parameter {param} not found in model file")
            params[param] = 0
    
    # Handle cost parameters with fallbacks
    cost_params = {
        'quarantined_animal_prob': 0.0008,
        'quarantined_animal_cost': 140.00,
        'lab_test_prob': 0.011333333,
        'lab_test_cost': 26.49,
        'bite_investigation_prob': 0.466666667,
        'bite_investigation_cost': 3.25,
        'vaccination_cost_per_dog': 2.45,
        'pep_and_other_costs': 17.40,
        'pep_prob_no_campaign': 0.25,
        'pep_prob_annual_campaign': 0.5,
        'YLL': 26.32
    }
    
    for param, default_value in cost_params.items():
        try:
            params[param] = model_parameters.query(f"Parameters == '{param}'")["Values"].iloc[0]
        except IndexError:
            params[param] = default_value
    
    return params

def create_parameter_inputs(model_parameters):
    """Create interactive parameter inputs in the sidebar based on parameter categories"""
    st.sidebar.header("🎛️ Model Parameters")
    
    # Create tabs for different parameter categories
    param_tab1, param_tab2, param_tab3 = st.sidebar.tabs(["📊 Variables", "⚙️ Constants", "🔢 Calculated"])
    
    with param_tab1:
        st.subheader("Key Variables")
        st.info("💡 These are the primary parameters you can modify for scenario analysis")
        
        # Load default values from Excel
        defaults = extract_model_parameters(model_parameters)
        
        # VARIABLE PARAMETERS (from sheet: variable type)
        st.markdown("**📍 Geographic & Population Variables**")
        
        # Add country preset dropdown
        country_presets = {
            "Custom (Manual Entry)": {
                "area": float(defaults['Km2_of_program_area']),
                "population": int(defaults['Human_population']),
                "hdr": float(defaults['Humans_per_free_roaming_dog']),
                "adjustment": 1.05
            },
            "Kenya": {
                "area": 54560.0,
                "population": 41136000,
                "hdr": 13.0,
                "adjustment": 1.05
            },
            "Cameroon": {
                "area": 12766.0,
                "population": 841000,
                "hdr": 16.0,
                "adjustment": 1.05
            },
            "Malawi": {
                "area": 17960.0,
                "population": 15150000,
                "hdr": 15.0,
                "adjustment": 1.05
            }
        }
        
        selected_country = st.selectbox(
            "Select Program Location",
            options=list(country_presets.keys()),
            index=0,  # Default to "Custom"
            help="Choose a preset country or select 'Custom' to enter your own values"
        )
        
        # Show country info if not custom
        if selected_country != "Custom (Manual Entry)":
            st.info(f"📍 **{selected_country}** preset selected - values auto-filled below")
        
        # Use preset values or allow manual entry
        if selected_country == "Custom (Manual Entry)":
            # Manual entry fields
            Km2_of_program_area = st.number_input(
                "Program Area (km²)",
                min_value=1000.0,
                max_value=100000.0,
                value=float(defaults['Km2_of_program_area']),
                step=1000.0,
                help="Square kilometers (km2) of program area"
            )
            
            Human_population = st.number_input(
                "Human Population",
                min_value=100000,
                max_value=100000000,
                value=int(defaults['Human_population']),
                step=100000,
                help="Human population in the program area"
            )
            
            Humans_per_free_roaming_dog = st.number_input(
                "Humans per Free-Roaming Dog (HDR)",
                min_value=1.0,
                max_value=50.0,
                value=float(defaults['Humans_per_free_roaming_dog']),
                step=1.0,
                help="Number of humans per FREE ROAMING dog (Human-to-Dog Ratio)"
            )
        else:
            # Use preset values but allow override
            preset = country_presets[selected_country]
            
            Km2_of_program_area = st.number_input(
                "Program Area (km²)",
                min_value=1000.0,
                max_value=100000.0,
                value=preset["area"],
                step=1000.0,
                help=f"Square kilometers for {selected_country} (you can modify this value)"
            )
            
            Human_population = st.number_input(
                "Human Population",
                min_value=100000,
                max_value=100000000,
                value=preset["population"],
                step=100000,
                help=f"Human population for {selected_country} (you can modify this value)"
            )
            
            Humans_per_free_roaming_dog = st.number_input(
                "Humans per Free-Roaming Dog (HDR)",
                min_value=1.0,
                max_value=50.0,
                value=preset["hdr"],
                step=1.0,
                help=f"HDR for {selected_country} (you can modify this value)"
            )
        
        # DOG DENSITY ADJUSTMENT FACTOR - separate section
        st.markdown("**🎯 Model Adjustment Parameters**")
        
        # Use preset or manual for adjustment factor
        if selected_country == "Custom (Manual Entry)":
            default_adjustment = 1.05
        else:
            default_adjustment = country_presets[selected_country]["adjustment"]
        
        dog_density_adjustment_factor = st.selectbox(
            "Dog Density Adjustment Factor",
            options=[0.95, 1.0, 1.05],
            index=[0.95, 1.0, 1.05].index(default_adjustment),
            help="Adjustment factor for dog population density and carrying capacity (K). Controls dog population dynamics. 0.95=Lower density, 1.0=Neutral, 1.05=Higher density"
        )
    
    with param_tab2:
        st.subheader("Model Constants")
        st.warning("⚠️ Advanced parameters. Modify only if you understand their epidemiological significance.")
        
        # CONSTANT PARAMETERS (from sheet: constant type)
        
        st.markdown("**👥 Demographic Constants**")
        Human_birth = st.number_input(
            "Human Birth Rate (per 1,000/year)",
            min_value=5.0,
            max_value=50.0,
            value=float(defaults['Human_birth']),
            step=1.0,
            help="Human birth rate per 1,000 population (suggested 17)"
        )
        
        Human_life_expectancy = st.number_input(
            "Human Life Expectancy (years)",
            min_value=40.0,
            max_value=90.0,
            value=float(defaults['Human_life_expectancy']),
            step=1.0,
            help="Human life expectancy in years"
        )
        
        Dog_birth_rate_per_1000_dogs = st.number_input(
            "Dog Birth Rate (per 1,000 dogs/year)",
            min_value=100.0,
            max_value=1000.0,
            value=float(defaults['Dog_birth_rate_per_1000_dogs']),
            step=10.0,
            help="Dog birth rate per 1,000 dogs (suggested 750)"
        )
        
        Dog_life_expectancy = st.number_input(
            "Dog Life Expectancy (years)",
            min_value=1.0,
            max_value=10.0,
            value=float(defaults['Dog_life_expectancy']),
            step=0.1,
            help="Dog life expectancy in years"
        )
        
        st.markdown("**🦠 Transmission Risk Constants**")
        Annual_dog_bite_risk = st.number_input(
            "Annual Dog Bite Risk",
            min_value=0.001,
            max_value=0.1,
            value=float(defaults['Annual_dog_bite_risk']),
            step=0.001,
            format="%.4f",
            help="Annual dog bite risk (suggested 1% - 3%)"
        )
        
        Probability_of_rabies_in_biting_dogs = st.number_input(
            "Probability of Rabies in Biting Dogs",
            min_value=0.0001,
            max_value=0.5,
            value=float(defaults['Probability_of_rabies_in_biting_dogs']),
            step=0.001,
            format="%.4f",
            help="Probability of rabies in biting dogs (suggested 0.1% - 5%)"
        )
        
        Probability_of_human_developing_rabies = st.number_input(
            "Probability of Human Developing Rabies",
            min_value=0.01,
            max_value=1.0,
            value=float(defaults['Probability_of_human_developing_rabies']),
            step=0.01,
            format="%.3f",
            help="Probability of human developing rabies (suggested 17%)"
        )
        
        Dog_Human_transmission_rate = st.number_input(
            "Dog-Human Transmission Rate",
            min_value=0.000001,
            max_value=0.001,
            value=float(defaults['Dog_Human_transmission_rate']),
            step=0.000001,
            format="%.8f",
            help="Dog-Human transmission rate (suggested 0.000034)"
        )
        
        st.markdown("**📈 Exposure & Economic Constants**")
        inflation_factor_for_the_suspect_exposure = st.number_input(
            "Suspect Exposure Inflation Factor",
            min_value=1.0,
            max_value=20.0,
            value=float(defaults['inflation_factor_for_the_suspect_exposure']),
            step=1.0,
            help="Inflation factor for the suspect exposure (>=1)"
        )
        
        post_elimination_pep_reduction = st.number_input(
            "Post-Elimination PEP Reduction",
            min_value=0.0,
            max_value=0.9,
            value=float(defaults['post_elimination_pep_reduction']),
            step=0.05,
            format="%.2f",
            help="Post-Elimination PEP Reduction (%)"
        )
        
        st.markdown("**💰 Cost Constants**")
        vaccination_cost_per_dog = st.number_input(
            "Vaccination Cost per Dog ($)",
            min_value=0.5,
            max_value=20.0,
            value=float(defaults['vaccination_cost_per_dog']),
            step=0.25,
            help="Average cost per dog vaccinated"
        )
        
        pep_and_other_costs = st.number_input(
            "PEP and Other Costs ($)",
            min_value=5.0,
            max_value=100.0,
            value=float(defaults['pep_and_other_costs']),
            step=1.0,
            help="PEP cost & Other Costs per exposure"
        )
        
        pep_prob_no_campaign = st.number_input(
            "PEP Probability (No Campaign)",
            min_value=0.1,
            max_value=1.0,
            value=float(defaults['pep_prob_no_campaign']),
            step=0.05,
            format="%.2f",
            help="Probability of receiving PEP, post-exposure (no Vaccination program)"
        )
        
        pep_prob_annual_campaign = st.number_input(
            "PEP Probability (Annual Campaign)",
            min_value=0.1,
            max_value=1.0,
            value=float(defaults['pep_prob_annual_campaign']),
            step=0.05,
            format="%.2f",
            help="Probability of receiving PEP, post-exposure (with Vaccination program)"
        )
        
        YLL = st.number_input(
            "Years of Life Lost (YLL)",
            min_value=10.0,
            max_value=50.0,
            value=float(defaults['YLL']),
            step=1.0,
            help="Years of Life Lost (YLL) per death"
        )
        
        # Additional cost parameters (if they exist in the model)
        try:
            quarantined_animal_prob = st.number_input(
                "Quarantined Animal Probability",
                min_value=0.0001,
                max_value=0.01,
                value=float(defaults.get('quarantined_animal_prob', 0.0008)),
                step=0.0001,
                format="%.5f"
            )
            
            quarantined_animal_cost = st.number_input(
                "Quarantined Animal Cost ($)",
                min_value=50.0,
                max_value=500.0,
                value=float(defaults.get('quarantined_animal_cost', 140.0)),
                step=10.0
            )
            
            lab_test_prob = st.number_input(
                "Lab Test Probability",
                min_value=0.001,
                max_value=0.1,
                value=float(defaults.get('lab_test_prob', 0.011333333)),
                step=0.001,
                format="%.6f"
            )
            
            lab_test_cost = st.number_input(
                "Lab Test Cost ($)",
                min_value=10.0,
                max_value=100.0,
                value=float(defaults.get('lab_test_cost', 26.49)),
                step=1.0
            )
            
            bite_investigation_prob = st.number_input(
                "Bite Investigation Probability",
                min_value=0.1,
                max_value=1.0,
                value=float(defaults.get('bite_investigation_prob', 0.466666667)),
                step=0.01,
                format="%.3f"
            )
            
            bite_investigation_cost = st.number_input(
                "Bite Investigation Cost ($)",
                min_value=1.0,
                max_value=20.0,
                value=float(defaults.get('bite_investigation_cost', 3.25)),
                step=0.25
            )
        except:
            # Use defaults if not in Excel file
            quarantined_animal_prob = 0.0008
            quarantined_animal_cost = 140.0
            lab_test_prob = 0.011333333
            lab_test_cost = 26.49
            bite_investigation_prob = 0.466666667
            bite_investigation_cost = 3.25
    
    with param_tab3:
        st.subheader("Calculated Values")
        st.info("📊 These values are automatically calculated from your input parameters")
        
        # CALCULATED PARAMETERS (derived from inputs)
        
        # Calculate Free_roaming_dog_population from HDR (as per Excel: Human_population / Humans_per_free_roaming_dog)
        Free_roaming_dog_population = Human_population / Humans_per_free_roaming_dog
        
        # Calculate geographic densities
        Humans_per_km2 = Human_population / Km2_of_program_area
        Free_roaming_dogs_per_km2 = Free_roaming_dog_population / Km2_of_program_area
        
        # Calculate R0_dog_to_dog (this would need the actual formula from your model)
        # For now, using the Excel value as placeholder - you'll need to implement the actual calculation
        R0_dog_to_dog = 1.38053211  # This should be calculated based on your epidemiological model
        
        st.markdown("**📊 Population Metrics**")
        st.metric("Humans per km²", f"{Humans_per_km2:,.1f}")
        st.metric("Free-roaming Dog Population", f"{Free_roaming_dog_population:,.0f}")
        st.metric("Free-roaming Dogs per km²", f"{Free_roaming_dogs_per_km2:.2f}")
        
        st.markdown("**🦠 Disease Metrics**")
        st.metric("R₀ (Dog-to-Dog)", f"{R0_dog_to_dog:.8f}")
        
        # Show K calculation with dog density adjustment factor
        st.markdown("**🔧 Model Parameters**")
        calculated_k = Free_roaming_dogs_per_km2 * (1 + 1 / np.log(Free_roaming_dog_population)) * dog_density_adjustment_factor
        st.metric("Carrying Capacity (K)", f"{calculated_k:.6f}")
        st.caption(f"K = Nd × (1 + 1/ln(Population)) × {dog_density_adjustment_factor}")
        
        st.markdown("**🔗 Transmission Verification**")
        calculated_transmission_rate = Annual_dog_bite_risk * Probability_of_rabies_in_biting_dogs * Probability_of_human_developing_rabies
        st.metric("Calculated Transmission Rate", f"{calculated_transmission_rate:.8f}")
        st.metric("Manual Transmission Rate", f"{Dog_Human_transmission_rate:.8f}")
        
        if abs(calculated_transmission_rate - Dog_Human_transmission_rate) > 0.000001:
            st.warning("⚠️ Manual transmission rate differs from calculated value!")
    
    # Parameter validation section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Parameter Validation")
    
    # Calculate values for validation
    Free_roaming_dog_population = Human_population / Humans_per_free_roaming_dog
    Free_roaming_dogs_per_km2 = Free_roaming_dog_population / Km2_of_program_area
    
    validation_warnings = []
    
    # Basic validation checks
    if Free_roaming_dogs_per_km2 > 100:
        validation_warnings.append("⚠️ Very high dog density (>100/km²)")
    if Free_roaming_dogs_per_km2 < 10:
        validation_warnings.append("⚠️ Very low dog density (<10/km²)")
    if vaccination_cost_per_dog > pep_and_other_costs:
        validation_warnings.append("⚠️ Vaccination cost > PEP cost")
    if Human_population / Km2_of_program_area < 10:
        validation_warnings.append("⚠️ Very low population density")
    if Humans_per_free_roaming_dog < 5:
        validation_warnings.append("⚠️ Very high dog-to-human ratio")
    if Humans_per_free_roaming_dog > 30:
        validation_warnings.append("⚠️ Very low dog-to-human ratio")
    
    # Add Dog Density Adjustment Factor validation
    if dog_density_adjustment_factor != 1.05:
        if dog_density_adjustment_factor == 0.95:
            validation_warnings.append("ℹ️ Using lower dog density (K×0.95)")
        elif dog_density_adjustment_factor == 1.0:
            validation_warnings.append("ℹ️ Using neutral dog density (K×1.0)")
    
    if validation_warnings:
        for warning in validation_warnings:
            if warning.startswith("ℹ️"):
                st.sidebar.info(warning)
            else:
                st.sidebar.warning(warning)
    else:
        st.sidebar.success("✅ Parameters look reasonable")
    
    # Reset button
    if st.sidebar.button("🔄 Reset to Excel Defaults", type="secondary"):
        st.rerun()
    
    # Return the complete parameters dictionary matching the extract_model_parameters format
    return {
        # Geographic and population (variables)
        'Km2_of_program_area': Km2_of_program_area,
        'Human_population': Human_population,
        'Humans_per_free_roaming_dog': Humans_per_free_roaming_dog,
        
        # Model adjustment parameters
        'dog_density_adjustment_factor': dog_density_adjustment_factor,
        
        # Calculated values
        'Free_roaming_dog_population': Free_roaming_dog_population,
        'Free_roaming_dogs_per_km2': Free_roaming_dogs_per_km2,
        'Humans_per_km2': Human_population / Km2_of_program_area,
        'R0_dog_to_dog': R0_dog_to_dog,
        
        # Demographic constants
        'Human_birth': Human_birth,
        'Human_life_expectancy': Human_life_expectancy,
        'Dog_birth_rate_per_1000_dogs': Dog_birth_rate_per_1000_dogs,
        'Dog_life_expectancy': Dog_life_expectancy,
        
        # Disease transmission constants
        'Annual_dog_bite_risk': Annual_dog_bite_risk,
        'Probability_of_rabies_in_biting_dogs': Probability_of_rabies_in_biting_dogs,
        'Probability_of_human_developing_rabies': Probability_of_human_developing_rabies,
        'Dog_Human_transmission_rate': Dog_Human_transmission_rate,
        
        # Economic parameters
        'inflation_factor_for_the_suspect_exposure': inflation_factor_for_the_suspect_exposure,
        'post_elimination_pep_reduction': post_elimination_pep_reduction,
        'vaccination_cost_per_dog': vaccination_cost_per_dog,
        'pep_and_other_costs': pep_and_other_costs,
        'pep_prob_no_campaign': pep_prob_no_campaign,
        'pep_prob_annual_campaign': pep_prob_annual_campaign,
        'YLL': YLL,
        
        # Additional cost parameters
        'quarantined_animal_prob': quarantined_animal_prob,
        'quarantined_animal_cost': quarantined_animal_cost,
        'lab_test_prob': lab_test_prob,
        'lab_test_cost': lab_test_cost,
        'bite_investigation_prob': bite_investigation_prob,
        'bite_investigation_cost': bite_investigation_cost
    }

def run_initial_simulation(params):
    """Run the initial equilibrium simulation"""
    # Extract parameters
    Km2_of_program_area = params['Km2_of_program_area']
    Free_roaming_dogs_per_km2 = params['Free_roaming_dogs_per_km2']
    Free_roaming_dog_population = params['Free_roaming_dog_population']
    Dog_birth_rate_per_1000_dogs = params['Dog_birth_rate_per_1000_dogs']
    Dog_life_expectancy = params['Dog_life_expectancy']
    R0_dog_to_dog = params['R0_dog_to_dog']
    Human_population = params['Human_population']
    Human_birth = params['Human_birth']
    Human_life_expectancy = params['Human_life_expectancy']
    dog_density_adjustment_factor = params['dog_density_adjustment_factor']  # NEW: Extract Dog Density Adjustment Factor
    
    # Calculate derived parameters
    Humans_per_km2 = Human_population / Km2_of_program_area
    
    # Initial conditions
    Sd = (1-((1/52)/Km2_of_program_area))*Free_roaming_dogs_per_km2
    Ed = 0
    Id = Free_roaming_dogs_per_km2*((1/52)/Km2_of_program_area)
    Rd = 0
    Nd = Free_roaming_dogs_per_km2
    
    # Model parameters
    b_d = Dog_birth_rate_per_1000_dogs / 52 / 1000
    lambda_d1 = 0
    lambda_d2 = 0.0096
    i_d = 6.27
    sigma_d = 1 / i_d
    r_d = 0.45
    m_d = (1 / Dog_life_expectancy) / 52
    mu_d = (1 / 10) * 7
    
    beta_d = (R0_dog_to_dog * (((sigma_d) + m_d) * (mu_d + m_d)) / (sigma_d * r_d * Sd))
    
    # UPDATED: Use Dog Density Adjustment Factor
    K = Nd * (1 + 1 / np.log(Free_roaming_dog_population)) * dog_density_adjustment_factor
    
    v_d = 0.95
    Vaccination_coverage_per_campaign = 0.05
    alpha_d1 = 0.0163
    alpha_d2 = 0
    
    b_h = (Human_birth / 52) / 1000
    lambda_h = 0
    m_h = (1 / Human_life_expectancy) / 52
    v_h = 0.93
    alpha_h = 0
    beta_dh = 0.0000510
    P10 = 0.50
    mu_h = (1 / 10) * 7
    gamma_d = (b_d - m_d) / K
    
    # Initialize results
    results = {"time": [0], "Sd": [Sd], "Ed": [Ed], "Id": [Id], "Rd": [Rd], "Nd": [Nd]}
    
    # Run simulation
    for time in range(1, 10001):
        lambda_d = lambda_d1 if time < 27 else lambda_d2
        week = 52 if time % 52 == 0 else time % 52
        alpha_d = alpha_d1 if (abs(week - 22) + abs(31 - week)) <= 10 else alpha_d2
        
        percent_immunized = Rd / Nd
        target_status = 1 if percent_immunized < Vaccination_coverage_per_campaign else 0
        
        # Update compartments
        Sd_new = (Sd + (b_d * Nd) + (lambda_d * Rd) + (sigma_d * (1 - r_d) * Ed) 
                 - (m_d * Sd) - (beta_d * Sd * Id) - (gamma_d * Nd * Sd) 
                 - (target_status * (v_d * alpha_d * Sd)))
        
        Ed_new = (Ed + (beta_d * Sd * Id) - (m_d * Ed) - (gamma_d * Nd * Ed) 
                 - (sigma_d * (1 - r_d) * Ed) - (target_status * (v_d * alpha_d * Ed)) 
                 - (sigma_d * r_d * Ed))
        
        Id_new = Id + (sigma_d * r_d * Ed) - (m_d * Id) - (gamma_d * Nd * Id) - (mu_d * Id)
        
        Rd_new = (Rd + (target_status * (v_d * alpha_d * (Sd + Ed))) 
                 - (m_d * Rd) - (gamma_d * Nd * Rd) - (lambda_d * Rd))
        
        # Update values
        Sd, Ed, Id, Rd = Sd_new, Ed_new, Id_new, Rd_new
        Nd = Sd + Ed + Id + Rd
        
        # Store results
        results["time"].append(time)
        results["Sd"].append(Sd)
        results["Ed"].append(Ed)
        results["Id"].append(Id)
        results["Rd"].append(Rd)
        results["Nd"].append(Nd)
    
    initial_run = pd.DataFrame(results)
    initial_run["week"] = initial_run["time"].apply(lambda x: 52 if x % 52 == 0 else x % 52)
    
    return initial_run

def run_scenario_simulation(initial_run, params, coverage_data, scenario_type="no_annual_vaccination"):
    """Run scenario simulation (with or without vaccination)"""
    # Get initial conditions from equilibrium run
    Sd = initial_run.iloc[-1]["Sd"]
    Ed = initial_run.iloc[-1]["Ed"]
    Id = initial_run.iloc[-1]["Id"]
    C_rd = initial_run.iloc[-1]["Id"]
    Rd = 0
    Nd = Sd + Ed + Id + Rd
    
    # Extract parameters
    Km2_of_program_area = params['Km2_of_program_area']
    Human_population = params['Human_population']
    Human_birth = params['Human_birth']
    Human_life_expectancy = params['Human_life_expectancy']
    Dog_birth_rate_per_1000_dogs = params['Dog_birth_rate_per_1000_dogs']
    Dog_life_expectancy = params['Dog_life_expectancy']
    Free_roaming_dog_population = params['Free_roaming_dog_population']
    dog_density_adjustment_factor = params['dog_density_adjustment_factor']  # NEW: Extract Dog Density Adjustment Factor
    
    Humans_per_km2 = Human_population / Km2_of_program_area
    
    # Human parameters
    Nh = Humans_per_km2
    Sh = Nh
    Eh = 0
    Ih = 0
    Rh = 0
    Dh = 0
    Dd = 0  # Dead dogs due to rabies
    new_expo = Eh
    
    # Model parameters
    b_d = Dog_birth_rate_per_1000_dogs / 52 / 1000
    lambda_d1 = 0
    lambda_d2 = 0.0096
    i_d = 6.27
    sigma_d = 1 / i_d
    r_d = 0.45
    m_d = (1 / Dog_life_expectancy) / 52
    mu_d = (1 / 10) * 7
    
    # Calculate beta_d and K from initial run
    R0_dog_to_dog = params['R0_dog_to_dog']
    beta_d = (R0_dog_to_dog * (((sigma_d) + m_d) * (mu_d + m_d)) / (sigma_d * r_d * initial_run.iloc[0]["Sd"]))
    # UPDATED: Use Dog Density Adjustment Factor
    K = initial_run.iloc[0]["Nd"] * (1 + 1 / np.log(Free_roaming_dog_population)) * dog_density_adjustment_factor
    
    v_d = 0.95
    alpha_d1 = 0.0163
    alpha_d2 = 0
    
    b_h = (Human_birth / 52) / 1000
    lambda_h = 0
    m_h = (1 / Human_life_expectancy) / 52
    v_h = 0.93
    alpha_h = 0
    beta_dh = 0.0000510
    mu_h = (1 / 10) * 7
    gamma_d = (b_d - m_d) / K
    
    # Missing parameters
    p_ExptoNoInf = 0.097
    p_ExptoInf = 0.025
    
    # Initialize results
    results = {
        "time": [0], "week": [0], "Sd": [Sd], "Ed": [Ed], "Id": [Id], "C_rd": [C_rd],
        "Rd": [Rd], "Nd": [Nd], "Sh": [Sh], "Eh": [Eh], "Ih": [Ih], "Dh": [Dh],
        "Dd": [Dd], "Rh": [Rh], "Nh": [Nh], "new_expo": [new_expo]
    }
    
    # Run simulation
    for time in range(1, 2300):
        current_year = (time // 52) + 1
        
        # Get time-varying coverage
        Vaccination_coverage_per_campaign = get_vaccination_coverage(current_year, scenario_type, coverage_data)
        P10_step = get_pep_coverage(current_year, scenario_type, coverage_data)
        
        # Calculate alpha_d1 for annual vaccination scenario
        if scenario_type == "annual_vaccination":
            alpha_d1_current = -(1 / 10) * np.log(1 - Vaccination_coverage_per_campaign)
        else:
            alpha_d1_current = alpha_d1
        
        lambda_d = lambda_d1 if time < 27 else lambda_d2
        week = 52 if time % 52 == 0 else time % 52
        
        # Determine alpha_d based on scenario
        if scenario_type == "no_annual_vaccination":
            if Vaccination_coverage_per_campaign > 0:
                alpha_d = alpha_d1 if (abs(week - 22) + abs(31 - week)) <= 10 else alpha_d2
            else:
                alpha_d = 0
        else:
            alpha_d = alpha_d1_current if (abs(week - 22) + abs(31 - week)) <= 10 else alpha_d2
        
        # Calculate vaccination status for annual vaccination
        if scenario_type == "annual_vaccination":
            percent_immunized = Rd / Nd
            target_status = 1 if percent_immunized < Vaccination_coverage_per_campaign else 0
        else:
            target_status = 1  # Always apply vaccination in no_annual scenario
        
        # Update dog compartments
        if scenario_type == "no_annual_vaccination":
            # Direct vaccination without target_status
            Sd_new = (Sd + (b_d * Nd) + (lambda_d * Rd) + (sigma_d * (1 - r_d) * Ed) 
                     - (m_d * Sd) - (beta_d * Sd * Id) - (gamma_d * Nd * Sd) - (v_d * alpha_d * Sd))
            Ed_new = (Ed + (beta_d * Sd * Id) - (m_d * Ed) - (gamma_d * Nd * Ed) 
                     - (sigma_d * (1 - r_d) * Ed) - (v_d * alpha_d * Ed) - (sigma_d * r_d * Ed))
            Rd_new = (Rd + (v_d * alpha_d * (Sd + Ed)) - (m_d * Rd) - (gamma_d * Nd * Rd) - (lambda_d * Rd))
        else:
            # With target_status
            Sd_new = (Sd + (b_d * Nd) + (lambda_d * Rd) + (sigma_d * (1 - r_d) * Ed) 
                     - (m_d * Sd) - (beta_d * Sd * Id) - (gamma_d * Nd * Sd) 
                     - (target_status * (v_d * alpha_d * Sd)))
            Ed_new = (Ed + (beta_d * Sd * Id) - (m_d * Ed) - (gamma_d * Nd * Ed) 
                     - (sigma_d * (1 - r_d) * Ed) - (target_status * (v_d * alpha_d * Ed)) 
                     - (sigma_d * r_d * Ed))
            Rd_new = (Rd + (target_status * (v_d * alpha_d * (Sd + Ed))) - (m_d * Rd) 
                     - (gamma_d * Nd * Rd) - (lambda_d * Rd))
        
        Id_new = Id + (sigma_d * r_d * Ed) - (m_d * Id) - (gamma_d * Nd * Id) - (mu_d * Id)
        C_rd_new = C_rd + (sigma_d * r_d * Ed)
        Nd_new = Sd_new + Ed_new + Id_new + Rd_new
        
        # Update human compartments
        Sh_new = (Sh + (b_h * (Sh + Eh + Rh)) + (lambda_h * Rh) + (Eh * p_ExptoNoInf) 
                 - (m_h * Sh) - (v_h * alpha_h * Sh) - (beta_dh * Sh * Id))
        
        Eh_new = (Eh + (beta_dh * Sh * Id) - (m_h * Eh) - (Eh * p_ExptoInf * P10_step * v_h) 
                 - (Eh * p_ExptoInf * (1 - P10_step * v_h)) - (Eh * p_ExptoNoInf))
        
        Ih_new = Ih + (Eh * p_ExptoInf * (1 - P10_step * v_h)) - (m_h * Ih) - (mu_h * Ih)
        Dh_new = Dh + (Eh * p_ExptoInf * (1 - P10_step * v_h))
        
        Rh_new = (Rh + (Eh * p_ExptoInf * P10_step * v_h) + (v_h * alpha_h * Sh) 
                 - (m_h * Rh) - (lambda_h * Rh))
        
        Nh_new = Sh_new + Eh_new + Ih_new + Dh_new + Rh_new
        new_expo_new = beta_dh * Sh * Id
        
        # Update all values
        Sd, Ed, Id, C_rd, Rd, Nd = Sd_new, Ed_new, Id_new, C_rd_new, Rd_new, Nd_new
        Sh, Eh, Ih, Dh, Rh, Nh = Sh_new, Eh_new, Ih_new, Dh_new, Rh_new, Nh_new
        Dd = Dd + (mu_d * Id)  # Accumulate dog deaths from rabies
        new_expo = new_expo_new
        
        # Store results
        results["time"].append(time)
        results["week"].append(week)
        results["Sd"].append(Sd)
        results["Ed"].append(Ed)
        results["Id"].append(Id)
        results["C_rd"].append(C_rd)
        results["Rd"].append(Rd)
        results["Nd"].append(Nd)
        results["Sh"].append(Sh)
        results["Eh"].append(Eh)
        results["Ih"].append(Ih)
        results["Dh"].append(Dh)
        results["Dd"].append(Dd)
        results["Rh"].append(Rh)
        results["Nh"].append(Nh)
        results["new_expo"].append(new_expo)
    
    # Convert to DataFrame and add derived columns
    scenario_df = pd.DataFrame(results)
    scenario_df["year"] = [1] + [year_val for year_val in range(1, 101) for _ in range(52)][:len(scenario_df) - 1]
    scenario_df["Cu_new_expo"] = scenario_df["new_expo"].cumsum()
    
    return scenario_df

def extract_summary_values(df, scenario_name, params, coverage_data, years=list(range(0,31))):
    """Extract key summary values matching Excel formula logic"""
    # Extract required parameters
    Km2_of_program_area = params['Km2_of_program_area']
    inflation_factor_for_the_suspect_exposure = params['inflation_factor_for_the_suspect_exposure']
    post_elimination_pep_reduction = params['post_elimination_pep_reduction']
    cost_per_suspect_exposure = (
        params['quarantined_animal_prob'] * params['quarantined_animal_cost'] +
        params['lab_test_prob'] * params['lab_test_cost'] +
        params['bite_investigation_prob'] * params['bite_investigation_cost']
    )
    vaccination_cost_per_dog = params['vaccination_cost_per_dog']
    pep_and_other_costs = params['pep_and_other_costs']
    pep_prob_no_campaign = params['pep_prob_no_campaign']
    pep_prob_annual_campaign = params['pep_prob_annual_campaign']
    YLL = params['YLL']
    
    summary_data = []
    
    for year in years:
        if year == 0:
            # Year 0: Return 0
            canine_population = 0
            canine_rabies_cumulative = 0
            canine_rabies_annual = 0
            human_population = 0
            human_rabies_cumulative = 0
            human_rabies_annual = 0
            dog_deaths_cumulative = 0
            dog_deaths_annual = 0
            exposure_cumulative = 0
            exposure_annual = 0
        else:
            # Convert year to time step (year * 52)
            time_step = year * 52
            
            if time_step < len(df):
                row_data = df.iloc[time_step]
                
                # Extract values and scale by program area
                canine_population = row_data["Nd"] * Km2_of_program_area
                canine_rabies_cumulative = row_data["C_rd"] * Km2_of_program_area
                human_population = row_data["Nh"] * Km2_of_program_area
                human_rabies_cumulative = row_data["Dh"] * Km2_of_program_area
                dog_deaths_cumulative = row_data["Dd"] * Km2_of_program_area
                exposure_cumulative = row_data["Cu_new_expo"] * Km2_of_program_area
                
                # Calculate annual values
                if year == 1:
                    exposure_annual = exposure_cumulative
                    canine_rabies_annual = canine_rabies_cumulative
                    human_rabies_annual = human_rabies_cumulative
                    dog_deaths_annual = dog_deaths_cumulative
                else:
                    prev_time = (year - 1) * 52
                    if prev_time < len(df):
                        prev_exposure = df.iloc[prev_time]["Cu_new_expo"] * Km2_of_program_area
                        prev_canine_rabies = df.iloc[prev_time]["C_rd"] * Km2_of_program_area
                        prev_human_rabies = df.iloc[prev_time]["Dh"] * Km2_of_program_area
                        prev_dog_deaths = df.iloc[prev_time]["Dd"] * Km2_of_program_area
                        exposure_annual = exposure_cumulative - prev_exposure
                        canine_rabies_annual = canine_rabies_cumulative - prev_canine_rabies
                        human_rabies_annual = human_rabies_cumulative - prev_human_rabies
                        dog_deaths_annual = dog_deaths_cumulative - prev_dog_deaths
                    else:
                        exposure_annual = 0
                        canine_rabies_annual = 0
                        human_rabies_annual = 0
                        dog_deaths_annual = 0
            else:
                canine_population = 0
                canine_rabies_cumulative = 0
                canine_rabies_annual = 0
                human_population = 0
                human_rabies_cumulative = 0
                human_rabies_annual = 0
                dog_deaths_cumulative = 0
                dog_deaths_annual = 0
                exposure_cumulative = 0
                exposure_annual = 0
        
        # Calculate suspect exposures based on scenario
        if scenario_name == "No Annual Vaccination":
            suspect_exposure_cumulative = inflation_factor_for_the_suspect_exposure * exposure_cumulative
            suspect_exposure_annual = inflation_factor_for_the_suspect_exposure * exposure_annual
        else:
            if year == 1:
                suspect_exposure_cumulative = inflation_factor_for_the_suspect_exposure * exposure_cumulative
                suspect_exposure_annual = inflation_factor_for_the_suspect_exposure * exposure_annual
            else:
                prev_suspect_cum = summary_data[-1]['Suspect_exposure_cumulative'] if summary_data else 0
                prev_suspect_ann = summary_data[-1]['Suspect_exposure_annual'] if summary_data else 0
                
                reduced_prev_cum = (1 - post_elimination_pep_reduction) * prev_suspect_cum
                reduced_prev_ann = (1 - post_elimination_pep_reduction) * prev_suspect_ann
                
                inflated_current_cum = inflation_factor_for_the_suspect_exposure * exposure_cumulative
                inflated_current_ann = inflation_factor_for_the_suspect_exposure * exposure_annual
                
                suspect_exposure_cumulative = max(reduced_prev_cum, inflated_current_cum)
                suspect_exposure_annual = max(reduced_prev_ann, inflated_current_ann)

        # Calculate costs
        suspect_exposure_cost_cumulative = suspect_exposure_cumulative * cost_per_suspect_exposure
        suspect_exposure_cost_annual = suspect_exposure_annual * cost_per_suspect_exposure

        # Vaccination costs
        try:
            if scenario_name == "No Annual Vaccination":
                vaccination_coverage = get_vaccination_coverage(year, "no_annual_vaccination", coverage_data)
            else:
                vaccination_coverage = get_vaccination_coverage(year, "annual_vaccination", coverage_data)
            
            vaccination_cost_annual = canine_population * vaccination_coverage * vaccination_cost_per_dog
            
            if year == 1:
                vaccination_cost_cumulative = vaccination_cost_annual
            else:
                prev_cum_cost = summary_data[-1]['Vaccination_cost_cumulative'] if summary_data else 0
                vaccination_cost_cumulative = prev_cum_cost + vaccination_cost_annual
        except:
            vaccination_cost_cumulative = 0
            vaccination_cost_annual = 0

        # PEP costs
        if scenario_name == "No Annual Vaccination":
            pep_cost_annual = suspect_exposure_annual * pep_prob_no_campaign * pep_and_other_costs
            pep_cost_cumulative = suspect_exposure_cumulative * pep_prob_no_campaign * pep_and_other_costs
        else:
            pep_cost_annual = suspect_exposure_annual * pep_prob_annual_campaign * pep_and_other_costs
            pep_cost_cumulative = suspect_exposure_cumulative * pep_prob_annual_campaign * pep_and_other_costs

        summary_data.append({
            'Year': year,
            'Canine_population': canine_population,
            'Canine_rabies_cumulative': canine_rabies_cumulative,
            'Canine_rabies_annual': canine_rabies_annual,
            'Human_population': human_population,
            'Human_rabies_cumulative': human_rabies_cumulative,
            'Human_rabies_annual': human_rabies_annual,
            'Dog_deaths_cumulative': dog_deaths_cumulative,
            'Dog_deaths_annual': dog_deaths_annual,
            'Exposure_cumulative': exposure_cumulative,
            'Exposure_annual': exposure_annual,
            'Suspect_exposure_cumulative': suspect_exposure_cumulative,
            'Suspect_exposure_annual': suspect_exposure_annual,
            'Suspect_exposure_cost_cumulative': suspect_exposure_cost_cumulative,
            'Suspect_exposure_cost_annual': suspect_exposure_cost_annual,
            'Vaccination_cost_cumulative': vaccination_cost_cumulative,
            'Vaccination_cost_annual': vaccination_cost_annual,
            'PEP_cost_cumulative': pep_cost_cumulative,
            'PEP_cost_annual': pep_cost_annual
        })
    
    return pd.DataFrame(summary_data)

def create_program_summary_table(no_annual_summary, annual_summary, params):
    """Create comprehensive program summary table"""
    time_periods = [5, 10, 30]
    YLL = params['YLL']
    
    # Calculate suspect exposure rates per 100,000 persons for year 1
    year1_no_vacc_suspect_rate = (no_annual_summary.iloc[1]['Suspect_exposure_annual'] / 
                                  no_annual_summary.iloc[1]['Human_population']) * 100000
    year1_vacc_suspect_rate = (annual_summary.iloc[1]['Suspect_exposure_annual'] / 
                               annual_summary.iloc[1]['Human_population']) * 100000
    
    # Create summary data
    summary_rows = []
    
    for period in time_periods:
        # Get data for this period
        no_vacc_data = no_annual_summary.iloc[period]
        vacc_data = annual_summary.iloc[period]
        
        # Calculate metrics
        no_vacc_rabid_ann = int(no_vacc_data['Canine_rabies_annual'])
        no_vacc_rabid_cum = int(no_vacc_data['Canine_rabies_cumulative'])
        vacc_rabid_ann = int(vacc_data['Canine_rabies_annual'])
        vacc_rabid_cum = int(vacc_data['Canine_rabies_cumulative'])
        
        no_vacc_deaths_ann = int(no_vacc_data['Human_rabies_annual'])
        no_vacc_deaths_cum = int(no_vacc_data['Human_rabies_cumulative'])
        vacc_deaths_ann = int(vacc_data['Human_rabies_annual'])
        vacc_deaths_cum = int(vacc_data['Human_rabies_cumulative'])
        
        no_vacc_cost_ann = int(no_vacc_data['Vaccination_cost_annual'] + 
                              no_vacc_data['Suspect_exposure_cost_annual'] + 
                              no_vacc_data['PEP_cost_annual'])
        no_vacc_cost_cum = int(no_vacc_data['Vaccination_cost_cumulative'] + 
                              no_vacc_data['Suspect_exposure_cost_cumulative'] + 
                              no_vacc_data['PEP_cost_cumulative'])
        vacc_cost_ann = int(vacc_data['Vaccination_cost_annual'] + 
                           vacc_data['Suspect_exposure_cost_annual'] + 
                           vacc_data['PEP_cost_annual'])
        vacc_cost_cum = int(vacc_data['Vaccination_cost_cumulative'] + 
                           vacc_data['Suspect_exposure_cost_cumulative'] + 
                           vacc_data['PEP_cost_cumulative'])
        
        # Cost-effectiveness metrics
        deaths_averted_annual = no_vacc_deaths_ann - vacc_deaths_ann
        deaths_averted_cumulative = no_vacc_deaths_cum - vacc_deaths_cum
        additional_cost_annual = vacc_cost_ann - no_vacc_cost_ann
        additional_cost_cumulative = vacc_cost_cum - no_vacc_cost_cum
        
        cost_per_death_annual = int(additional_cost_annual / deaths_averted_annual) if deaths_averted_annual > 0 else None
        cost_per_death_cumulative = int(additional_cost_cumulative / deaths_averted_cumulative) if deaths_averted_cumulative > 0 else None
        
        dalys_averted_annual = deaths_averted_annual * YLL
        dalys_averted_cumulative = deaths_averted_cumulative * YLL
        
        cost_per_daly_annual = int(additional_cost_annual / dalys_averted_annual) if dalys_averted_annual > 0 else None
        cost_per_daly_cumulative = int(additional_cost_cumulative / dalys_averted_cumulative) if dalys_averted_cumulative > 0 else None
        
        summary_rows.append({
            'Time_Period': f'Year {period}',
            'Metric': 'Rabid dogs',
            'No_Vacc_Ann': f"{no_vacc_rabid_ann:,}",
            'No_Vacc_Cum': f"{no_vacc_rabid_cum:,}",
            'Vacc1_Ann': f"{vacc_rabid_ann:,}",
            'Vacc1_Cum': f"{vacc_rabid_cum:,}"
        })
        
        summary_rows.append({
            'Time_Period': f'Year {period}',
            'Metric': 'Human deaths',
            'No_Vacc_Ann': f"{no_vacc_deaths_ann:,}",
            'No_Vacc_Cum': f"{no_vacc_deaths_cum:,}",
            'Vacc1_Ann': f"{vacc_deaths_ann:,}",
            'Vacc1_Cum': f"{vacc_deaths_cum:,}"
        })
        
        # Add dog deaths metrics
        no_vacc_dog_deaths_ann = int(no_vacc_data['Dog_deaths_annual'])
        no_vacc_dog_deaths_cum = int(no_vacc_data['Dog_deaths_cumulative'])
        vacc_dog_deaths_ann = int(vacc_data['Dog_deaths_annual'])
        vacc_dog_deaths_cum = int(vacc_data['Dog_deaths_cumulative'])
        
        summary_rows.append({
            'Time_Period': f'Year {period}',
            'Metric': 'Dog deaths from rabies',
            'No_Vacc_Ann': f"{no_vacc_dog_deaths_ann:,}",
            'No_Vacc_Cum': f"{no_vacc_dog_deaths_cum:,}",
            'Vacc1_Ann': f"{vacc_dog_deaths_ann:,}",
            'Vacc1_Cum': f"{vacc_dog_deaths_cum:,}"
        })
        
        summary_rows.append({
            'Time_Period': f'Year {period}',
            'Metric': 'Program costs',
            'No_Vacc_Ann': f"${no_vacc_cost_ann:,}",
            'No_Vacc_Cum': f"${no_vacc_cost_cum:,}",
            'Vacc1_Ann': f"${vacc_cost_ann:,}",
            'Vacc1_Cum': f"${vacc_cost_cum:,}"
        })
        
        summary_rows.append({
            'Time_Period': f'Year {period}',
            'Metric': 'Cost per death averted',
            'No_Vacc_Ann': 'N/A',
            'No_Vacc_Cum': 'N/A',
            'Vacc1_Ann': f"${cost_per_death_annual:,}" if cost_per_death_annual else 'N/A',
            'Vacc1_Cum': f"${cost_per_death_cumulative:,}" if cost_per_death_cumulative else 'N/A'
        })
        
        summary_rows.append({
            'Time_Period': f'Year {period}',
            'Metric': 'Cost per DALY averted',
            'No_Vacc_Ann': 'N/A',
            'No_Vacc_Cum': 'N/A',
            'Vacc1_Ann': f"${cost_per_daly_annual:,}" if cost_per_daly_annual else 'N/A',
            'Vacc1_Cum': f"${cost_per_daly_cumulative:,}" if cost_per_daly_cumulative else 'N/A'
        })
    
    return pd.DataFrame(summary_rows), year1_no_vacc_suspect_rate, year1_vacc_suspect_rate

def create_visualization_plots(no_annual_summary, annual_summary):
    """Create the 2x2 visualization plots"""
    # Filter data to start from year 1
    no_annual_filtered = no_annual_summary[no_annual_summary["Year"] >= 1].iloc[:30]  # Years 1-30
    annual_filtered = annual_summary[annual_summary["Year"] >= 1].iloc[:30]  # Years 1-30
    
    # Create 2x2 subplot grid with smaller figure size to fit screen better
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Rabid dogs (annual) - Top Left
    axes[0,0].plot(no_annual_filtered["Year"], no_annual_filtered["Canine_rabies_annual"], 
                   linewidth=2.5, color='red', label='No vaccination campaign')
    axes[0,0].plot(annual_filtered["Year"], annual_filtered["Canine_rabies_annual"], 
                   linewidth=2.5, color='green', label='Annual vaccination campaign')
    axes[0,0].set_title("Rabid dogs (annual)", fontsize=11, fontweight='bold')
    axes[0,0].set_xlabel("Year", fontsize=10)
    axes[0,0].set_ylabel("Canine rabies cases", fontsize=10)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlim(1, 30)
    axes[0,0].tick_params(axis='both', which='major', labelsize=9)
    
    # Plot 2: Canine rabies cases (cumulative) - Top Right
    axes[0,1].plot(no_annual_filtered["Year"], no_annual_filtered["Canine_rabies_cumulative"], 
                   linewidth=2.5, color='red', label='No vaccination campaign')
    axes[0,1].plot(annual_filtered["Year"], annual_filtered["Canine_rabies_cumulative"], 
                   linewidth=2.5, color='green', label='Annual vaccination campaign')
    axes[0,1].set_title("Canine rabies cases (cumulative)", fontsize=11, fontweight='bold')
    axes[0,1].set_xlabel("Year", fontsize=10)
    axes[0,1].set_ylabel("Cumulative canine cases", fontsize=10)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xlim(1, 30)
    axes[0,1].tick_params(axis='both', which='major', labelsize=9)
    
    # Plot 3: Human deaths due to rabies (annual) - Bottom Left
    axes[1,0].plot(no_annual_filtered["Year"], no_annual_filtered["Human_rabies_annual"], 
                   linewidth=2.5, color='red', label='No vaccination campaign')
    axes[1,0].plot(annual_filtered["Year"], annual_filtered["Human_rabies_annual"], 
                   linewidth=2.5, color='green', label='Annual vaccination campaign')
    axes[1,0].set_title("Human deaths due to rabies (annual)", fontsize=11, fontweight='bold')
    axes[1,0].set_xlabel("Year", fontsize=10)
    axes[1,0].set_ylabel("Human deaths", fontsize=10)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlim(1, 30)
    axes[1,0].tick_params(axis='both', which='major', labelsize=9)
    
    # Plot 4: Human deaths (cumulative) - Bottom Right
    axes[1,1].plot(no_annual_filtered["Year"], no_annual_filtered["Human_rabies_cumulative"], 
                   linewidth=2.5, color='red', label='No vaccination campaign')
    axes[1,1].plot(annual_filtered["Year"], annual_filtered["Human_rabies_cumulative"], 
                   linewidth=2.5, color='green', label='Annual vaccination campaign')
    axes[1,1].set_title("Human deaths (cumulative)", fontsize=11, fontweight='bold')
    axes[1,1].set_xlabel("Year", fontsize=10)
    axes[1,1].set_ylabel("Cumulative human deaths", fontsize=10)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlim(1, 30)
    axes[1,1].tick_params(axis='both', which='major', labelsize=9)
    
    # Add a single legend at the bottom
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10)
    
    # Adjust layout with tighter spacing
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.3, wspace=0.25)
    
    return fig

def create_mortality_rate_plots(no_annual_summary, annual_summary):
    """Create side-by-side plots for dog and human mortality rates"""
    from matplotlib.ticker import FuncFormatter
    
    # Create figure with side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get years data (skip year 0)
    years = range(1, 31)
    
    # Left plot: Dog deaths as percentage of population
    # Calculate dog deaths percentage
    no_dog_deaths_pct = []
    annual_dog_deaths_pct = []
    
    for year in years:
        if year < len(no_annual_summary):
            # Get dog population and deaths for this year
            no_dog_pop = no_annual_summary.iloc[year]['Canine_population']
            no_dog_deaths = no_annual_summary.iloc[year]['Dog_deaths_annual']
            no_pct = (no_dog_deaths / no_dog_pop * 100) if no_dog_pop > 0 else 0
            no_dog_deaths_pct.append(no_pct)
            
            annual_dog_pop = annual_summary.iloc[year]['Canine_population']
            annual_dog_deaths = annual_summary.iloc[year]['Dog_deaths_annual']
            annual_pct = (annual_dog_deaths / annual_dog_pop * 100) if annual_dog_pop > 0 else 0
            annual_dog_deaths_pct.append(annual_pct)
        else:
            no_dog_deaths_pct.append(0)
            annual_dog_deaths_pct.append(0)
    
    axes[0].plot(years, no_dog_deaths_pct, linewidth=2.5, color='red', 
                 label='No vaccination campaign', marker='o', markersize=3)
    axes[0].plot(years, annual_dog_deaths_pct, linewidth=2.5, color='green', 
                 label='Annual vaccination campaign', marker='s', markersize=3)
    axes[0].set_title('Dog Deaths Due to Rabies\n(% of Dog Population)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Year', fontsize=10)
    axes[0].set_ylabel('Dog Deaths (%)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(1, 30)
    axes[0].tick_params(axis='both', which='major', labelsize=9)
    axes[0].legend(loc='upper right', fontsize=9)
    
    # Format y-axis as percentage
    def percentage_formatter(x, pos):
        return f'{x:.3f}%'
    axes[0].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    
    # Right plot: Human deaths per 100,000 population
    no_human_deaths_rate = []
    annual_human_deaths_rate = []
    
    for year in years:
        if year < len(no_annual_summary):
            # Get human population and deaths for this year
            no_human_pop = no_annual_summary.iloc[year]['Human_population']
            no_human_deaths = no_annual_summary.iloc[year]['Human_rabies_annual']
            no_rate = (no_human_deaths / no_human_pop * 100000) if no_human_pop > 0 else 0
            no_human_deaths_rate.append(no_rate)
            
            annual_human_pop = annual_summary.iloc[year]['Human_population']
            annual_human_deaths = annual_summary.iloc[year]['Human_rabies_annual']
            annual_rate = (annual_human_deaths / annual_human_pop * 100000) if annual_human_pop > 0 else 0
            annual_human_deaths_rate.append(annual_rate)
        else:
            no_human_deaths_rate.append(0)
            annual_human_deaths_rate.append(0)
    
    axes[1].plot(years, no_human_deaths_rate, linewidth=2.5, color='red', 
                 label='No vaccination campaign', marker='o', markersize=3)
    axes[1].plot(years, annual_human_deaths_rate, linewidth=2.5, color='green', 
                 label='Annual vaccination campaign', marker='s', markersize=3)
    axes[1].set_title('Human Deaths Due to Rabies\n(per 100,000 Population)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Year', fontsize=10)
    axes[1].set_ylabel('Deaths per 100,000', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(1, 30)
    axes[1].tick_params(axis='both', which='major', labelsize=9)
    axes[1].legend(loc='upper right', fontsize=9)
    
    # Format y-axis for rate
    def rate_formatter(x, pos):
        return f'{x:.2f}'
    axes[1].yaxis.set_major_formatter(FuncFormatter(rate_formatter))
    
    plt.tight_layout()
    return fig

# Main Streamlit App
def main():
    st.title("🐕 Rabies Economic Analysis Model")
    st.markdown("### Comprehensive Economic Impact Assessment of Rabies Vaccination Programs")
    
    # Load data
    with st.spinner("Loading model data..."):
        coverage_data, model_parameters = load_data()
    
    st.success("✅ Data loaded successfully")
    
    # Create interactive parameter inputs FIRST (so they appear at top of sidebar)
    params = create_parameter_inputs(model_parameters)
    
    # Create coverage data editor (this will modify coverage_data if custom data is used)
    coverage_data = create_coverage_data_editor(coverage_data)
    
    # Add parameter summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Current Parameters")
    st.sidebar.metric("Program Area (km²)", f"{params['Km2_of_program_area']:,.0f}")
    st.sidebar.metric("Human Population", f"{params['Human_population']:,.0f}")
    st.sidebar.metric("Humans per Free-Roaming Dog (HDR)", f"{params['Humans_per_free_roaming_dog']:.1f}")
    st.sidebar.metric("Dog Population", f"{params['Free_roaming_dog_population']:,.0f}")
    st.sidebar.metric("Dogs per km²", f"{params['Free_roaming_dogs_per_km2']:,.1f}")
    st.sidebar.metric("R₀ (Dog-to-Dog)", f"{params['R0_dog_to_dog']:.3f}")
    
    # Run analysis button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        
        # Validate critical parameters
        if params['Km2_of_program_area'] <= 0:
            st.error("❌ Program area must be greater than 0")
            return
        if params['Human_population'] <= 0:
            st.error("❌ Human population must be greater than 0")
            return
        if params['Humans_per_free_roaming_dog'] <= 0:
            st.error("❌ Humans per dog ratio must be greater than 0")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initial equilibrium simulation
        status_text.text("Running initial equilibrium simulation...")
        progress_bar.progress(20)
        initial_run = run_initial_simulation(params)
        
        # Step 2: No vaccination scenario
        status_text.text("Simulating no vaccination scenario...")
        progress_bar.progress(40)
        no_annual_vaccination = run_scenario_simulation(initial_run, params, coverage_data, "no_annual_vaccination")
        
        # Step 3: Annual vaccination scenario
        status_text.text("Simulating annual vaccination scenario...")
        progress_bar.progress(60)
        annual_vaccination = run_scenario_simulation(initial_run, params, coverage_data, "annual_vaccination")
        
        # Step 4: Extract summary values
        status_text.text("Extracting summary statistics...")
        progress_bar.progress(80)
        no_annual_summary = extract_summary_values(no_annual_vaccination, "No Annual Vaccination", params, coverage_data)
        annual_summary = extract_summary_values(annual_vaccination, "Annual Vaccination", params, coverage_data)
        
        # Step 5: Create visualizations
        status_text.text("Generating visualizations...")
        progress_bar.progress(100)
        
        status_text.text("Analysis complete!")
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.success("🎉 Analysis completed successfully!")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Executive Summary", "📊 Program Summary", "📋 Detailed Results", "📈 Visualizations"])
        
        with tab1:
            st.header("Executive Summary")
            st.markdown("### 📊 Impact Analysis by Vaccination Program Phase")
            
            # Create phase-based summary
            phases = [
                {"name": "Phase I (Years 1-3)", "years": [1, 2, 3], "color": "#1e40af", "description": "Initial mass vaccination"},
                {"name": "Phase II (Years 4-6)", "years": [4, 5, 6], "color": "#7c2d12", "description": "Intensified efforts"},
                {"name": "Phase III (Years 7-13)", "years": [7, 8, 9, 10, 11, 12, 13], "color": "#065f46", "description": "High coverage maintenance"},
                {"name": "Phase IV (Years 14-20)", "years": [14, 15, 16, 17, 18, 19, 20], "color": "#581c87", "description": "Post-elimination maintenance"}
            ]
            
            # Display phase results in tabs
            phase_tabs = st.tabs([phase["name"] for phase in phases])
            
            for i, phase in enumerate(phases):
                with phase_tabs[i]:
                    st.markdown(f"**{phase['description']}**")
                    
                    # Calculate cumulative results for this phase
                    phase_end_year = max(phase["years"])
                    if phase_end_year <= len(no_annual_summary) - 1:
                        year_no_vacc = no_annual_summary.iloc[phase_end_year]
                        year_vacc = annual_summary.iloc[phase_end_year]
                        
                        # Create metrics for this phase
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            deaths_averted = year_no_vacc['Human_rabies_cumulative'] - year_vacc['Human_rabies_cumulative']
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {phase['color']} 0%, {phase['color']}aa 100%); 
                                       padding: 1.2rem; border-radius: 10px; text-align: center; 
                                       box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 0.5rem 0;">
                                <h4 style="color: #e0e7ff; margin: 0; font-size: 0.9rem; font-weight: 600;">Deaths Averted</h4>
                                <h2 style="color: white; margin: 0.5rem 0; font-size: 1.8rem; font-weight: bold;">{deaths_averted:.0f}</h2>
                                <p style="color: #86efac; margin: 0; font-size: 0.8rem;">Cumulative by Year {phase_end_year}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            cost_diff = (year_vacc['Vaccination_cost_cumulative'] + 
                                       year_vacc['Suspect_exposure_cost_cumulative'] + 
                                       year_vacc['PEP_cost_cumulative']) - \
                                      (year_no_vacc['Vaccination_cost_cumulative'] + 
                                       year_no_vacc['Suspect_exposure_cost_cumulative'] + 
                                       year_no_vacc['PEP_cost_cumulative'])
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #dc2626 0%, #dc2626aa 100%); 
                                       padding: 1.2rem; border-radius: 10px; text-align: center; 
                                       box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 0.5rem 0;">
                                <h4 style="color: #fecaca; margin: 0; font-size: 0.9rem; font-weight: 600;">Additional Cost</h4>
                                <h2 style="color: white; margin: 0.5rem 0; font-size: 1.8rem; font-weight: bold;">${cost_diff:,.0f}</h2>
                                <p style="color: #fbbf24; margin: 0; font-size: 0.8rem;">Investment required</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            cost_per_death = cost_diff / deaths_averted if deaths_averted > 0 else 0
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #059669 0%, #059669aa 100%); 
                                       padding: 1.2rem; border-radius: 10px; text-align: center; 
                                       box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 0.5rem 0;">
                                <h4 style="color: #a7f3d0; margin: 0; font-size: 0.9rem; font-weight: 600;">Cost per Death Averted</h4>
                                <h2 style="color: white; margin: 0.5rem 0; font-size: 1.8rem; font-weight: bold;">${cost_per_death:,.0f}</h2>
                                <p style="color: #86efac; margin: 0; font-size: 0.8rem;">Cost-effectiveness</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            dalys_averted = deaths_averted * params['YLL']
                            cost_per_daly = cost_diff / dalys_averted if dalys_averted > 0 else 0
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #7c3aed 0%, #7c3aedaa 100%); 
                                       padding: 1.2rem; border-radius: 10px; text-align: center; 
                                       box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 0.5rem 0;">
                                <h4 style="color: #ddd6fe; margin: 0; font-size: 0.9rem; font-weight: 600;">Cost per DALY Averted</h4>
                                <h2 style="color: white; margin: 0.5rem 0; font-size: 1.8rem; font-weight: bold;">${cost_per_daly:,.0f}</h2>
                                <p style="color: #c084fc; margin: 0; font-size: 0.8rem;">WHO threshold comparison</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Phase-specific insights
                        st.markdown("#### Phase Insights")
                        
                        if i == 0:  # Phase I
                            st.info("""
                            🎯 **Phase I Focus**: Initial mass vaccination establishes baseline immunity
                            - Foundation period for building dog population immunity
                            - Higher costs as program infrastructure is established
                            - Early impact on reducing rabies transmission
                            """)
                        elif i == 1:  # Phase II
                            st.warning("""
                            ⚡ **Phase II Focus**: Intensified efforts to reach elimination targets
                            - Critical period for achieving high vaccination coverage
                            - Significant reduction in human rabies cases begins
                            - Cost-effectiveness improves as coverage increases
                            """)
                        elif i == 2:  # Phase III
                            st.success("""
                            🏆 **Phase III Focus**: High coverage maintenance and elimination approach
                            - Optimal period for rabies elimination in dog populations
                            - Maximum public health impact achieved
                            - Best cost-effectiveness ratios typically seen
                            """)
                        else:  # Phase IV
                            st.info("""
                            🛡️ **Phase IV Focus**: Post-elimination maintenance and surveillance
                            - Sustained low-level vaccination to prevent reintroduction
                            - Minimal human cases, focus on maintaining elimination
                            - Long-term cost savings from reduced PEP needs
                            """)
                    else:
                        st.warning(f"Data not available for {phase['name']} (extends beyond simulation period)")
            
            # Overall program summary
            st.markdown("---")
            st.subheader("🎯 Overall Program Impact (30 Years)")
            
            # Final year comparison
            final_year = len(no_annual_summary) - 1
            final_no_vacc = no_annual_summary.iloc[final_year]
            final_vacc = annual_summary.iloc[final_year]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_deaths_averted = final_no_vacc['Human_rabies_cumulative'] - final_vacc['Human_rabies_cumulative']
                st.metric(
                    "Total Deaths Averted", 
                    f"{total_deaths_averted:,.0f}", 
                    f"{total_deaths_averted:,.0f} lives saved over 30 years"
                )
            
            with col2:
                total_cost_diff = (final_vacc['Vaccination_cost_cumulative'] + 
                                 final_vacc['Suspect_exposure_cost_cumulative'] + 
                                 final_vacc['PEP_cost_cumulative']) - \
                                (final_no_vacc['Vaccination_cost_cumulative'] + 
                                 final_no_vacc['Suspect_exposure_cost_cumulative'] + 
                                 final_no_vacc['PEP_cost_cumulative'])
                st.metric(
                    "Total Additional Investment", 
                    f"${total_cost_diff:,.0f}", 
                    "Total program cost over 30 years"
                )
            
            with col3:
                overall_cost_per_death = total_cost_diff / total_deaths_averted if total_deaths_averted > 0 else 0
                st.metric(
                    "Overall Cost-Effectiveness", 
                    f"${overall_cost_per_death:,.0f}", 
                    "Per death averted (30-year program)"
                )
            
            # Add visualization plots to Executive Summary
            st.subheader("📈 Impact Visualization")
            st.markdown("Visual comparison of rabies outcomes with and without vaccination programs over 30 years:")
            
            # Create and display the visualization plots
            fig = create_visualization_plots(no_annual_summary, annual_summary)
            st.pyplot(fig)
            
            # Add interpretation text below the plots
            st.markdown("""
            **Chart Interpretation:**
            - **Top row**: Shows dramatic reduction in canine rabies cases with vaccination
            - **Bottom row**: Demonstrates significant prevention of human deaths
            - **Red lines**: No vaccination program (status quo)
            - **Green lines**: Annual vaccination program (intervention)
            """)
        
        with tab2:
            st.header("Program Summary Table")
            
            # Create program summary
            summary_df, year1_no_vacc_rate, year1_vacc_rate = create_program_summary_table(
                no_annual_summary, annual_summary, params
            )
            
            # Program definition
            st.subheader("Program Definition")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **No Vaccination Program:**
                - Single/one-time vaccination
                - 0% vaccination coverage
                - 25% human exposures receive PEP
                - 0% female dogs spayed annually
                - 0% male dogs neutered annually
                """)
            
            with col2:
                st.markdown("""
                **Vaccination Option 1:**
                - Annual vaccination program
                - Varying vaccination coverage
                - 50% human exposures receive PEP
                - 0% female dogs spayed annually
                - 0% male dogs neutered annually
                """)
            
            # Exposure rates
            st.subheader("Suspect Human Rabies Exposures (per 100,000 persons) in Year 1")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #7c2d12 0%, #dc2626 100%); 
                           padding: 1.2rem; border-radius: 10px; text-align: center; 
                           box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 0.5rem 0;">
                    <h4 style="color: #fecaca; margin: 0; font-size: 1rem; font-weight: 600;">No Vaccination Program</h4>
                    <h2 style="color: white; margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">{year1_no_vacc_rate:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #065f46 0%, #059669 100%); 
                           padding: 1.2rem; border-radius: 10px; text-align: center; 
                           box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 0.5rem 0;">
                    <h4 style="color: #a7f3d0; margin: 0; font-size: 1rem; font-weight: 600;">Vaccination Option 1</h4>
                    <h2 style="color: white; margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">{year1_vacc_rate:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Summary table
            st.subheader("Fixed Timeframes Analysis (Years 5, 10, 30)")
            st.dataframe(
                summary_df,
                column_config={
                    "Time_Period": "Time Period",
                    "Metric": "Metric",
                    "No_Vacc_Ann": "No Vacc (Annual)",
                    "No_Vacc_Cum": "No Vacc (Cumulative)",
                    "Vacc1_Ann": "Vaccination (Annual)",
                    "Vacc1_Cum": "Vaccination (Cumulative)"
                },
                hide_index=True,
                use_container_width=True
            )
        
        with tab3:
            st.header("Detailed Results")
            
            # Year selector
            year_options = list(range(1, 31))
            selected_years = st.multiselect(
                "Select years to display:",
                year_options,
                default=[1, 5, 10, 15, 20, 25, 30]
            )
            
            if selected_years:
                # Create detailed comparison table
                comparison_data = []
                for year in selected_years:
                    no_vacc_data = no_annual_summary.iloc[year]
                    vacc_data = annual_summary.iloc[year]
                    
                    comparison_data.append({
                        'Year': year,
                        'No_Vacc_Canine_Deaths': f"{no_vacc_data['Canine_rabies_cumulative']:,.0f}",
                        'Vacc_Canine_Deaths': f"{vacc_data['Canine_rabies_cumulative']:,.0f}",
                        'No_Vacc_Human_Deaths': f"{no_vacc_data['Human_rabies_cumulative']:,.0f}",
                        'Vacc_Human_Deaths': f"{vacc_data['Human_rabies_cumulative']:,.0f}",
                        'Deaths_Averted': f"{no_vacc_data['Human_rabies_cumulative'] - vacc_data['Human_rabies_cumulative']:,.0f}",
                        'No_Vacc_Total_Cost': f"${(no_vacc_data['Vaccination_cost_cumulative'] + no_vacc_data['Suspect_exposure_cost_cumulative'] + no_vacc_data['PEP_cost_cumulative']):,.0f}",
                        'Vacc_Total_Cost': f"${(vacc_data['Vaccination_cost_cumulative'] + vacc_data['Suspect_exposure_cost_cumulative'] + vacc_data['PEP_cost_cumulative']):,.0f}",
                        'Additional_Cost': f"${((vacc_data['Vaccination_cost_cumulative'] + vacc_data['Suspect_exposure_cost_cumulative'] + vacc_data['PEP_cost_cumulative']) - (no_vacc_data['Vaccination_cost_cumulative'] + no_vacc_data['Suspect_exposure_cost_cumulative'] + no_vacc_data['PEP_cost_cumulative'])):,.0f}"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                st.dataframe(
                    comparison_df,
                    column_config={
                        "Year": "Year",
                        "No_Vacc_Canine_Deaths": "Canine Deaths (No Vacc)",
                        "Vacc_Canine_Deaths": "Canine Deaths (Vacc)",
                        "No_Vacc_Human_Deaths": "Human Deaths (No Vacc)",
                        "Vacc_Human_Deaths": "Human Deaths (Vacc)",
                        "Deaths_Averted": "Deaths Averted",
                        "No_Vacc_Total_Cost": "Total Cost (No Vacc)",
                        "Vacc_Total_Cost": "Total Cost (Vacc)",
                        "Additional_Cost": "Additional Cost"
                    },
                    hide_index=True,
                    use_container_width=True
                )
        
        with tab4:
            st.header("Impact Visualizations")
            
            # Create and display plots
            fig = create_visualization_plots(no_annual_summary, annual_summary)
            st.pyplot(fig)
            
            # Mortality Rate Analysis
            st.subheader("Mortality Rate Analysis")
            
            # Create and display mortality rate plots
            fig_mortality = create_mortality_rate_plots(no_annual_summary, annual_summary)
            st.pyplot(fig_mortality)

if __name__ == "__main__":
    main()