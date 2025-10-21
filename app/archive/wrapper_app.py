"""
Rabies Economic Analysis - Wrapper for initial_run.py
====================================================

A simple Streamlit app that runs the working initial_run.py and displays results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import sys
from pathlib import Path
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Rabies Economic Analysis",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

def run_initial_script():
    """Run the working initial_run.py script and return results"""
    
    project_root = Path(__file__).parent.parent
    script_path = project_root / "notebooks" / "initial_run.py"
    
    if not script_path.exists():
        st.error(f"Script not found: {script_path}")
        return None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Running initial_run.py script...")
        progress_bar.progress(0.1)
        
        # Change to project directory
        original_cwd = os.getcwd()
        os.chdir(str(project_root))
        
        # Run the script
        result = subprocess.run([
            sys.executable, 
            str(script_path)
        ], capture_output=True, text=True, cwd=str(project_root))
        
        progress_bar.progress(0.5)
        status_text.text("Processing results...")
        
        if result.returncode != 0:
            st.error(f"Script execution failed:")
            st.code(result.stderr)
            return None
        
        # The script should have generated some output files or we can parse the output
        st.success("Script executed successfully!")
        
        # Display the output
        if result.stdout:
            st.subheader("📋 Script Output")
            st.text(result.stdout)
        
        if result.stderr:
            st.subheader("⚠️ Warnings")
            st.text(result.stderr)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Completed!")
        
        return result
        
    except Exception as e:
        st.error(f"Error running script: {e}")
        return None
    finally:
        os.chdir(original_cwd)


def create_manual_plots():
    """Create plots using the known structure from your working model"""
    
    st.header("📊 Manual Plot Creation")
    st.info("Since we can see your original graphs work, let's create similar plots with some sample data to test the visualization framework.")
    
    # Create sample data that matches your expected output structure
    years = np.arange(1, 31)
    
    # Sample data based on your screenshots (rough approximation)
    # No vaccination scenario - higher values
    no_vacc_canine_annual = np.array([1200, 1100, 1000, 950, 900, 850, 800, 750, 700, 650] + 
                                    [600, 580, 560, 540, 520, 500, 480, 460, 440, 420] +
                                    [400, 390, 380, 370, 360, 350, 340, 330, 320, 310])
    
    no_vacc_canine_cumulative = np.cumsum(no_vacc_canine_annual)
    
    # Vaccination scenario - lower values  
    vacc_canine_annual = np.array([300, 250, 200, 180, 160, 140, 120, 100, 90, 80] +
                                 [75, 70, 65, 60, 55, 50, 45, 40, 35, 30] +
                                 [28, 26, 24, 22, 20, 18, 16, 14, 12, 10])
    
    vacc_canine_cumulative = np.cumsum(vacc_canine_annual)
    
    # Human deaths (proportional to canine cases)
    no_vacc_human_annual = no_vacc_canine_annual * 0.5  # Rough scaling
    no_vacc_human_cumulative = np.cumsum(no_vacc_human_annual)
    
    vacc_human_annual = vacc_canine_annual * 0.3  # Lower impact with vaccination
    vacc_human_cumulative = np.cumsum(vacc_human_annual)
    
    # Create the 2x2 plot
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
    
    # Plot 1: Rabid dogs (annual)
    fig.add_trace(go.Scatter(x=years, y=no_vacc_canine_annual, mode='lines', name='No vaccination campaign', line=dict(color='red', width=3), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=years, y=vacc_canine_annual, mode='lines', name='Annual vaccination campaign', line=dict(color='green', width=3), showlegend=True), row=1, col=1)
    
    # Plot 2: Canine rabies cases (cumulative)
    fig.add_trace(go.Scatter(x=years, y=no_vacc_canine_cumulative, mode='lines', name='No vaccination campaign', line=dict(color='red', width=3), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=years, y=vacc_canine_cumulative, mode='lines', name='Annual vaccination campaign', line=dict(color='green', width=3), showlegend=False), row=1, col=2)
    
    # Plot 3: Human deaths (annual)
    fig.add_trace(go.Scatter(x=years, y=no_vacc_human_annual, mode='lines', name='No vaccination campaign', line=dict(color='red', width=3), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=years, y=vacc_human_annual, mode='lines', name='Annual vaccination campaign', line=dict(color='green', width=3), showlegend=False), row=2, col=1)
    
    # Plot 4: Human deaths (cumulative)
    fig.add_trace(go.Scatter(x=years, y=no_vacc_human_cumulative, mode='lines', name='No vaccination campaign', line=dict(color='red', width=3), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=years, y=vacc_human_cumulative, mode='lines', name='Annual vaccination campaign', line=dict(color='green', width=3), showlegend=False), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        title_text="Rabies Epidemiological Impact: Vaccination vs No Vaccination"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="Canine rabies cases", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative canine cases", row=1, col=2)
    fig.update_yaxes(title_text="Human deaths", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative human deaths", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show key insights
    st.subheader("💡 Key Insights")
    
    year_10_idx = 9  # Year 10 (0-indexed)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        deaths_averted = no_vacc_human_cumulative[year_10_idx] - vacc_human_cumulative[year_10_idx]
        st.metric("Human Deaths Averted (10 yrs)", f"{deaths_averted:,.0f}")
    
    with col2:
        cases_averted = no_vacc_canine_cumulative[year_10_idx] - vacc_canine_cumulative[year_10_idx]
        st.metric("Canine Cases Averted (10 yrs)", f"{cases_averted:,.0f}")
    
    with col3:
        # Rough cost estimates
        additional_cost = 1000000  # $1M additional cost for vaccination program
        if deaths_averted > 0:
            cost_per_death = additional_cost / deaths_averted
            st.metric("Est. Cost per Death Averted", f"${cost_per_death:,.0f}")
    
    with col4:
        if deaths_averted > 0:
            yll = 37  # Years of life lost per death
            cost_per_daly = cost_per_death / yll
            st.metric("Est. Cost per DALY Averted", f"${cost_per_daly:,.0f}")


def create_sample_summary_table():
    """Create a sample summary table matching your format"""
    
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
    
    # Sample metrics table
    st.subheader("📈 Key Metrics Comparison")
    
    # Sample data matching your table format
    table_data = [
        {'Metric': 'Rabid dogs', 'Time Period': 'Year 5', 'No Vacc Annual': '24,995', 'No Vacc Cumulative': '168,292', 'Vacc1 Annual': '11,169', 'Vacc1 Cumulative': '155,900'},
        {'Metric': 'Human deaths', 'Time Period': 'Year 5', 'No Vacc Annual': '199', 'No Vacc Cumulative': '1,345', 'Vacc1 Annual': '71', 'Vacc1 Cumulative': '884'},
        {'Metric': 'Program costs', 'Time Period': 'Year 5', 'No Vacc Annual': '$181,291', 'No Vacc Cumulative': '$1,017,325', 'Vacc1 Annual': '$819,441', 'Vacc1 Cumulative': '$2,859,043'},
        {'Metric': 'Cost per death averted', 'Time Period': 'Year 5', 'No Vacc Annual': 'N/A', 'No Vacc Cumulative': 'N/A', 'Vacc1 Annual': '$4985', 'Vacc1 Cumulative': '$3995'},
        {'Metric': 'Cost per DALY averted', 'Time Period': 'Year 5', 'No Vacc Annual': 'N/A', 'No Vacc Cumulative': 'N/A', 'Vacc1 Annual': '$189', 'Vacc1 Cumulative': '$151'},
        
        {'Metric': 'Program costs', 'Time Period': 'Year 10', 'No Vacc Annual': '$289,555', 'No Vacc Cumulative': '$2,353,958', 'Vacc1 Annual': '$792,724', 'Vacc1 Cumulative': '$6,302,659'},
        {'Metric': 'Cost per death averted', 'Time Period': 'Year 10', 'No Vacc Annual': 'N/A', 'No Vacc Cumulative': 'N/A', 'Vacc1 Annual': '$1138', 'Vacc1 Cumulative': '$1691'},
        {'Metric': 'Cost per DALY averted', 'Time Period': 'Year 10', 'No Vacc Annual': 'N/A', 'No Vacc Cumulative': 'N/A', 'Vacc1 Annual': '$43', 'Vacc1 Cumulative': '$64'},
    ]
    
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    return df_table


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🐕 Rabies Economic Analysis Dashboard")
    st.markdown("""
    This application provides a working interface for your rabies economic analysis.
    Since your `initial_run.py` works correctly, we can either run it directly or use its output structure.
    """)
    
    # Create tabs for different approaches
    tab1, tab2, tab3 = st.tabs(["🚀 Run Original Script", "📊 Sample Visualization", "📋 Sample Table"])
    
    with tab1:
        st.subheader("Run Your Working initial_run.py Script")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Execute initial_run.py", type="primary", use_container_width=True):
                result = run_initial_script()
                
                if result:
                    st.info("The script has been executed. Check the output above and any generated files in your project directory.")
    
    with tab2:
        create_manual_plots()
    
    with tab3:
        create_sample_summary_table()
    
    # Information section
    st.markdown("---")
    st.subheader("📖 Next Steps")
    st.markdown("""
    **To get your exact results in Streamlit:**
    
    1. **Run your working script** using the first tab to confirm it works
    2. **Identify the data structure** - we need to see exactly what variables and DataFrames your `initial_run.py` creates
    3. **Extract the results** - modify your script to save results as pickle/CSV files that Streamlit can load
    4. **Create the interface** - build Streamlit widgets around your working model
    
    **The sample visualization** (tab 2) shows the correct chart structure that matches your screenshots.
    **The sample table** (tab 3) shows the format you're looking for.
    
    The key is to use your working model as-is, rather than trying to recreate it!
    """)


if __name__ == "__main__":
    main()