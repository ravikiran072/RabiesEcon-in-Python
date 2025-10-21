import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os

st.set_page_config(page_title="Rabies Economic Analysis", layout="wide")

st.title("Rabies Economic Model - Exact Results")
st.write("This app shows the exact results from the working initial_run.py model")

# Button to run the model and generate fresh results
if st.button("Run Model (Generate Fresh Results)"):
    with st.spinner("Running rabies economic model..."):
        try:
            # Change to notebooks directory and run initial_run.py
            os.chdir("notebooks")
            result = subprocess.run(["python", "initial_run.py"], 
                                  capture_output=True, text=True, timeout=300)
            os.chdir("..")
            
            if result.returncode == 0:
                st.success("Model completed successfully!")
                if result.stdout:
                    st.text("Model output:")
                    st.code(result.stdout)
            else:
                st.error("Model failed to run")
                st.code(result.stderr)
        except subprocess.TimeoutExpired:
            st.error("Model execution timed out (5 minutes)")
        except Exception as e:
            st.error(f"Error running model: {e}")

# Try to load saved results
results_file = "notebooks/streamlit_results.pkl"
if os.path.exists(results_file):
    try:
        with open(results_file, 'rb') as f:
            saved_results = pickle.load(f)
        
        plot_data = saved_results['plot_data']
        summary_data = saved_results['summary_data']
        
        st.success("Loaded exact results from model!")
        
        # Create the exact 2x2 plot layout matching initial_run.py
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Rabid dogs (annual)", "Canine rabies cases (cumulative)",
                          "Human deaths due to rabies (annual)", "Human deaths (cumulative)"),
            vertical_spacing=0.12
        )
        
        years = plot_data['years']
        
        # Top left: Annual rabid dogs
        fig.add_trace(
            go.Scatter(x=years, y=plot_data['annual_rabid_dogs_no_vac'],
                      mode='lines', name='No vaccination', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=years, y=plot_data['annual_rabid_dogs_vac'],
                      mode='lines', name='Annual vaccination', line=dict(color='green')),
            row=1, col=1
        )
        
        # Top right: Cumulative canine rabies cases
        fig.add_trace(
            go.Scatter(x=years, y=plot_data['cumul_rabies_cases_no_vac'],
                      mode='lines', name='No vaccination', line=dict(color='red'), showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=years, y=plot_data['cumul_rabies_cases_vac'],
                      mode='lines', name='Annual vaccination', line=dict(color='green'), showlegend=False),
            row=1, col=2
        )
        
        # Bottom left: Annual human deaths
        fig.add_trace(
            go.Scatter(x=years, y=plot_data['annual_human_deaths_no_vac'],
                      mode='lines', name='No vaccination', line=dict(color='red'), showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=years, y=plot_data['annual_human_deaths_vac'],
                      mode='lines', name='Annual vaccination', line=dict(color='green'), showlegend=False),
            row=2, col=1
        )
        
        # Bottom right: Cumulative human deaths
        fig.add_trace(
            go.Scatter(x=years, y=plot_data['cumul_human_deaths_no_vac'],
                      mode='lines', name='No vaccination', line=dict(color='red'), showlegend=False),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=years, y=plot_data['cumul_human_deaths_vac'],
                      mode='lines', name='Annual vaccination', line=dict(color='green'), showlegend=False),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text="Rabies Economic Model Results - 30 Year Projection",
            showlegend=True
        )
        
        # Update x-axis labels
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Years", row=i, col=j)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the exact summary tables
        st.subheader("Program Summary Tables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**No Annual Vaccination Scenario**")
            if 'no_annual_summary' in summary_data:
                st.dataframe(summary_data['no_annual_summary'])
            
        with col2:
            st.write("**Annual Vaccination Scenario**")
            if 'annual_summary' in summary_data:
                st.dataframe(summary_data['annual_summary'])
        
        # Show data statistics
        st.subheader("Data Summary")
        st.write(f"**Total years analyzed:** {len(years)}")
        st.write(f"**Annual rabid dogs (no vac) - Min:** {min(plot_data['annual_rabid_dogs_no_vac']):.1f}, **Max:** {max(plot_data['annual_rabid_dogs_no_vac']):.1f}")
        st.write(f"**Annual rabid dogs (with vac) - Min:** {min(plot_data['annual_rabid_dogs_vac']):.1f}, **Max:** {max(plot_data['annual_rabid_dogs_vac']):.1f}")
        st.write(f"**Cumulative human deaths (no vac) at year 30:** {plot_data['cumul_human_deaths_no_vac'][-1]:.1f}")
        st.write(f"**Cumulative human deaths (with vac) at year 30:** {plot_data['cumul_human_deaths_vac'][-1]:.1f}")
        
    except Exception as e:
        st.error(f"Error loading saved results: {e}")
        st.write("Please run the model first to generate results.")

else:
    st.warning("No saved results found. Please click 'Run Model' to generate fresh results.")
    st.info("The model will run initial_run.py and save the exact results for display here.")