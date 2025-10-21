"""
Streamlit Integration Utilities for Model Parameters
===================================================

This module provides Streamlit-specific utilities for the parameter management system,
including widgets, validation, and display components.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Tuple, Optional, List
from .model_parameters import ModelParameters, load_parameters_from_excel, create_parameter_scenarios


class ParameterManager:
    """Streamlit interface manager for model parameters."""
    
    def __init__(self, session_key: str = "model_params"):
        self.session_key = session_key
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state with default parameters."""
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = load_parameters_from_excel()
    
    @property
    def params(self) -> ModelParameters:
        """Get current parameters from session state."""
        return st.session_state[self.session_key]
    
    def update_params(self, new_params: ModelParameters):
        """Update parameters in session state."""
        st.session_state[self.session_key] = new_params
    
    def render_parameter_sidebar(self) -> ModelParameters:
        """
        Render parameter controls in Streamlit sidebar.
        
        Returns:
            Updated ModelParameters instance
        """
        st.sidebar.header("🎛️ Model Parameters")
        
        # Scenario selection
        scenarios = create_parameter_scenarios()
        scenario_names = list(scenarios.keys())
        
        selected_scenario = st.sidebar.selectbox(
            "📋 Select Scenario:",
            scenario_names,
            index=0,
            help="Choose a predefined scenario or 'Custom' to modify parameters manually"
        )
        
        if selected_scenario != "Default (Excel Values)" or "current_scenario" not in st.session_state:
            st.session_state.current_scenario = selected_scenario
            self.update_params(scenarios[selected_scenario])
        
        # Parameter categories
        param_info = self.params.get_parameter_info()
        
        # Variable parameters (user can modify)
        st.sidebar.subheader("📊 Adjustable Parameters")
        
        updated_params = {}
        
        for category_name, parameters in param_info["variable_parameters"].items():
            with st.sidebar.expander(f"📁 {category_name}"):
                for param_name, param_data in parameters.items():
                    if param_data["unit"] == "probability":
                        # Use percentage slider for probabilities
                        if param_data["value"] <= 1.0:
                            display_value = param_data["value"] * 100
                            min_val = param_data["min"] * 100
                            max_val = param_data["max"] * 100
                            step_val = param_data["step"] * 100
                        else:
                            display_value = param_data["value"]
                            min_val = param_data["min"]
                            max_val = param_data["max"]
                            step_val = param_data["step"]
                        
                        new_value = st.slider(
                            param_data["description"],
                            min_value=min_val,
                            max_value=max_val,
                            value=display_value,
                            step=step_val,
                            key=f"slider_{param_name}",
                            help=f"Unit: {param_data['unit']}"
                        )
                        
                        # Convert back to decimal if needed
                        if param_data["unit"] == "probability" and param_data["value"] <= 1.0:
                            updated_params[param_name] = new_value / 100
                        else:
                            updated_params[param_name] = new_value
                            
                    else:
                        # Regular number input or slider
                        if param_data["unit"] in ["$", "persons", "dogs", "km²"]:
                            new_value = st.number_input(
                                param_data["description"],
                                min_value=param_data["min"],
                                max_value=param_data["max"],
                                value=param_data["value"],
                                step=param_data["step"],
                                key=f"number_{param_name}",
                                help=f"Unit: {param_data['unit']}"
                            )
                        else:
                            new_value = st.slider(
                                param_data["description"],
                                min_value=param_data["min"],
                                max_value=param_data["max"],
                                value=param_data["value"],
                                step=param_data["step"],
                                key=f"slider_{param_name}",
                                help=f"Unit: {param_data['unit']}"
                            )
                        
                        updated_params[param_name] = new_value
        
        # Update parameters if any changed
        params_changed = False
        for param_name, new_value in updated_params.items():
            if getattr(self.params, param_name) != new_value:
                self.params.update_parameter(param_name, new_value)
                params_changed = True
        
        if params_changed:
            st.sidebar.success("✅ Parameters updated!")
        
        # Display constant parameters
        with st.sidebar.expander("📋 View Constant Parameters"):
            st.write("**Human Demographics:**")
            st.write(f"• Birth rate: {self.params.Human_birth} per 1,000")
            st.write(f"• Life expectancy: {self.params.Human_life_expectancy} years")
            
            st.write("**Dog Demographics:**")
            st.write(f"• Birth rate: {self.params.Dog_birth_rate_per_1000_dogs} per 1,000")
            st.write(f"• Life expectancy: {self.params.Dog_life_expectancy} years")
            
            st.write("**Disease Parameters:**")
            st.write(f"• Years of Life Lost: {self.params.YLL} years")
            st.write(f"• Dog-Human transmission: {self.params.Dog_Human_transmission_rate:.8f}")
        
        return self.params
    
    def render_calculated_parameters(self):
        """Display calculated parameters in main area."""
        st.header("📊 Calculated Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Human Density", 
                f"{self.params.Humans_per_km2:,.0f} per km²",
                help="Calculated from Human Population / Program Area"
            )
            
            st.metric(
                "Total Dogs", 
                f"{self.params.Free_roaming_dog_population:,.0f}",
                help="Calculated from Dogs per km² × Program Area"
            )
        
        with col2:
            human_dog_ratio = self.params.Human_population / self.params.Free_roaming_dog_population
            st.metric(
                "Human:Dog Ratio", 
                f"{human_dog_ratio:.1f}:1",
                help="Ratio of humans to free-roaming dogs"
            )
            
            st.metric(
                "Cost per Suspect Exposure", 
                f"${self.params.cost_per_suspect_exposure:.2f}",
                help="Weighted average of quarantine, testing, and investigation costs"
            )
        
        with col3:
            program_coverage = (self.params.Km2_of_program_area / 100000) * 100  # Assuming country size
            st.metric(
                "Population Density",
                f"{self.params.Human_population / self.params.Km2_of_program_area:.0f} persons/km²"
            )
            
            dogs_per_1000_humans = (self.params.Free_roaming_dog_population / self.params.Human_population) * 1000
            st.metric(
                "Dogs per 1,000 Humans",
                f"{dogs_per_1000_humans:.1f}",
                help="Free-roaming dogs per 1,000 humans"
            )
    
    def render_parameter_comparison(self, scenario_a: str, scenario_b: str):
        """
        Render side-by-side parameter comparison.
        
        Args:
            scenario_a: Name of first scenario
            scenario_b: Name of second scenario
        """
        scenarios = create_parameter_scenarios()
        
        if scenario_a in scenarios and scenario_b in scenarios:
            params_a = scenarios[scenario_a]
            params_b = scenarios[scenario_b]
            
            st.subheader(f"📊 Comparison: {scenario_a} vs {scenario_b}")
            
            # Create comparison DataFrame
            comparison_data = []
            param_info_a = params_a.get_parameter_info()
            param_info_b = params_b.get_parameter_info()
            
            for category_name, categories in param_info_a["variable_parameters"].items():
                for param_name, param_data_a in categories.items():
                    param_data_b = param_info_b["variable_parameters"][category_name][param_name]
                    
                    difference = param_data_b["value"] - param_data_a["value"]
                    percent_change = (difference / param_data_a["value"]) * 100 if param_data_a["value"] != 0 else 0
                    
                    comparison_data.append({
                        "Parameter": param_data_a["description"],
                        f"{scenario_a}": param_data_a["value"],
                        f"{scenario_b}": param_data_b["value"],
                        "Difference": difference,
                        "% Change": percent_change,
                        "Unit": param_data_a["unit"]
                    })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Style the dataframe
            def highlight_differences(row):
                if abs(row["% Change"]) > 10:
                    return ['background-color: #ffeeee'] * len(row)
                elif abs(row["% Change"]) > 5:
                    return ['background-color: #fff8e1'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = df_comparison.style.apply(highlight_differences, axis=1)
            st.dataframe(styled_df, use_container_width=True)
    
    def render_sensitivity_analysis(self, parameter_name: str, range_percent: float = 20):
        """
        Render sensitivity analysis for a specific parameter.
        
        Args:
            parameter_name: Name of parameter to analyze
            range_percent: Percentage range around current value
        """
        if not hasattr(self.params, parameter_name):
            st.error(f"Parameter '{parameter_name}' not found!")
            return
        
        current_value = getattr(self.params, parameter_name)
        min_val = current_value * (1 - range_percent / 100)
        max_val = current_value * (1 + range_percent / 100)
        
        # Create range of values
        values = []
        results = []
        
        n_points = 20
        for i in range(n_points):
            test_value = min_val + (max_val - min_val) * i / (n_points - 1)
            values.append(test_value)
            
            # Create temporary params for testing
            temp_params = ModelParameters()
            temp_params.update_parameter(parameter_name, test_value)
            
            # Calculate a simple outcome metric (you can expand this)
            if parameter_name == "R0_dog_to_dog":
                outcome = test_value  # Simple example
            elif parameter_name == "vaccination_cost_per_dog":
                outcome = test_value * temp_params.Free_roaming_dog_population
            else:
                outcome = test_value  # Default
            
            results.append(outcome)
        
        # Create sensitivity plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=values,
            y=results,
            mode='lines+markers',
            name=f'Sensitivity of {parameter_name}',
            line=dict(width=3)
        ))
        
        # Add current value line
        fig.add_vline(
            x=current_value,
            line_dash="dash",
            line_color="red",
            annotation_text="Current Value"
        )
        
        fig.update_layout(
            title=f"Sensitivity Analysis: {parameter_name}",
            xaxis_title=parameter_name,
            yaxis_title="Outcome Metric",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def export_current_parameters(self) -> bytes:
        """
        Export current parameters to Excel format.
        
        Returns:
            Excel file as bytes
        """
        import io
        from .model_parameters import export_parameters_to_excel
        
        output = io.BytesIO()
        
        # Create temporary file path
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            if export_parameters_to_excel(self.params, tmp_file.name):
                with open(tmp_file.name, 'rb') as f:
                    output.write(f.read())
                output.seek(0)
                return output.getvalue()
        
        return b""
    
    def validate_parameters(self) -> List[str]:
        """
        Validate current parameters and return list of warnings/errors.
        
        Returns:
            List of validation messages
        """
        warnings = []
        
        # Check for unrealistic ratios
        if self.params.Humans_per_free_roaming_dog < 5:
            warnings.append("⚠️ Very low human:dog ratio - may not be realistic")
        
        if self.params.Humans_per_free_roaming_dog > 30:
            warnings.append("⚠️ Very high human:dog ratio - check dog population estimates")
        
        # Check dog density
        if self.params.Free_roaming_dogs_per_km2 > 100:
            warnings.append("🚨 Extremely high dog density - verify data")
        
        # Check R0
        if self.params.R0_dog_to_dog < 1.0:
            warnings.append("ℹ️ R0 < 1: Disease will naturally fade out")
        elif self.params.R0_dog_to_dog > 2.5:
            warnings.append("🚨 Very high R0: Rapid disease spread expected")
        
        # Check economic parameters
        if self.params.vaccination_cost_per_dog > self.params.pep_and_other_costs * 0.5:
            warnings.append("💰 Vaccination cost is high relative to PEP cost")
        
        return warnings


def create_parameter_dashboard():
    """Create a comprehensive parameter dashboard for Streamlit."""
    
    st.title("🎛️ Model Parameters Dashboard")
    
    # Initialize parameter manager
    param_manager = ParameterManager()
    
    # Sidebar for parameter controls
    current_params = param_manager.render_parameter_sidebar()
    
    # Main area tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Analysis", "⚖️ Comparison", "📤 Export"])
    
    with tab1:
        param_manager.render_calculated_parameters()
        
        # Validation warnings
        warnings = param_manager.validate_parameters()
        if warnings:
            st.subheader("⚠️ Parameter Validation")
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("✅ All parameters look reasonable!")
    
    with tab2:
        st.subheader("🔍 Sensitivity Analysis")
        
        # Parameter selection for sensitivity analysis
        variable_params = list(current_params.get_parameter_info()["variable_parameters"].keys())
        all_params = []
        for category in current_params.get_parameter_info()["variable_parameters"].values():
            all_params.extend(category.keys())
        
        selected_param = st.selectbox(
            "Select parameter for sensitivity analysis:",
            all_params,
            help="Choose a parameter to see how changes affect outcomes"
        )
        
        range_percent = st.slider(
            "Analysis range (% around current value):",
            5, 50, 20,
            help="Percentage range to test around the current parameter value"
        )
        
        if st.button("Run Sensitivity Analysis"):
            param_manager.render_sensitivity_analysis(selected_param, range_percent)
    
    with tab3:
        st.subheader("⚖️ Scenario Comparison")
        
        scenarios = list(create_parameter_scenarios().keys())
        
        col1, col2 = st.columns(2)
        with col1:
            scenario_a = st.selectbox("Scenario A:", scenarios, index=0)
        with col2:
            scenario_b = st.selectbox("Scenario B:", scenarios, index=1 if len(scenarios) > 1 else 0)
        
        if scenario_a != scenario_b:
            param_manager.render_parameter_comparison(scenario_a, scenario_b)
    
    with tab4:
        st.subheader("📤 Export Parameters")
        
        st.write("Download current parameter configuration:")
        
        # Export button
        excel_data = param_manager.export_current_parameters()
        if excel_data:
            st.download_button(
                label="📥 Download Parameters (Excel)",
                data=excel_data,
                file_name="model_parameters_custom.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Parameter summary
        st.subheader("📋 Current Configuration Summary")
        param_info = current_params.get_parameter_info()
        
        summary_data = []
        for category_name, categories in param_info["variable_parameters"].items():
            for param_name, param_data in categories.items():
                summary_data.append({
                    "Category": category_name,
                    "Parameter": param_data["description"],
                    "Value": param_data["value"],
                    "Unit": param_data["unit"]
                })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
    
    return param_manager