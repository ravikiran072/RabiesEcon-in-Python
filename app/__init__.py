"""
Rabies Economic Analysis Streamlit App
=====================================

This package contains the Streamlit application for rabies economic analysis,
including comprehensive parameter management and model integration.

Key Components:
- model_parameters.py: Core parameter management with Excel fidelity
- streamlit_utils.py: Streamlit-specific utilities and widgets
- main_app.py: Main Streamlit application (to be created)

Usage:
    from app.model_parameters import ModelParameters, load_parameters_from_excel
    from app.streamlit_utils import ParameterManager, create_parameter_dashboard
"""

from .model_parameters import (
    ModelParameters,
    load_parameters_from_excel,
    create_parameter_scenarios,
    export_parameters_to_excel,
    DEFAULT_PARAMETERS
)

# Only import Streamlit utilities if streamlit is available
try:
    from .streamlit_utils import ParameterManager, create_parameter_dashboard
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False
    ParameterManager = None
    create_parameter_dashboard = None

__version__ = "1.0.0"
__author__ = "Rabies Economic Analysis Team"

__all__ = [
    "ModelParameters",
    "load_parameters_from_excel", 
    "create_parameter_scenarios",
    "export_parameters_to_excel",
    "DEFAULT_PARAMETERS",
    "ParameterManager",
    "create_parameter_dashboard"
]