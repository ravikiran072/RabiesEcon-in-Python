# Archive Folder# Archive Folder# Model Parameters Documentation



This folder contains archived/obsolete files from the app development process.



## Archived FilesThis folder contains archived/obsolete files from the app development process.## Overview



### Old App Versions:

- `main_app.py` - Original main application file

- `main_app_full_model.py` - Full model version of main app## Archived FilesThe `app/model_parameters.py` module provides a comprehensive parameter management system for the Rabies Economic Analysis that maintains **exact fidelity** to the original `model_parameters.xlsx` file while providing flexibility for Streamlit app customization.

- `simple_app.py` - Simple wrapper-based app

- `working_app.py` - Working version during development

- `wrapper_app.py` - Wrapper-function based app

- `exact_results_app.py` - App focused on exact results matching### Old App Versions:## Key Features



### Utilities and Parameters:- `main_app.py` - Original main application file

- `streamlit_utils.py` - Utility functions for Streamlit apps

- `model_parameters.py` - Parameter handling functions# Archive Folder

- `test_parameters.py` - Parameter testing utilities

This folder contains archived/obsolete files from the app development process.

### Documentation:

- `OLD_README.md` - Old README file for the app folder (renamed from README.md)## Archived Files



### Configuration:### Old App Versions:

- `.gitignore` - Git ignore rules for archived files and temporary files- `main_app.py` - Original main application file

- `main_app_full_model.py` - Full model version of main app

### Cache:- `simple_app.py` - Simple wrapper-based app

- `__pycache__/` - Compiled Python bytecode cache- `working_app.py` - Working version during development

- `wrapper_app.py` - Wrapper-function based app

## Current Active Files (in parent app/ folder):- `exact_results_app.py` - App focused on exact results matching

- `comprehensive_rabies_app.py` - **Main application** - Complete internal analysis with visualization

- `COMPREHENSIVE_APP_README.md` - Documentation for the main application- `simple_app.py` - Simple wrapper-based app- All parameter values match the Excel file exactly (verified by automated tests)

- `__init__.py` - Python package initialization file

- `working_app.py` - Working version during development- Maintains precision of original calculations

## Notes:

- All archived files represent previous iterations of the development process- `wrapper_app.py` - Wrapper-function based app- Handles missing parameters gracefully with documented fallback values

- The current working application is `comprehensive_rabies_app.py` in the parent directory

- These files are kept for reference but are no longer actively maintained- `exact_results_app.py` - App focused on exact results matching

- The `.gitignore` file helps prevent tracking of temporary and cache files

- Created on: October 21, 2025### 📊 **Parameter Categories**

### Utilities and Parameters:1. **Variable Parameters**: Can be modified by users (geographic, epidemiological, economic)

- `streamlit_utils.py` - Utility functions for Streamlit apps2. **Constant Parameters**: Fixed biological/demographic values

- `model_parameters.py` - Parameter handling functions3. **Calculated Parameters**: Automatically derived from other parameters

- `test_parameters.py` - Parameter testing utilities

### 🎛️ **Streamlit Integration**

### Documentation:- Ready-to-use widgets for parameter input

- `OLD_README.md` - Old README file for the app folder (renamed from README.md)- Validation and warning system

- Scenario comparison tools

### Cache:- Export functionality

- `__pycache__/` - Compiled Python bytecode cache

## Usage Examples

## Current Active Files (in parent app/ folder):

- `comprehensive_rabies_app.py` - **Main application** - Complete internal analysis with visualization### Basic Usage

- `COMPREHENSIVE_APP_README.md` - Documentation for the main application

- `__init__.py` - Python package initialization file```python

from app.model_parameters import ModelParameters, load_parameters_from_excel

## Notes:

- All archived files represent previous iterations of the development process# Load parameters from Excel (maintains exact fidelity)

- The current working application is `comprehensive_rabies_app.py` in the parent directoryparams = load_parameters_from_excel()

- These files are kept for reference but are no longer actively maintained

- Created on: October 21, 2025# Access parameters
print(f"Program Area: {params.Km2_of_program_area:,.0f} km²")
print(f"Human Population: {params.Human_population:,.0f}")
print(f"R0: {params.R0_dog_to_dog:.6f}")

# Calculated parameters are automatically updated
print(f"Humans per km²: {params.Humans_per_km2:,.0f}")
print(f"Cost per suspect exposure: ${params.cost_per_suspect_exposure:.2f}")
```

### Updating Parameters

```python
# Update a parameter (recalculates derived values automatically)
params.update_parameter("Km2_of_program_area", 20000.0)
print(f"Updated Humans per km²: {params.Humans_per_km2:,.0f}")

# Direct assignment also works
params.Human_population = 15000000
params._calculate_derived_parameters()  # Manual recalculation if needed
```

### Creating Custom Scenarios

```python
from app.model_parameters import create_parameter_scenarios

# Get predefined scenarios
scenarios = create_parameter_scenarios()
urban_params = scenarios["Urban High-Density"]

# Create custom scenario
custom_params = ModelParameters(
    Km2_of_program_area=5000.0,
    Human_population=1000000.0,
    Free_roaming_dogs_per_km2=40.0,
    vaccination_cost_per_dog=3.0
)
```

### Streamlit Integration

```python
import streamlit as st
from app.streamlit_utils import ParameterManager, create_parameter_dashboard

# Simple parameter management
param_manager = ParameterManager()
current_params = param_manager.render_parameter_sidebar()

# Full parameter dashboard
if __name__ == "__main__":
    create_parameter_dashboard()
```

## Parameter Categories

### Variable Parameters (User Modifiable)

#### Geographic & Program
- `Km2_of_program_area`: Program area in km²
- `Human_population`: Total human population
- `Humans_per_free_roaming_dog`: Human to dog ratio
- `Free_roaming_dogs_per_km2`: Dog density

#### Epidemiological
- `Annual_dog_bite_risk`: Annual dog bite risk probability
- `Probability_of_rabies_in_biting_dogs`: Rabies probability in biting dogs
- `Probability_of_human_developing_rabies`: Human rabies development probability
- `R0_dog_to_dog`: Basic reproduction number

#### Economic
- `vaccination_cost_per_dog`: Cost per dog vaccination
- `pep_and_other_costs`: PEP treatment cost
- `pep_prob_no_campaign`: PEP probability (no vaccination program)
- `pep_prob_annual_campaign`: PEP probability (with vaccination)

#### Suspect Exposure & Costs
- Various probabilities and costs for quarantine, testing, and investigation

### Constant Parameters (Fixed)
- `Human_birth`: Birth rate (17 per 1,000)
- `Human_life_expectancy`: Life expectancy (65 years)
- `Dog_birth_rate_per_1000_dogs`: Dog birth rate (750 per 1,000)
- `Dog_life_expectancy`: Dog life expectancy (2.5 years)
- `YLL`: Years of Life Lost per death (26.32 years)
- `Dog_Human_transmission_rate`: Transmission rate (0.0000510)

### Calculated Parameters (Automatic)
- `Humans_per_km2`: Population density
- `Free_roaming_dog_population`: Total dog population
- `cost_per_suspect_exposure`: Average suspect exposure cost

## Integration with Main Model

The parameter system is designed to integrate seamlessly with your existing model:

```python
from app.model_parameters import load_parameters_from_excel

# Load parameters for model
params = load_parameters_from_excel()

# Use in model (example from initial_run.py)
Km2_of_program_area = params.Km2_of_program_area
Human_population = params.Human_population
Human_birth = params.Human_birth
# ... etc for all parameters

# Calculate model-specific derived values
Humans_per_km2 = params.Humans_per_km2  # Already calculated
b_h = (params.Human_birth / 52) / 1000  # Human birth rate
```

## Validation and Safety

The system includes comprehensive validation:

```python
# Parameter validation
warnings = param_manager.validate_parameters()
for warning in warnings:
    print(warning)

# Example warnings:
# "⚠️ Very low human:dog ratio - may not be realistic"
# "🚨 Very high dog density - verify data"
# "ℹ️ R0 < 1: Disease will naturally fade out"
```

## Export and Import

```python
# Export current configuration
success = export_parameters_to_excel(params, "custom_parameters.xlsx")

# Load from custom file
custom_params = load_parameters_from_excel("custom_parameters.xlsx")
```

## Testing

Run the test suite to verify everything works:

```bash
python app/test_parameters.py
```

The tests verify:
- ✅ Parameter loading from Excel
- ✅ Calculated parameter accuracy
- ✅ Parameter update functionality
- ✅ Predefined scenarios
- ✅ Parameter info structure
- ✅ **Exact fidelity to Excel values**

## Best Practices

### For Streamlit App Development
1. Use `ParameterManager` for session state management
2. Always validate parameters before running models
3. Provide clear descriptions and units in widgets
4. Use scenarios for quick setup

### For Model Integration
1. Load parameters once at startup
2. Use the parameter object throughout the model
3. Don't hard-code values - always use parameter attributes
4. Maintain parameter updates through the object methods

### For Maintenance
1. Test parameter changes with the test suite
2. Update fallback values when Excel file changes
3. Document any new parameters in both places
4. Verify exact fidelity after any changes

## Architecture Benefits

1. **Single Source of Truth**: Excel file remains authoritative
2. **Type Safety**: Dataclass with proper types
3. **Automatic Calculations**: Derived parameters always consistent
4. **Streamlit Ready**: Built-in widget support
5. **Extensible**: Easy to add new parameters or categories
6. **Testable**: Comprehensive test suite ensures reliability
7. **User Friendly**: Clear categories and validation

This system provides the perfect balance between maintaining exact fidelity to your Excel model while providing the flexibility needed for a sophisticated Streamlit application.