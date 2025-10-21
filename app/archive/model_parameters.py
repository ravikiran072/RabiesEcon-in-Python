"""
Model Parameters Management for Rabies Economic Analysis
======================================================

This module provides a comprehensive interface for managing model parameters
while maintaining exact fidelity to the original model_parameters.xlsx file.

The parameters are categorized into:
1. Variable parameters (user can modify)
2. Constant parameters (fixed values)
3. Calculated parameters (derived from other parameters)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import warnings


@dataclass
class ModelParameters:
    """
    Comprehensive model parameters class that maintains exact Excel file values
    while providing flexibility for Streamlit app customization.
    """
    
    # ==================== VARIABLE PARAMETERS ====================
    # These can be modified by users in the Streamlit app
    
    # Geographic & Program Parameters
    Km2_of_program_area: float = 17960.0
    Human_population: float = 13125164.0
    
    # Dog Population Parameters  
    Humans_per_free_roaming_dog: float = 15.6
    Free_roaming_dogs_per_km2: float = 46.84613957
    
    # Epidemiological Parameters
    Annual_dog_bite_risk: float = 0.03
    Probability_of_rabies_in_biting_dogs: float = 0.01
    Probability_of_human_developing_rabies: float = 0.17
    R0_dog_to_dog: float = 1.307935326
    
    # Economic Parameters
    vaccination_cost_per_dog: float = 2.45
    pep_and_other_costs: float = 17.40
    pep_prob_no_campaign: float = 0.25
    pep_prob_annual_campaign: float = 0.5
    
    # Suspect Exposure Parameters
    inflation_factor_for_the_suspect_exposure: float = 10.0
    post_elimination_pep_reduction: float = 0.1
    
    # Suspect Animal Cost Parameters
    quarantined_animal_prob: float = 0.0008
    quarantined_animal_cost: float = 140.00
    lab_test_prob: float = 0.011333333
    lab_test_cost: float = 26.49
    bite_investigation_prob: float = 0.466666667
    bite_investigation_cost: float = 3.25
    
    # ==================== CONSTANT PARAMETERS ====================
    # These should NOT be modified (fixed biological/demographic constants)
    
    # Human Demographics (Constants)
    Human_birth: float = 17.0  # per 1,000 population
    Human_life_expectancy: float = 65.0  # years
    
    # Dog Demographics (Constants)
    Dog_birth_rate_per_1000_dogs: float = 750.0
    Dog_life_expectancy: float = 2.5  # years
    
    # Disease Parameters (Constants)
    YLL: float = 26.32  # Years of Life Lost per death
    
    # Model Constants (should not change)
    Dog_Human_transmission_rate: float = 0.0000510
    
    # ==================== CALCULATED PARAMETERS ====================
    # These are computed from other parameters
    
    Humans_per_km2: float = field(init=False)
    Free_roaming_dog_population: float = field(init=False)
    cost_per_suspect_exposure: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        self._calculate_derived_parameters()
    
    def _calculate_derived_parameters(self):
        """Calculate all derived parameters based on current values."""
        # Geographic calculations
        self.Humans_per_km2 = self.Human_population / self.Km2_of_program_area
        self.Free_roaming_dog_population = self.Free_roaming_dogs_per_km2 * self.Km2_of_program_area
        
        # Cost calculations
        self.cost_per_suspect_exposure = (
            self.quarantined_animal_prob * self.quarantined_animal_cost +
            self.lab_test_prob * self.lab_test_cost +
            self.bite_investigation_prob * self.bite_investigation_cost
        )
    
    def update_parameter(self, parameter_name: str, new_value: float) -> bool:
        """
        Safely update a parameter and recalculate derived values.
        
        Args:
            parameter_name: Name of the parameter to update
            new_value: New value for the parameter
            
        Returns:
            True if parameter was updated successfully, False otherwise
        """
        if hasattr(self, parameter_name):
            setattr(self, parameter_name, new_value)
            self._calculate_derived_parameters()
            return True
        else:
            warnings.warn(f"Parameter '{parameter_name}' does not exist.")
            return False
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive parameter information for the Streamlit app.
        
        Returns:
            Dictionary containing parameter categories and metadata
        """
        return {
            "variable_parameters": {
                "Geographic & Program": {
                    "Km2_of_program_area": {
                        "value": self.Km2_of_program_area,
                        "description": "Square kilometers of program area",
                        "unit": "km²",
                        "min": 100,
                        "max": 50000,
                        "step": 100
                    },
                    "Human_population": {
                        "value": self.Human_population,
                        "description": "Total human population in program area",
                        "unit": "persons",
                        "min": 100000,
                        "max": 50000000,
                        "step": 10000
                    },
                    "Humans_per_free_roaming_dog": {
                        "value": self.Humans_per_free_roaming_dog,
                        "description": "Number of humans per free roaming dog (HDR)",
                        "unit": "ratio",
                        "min": 5.0,
                        "max": 30.0,
                        "step": 0.1
                    },
                    "Free_roaming_dogs_per_km2": {
                        "value": self.Free_roaming_dogs_per_km2,
                        "description": "Free roaming dogs per square kilometer",
                        "unit": "dogs/km²",
                        "min": 10.0,
                        "max": 100.0,
                        "step": 0.1
                    }
                },
                "Epidemiological": {
                    "Annual_dog_bite_risk": {
                        "value": self.Annual_dog_bite_risk,
                        "description": "Annual dog bite risk (suggested 1% - 3%)",
                        "unit": "probability",
                        "min": 0.005,
                        "max": 0.05,
                        "step": 0.001
                    },
                    "Probability_of_rabies_in_biting_dogs": {
                        "value": self.Probability_of_rabies_in_biting_dogs,
                        "description": "Probability of rabies in biting dogs (suggested 0.1% - 5%)",
                        "unit": "probability",
                        "min": 0.001,
                        "max": 0.05,
                        "step": 0.001
                    },
                    "Probability_of_human_developing_rabies": {
                        "value": self.Probability_of_human_developing_rabies,
                        "description": "Probability of human developing rabies (suggested 17%)",
                        "unit": "probability",
                        "min": 0.1,
                        "max": 0.3,
                        "step": 0.01
                    },
                    "R0_dog_to_dog": {
                        "value": self.R0_dog_to_dog,
                        "description": "Basic reproduction number for dog-to-dog transmission",
                        "unit": "ratio",
                        "min": 0.5,
                        "max": 3.0,
                        "step": 0.01
                    }
                },
                "Economic": {
                    "vaccination_cost_per_dog": {
                        "value": self.vaccination_cost_per_dog,
                        "description": "Average cost per dog vaccinated",
                        "unit": "$",
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.05
                    },
                    "pep_and_other_costs": {
                        "value": self.pep_and_other_costs,
                        "description": "PEP cost & Other Costs per treatment",
                        "unit": "$",
                        "min": 10.0,
                        "max": 50.0,
                        "step": 0.50
                    },
                    "pep_prob_no_campaign": {
                        "value": self.pep_prob_no_campaign,
                        "description": "Probability of receiving PEP, post-exposure (no Vaccination program)",
                        "unit": "probability",
                        "min": 0.1,
                        "max": 0.5,
                        "step": 0.01
                    },
                    "pep_prob_annual_campaign": {
                        "value": self.pep_prob_annual_campaign,
                        "description": "Probability of receiving PEP, post-exposure (with Vaccination program)",
                        "unit": "probability",
                        "min": 0.3,
                        "max": 0.8,
                        "step": 0.01
                    }
                },
                "Suspect Exposure": {
                    "inflation_factor_for_the_suspect_exposure": {
                        "value": self.inflation_factor_for_the_suspect_exposure,
                        "description": "Inflation factor for suspect exposure (>=1)",
                        "unit": "multiplier",
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1
                    },
                    "post_elimination_pep_reduction": {
                        "value": self.post_elimination_pep_reduction,
                        "description": "Post-Elimination PEP Reduction (%)",
                        "unit": "probability",
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01
                    }
                },
                "Suspect Animal Costs": {
                    "quarantined_animal_prob": {
                        "value": self.quarantined_animal_prob,
                        "description": "Probability of animal quarantine per exposure",
                        "unit": "probability",
                        "min": 0.0001,
                        "max": 0.01,
                        "step": 0.0001
                    },
                    "quarantined_animal_cost": {
                        "value": self.quarantined_animal_cost,
                        "description": "Cost per quarantined animal",
                        "unit": "$",
                        "min": 50.0,
                        "max": 300.0,
                        "step": 1.0
                    },
                    "lab_test_prob": {
                        "value": self.lab_test_prob,
                        "description": "Probability of laboratory testing per exposure",
                        "unit": "probability",
                        "min": 0.001,
                        "max": 0.05,
                        "step": 0.001
                    },
                    "lab_test_cost": {
                        "value": self.lab_test_cost,
                        "description": "Cost per laboratory test",
                        "unit": "$",
                        "min": 10.0,
                        "max": 100.0,
                        "step": 0.50
                    },
                    "bite_investigation_prob": {
                        "value": self.bite_investigation_prob,
                        "description": "Probability of bite investigation per exposure",
                        "unit": "probability",
                        "min": 0.1,
                        "max": 0.8,
                        "step": 0.01
                    },
                    "bite_investigation_cost": {
                        "value": self.bite_investigation_cost,
                        "description": "Cost per bite investigation",
                        "unit": "$",
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.25
                    }
                }
            },
            "constant_parameters": {
                "Human Demographics": {
                    "Human_birth": {
                        "value": self.Human_birth,
                        "description": "Human birth rate per 1,000 population (suggested 17)",
                        "unit": "per 1,000"
                    },
                    "Human_life_expectancy": {
                        "value": self.Human_life_expectancy,
                        "description": "Human life expectancy in years",
                        "unit": "years"
                    }
                },
                "Dog Demographics": {
                    "Dog_birth_rate_per_1000_dogs": {
                        "value": self.Dog_birth_rate_per_1000_dogs,
                        "description": "Dog birth rate per 1,000 dogs (suggested 750)",
                        "unit": "per 1,000"
                    },
                    "Dog_life_expectancy": {
                        "value": self.Dog_life_expectancy,
                        "description": "Dog life expectancy in years",
                        "unit": "years"
                    }
                },
                "Disease Parameters": {
                    "YLL": {
                        "value": self.YLL,
                        "description": "Years of Life Lost (YLL) per death",
                        "unit": "years"
                    },
                    "Dog_Human_transmission_rate": {
                        "value": self.Dog_Human_transmission_rate,
                        "description": "Dog-Human transmission rate (suggested 0.000034)",
                        "unit": "rate"
                    }
                }
            },
            "calculated_parameters": {
                "Derived Values": {
                    "Humans_per_km2": {
                        "value": self.Humans_per_km2,
                        "description": "Calculated humans per km²",
                        "unit": "persons/km²",
                        "formula": "Human_population / Km2_of_program_area"
                    },
                    "Free_roaming_dog_population": {
                        "value": self.Free_roaming_dog_population,
                        "description": "Total free roaming dog population",
                        "unit": "dogs",
                        "formula": "Free_roaming_dogs_per_km2 * Km2_of_program_area"
                    },
                    "cost_per_suspect_exposure": {
                        "value": self.cost_per_suspect_exposure,
                        "description": "Average cost per suspect exposure incident",
                        "unit": "$",
                        "formula": "sum of (probability × cost) for all suspect animal activities"
                    }
                }
            }
        }


def load_parameters_from_excel(excel_path: Optional[str] = None) -> ModelParameters:
    """
    Load parameters from Excel file while maintaining exact fidelity.
    
    Args:
        excel_path: Path to the model_parameters.xlsx file
        
    Returns:
        ModelParameters instance with values from Excel
    """
    if excel_path is None:
        # Default path relative to project root
        excel_path = Path(__file__).parent.parent / "data" / "model_parameters.xlsx"
    
    try:
        model_params_df = pd.read_excel(excel_path)
        
        # Create parameter extraction function for safety
        def get_param_value(param_name: str, default_value: float = None) -> float:
            try:
                return model_params_df.query(f"Parameters == '{param_name}'")["Values"].iloc[0]
            except (IndexError, KeyError):
                if default_value is not None:
                    warnings.warn(f"Parameter '{param_name}' not found, using default: {default_value}")
                    return default_value
                else:
                    raise ValueError(f"Required parameter '{param_name}' not found in Excel file")
        
        # Load all parameters with exact Excel values
        params = ModelParameters(
            # Variable parameters
            Km2_of_program_area=get_param_value("Km2_of_program_area"),
            Human_population=get_param_value("Human_population"),
            Humans_per_free_roaming_dog=get_param_value("Humans_per_free_roaming_dog"),
            Free_roaming_dogs_per_km2=get_param_value("Free_roaming_dogs_per_km2"),
            Annual_dog_bite_risk=get_param_value("Annual_dog_bite_risk"),
            Probability_of_rabies_in_biting_dogs=get_param_value("Probability_of_rabies_in_biting_dogs"),
            Probability_of_human_developing_rabies=get_param_value("Probability_of_human_developing_rabies"),
            R0_dog_to_dog=get_param_value("R0_dog_to_dog"),
            vaccination_cost_per_dog=get_param_value("vaccination_cost_per_dog", 2.45),
            pep_and_other_costs=get_param_value("pep_and_other_costs", 17.40),
            pep_prob_no_campaign=get_param_value("pep_prob_no_campaign", 0.25),
            pep_prob_annual_campaign=get_param_value("pep_prob_annual_campaign", 0.5),
            inflation_factor_for_the_suspect_exposure=get_param_value("inflation_factor_for_the_suspect_exposure"),
            post_elimination_pep_reduction=get_param_value("post_elimination_pep_reduction"),
            quarantined_animal_prob=get_param_value("quarantined_animal_prob", 0.0008),
            quarantined_animal_cost=get_param_value("quarantined_animal_cost", 140.00),
            lab_test_prob=get_param_value("lab_test_prob", 0.011333333),
            lab_test_cost=get_param_value("lab_test_cost", 26.49),
            bite_investigation_prob=get_param_value("bite_investigation_prob", 0.466666667),
            bite_investigation_cost=get_param_value("bite_investigation_cost", 3.25),
            
            # Constant parameters
            Human_birth=get_param_value("Human_birth"),
            Human_life_expectancy=get_param_value("Human_life_expectancy"),
            Dog_birth_rate_per_1000_dogs=get_param_value("Dog_birth_rate_per_1000_dogs"),
            Dog_life_expectancy=get_param_value("Dog_life_expectancy"),
            YLL=get_param_value("YLL", 26.32),
            Dog_Human_transmission_rate=get_param_value("Dog_Human_transmission_rate")
        )
        
        return params
        
    except Exception as e:
        warnings.warn(f"Could not load Excel file: {e}. Using default values.")
        return ModelParameters()  # Return with default values


def create_parameter_scenarios() -> Dict[str, ModelParameters]:
    """
    Create predefined parameter scenarios for common use cases.
    
    Returns:
        Dictionary of scenario names and their parameter configurations
    """
    base_params = load_parameters_from_excel()
    
    scenarios = {
        "Default (Excel Values)": base_params,
        
        "Urban High-Density": ModelParameters(
            Km2_of_program_area=1000.0,
            Human_population=2000000.0,
            Free_roaming_dogs_per_km2=60.0,
            Humans_per_free_roaming_dog=25.0,
            Annual_dog_bite_risk=0.04,
            R0_dog_to_dog=1.5,
            vaccination_cost_per_dog=3.0,
            pep_and_other_costs=20.0
        ),
        
        "Rural Low-Density": ModelParameters(
            Km2_of_program_area=15000.0,
            Human_population=800000.0,
            Free_roaming_dogs_per_km2=25.0,
            Humans_per_free_roaming_dog=12.0,
            Annual_dog_bite_risk=0.02,
            R0_dog_to_dog=1.2,
            vaccination_cost_per_dog=2.0,
            pep_and_other_costs=15.0
        ),
        
        "Island Setting": ModelParameters(
            Km2_of_program_area=500.0,
            Human_population=150000.0,
            Free_roaming_dogs_per_km2=35.0,
            Humans_per_free_roaming_dog=8.0,
            Annual_dog_bite_risk=0.025,
            R0_dog_to_dog=1.8,
            vaccination_cost_per_dog=4.0,
            pep_and_other_costs=25.0
        ),
        
        "Conflict/Emergency Zone": ModelParameters(
            Km2_of_program_area=5000.0,
            Human_population=500000.0,
            Free_roaming_dogs_per_km2=80.0,
            Humans_per_free_roaming_dog=6.0,
            Annual_dog_bite_risk=0.05,
            R0_dog_to_dog=2.0,
            vaccination_cost_per_dog=1.5,
            pep_and_other_costs=30.0
        )
    }
    
    return scenarios


def export_parameters_to_excel(params: ModelParameters, output_path: str) -> bool:
    """
    Export current parameter configuration to Excel format.
    
    Args:
        params: ModelParameters instance to export
        output_path: Path where to save the Excel file
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        # Create DataFrame matching original Excel structure
        param_data = []
        
        # Add all parameters
        param_info = params.get_parameter_info()
        
        for category_name, categories in param_info.items():
            for subcategory_name, parameters in categories.items():
                for param_name, param_data_dict in parameters.items():
                    param_data.append({
                        "Category": category_name,
                        "Type": subcategory_name,
                        "Parameters": param_name,
                        "Values": param_data_dict["value"],
                        "Explanation": param_data_dict["description"]
                    })
        
        df = pd.DataFrame(param_data)
        df.to_excel(output_path, index=False)
        return True
        
    except Exception as e:
        warnings.warn(f"Could not export parameters: {e}")
        return False


# Create a global instance for easy access
DEFAULT_PARAMETERS = load_parameters_from_excel()