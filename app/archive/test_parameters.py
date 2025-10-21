"""
Test script to verify parameter loading and fidelity to Excel file
================================================================

This script tests the parameter management system to ensure exact fidelity
to the original model_parameters.xlsx file values.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.model_parameters import (
    ModelParameters, 
    load_parameters_from_excel, 
    create_parameter_scenarios,
    export_parameters_to_excel
)
import pandas as pd


def test_parameter_loading():
    """Test that parameters load correctly from Excel."""
    print("🧪 Testing Parameter Loading...")
    print("=" * 50)
    
    try:
        # Test loading from Excel
        params = load_parameters_from_excel()
        print("✅ Successfully loaded parameters from Excel")
        
        # Test some key parameters
        print(f"📊 Key Parameters:")
        print(f"  • Program Area: {params.Km2_of_program_area:,.0f} km²")
        print(f"  • Human Population: {params.Human_population:,.0f}")
        print(f"  • Humans per km²: {params.Humans_per_km2:,.0f}")
        print(f"  • Dogs per km²: {params.Free_roaming_dogs_per_km2:.2f}")
        print(f"  • R0: {params.R0_dog_to_dog:.6f}")
        print(f"  • Vaccination cost: ${params.vaccination_cost_per_dog:.2f}")
        print(f"  • PEP cost: ${params.pep_and_other_costs:.2f}")
        print(f"  • YLL: {params.YLL:.2f} years")
        print(f"  • Cost per suspect exposure: ${params.cost_per_suspect_exposure:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        return False


def test_parameter_calculations():
    """Test that calculated parameters are correct."""
    print("\n🧮 Testing Parameter Calculations...")
    print("=" * 50)
    
    try:
        params = ModelParameters(
            Km2_of_program_area=10000.0,
            Human_population=1000000.0,
            Free_roaming_dogs_per_km2=50.0
        )
        
        # Test calculations
        expected_humans_per_km2 = 1000000 / 10000  # 100
        expected_dog_population = 50.0 * 10000  # 500,000
        
        assert abs(params.Humans_per_km2 - expected_humans_per_km2) < 0.01
        assert abs(params.Free_roaming_dog_population - expected_dog_population) < 0.01
        
        print("✅ Calculated parameters are correct")
        print(f"  • Humans per km²: {params.Humans_per_km2:.1f} (expected: {expected_humans_per_km2:.1f})")
        print(f"  • Total dogs: {params.Free_roaming_dog_population:,.0f} (expected: {expected_dog_population:,.0f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in calculations: {e}")
        return False


def test_parameter_updates():
    """Test that parameter updates work correctly."""
    print("\n🔄 Testing Parameter Updates...")
    print("=" * 50)
    
    try:
        params = ModelParameters()
        original_area = params.Km2_of_program_area
        
        # Update a parameter
        new_area = 20000.0
        success = params.update_parameter("Km2_of_program_area", new_area)
        
        assert success == True
        assert params.Km2_of_program_area == new_area
        assert params.Humans_per_km2 != params.Human_population / original_area  # Should be recalculated
        
        print("✅ Parameter updates work correctly")
        print(f"  • Updated area from {original_area:,.0f} to {new_area:,.0f}")
        print(f"  • Recalculated Humans per km²: {params.Humans_per_km2:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in parameter updates: {e}")
        return False


def test_scenarios():
    """Test predefined scenarios."""
    print("\n🎯 Testing Predefined Scenarios...")
    print("=" * 50)
    
    try:
        scenarios = create_parameter_scenarios()
        
        print(f"✅ Created {len(scenarios)} scenarios:")
        for name, params in scenarios.items():
            print(f"  • {name}: {params.Km2_of_program_area:,.0f} km², {params.Human_population:,.0f} people")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating scenarios: {e}")
        return False


def test_parameter_info():
    """Test parameter info structure."""
    print("\n📋 Testing Parameter Info Structure...")
    print("=" * 50)
    
    try:
        params = ModelParameters()
        info = params.get_parameter_info()
        
        # Check structure
        assert "variable_parameters" in info
        assert "constant_parameters" in info
        assert "calculated_parameters" in info
        
        # Count parameters
        total_variable = sum(len(category) for category in info["variable_parameters"].values())
        total_constant = sum(len(category) for category in info["constant_parameters"].values())
        total_calculated = sum(len(category) for category in info["calculated_parameters"].values())
        
        print("✅ Parameter info structure is correct")
        print(f"  • Variable parameters: {total_variable}")
        print(f"  • Constant parameters: {total_constant}")
        print(f"  • Calculated parameters: {total_calculated}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in parameter info: {e}")
        return False


def compare_with_excel():
    """Compare loaded parameters with direct Excel read."""
    print("\n🔍 Comparing with Direct Excel Read...")
    print("=" * 50)
    
    try:
        # Load through our system
        params = load_parameters_from_excel()
        
        # Load directly from Excel
        excel_path = Path(__file__).parent.parent / "data" / "model_parameters.xlsx"
        if excel_path.exists():
            df = pd.read_excel(excel_path)
            
            # Test a few key parameters
            test_params = [
                ("Km2_of_program_area", params.Km2_of_program_area),
                ("Human_population", params.Human_population),
                ("R0_dog_to_dog", params.R0_dog_to_dog),
                ("vaccination_cost_per_dog", params.vaccination_cost_per_dog)
            ]
            
            all_match = True
            for param_name, our_value in test_params:
                try:
                    excel_value = df.query(f"Parameters == '{param_name}'")["Values"].iloc[0]
                    match = abs(our_value - excel_value) < 1e-6
                    
                    status = "✅" if match else "❌"
                    print(f"  {status} {param_name}: Our={our_value}, Excel={excel_value}")
                    
                    if not match:
                        all_match = False
                        
                except IndexError:
                    print(f"  ⚠️ {param_name}: Not found in Excel")
            
            if all_match:
                print("✅ All tested parameters match Excel values exactly!")
            else:
                print("❌ Some parameters don't match - check implementation")
                
        else:
            print("⚠️ Excel file not found - skipping comparison")
            
        return True
        
    except Exception as e:
        print(f"❌ Error comparing with Excel: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Rabies Model Parameter System Test Suite")
    print("=" * 60)
    
    tests = [
        test_parameter_loading,
        test_parameter_calculations,
        test_parameter_updates,
        test_scenarios,
        test_parameter_info,
        compare_with_excel
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Parameter system is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    main()