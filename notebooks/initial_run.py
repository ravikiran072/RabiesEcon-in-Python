import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time as tm
import os

# Set working directory to project root (one level up from notebooks)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

coverage_data = pd.read_csv("data/coverage_data.csv")
model_parameters = pd.read_excel("data/model_parameters.xlsx")


def get_vaccination_coverage(year, scenario="annual_vaccination"):
    """
    Get vaccination coverage for a specific year and scenario from CSV data
    """
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


def get_pep_coverage(year, scenario="annual_vaccination"):
    """
    Get PEP coverage for a specific year and scenario from CSV data
    """
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


# Initial parameters
Km2_of_program_area = model_parameters.query("Parameters == 'Km2_of_program_area'")[
    "Values"
].iloc[0]
Human_population = model_parameters.query("Parameters == 'Human_population'")[
    "Values"
].iloc[0]
Humans_per_km2 = Human_population / Km2_of_program_area
Human_birth = model_parameters.query("Parameters == 'Human_birth'")["Values"].iloc[0]
Human_life_expectancy = model_parameters.query("Parameters == 'Human_life_expectancy'")[
    "Values"
].iloc[0]
Humans_per_free_roaming_dog = model_parameters.query(
    "Parameters == 'Humans_per_free_roaming_dog'"
)["Values"].iloc[0]
Free_roaming_dog_population = model_parameters.query(
    "Parameters == 'Free_roaming_dog_population'"
)["Values"].iloc[0]
Free_roaming_dogs_per_km2 = model_parameters.query(
    "Parameters == 'Free_roaming_dogs_per_km2'"
)["Values"].iloc[0]
Dog_birth_rate_per_1000_dogs = model_parameters.query(
    "Parameters == 'Dog_birth_rate_per_1000_dogs'"
)["Values"].iloc[0]
Dog_life_expectancy = model_parameters.query("Parameters == 'Dog_life_expectancy'")[
    "Values"
].iloc[0]

Annual_dog_bite_risk = model_parameters.query("Parameters == 'Annual_dog_bite_risk'")[
    "Values"
].iloc[0]
Probability_of_rabies_in_biting_dogs = model_parameters.query(
    "Parameters == 'Probability_of_rabies_in_biting_dogs'"
)["Values"].iloc[0]
Probability_of_human_developing_rabies = model_parameters.query(
    "Parameters == 'Probability_of_human_developing_rabies'"
)["Values"].iloc[0]
Dog_Human_transmission_rate = model_parameters.query(
    "Parameters == 'Dog_Human_transmission_rate'"
)["Values"].iloc[0]
R0_dog_to_dog = model_parameters.query(
    "Parameters == 'R0_dog_to_dog'"
)["Values"].iloc[0]

# Model parameters
Program_Area = Km2_of_program_area  # (REQUIRES INPUT) Km2_of_program_area
R0 = R0_dog_to_dog  # Effective reproductive number at t0
Sd = (1-((1/52)/Km2_of_program_area))*Free_roaming_dogs_per_km2 #Free_roaming_dogs_per_km2 * (1 - (1 / 52))  # Susceptible - Fixed scaling
Ed = 0  # Exposed at t0
Id = Free_roaming_dogs_per_km2*((1/52)/Km2_of_program_area) #Free_roaming_dogs_per_km2 * (1 / 52)  # Infectious/Rabid at t0 - Fixed formula
Rd = 0  # Immune at t0
Nd = Free_roaming_dogs_per_km2  # Population at t0

Nh = Humans_per_km2  # Population at t0
Sh = Nh  # Susceptible at t0
Eh = 0  # Exposed at t0
Ih = 0  # Infectious/Rabid at t0
Rh = 0  # Immune at t0

b_d = Dog_birth_rate_per_1000_dogs / 52 / 1000  # Dog birth rate
lambda_d1 = 0  # (REQUIRES INPUT) Loss of vaccination immunity (first 26 weeks after vaccination)
lambda_d2 = 0.0096  # (REQUIRES INPUT) Loss of vaccination immunity (last 26 weeks after vaccination)
i_d = 6.27  # (REQUIRES INPUT) Dog incubation period
sigma_d = 1 / i_d  # Inverse of average incubation period
r_d = 0.45  # (REQUIRES INPUT) Risk of clinical outcome
m_d = (1 / Dog_life_expectancy) / 52  # Death rate
mu_d = (
    1 / 10
) * 7  # (REQUIRES INPUT) Inverse of average infective period, rabid mortality rate

beta_d = (
    R0_dog_to_dog * (((sigma_d) + m_d) * (mu_d + m_d)) / (sigma_d * r_d * Sd)
)  # Transmission coefficient
K = Nd * (1 + 1 / np.log(Free_roaming_dog_population)) * 1.05  # Mean carrying capacity

v_d = 0.95  # (REQUIRES INPUT) Dog vaccine efficacy
Vaccination_coverage_per_campaign = 0.05
alpha_d1 = 0.0163  # Dog vaccination rate 1%
alpha_d2 = 0  # Dog vaccination rate 0%

b_h = (Human_birth / 52) / 1000  # Human birth rate
lambda_h = 0  # Human loss of vaccination immunity rate
m_h = (1 / Human_life_expectancy) / 52  # Human mortality rate
v_h = 0.93  # (REQUIRES INPUT) Human vaccine efficacy
alpha_h = 0  # Human prophylactic rate
beta_dh = 0.0000510  # (REQUIRES INPUT) Dog human transmission rate
P10 = 0.50  # (REQUIRES INPUT) PEP vaccination rate

mu_h = (
    (1 / 10) * 7
)  # (REQUIRES INPUT) Inverse of average infective period, rabid human mortality rate
gamma_d = (b_d - m_d) / K  # Dog density dependent mortality

# Initialize results dataframe
initial_run = pd.DataFrame(
    {"time": [0], "Sd": [Sd], "Ed": [Ed], "Id": [Id], "Rd": [Rd], "Nd": [Nd]}
)

# Convert to lists for easier manipulation during loop
results = {"time": [0], "Sd": [Sd], "Ed": [Ed], "Id": [Id], "Rd": [Rd], "Nd": [Nd]}

# Loop over step function
for time in range(1, 10001):
    # Determine lambda_d based on time
    lambda_d = lambda_d1 if time < 27 else lambda_d2

    # Calculate week (equivalent to R's modulo operation)
    week = 52 if time % 52 == 0 else time % 52

    # Determine alpha_d based on week
    alpha_d = alpha_d1 if (abs(week - 22) + abs(31 - week)) <= 10 else alpha_d2

    # Calculate vaccination status
    percent_immunized = Rd / Nd
    target_status = 1 if percent_immunized < Vaccination_coverage_per_campaign else 0

    # Update compartments
    Sd_new = (
        Sd
        + (b_d * Nd)
        + (lambda_d * Rd)
        + (sigma_d * (1 - r_d) * Ed)
        - (m_d * Sd)
        - (beta_d * Sd * Id)
        - (gamma_d * Nd * Sd)
        - (target_status * (v_d * alpha_d * Sd))
    )

    Ed_new = (
        Ed
        + (beta_d * Sd * Id)
        - (m_d * Ed)
        - (gamma_d * Nd * Ed)
        - (sigma_d * (1 - r_d) * Ed)
        - (target_status * (v_d * alpha_d * Ed))
        - (sigma_d * r_d * Ed)
    )

    Id_new = Id + (sigma_d * r_d * Ed) - (m_d * Id) - (gamma_d * Nd * Id) - (mu_d * Id)

    Rd_new = (
        Rd
        + (target_status * (v_d * alpha_d * (Sd + Ed)))
        - (m_d * Rd)
        - (gamma_d * Nd * Rd)
        - (lambda_d * Rd)
    )

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

# Convert results to DataFrame
initial_run = pd.DataFrame(results)

# Add week column
initial_run["week"] = initial_run["time"].apply(lambda x: 52 if x % 52 == 0 else x % 52)


# Create visualization (equivalent to ggplot)
# Melt the dataframe for plotting
plot_data = initial_run.melt(
    id_vars=["time"],
    value_vars=["Sd", "Ed", "Id", "Rd", "Nd"],
    var_name="variable",
    value_name="value",
)

# Create the plot
plt.figure(figsize=(12, 8))
for variable in ["Sd", "Ed", "Id", "Rd", "Nd"]:
    subset = plot_data[plot_data["variable"] == variable]
    plt.plot(subset["time"], subset["value"], label=variable, linewidth=2)

plt.xlabel("Time Step", fontsize=15)
plt.ylabel("# Indv.", fontsize=15)
plt.legend(title="Disease State")
plt.grid(True, alpha=0.3)
plt.title("Rabies Model Dynamics Over Time", fontsize=16)


# Set colors similar to R's default
colors = ["blue", "orange", "green", "red", "black"]
for i, line in enumerate(plt.gca().lines):
    line.set_color(colors[i])

plt.tight_layout()
plt.show()

# Optional: Save the plot
# plt.savefig('rabies_model_dynamics.png', dpi=300, bbox_inches='tight')


#### No Annual Vaccination (No Intervention) ####

# Get initial conditions from the previous run (last row of initial_run)
Program_Area = Km2_of_program_area  # (REQUIRES INPUT) Km2_of_program_area

Sd = initial_run.iloc[-1]["Sd"]  # Susceptible
Ed = initial_run.iloc[-1]["Ed"]  # Exposed at t0
Id = initial_run.iloc[-1]["Id"]  # Infectious/Rabid at t0

C_rd = initial_run.iloc[-1]["Id"]

Rd = 0 #initial_run.iloc[-1]["Rd"]  # Immune at t0
Nd = Sd + Ed + Id + Rd  # Population at t0

# Human parameters
Nh = Humans_per_km2  # Population at t0
Sh = Nh  # Susceptible at t0
Eh = 0  # Exposed at t0
Ih = 0  # Infectious/Rabid at t0
Rh = 0  # Immune at t0
Dh = 0
new_expo = Eh

# Model parameters (updated for no intervention)
b_d = Dog_birth_rate_per_1000_dogs / 52 / 1000  # Dog birth rate
lambda_d1 = 0  # Loss of vaccination immunity (first 26 weeks after vaccination)
lambda_d2 = 0.0096  # Loss of vaccination immunity (last 26 weeks after vaccination)
i_d = 6.27  # Dog incubation period
sigma_d = 1 / i_d  # Inverse of average incubation period
r_d = 0.45  # Risk of clinical outcome
m_d = (1 / Dog_life_expectancy) / 52  # Death rate
mu_d = (1 / 10) * 7  # Inverse of average infective period, rabid mortality rate

beta_d_1 = beta_d  # Transmission coefficient
K_1 = K  # Mean carrying capacity

v_d = 0.95  # Dog vaccine efficacy
# Vaccination_coverage_per_campaign will be loaded from CSV for each year
alpha_d1 = 0.0163  # Dog vaccination rate 5% (STATUS QUO - matches R code!)
alpha_d2 = 0  # Dog vaccination rate 0%

b_h = (Human_birth / 52) / 1000  # Human birth rate
lambda_h = 0  # Human loss of vaccination immunity rate
m_h = (1 / Human_life_expectancy) / 52  # Human mortality rate
v_h = 0.93  # Human vaccine efficacy
alpha_h = 0  # Human prophylactic rate
beta_dh = 0.0000510	  # Dog human transmission rate (No annual vaccination value) 
                      # Probability of rabies in biting dogs (suggested 0.1% - 5%) * Probability of rabies in biting dogs (suggested 0.1% - 5%) *Probability of human developing rabies (suggested 17%)
P10 = 0.50  # PEP vaccination rate

mu_h = (1 / 10) * 7  # Inverse of average infective period, rabid human mortality rate
gamma_d = (b_d - m_d) / K_1  # Dog density dependent mortality
R0 = (sigma_d * r_d * beta_d * Sd) / (
    (sigma_d + m_d) * (mu_d + m_d)
)  # Effective reproductive number

# Define missing parameters (from definition.py)
p_ExptoNoInf = 0.097  # Rates of not developing rabies from exposure (bites), per week
p_ExptoInf = 0.025  # Rates of developing rabies from exposure (bites), per week

# Initialize no_annual_vaccination dataframe
no_annual_vaccination = pd.DataFrame(
    {
        "time": [0],
        "week": [0],
        "Sd": [Sd],
        "Ed": [Ed],
        "Id": [Id],
        "C_rd": [C_rd],
        "Rd": [Rd],
        "Nd": [Nd],
        "Sh": [Sh],
        "Eh": [Eh],
        "Ih": [Ih],
        "Dh": [Dh],
        "Rh": [Rh],
        "Nh": [Nh],
        "new_expo": [new_expo],
    }
)

# Convert to lists for easier manipulation during loop
results_no_annual = {
    "time": [0],
    "week": [0],
    "Sd": [Sd],
    "Ed": [Ed],
    "Id": [Id],
    "C_rd": [C_rd],
    "Rd": [Rd],
    "Nd": [Nd],
    "Sh": [Sh],
    "Eh": [Eh],
    "Ih": [Ih],
    "Dh": [Dh],
    "Rh": [Rh],
    "Nh": [Nh],
    "new_expo": [new_expo],
}

# Loop over step function (NO INTERVENTION - with time-varying coverage from CSV)
for time in range(1, 2300):  # 
    # Calculate current year
    current_year = (time // 52) + 1

    # Get time-varying vaccination coverage from CSV
    Vaccination_coverage_per_campaign = get_vaccination_coverage(
        current_year, "no_annual_vaccination"
    )
    
    # Get time-varying PEP coverage from CSV
    P10_step = get_pep_coverage(current_year, "no_annual_vaccination")

    # Determine lambda_d based on time
    lambda_d = lambda_d1 if time < 27 else lambda_d2

    # Calculate week
    week = 52 if time % 52 == 0 else time % 52

    # Determine alpha_d based on week - only if there's actual vaccination coverage
    if Vaccination_coverage_per_campaign > 0:
        alpha_d = alpha_d1 if (abs(week - 22) + abs(31 - week)) <= 10 else alpha_d2
    else:
        alpha_d = 0  # No vaccination if coverage is 0

    # Update dog compartments (WITH vaccination - apply vaccination directly like R No_GAVI code)
    Sd_new = (
        Sd
        + (b_d * Nd)
        + (lambda_d * Rd)
        + (sigma_d * (1 - r_d) * Ed)
        - (m_d * Sd)
        - (beta_d_1 * Sd * Id)
        - (gamma_d * Nd * Sd)
        - (v_d * alpha_d * Sd)
    )  # Direct vaccination without target_status (like R code)

    Ed_new = (
        Ed
        + (beta_d_1 * Sd * Id)
        - (m_d * Ed)
        - (gamma_d * Nd * Ed)
        - (sigma_d * (1 - r_d) * Ed)
        - (v_d * alpha_d * Ed)
        - (sigma_d * r_d * Ed)
    )  # Direct vaccination without target_status (like R code)

    Id_new = Id + (sigma_d * r_d * Ed) - (m_d * Id) - (gamma_d * Nd * Id) - (mu_d * Id)

    C_rd_new = C_rd + (sigma_d * r_d * Ed)

    Rd_new = (
        Rd
        + (v_d * alpha_d * (Sd + Ed))
        - (m_d * Rd)
        - (gamma_d * Nd * Rd)
        - (lambda_d * Rd)
    )  # Direct vaccination without target_status (like R code)

    Nd_new = Sd_new + Ed_new + Id_new + Rd_new

    # Update human compartments (same as GAVI scenario)
    Sh_new = (
        Sh
        + (b_h * (Sh + Eh + Rh))
        + (lambda_h * Rh)
        + (Eh * p_ExptoNoInf)
        - (m_h * Sh)
        - (v_h * alpha_h * Sh)
        - (beta_dh * Sh * Id)
    )

    Eh_new = (
        Eh
        + (beta_dh * Sh * Id)
        - (m_h * Eh)
        - (Eh * p_ExptoInf * P10_step * v_h)
        - (Eh * p_ExptoInf * (1 - P10_step * v_h))
        - (Eh * p_ExptoNoInf)
    )

    Ih_new = Ih + (Eh * p_ExptoInf * (1 - P10_step * v_h)) - (m_h * Ih) - (mu_h * Ih)

    Dh_new = Dh + (Eh * p_ExptoInf * (1 - P10_step * v_h))

    Rh_new = (
        Rh
        + (Eh * p_ExptoInf * P10_step * v_h)
        + (v_h * alpha_h * Sh)
        - (m_h * Rh)
        - (lambda_h * Rh)
    )

    Nh_new = Sh_new + Eh_new + Ih_new + Dh_new + Rh_new

    new_expo_new = beta_dh * Sh * Id

    # Update all values
    Sd, Ed, Id, C_rd, Rd, Nd = Sd_new, Ed_new, Id_new, C_rd_new, Rd_new, Nd_new
    Sh, Eh, Ih, Dh, Rh, Nh = Sh_new, Eh_new, Ih_new, Dh_new, Rh_new, Nh_new
    new_expo = new_expo_new

    # Store results
    results_no_annual["time"].append(time)
    results_no_annual["week"].append(week)
    results_no_annual["Sd"].append(Sd)
    results_no_annual["Ed"].append(Ed)
    results_no_annual["Id"].append(Id)
    results_no_annual["C_rd"].append(C_rd)
    results_no_annual["Rd"].append(Rd)
    results_no_annual["Nd"].append(Nd)
    results_no_annual["Sh"].append(Sh)
    results_no_annual["Eh"].append(Eh)
    results_no_annual["Ih"].append(Ih)
    results_no_annual["Dh"].append(Dh)
    results_no_annual["Rh"].append(Rh)
    results_no_annual["Nh"].append(Nh)
    results_no_annual["new_expo"].append(new_expo)

# Convert results to DataFrame
no_annual_vaccination = pd.DataFrame(results_no_annual)

# Add year column
no_annual_vaccination["year"] = [1] + [
    year_val for year_val in range(1, 101) for _ in range(52)
][: len(no_annual_vaccination) - 1]

# Calculate cumulative new exposures
no_annual_vaccination["Cu_new_expo"] = no_annual_vaccination["new_expo"].cumsum()


# Create result summary by year
result_no_annual_vaccination = pd.DataFrame(
    columns=[
        "year",
        "canine_popn",
        "canine_rabies_cumulative",
        "canine_rabies_annual",
        "hum_popn",
        "hum_rabies_cases_cumulative",
        "hum_exposure_cumulative",
        "human_rabies_annual",
    ]
)

for year in range(1, 32):  # Changed to start from year 1
    time_point = (year - 1) * 52  # Adjust time_point calculation
    if time_point < len(no_annual_vaccination):
        row_data = no_annual_vaccination.iloc[time_point]

        canine_popn = row_data["Nd"] * Km2_of_program_area
        canine_rabies_cumulative = row_data["C_rd"] * Km2_of_program_area
        hum_popn = row_data["Nh"] * Km2_of_program_area
        hum_rabies_cases_cumulative = row_data["Dh"] * Km2_of_program_area
        hum_exposure_cumulative = row_data["Cu_new_expo"] * Km2_of_program_area

        new_row = pd.DataFrame(
            {
                "year": [year],
                "canine_popn": [canine_popn],
                "canine_rabies_cumulative": [canine_rabies_cumulative],
                "canine_rabies_annual": [np.nan],
                "hum_popn": [hum_popn],
                "hum_rabies_cases_cumulative": [hum_rabies_cases_cumulative],
                "hum_exposure_cumulative": [hum_exposure_cumulative],
                "human_rabies_annual": [np.nan],
            }
        )

        result_no_annual_vaccination = pd.concat(
            [result_no_annual_vaccination, new_row], ignore_index=True
        )

# Calculate annual differences
result_no_annual_vaccination["canine_rabies_annual"] = [
    result_no_annual_vaccination["canine_rabies_cumulative"].iloc[0]
] + list(np.diff(result_no_annual_vaccination["canine_rabies_cumulative"]))

result_no_annual_vaccination["human_rabies_annual"] = [
    result_no_annual_vaccination["hum_rabies_cases_cumulative"].iloc[0]
] + list(np.diff(result_no_annual_vaccination["hum_rabies_cases_cumulative"]))


#### Annual Vaccination Plan ####

# Get initial conditions from the previous run (last row of initial_run)
Program_Area = Km2_of_program_area  # (REQUIRES INPUT) Km2_of_program_area


Sd = initial_run.iloc[-1]["Sd"]  # Susceptible
Ed = initial_run.iloc[-1]["Ed"]  # Exposed at t0
Id = initial_run.iloc[-1]["Id"]  # Infectious/Rabid at t0


C_rd = initial_run.iloc[-1]["Id"]

Rd = 0  # Immune at t0
Nd = Sd + Ed + Id + Rd  # Population at t0

# Human parameters
Nh = Humans_per_km2  # Population at t0
Sh = Nh  # Susceptible at t0
Eh = 0  # Exposed at t0
Ih = 0  # Infectious/Rabid at t0
Rh = 0  # Immune at t0
Dh = 0
new_expo = Eh

# Model parameters (updated)
b_d = Dog_birth_rate_per_1000_dogs / 52 / 1000  # Dog birth rate
lambda_d1 = 0  # Loss of vaccination immunity (first 26 weeks after vaccination)
lambda_d2 = 0.0096  # Loss of vaccination immunity (last 26 weeks after vaccination)
i_d = 6.27  # Dog incubation period
sigma_d = 1 / i_d  # Inverse of average incubation period
r_d = 0.45  # Risk of clinical outcome
m_d = (1 / Dog_life_expectancy) / 52  # Death rate
mu_d = (1 / 10) * 7  # Inverse of average infective period, rabid mortality rate

beta_d_1 = beta_d  # Transmission coefficient
K_1 = K  # Mean carrying capacity

v_d = 0.95  # Dog vaccine efficacy
# Vaccination_coverage_per_campaign will be loaded from CSV for each year
# alpha_d1 will be calculated dynamically based on current coverage target
alpha_d2 = 0  # Dog vaccination rate 0%

b_h = (Human_birth / 52) / 1000  # Human birth rate
lambda_h = 0  # Human loss of vaccination immunity rate
m_h = (1 / Human_life_expectancy) / 52  # Human mortality rate
v_h = 0.93  # Human vaccine efficacy
alpha_h = 0  # Human prophylactic rate
beta_dh = 0.0000510  # Dog human transmission rate (Annual vaccination value)

# Create year array for P10 indexing - match R code exactly
year = [1] + [year_val for year_val in range(1, 101) for _ in range(52)][:2230]
# Load parameter values from our created file
try:
    parameter_values = coverage_data
    P10 = parameter_values["p_PEP_Exposed"].values
    print("Loaded time-varying P10 values successfully")
    # Ensure we have the right length and add indexing safety
    if len(P10) < len(year) + 1:
        P10 = np.pad(
            P10, (0, len(year) + 1 - len(P10)), "constant", constant_values=P10[-1]
        )
    print(
        f"Loaded parameter values: P10[0]={P10[0]:.3f}, P10[500]={P10[500]:.3f}, P10[-1]={P10[-1]:.3f}"
    )
except Exception as e:
    print(f"Warning: Could not load parameter values: {e}")
    # Create P10 array matching R structure: P10[1] = baseline, P10[time] = time-varying
    P10_baseline = 0.25  # This matches R's P10[1]
    P10 = np.full(len(year) + 1, P10_baseline)  # +1 to allow P10[1] indexing
    P10[0] = P10_baseline  # P10[0] for safety


mu_h = (1 / 10) * 7  # Inverse of average infective period, rabid human mortality rate
gamma_d = (b_d - m_d) / K_1  # Dog density dependent mortality
R0 = (sigma_d * r_d * beta_d * Sd) / (
    (sigma_d + m_d) * (mu_d + m_d)
)  # Effective reproductive number

# Define missing parameters (from definition.py)
p_ExptoNoInf = 0.097  # Rates of not developing rabies from exposure (bites), per week
p_ExptoInf = 0.025  # Rates of developing rabies from exposure (bites), per week

# Initialize annual_vaccination dataframe
annual_vaccination = pd.DataFrame(
    {
        "time": [0],
        "week": [0],
        "Sd": [Sd],
        "Ed": [Ed],
        "Id": [Id],
        "C_rd": [C_rd],
        "Rd": [Rd],
        "Nd": [Nd],
        "Sh": [Sh],
        "Eh": [Eh],
        "Ih": [Ih],
        "Dh": [Dh],
        "Rh": [Rh],
        "Nh": [Nh],
        "new_expo": [new_expo],
    }
)

# Convert to lists for easier manipulation during loop
results = {
    "time": [0],
    "week": [0],
    "Sd": [Sd],
    "Ed": [Ed],
    "Id": [Id],
    "C_rd": [C_rd],
    "Rd": [Rd],
    "Nd": [Nd],
    "Sh": [Sh],
    "Eh": [Eh],
    "Ih": [Ih],
    "Dh": [Dh],
    "Rh": [Rh],
    "Nh": [Nh],
    "new_expo": [new_expo],
}

# Loop over step function (WITH time-varying coverage from CSV)
for time in range(1, 2300):  # Changed from 2289 to 2300 to match Excel
    # Calculate current year
    current_year = (time // 52) + 1

    # Get time-varying vaccination coverage from CSV
    Vaccination_coverage_per_campaign = get_vaccination_coverage(
        current_year, "annual_vaccination"
    )
    
    # Get time-varying PEP coverage from CSV
    P10_step = get_pep_coverage(current_year, "annual_vaccination")

    # Calculate alpha_d1 based on current coverage target
    alpha_d1_current = -(1 / 10) * np.log(1 - Vaccination_coverage_per_campaign)

    # Determine lambda_d based on time
    lambda_d = lambda_d1 if time < 27 else lambda_d2

    # Calculate week
    week = 52 if time % 52 == 0 else time % 52

    # Determine alpha_d based on week (using current alpha_d1)
    alpha_d = alpha_d1_current if (abs(week - 22) + abs(31 - week)) <= 10 else alpha_d2

    # Calculate vaccination status
    percent_immunized = Rd / Nd
    target_status = 1 if percent_immunized < Vaccination_coverage_per_campaign else 0

    # Update dog compartments
    Sd_new = (
        Sd
        + (b_d * Nd)
        + (lambda_d * Rd)
        + (sigma_d * (1 - r_d) * Ed)
        - (m_d * Sd)
        - (beta_d_1 * Sd * Id)
        - (gamma_d * Nd * Sd)
        - (target_status * (v_d * alpha_d * Sd))
    )

    Ed_new = (
        Ed
        + (beta_d_1 * Sd * Id)
        - (m_d * Ed)
        - (gamma_d * Nd * Ed)
        - (sigma_d * (1 - r_d) * Ed)
        - (target_status * (v_d * alpha_d * Ed))
        - (sigma_d * r_d * Ed)
    )

    Id_new = Id + (sigma_d * r_d * Ed) - (m_d * Id) - (gamma_d * Nd * Id) - (mu_d * Id)

    C_rd_new = C_rd + (sigma_d * r_d * Ed)

    Rd_new = (
        Rd
        + (target_status * (v_d * alpha_d * (Sd + Ed)))
        - (m_d * Rd)
        - (gamma_d * Nd * Rd)
        - (lambda_d * Rd)
    )

    Nd_new = Sd_new + Ed_new + Id_new + Rd_new

    # Update human compartments
    Sh_new = (
        Sh
        + (b_h * (Sh + Eh + Rh))
        + (lambda_h * Rh)
        + (Eh * p_ExptoNoInf)
        - (m_h * Sh)
        - (v_h * alpha_h * Sh)
        - (beta_dh * Sh * Id)
    )

    Eh_new = (
        Eh
        + (beta_dh * Sh * Id)
        - (m_h * Eh)
        - (Eh * p_ExptoInf * P10_step * v_h)
        - (Eh * p_ExptoInf * (1 - P10_step * v_h))
        - (Eh * p_ExptoNoInf)
    )

    Ih_new = Ih + (Eh * p_ExptoInf * (1 - P10_step * v_h)) - (m_h * Ih) - (mu_h * Ih)

    Dh_new = Dh + (Eh * p_ExptoInf * (1 - P10_step * v_h))

    Rh_new = (
        Rh
        + (Eh * p_ExptoInf * P10_step * v_h)
        + (v_h * alpha_h * Sh)
        - (m_h * Rh)
        - (lambda_h * Rh)
    )

    Nh_new = Sh_new + Eh_new + Ih_new + Dh_new + Rh_new

    new_expo_new = beta_dh * Sh * Id

    # Update all values
    Sd, Ed, Id, C_rd, Rd, Nd = Sd_new, Ed_new, Id_new, C_rd_new, Rd_new, Nd_new
    Sh, Eh, Ih, Dh, Rh, Nh = Sh_new, Eh_new, Ih_new, Dh_new, Rh_new, Nh_new
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
    results["Rh"].append(Rh)
    results["Nh"].append(Nh)
    results["new_expo"].append(new_expo)

# Convert results to DataFrame
annual_vaccination = pd.DataFrame(results)

# Add year column
annual_vaccination["year"] = [1] + [
    year_val for year_val in range(1, 101) for _ in range(52)
][: len(annual_vaccination) - 1]

# Calculate cumulative new exposures
annual_vaccination["Cu_new_expo"] = annual_vaccination["new_expo"].cumsum()


# Create result summary by year
result_annual_vaccination = pd.DataFrame(
    columns=[
        "year",
        "canine_popn",
        "canine_rabies_cumulative",
        "canine_rabies_annual",
        "hum_popn",
        "hum_rabies_cases_cumulative",
        "hum_exposure_cumulative",
        "human_rabies_annual",
    ]
)

for year in range(1, 32):  # Changed to start from year 1
    time_point = (year - 1) * 52  # Adjust time_point calculation
    if time_point < len(annual_vaccination):
        row_data = annual_vaccination.iloc[time_point]

        canine_popn = row_data["Nd"] * Km2_of_program_area
        canine_rabies_cumulative = row_data["C_rd"] * Km2_of_program_area
        hum_popn = row_data["Nh"] * Km2_of_program_area
        hum_rabies_cases_cumulative = row_data["Dh"] * Km2_of_program_area
        hum_exposure_cumulative = row_data["Cu_new_expo"] * Km2_of_program_area

        new_row = pd.DataFrame(
            {
                "year": [year],
                "canine_popn": [canine_popn],
                "canine_rabies_cumulative": [canine_rabies_cumulative],
                "canine_rabies_annual": [np.nan],
                "hum_popn": [hum_popn],
                "hum_rabies_cases_cumulative": [hum_rabies_cases_cumulative],
                "hum_exposure_cumulative": [hum_exposure_cumulative],
                "human_rabies_annual": [np.nan],
            }
        )

        result_annual_vaccination = pd.concat(
            [result_annual_vaccination, new_row], ignore_index=True
        )




annual_vaccination
no_annual_vaccination






# Extract summary values function (updated)
def extract_summary_values(df, scenario_name, years = list(range(0,31))):
    """Extract key summary values matching Excel formula logic"""
    summary_data = []
    
    for year in years:
        if year == 0:
            # Year 0: Return 0 (matches IF($B$1=0,0,...) logic)
            canine_population = 0
            canine_rabies_cumulative = 0
            canine_rabies_annual = 0
            human_population = 0
            human_rabies_cumulative = 0
            human_rabies_annual = 0  # Add human rabies annual
            exposure_cumulative = 0
            exposure_annual = 0
        else:
            # Convert year to time step (year * 52)
            time_step = year * 52
            
            # VLOOKUP equivalent: find row with matching time step
            if time_step < len(df):
                row_data = df.iloc[time_step]
                
                # Extract values and scale by program area (equivalent to *Define_program!$C$8)
                canine_population = row_data["Nd"] * Km2_of_program_area
                canine_rabies_cumulative = row_data["C_rd"] * Km2_of_program_area
                human_population = row_data["Nh"] * Km2_of_program_area
                human_rabies_cumulative = row_data["Dh"] * Km2_of_program_area
                exposure_cumulative = row_data["Cu_new_expo"] * Km2_of_program_area
                
                # Calculate annual values
                if year == 1:
                    exposure_annual = exposure_cumulative
                    canine_rabies_annual = canine_rabies_cumulative
                    human_rabies_annual = human_rabies_cumulative  # Add human rabies annual calculation
                else:
                    prev_time = (year - 1) * 52
                    if prev_time < len(df):
                        prev_exposure = df.iloc[prev_time]["Cu_new_expo"] * Km2_of_program_area
                        prev_canine_rabies = df.iloc[prev_time]["C_rd"] * Km2_of_program_area
                        prev_human_rabies = df.iloc[prev_time]["Dh"] * Km2_of_program_area  # Add previous human rabies
                        exposure_annual = exposure_cumulative - prev_exposure
                        canine_rabies_annual = canine_rabies_cumulative - prev_canine_rabies
                        human_rabies_annual = human_rabies_cumulative - prev_human_rabies  # Calculate human rabies annual
                    else:
                        exposure_annual = 0
                        canine_rabies_annual = 0
                        human_rabies_annual = 0  # Add human rabies annual
            else:
                # If time step exceeds data length, use last available values
                canine_population = 0
                canine_rabies_cumulative = 0
                canine_rabies_annual = 0
                human_population = 0
                human_rabies_cumulative = 0
                human_rabies_annual = 0  # Add human rabies annual
                exposure_cumulative = 0
                exposure_annual = 0
        
        summary_data.append({
            'Year': year,
            'Canine_population': canine_population,
            'Canine_rabies_cumulative': canine_rabies_cumulative,
            'Canine_rabies_annual': canine_rabies_annual,
            'Human_population': human_population,
            'Human_rabies_cumulative': human_rabies_cumulative,
            'Human_rabies_annual': human_rabies_annual,  # Add human rabies annual to output
            'Exposure_cumulative': exposure_cumulative,
            'Exposure_annual': exposure_annual
        })
    
    return pd.DataFrame(summary_data)

# Create Excel-equivalent summary tables
def create_excel_equivalent_summary():
    """Create summary tables that match Excel output exactly"""
    
    # Extract values for years 0-5 for both scenarios
    no_annual_summary = extract_summary_values(no_annual_vaccination, "No Annual Vaccination", years=[0, 1, 2, 3, 4, 5])
    annual_summary = extract_summary_values(annual_vaccination, "Annual Vaccination", years=[0, 1, 2, 3, 4, 5])
    
    # Create comparison table matching your Excel structure
    excel_summary = pd.DataFrame({
        'Year': [0, 1, 2, 3, 4, 5],
        'No_Annual_Canine_Pop': no_annual_summary['Canine_population'].round(0),
        'Annual_Canine_Pop': annual_summary['Canine_population'].round(0),
        'No_Annual_Canine_Rabies_Cum': no_annual_summary['Canine_rabies_cumulative'].round(0),
        'Annual_Canine_Rabies_Cum': annual_summary['Canine_rabies_cumulative'].round(0),
        'No_Annual_Canine_Rabies_Ann': no_annual_summary['Canine_rabies_annual'].round(0),
        'Annual_Canine_Rabies_Ann': annual_summary['Canine_rabies_annual'].round(0),
        'No_Annual_Human_Pop': no_annual_summary['Human_population'].round(0),
        'Annual_Human_Pop': annual_summary['Human_population'].round(0),
        'No_Annual_Human_Deaths_Cum': no_annual_summary['Human_rabies_cumulative'].round(0),
        'Annual_Human_Deaths_Cum': annual_summary['Human_rabies_cumulative'].round(0),
        'No_Annual_Human_Deaths_Ann': no_annual_summary['Human_rabies_annual'].round(0),  # Add human rabies annual
        'Annual_Human_Deaths_Ann': annual_summary['Human_rabies_annual'].round(0),  # Add human rabies annual
        'No_Annual_Exposure_Cum': no_annual_summary['Exposure_cumulative'].round(0),
        'Annual_Exposure_Cum': annual_summary['Exposure_cumulative'].round(0),
        'No_Annual_Exposure_Ann': no_annual_summary['Exposure_annual'].round(0),
        'Annual_Exposure_Ann': annual_summary['Exposure_annual'].round(0)
    })
    
    return excel_summary

excel_equivalent = create_excel_equivalent_summary()



# Extract annual rabies data for both scenarios (years 1-30)
no_annual_summary = extract_summary_values(no_annual_vaccination, "No Annual Vaccination")
annual_summary = extract_summary_values(annual_vaccination, "Annual Vaccination")

# Filter data to start from year 1
no_annual_filtered = no_annual_summary[no_annual_summary["Year"] >= 1]
annual_filtered = annual_summary[annual_summary["Year"] >= 1]

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Rabid dogs (annual) - Top Left
axes[0,0].plot(no_annual_filtered["Year"], no_annual_filtered["Canine_rabies_annual"], 
               linewidth=3, color='red', label='No vaccination campaign')
axes[0,0].plot(annual_filtered["Year"], annual_filtered["Canine_rabies_annual"], 
               linewidth=3, color='green', label='Annual vaccination campaign')
axes[0,0].set_title("Rabid dogs (annual)", fontsize=14, fontweight='bold')
axes[0,0].set_xlabel("Year", fontsize=12)
axes[0,0].set_ylabel("Canine rabies cases", fontsize=12)
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_xlim(1, 30)

# Plot 2: Canine rabies cases (cumulative) - Top Right
axes[0,1].plot(no_annual_filtered["Year"], no_annual_filtered["Canine_rabies_cumulative"], 
               linewidth=3, color='red', label='No vaccination campaign')
axes[0,1].plot(annual_filtered["Year"], annual_filtered["Canine_rabies_cumulative"], 
               linewidth=3, color='green', label='Annual vaccination campaign')
axes[0,1].set_title("Canine rabies cases (cumulative)", fontsize=14, fontweight='bold')
axes[0,1].set_xlabel("Year", fontsize=12)
axes[0,1].set_ylabel("Cumulative canine cases", fontsize=12)
axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_xlim(1, 30)

# Plot 3: Human deaths due to rabies (annual) - Bottom Left
axes[1,0].plot(no_annual_filtered["Year"], no_annual_filtered["Human_rabies_annual"], 
               linewidth=3, color='red', label='No vaccination campaign')
axes[1,0].plot(annual_filtered["Year"], annual_filtered["Human_rabies_annual"], 
               linewidth=3, color='green', label='Annual vaccination campaign')
axes[1,0].set_title("Human deaths due to rabies (annual)", fontsize=14, fontweight='bold')
axes[1,0].set_xlabel("Year", fontsize=12)
axes[1,0].set_ylabel("Human deaths", fontsize=12)
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_xlim(1, 30)

# Plot 4: Human deaths (cumulative) - Bottom Right
axes[1,1].plot(no_annual_filtered["Year"], no_annual_filtered["Human_rabies_cumulative"], 
               linewidth=3, color='red', label='No vaccination campaign')
axes[1,1].plot(annual_filtered["Year"], annual_filtered["Human_rabies_cumulative"], 
               linewidth=3, color='green', label='Annual vaccination campaign')
axes[1,1].set_title("Human deaths (cumulative)", fontsize=14, fontweight='bold')
axes[1,1].set_xlabel("Year", fontsize=12)
axes[1,1].set_ylabel("Cumulative human deaths", fontsize=12)
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_xlim(1, 30)

# Add a single legend at the bottom
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=12)

# Adjust layout and show
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()

# ...existing code...