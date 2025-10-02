import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time as tm


os.chdir(r"C:\Users\mfl3\OneDrive - CDC\Rabies\RabioesEcon-Python")
coverage_data = pd.read_csv("data\\coverage_data.csv")
model_parameters = pd.read_excel("data\\model_parameters.xlsx")


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
Free_roaming_dog_population = Human_population / Humans_per_free_roaming_dog
Free_roaming_dogs_per_km2 = Free_roaming_dog_population / Km2_of_program_area
Dog_birth_rate_per_1000_dogs = model_parameters.query(
    "Parameters == 'Dog_birth_rate_per_1000_dogs'"
)["Values"].iloc[0]
Dog_life_expectancy = model_parameters.query("Parameters == 'Dog_life_expectancy'")[
    "Values"
].iloc[0]
R0_dog_to_dog = model_parameters.query("Parameters == 'R0_dog_to_dog'")["Values"].iloc[
    0
]
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

# Model parameters
Program_Area = Km2_of_program_area  # (REQUIRES INPUT) Km2_of_program_area
R0 = R0_dog_to_dog  # Effective reproductive number at t0
Sd = (1 - ((1 / 52) / Program_Area)) * Free_roaming_dogs_per_km2  # Susceptible
Ed = 0  # Exposed at t0
Id = Free_roaming_dogs_per_km2 * (
    (1 / 52) / Km2_of_program_area
)  # Infectious/Rabid at t0
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
i_d = 6  # (REQUIRES INPUT) Dog incubation period
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
alpha_d1 = 0.001  # Dog vaccination rate 1%
alpha_d2 = 0  # Dog vaccination rate 0%

b_h = (Human_birth / 52) / 1000  # Human birth rate
lambda_h = 0  # Human loss of vaccination immunity rate
m_h = (1 / Human_life_expectancy) / 52  # Human mortality rate
v_h = 0.969  # (REQUIRES INPUT) Human vaccine efficacy
alpha_h = 0  # Human prophylactic rate
beta_dh = 0.000016  # (REQUIRES INPUT) Dog human transmission rate
P10 = 0.25  # (REQUIRES INPUT) PEP vaccination rate

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
for time in range(1, 5001):
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
alpha_d1 = 0.05  # Dog vaccination rate 5% (STATUS QUO - matches R code!)
alpha_d2 = 0  # Dog vaccination rate 0%

b_h = (Human_birth / 52) / 1000  # Human birth rate
lambda_h = 0  # Human loss of vaccination immunity rate
m_h = (1 / Human_life_expectancy) / 52  # Human mortality rate
v_h = 0.969  # Human vaccine efficacy
alpha_h = 0  # Human prophylactic rate
beta_dh = 0.0000156  # Dog human transmission rate (No annual vaccination value)
P10 = 0.25  # PEP vaccination rate

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
for time in range(1, 2289):
    # Calculate current year
    current_year = (time // 52) + 1

    # Get time-varying vaccination coverage from CSV
    Vaccination_coverage_per_campaign = get_vaccination_coverage(
        current_year, "no_annual_vaccination"
    )

    # Determine lambda_d based on time
    lambda_d = lambda_d1 if time < 27 else lambda_d2

    # Calculate week
    week = 52 if time % 52 == 0 else time % 52

    # Determine alpha_d based on week (but no target_status check)
    alpha_d = alpha_d1 if (abs(week - 22) + abs(31 - week)) <= 10 else alpha_d2

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
        - (Eh * p_ExptoInf * P10 * v_h)
        - (Eh * p_ExptoInf * (1 - P10 * v_h))
        - (Eh * p_ExptoNoInf)
    )

    Ih_new = Ih + (Eh * p_ExptoInf * (1 - P10 * v_h)) - (m_h * Ih) - (mu_h * Ih)

    Dh_new = Dh + (Eh * p_ExptoInf * (1 - P10 * v_h))

    Rh_new = (
        Rh
        + (Eh * p_ExptoInf * P10 * v_h)
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

for year in range(0, 31):
    time_point = year * 52
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
v_h = 0.969  # Human vaccine efficacy
alpha_h = 0  # Human prophylactic rate
beta_dh = 0.000016  # Dog human transmission rate (Annual vaccination value)

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
for time in range(1, 2289):
    # Calculate current year
    current_year = (time // 52) + 1

    # Get time-varying vaccination coverage from CSV
    Vaccination_coverage_per_campaign = get_vaccination_coverage(
        current_year, "annual_vaccination"
    )

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

    # Get P10 step value - match R indexing P10[time] where time starts at 1
    P10_step = P10[min(time, len(P10) - 1)] if time < len(P10) else P10[-1]

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
        - (Eh * p_ExptoInf * P10[1] * v_h)
        - (Eh * p_ExptoInf * (1 - P10[1] * v_h))
        - (Eh * p_ExptoNoInf)
    )

    Ih_new = Ih + (Eh * p_ExptoInf * (1 - P10_step * v_h)) - (m_h * Ih) - (mu_h * Ih)

    Dh_new = Dh + (Eh * p_ExptoInf * (1 - P10_step * v_h))

    Rh_new = (
        Rh
        + (Eh * p_ExptoInf * P10[1] * v_h)
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

for year in range(0, 31):
    time_point = year * 52
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

# Calculate annual differences
result_annual_vaccination["canine_rabies_annual"] = [
    result_annual_vaccination["canine_rabies_cumulative"].iloc[0]
] + list(np.diff(result_annual_vaccination["canine_rabies_cumulative"]))

result_annual_vaccination["human_rabies_annual"] = [
    result_annual_vaccination["hum_rabies_cases_cumulative"].iloc[0]
] + list(np.diff(result_annual_vaccination["hum_rabies_cases_cumulative"]))


### Summary and plot

# Record start time for performance tracking
start_time = tm.time()

# Create comparison data for canine rabies (annual)
Rb_an_data = pd.DataFrame(
    {
        "year": result_no_annual_vaccination["year"],
        "no_annual_vaccination": result_no_annual_vaccination["canine_rabies_annual"],
        "annual_vaccination": result_annual_vaccination["canine_rabies_annual"],
    }
)

# Melt the data for plotting
Rb_an_data_melted = Rb_an_data.melt(
    id_vars=["year"],
    value_vars=["no_annual_vaccination", "annual_vaccination"],
    var_name="variable",
    value_name="value",
)

# Create comparison data for canine rabies (cumulative)
Rb_cu_data = pd.DataFrame(
    {
        "year": result_no_annual_vaccination["year"],
        "no_annual_vaccination": result_no_annual_vaccination[
            "canine_rabies_cumulative"
        ],
        "annual_vaccination": result_annual_vaccination["canine_rabies_cumulative"],
    }
)

Rb_cu_data_melted = Rb_cu_data.melt(
    id_vars=["year"],
    value_vars=["no_annual_vaccination", "annual_vaccination"],
    var_name="variable",
    value_name="value",
)

# Create comparison data for human rabies (annual)
Rb_an_data_hu = pd.DataFrame(
    {
        "year": result_no_annual_vaccination["year"],
        "no_annual_vaccination": result_no_annual_vaccination["human_rabies_annual"],
        "annual_vaccination": result_annual_vaccination["human_rabies_annual"],
    }
)

Rb_an_data_hu_melted = Rb_an_data_hu.melt(
    id_vars=["year"],
    value_vars=["no_annual_vaccination", "annual_vaccination"],
    var_name="variable",
    value_name="value",
)

# Create comparison data for human rabies (cumulative)
Rb_cu_data_hu = pd.DataFrame(
    {
        "year": result_no_annual_vaccination["year"],
        "no_annual_vaccination": result_no_annual_vaccination[
            "hum_rabies_cases_cumulative"
        ],
        "annual_vaccination": result_annual_vaccination["hum_rabies_cases_cumulative"],
    }
)

Rb_cu_data_hu_melted = Rb_cu_data_hu.melt(
    id_vars=["year"],
    value_vars=["no_annual_vaccination", "annual_vaccination"],
    var_name="variable",
    value_name="value",
)

# Set up the plot style
plt.style.use("default")
sns.set_palette(["red", "green"])

# Create figure with subplots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(
    "Rabies Model Comparison: Annual Vaccination vs No Annual Vaccination",
    fontsize=16,
    fontweight="bold",
)

# Plot 1: Canine rabies cases (annual)
ax1 = axes[0, 0]
for variable in ["no_annual_vaccination", "annual_vaccination"]:
    subset = Rb_an_data_melted[Rb_an_data_melted["variable"] == variable]
    color = "red" if variable == "no_annual_vaccination" else "green"
    ax1.plot(subset["year"], subset["value"], label=variable, color=color, linewidth=2)

ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Canine rabies cases", fontsize=12)
ax1.set_title("Rabid dogs (annual)", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Canine rabies cases (cumulative)
ax2 = axes[0, 1]
for variable in ["no_annual_vaccination", "annual_vaccination"]:
    subset = Rb_cu_data_melted[Rb_cu_data_melted["variable"] == variable]
    color = "red" if variable == "no_annual_vaccination" else "green"
    ax2.plot(subset["year"], subset["value"], label=variable, color=color, linewidth=2)

ax2.set_xlabel("Year", fontsize=12)
ax2.set_ylabel("Cumulative canine cases", fontsize=12)
ax2.set_title("Canine rabies cases (cumulative)", fontsize=14, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Human deaths (annual)
ax3 = axes[1, 0]
for variable in ["no_annual_vaccination", "annual_vaccination"]:
    subset = Rb_an_data_hu_melted[Rb_an_data_hu_melted["variable"] == variable]
    color = "red" if variable == "no_annual_vaccination" else "green"
    ax3.plot(subset["year"], subset["value"], label=variable, color=color, linewidth=2)

ax3.set_xlabel("Year", fontsize=12)
ax3.set_ylabel("Human deaths", fontsize=12)
ax3.set_title("Human deaths due to rabies (annual)", fontsize=14, fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Human deaths (cumulative)
ax4 = axes[1, 1]
for variable in ["no_annual_vaccination", "annual_vaccination"]:
    subset = Rb_cu_data_hu_melted[Rb_cu_data_hu_melted["variable"] == variable]
    color = "red" if variable == "no_annual_vaccination" else "green"
    ax4.plot(subset["year"], subset["value"], label=variable, color=color, linewidth=2)

ax4.set_xlabel("Year", fontsize=12)
ax4.set_ylabel("Cumulative human cases", fontsize=12)
ax4.set_title("Human deaths (cumulative)", fontsize=14, fontweight="bold")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# Alternative: Save the plot
# plt.savefig('rabies_comparison_plots.png', dpi=300, bbox_inches='tight')

# Calculate and display execution time
end_time = tm.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds")


# Optional: Create individual plots with better styling
def create_individual_plots():
    """Create individual plots with enhanced styling"""

    # Set style for publication-quality plots
    plt.style.use("seaborn-v0_8")

    # Plot 1: Annual canine cases
    plt.figure(figsize=(10, 6))
    plt.plot(
        Rb_an_data["year"],
        Rb_an_data["no_annual_vaccination"],
        "r-",
        linewidth=2.5,
        label="No Annual Vaccination",
    )
    plt.plot(
        Rb_an_data["year"],
        Rb_an_data["annual_vaccination"],
        "g-",
        linewidth=2.5,
        label="Annual Vaccination",
    )
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Annual Canine Rabies Cases", fontsize=14)
    plt.title(
        "Annual Rabid Dogs: Intervention Comparison", fontsize=16, fontweight="bold"
    )
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Cumulative human deaths
    plt.figure(figsize=(10, 6))
    plt.plot(
        Rb_cu_data_hu["year"],
        Rb_cu_data_hu["no_annual_vaccination"],
        "r-",
        linewidth=2.5,
        label="No Annual Vaccination",
    )
    plt.plot(
        Rb_cu_data_hu["year"],
        Rb_cu_data_hu["annual_vaccination"],
        "g-",
        linewidth=2.5,
        label="Annual Vaccination",
    )
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Cumulative Human Deaths", fontsize=14)
    plt.title(
        "Cumulative Human Deaths: Intervention Impact", fontsize=16, fontweight="bold"
    )
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Uncomment to create individual plots
# create_individual_plots()

# Print summary statistics
print("\n" + "=" * 50)
print("SUMMARY STATISTICS")
print("=" * 50)

print("\nFinal Year (Year 30):")
print("Canine rabies cases (cumulative):")
print(
    f"  No Annual Vaccination: {result_no_annual_vaccination.iloc[-1]['canine_rabies_cumulative']:,.0f}"
)
print(
    f"  Annual Vaccination: {result_annual_vaccination.iloc[-1]['canine_rabies_cumulative']:,.0f}"
)

print("\nHuman deaths (cumulative):")
print(
    f"  No Annual Vaccination: {result_no_annual_vaccination.iloc[-1]['hum_rabies_cases_cumulative']:,.0f}"
)
print(
    f"  Annual Vaccination: {result_annual_vaccination.iloc[-1]['hum_rabies_cases_cumulative']:,.0f}"
)

# Calculate percentage reduction
canine_reduction = (
    (
        result_no_annual_vaccination.iloc[-1]["canine_rabies_cumulative"]
        - result_annual_vaccination.iloc[-1]["canine_rabies_cumulative"]
    )
    / result_no_annual_vaccination.iloc[-1]["canine_rabies_cumulative"]
) * 100

human_reduction = (
    (
        result_no_annual_vaccination.iloc[-1]["hum_rabies_cases_cumulative"]
        - result_annual_vaccination.iloc[-1]["hum_rabies_cases_cumulative"]
    )
    / result_no_annual_vaccination.iloc[-1]["hum_rabies_cases_cumulative"]
) * 100

print("\nReduction due to annual vaccination intervention:")
print(f"  Canine cases: {canine_reduction:.1f}% reduction")
print(f"  Human deaths: {human_reduction:.1f}% reduction")
