import numpy as np
import pandas as pd
import os

# Set working directory to project root (one level up from notebooks)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)



Km2_of_program_area = 100
Human_population = 534722
Humans_per_km2 = Human_population / Km2_of_program_area
Human_birth = 18.5
Human_life_expectancy = 72
Humans_per_free_roaming_dog = 15.0
Free_roaming_dog_population = Human_population / Humans_per_free_roaming_dog
Free_roaming_dogs_per_km2 = Free_roaming_dog_population / Km2_of_program_area
Dog_birth_rate_per_1000_dogs = 530
Dog_life_expectancy = 3.0
R0_dog_to_dog = 1.2
Dog_Human_transmission_rate = 0.0002054

Annual_dog_bite_risk = 0.02
Probability_of_rabies_in_biting_dogs = 0.02
Probability_of_human_developing_rabies = 0.17

# Probability matrix for dog-human transmission by anatomical location and exposure type
Probability_dh = np.array([
    [0.070, 0.450, 3.14],   # h_n (head/neck)
    [0.384, 0.275, 8.57],   # u_e (upper extremity)
    [0.060, 0.050, 6.43],   # t (trunk)
    [0.486, 0.050, 10.71]   # l_e (lower extremity)
])

# Convert to pandas DataFrame with proper labels
Probability_dh = pd.DataFrame(
    Probability_dh,
    columns=["P_exp_by_loc", "P_rabies_from_exp", "incub"],
    index=["h_n", "u_e", "t", "l_e"]
)


p_ExptoNoInf = 0.097 #Rates of not developing rabies from exposure (bites), per week
p_ExptoInf = 0.025   #Rates of developing rabies from exposure (bites), per week
I_infective = 10.00  #Dog rabies infective period, life expectancy (days)
co_clinical_outcome = 0.45  #Risk of clinical outcome per bite (rabid dog-dog)
L_latent = 45.00            #Dog rabies incubation period (days) (3)
rab_vacc_efficacy = 0.95   #Efficacy of dog rabies vaccine
h_rab_inf_per = 7.00  #Human rabies infective period (days)
pep_efficacy = 0.97   #Efficacy of human rabies post exposure vaccine (PEP)

parameter_values = pd.read_excel("data/raw/parameter_values.xlsx")

