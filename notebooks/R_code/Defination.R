library(ggplot2)
library(dplyr)
library(reshape2) 
library(zoo) 
library(cowplot)
library(readxl)

setwd("C:/Users/mfl3/OneDrive - CDC/Rabies/Rabies_R0/RabiesEcon_in_R")

Km2_of_program_area <- 14000
Human_population <- 11172223 
Humans_per_km2 <- Human_population/Km2_of_program_area 
Human_birth <- 17.00
Human_life_expectancy <- 70
Humans_per_free_roaming_dog <- 15.0
Free_roaming_dog_population <-	 Human_population/Humans_per_free_roaming_dog
Free_roaming_dogs_per_km2 <- Free_roaming_dog_population / Km2_of_program_area
Dog_birth_rate_per_1000_dogs <-	750
Dog_life_expectancy <-	3.0
R0_dog_to_dog <- 1.7
Annual_dog_bite_risk <- 0.02
Probability_of_rabies_in_biting_dogs <-	0.02
Probability_of_human_developing_rabies <- 0.17
Dog_Human_transmission_rate <- 0.000016


Probability_dh <-  cbind(c(0.070, 0.384, 0.060, 0.486),
      c(0.450, 0.275, 0.050, 0.050),
      c(3.14, 8.57, 6.43, 10.71)
      )
colnames(Probability_dh) <- c("P_exp_by_loc", "P_rabies_from_exp", "incub")
rownames(Probability_dh) <- c("h_n", "u_e", "t", "l_e")

Probability_dh["h_n", "incub"]

p_ExptoNoInf <- 0.097 #Rates of not developing rabies from exposure (bites), per week
p_ExptoInf <-	0.025   #Rates of developing rabies from exposure (bites), per week

I_infective <- 10.00  #Dog rabies infective period, life expectancy (days)
co_clinical_outcome <- 0.45  #Risk of clinical outcome per bite (rabid dog-dog)
L_latent <- 45.00            #Dog rabies incubation period (days) (3)
rab_vacc_efficacy	 <- 0.95   #Efficacy of dog rabies vaccine

h_rab_inf_per <- 7.00  #Human rabies infective period (days)
pep_efficacy <-	0.97   #Efficacy of human rabies post exposure vaccine (PEP)




parameter_values <- read_excel("data/parameter_values.xlsx")
