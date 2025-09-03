

Km2_of_program_area <- 14060
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



Program_Area <- Km2_of_program_area	 #(REQUIRES INPUT)  Km2_of_program_area
R0 <-	R0_dog_to_dog  #Effective reproductive number at t0
Sd <- (1-((1/52)/Program_Area))*Free_roaming_dogs_per_km2 #Susceptible 
Ed <-	0                                                          #Exposed at  t0	0
Id <- Free_roaming_dogs_per_km2*((1/52)/Km2_of_program_area)	#Infectious/Rabid at t0	7.2456E-05
Rd <-	0	#Immune at t0	0
Nd <-	Free_roaming_dogs_per_km2 #Population at t0	53.0


Nh <- Humans_per_km2	#Population at t0	795
Sh <-  Nh #Susceptible at t0	795
Eh <- 0	#Exposed at  t0	0
Ih <- 0	#Infectious/Rabid at t0	0
Rh <- 0	#Immune at t0	0


b_d<- Dog_birth_rate_per_1000_dogs/52/1000	#Dog birth rate	 0.014 
lambda_d1<- 0	 #(REQUIRES INPUT) 	#Loss of vaccination immunity (first 26 weeks after vaccination)	
lambda_d2<-	0.0096   #(REQUIRES INPUT) Loss of vaccination immunity (last 26 weeks after vaccination)	
i_d<- 6	# (REQUIRES INPUT) Dog incubation period	6
sigma_d	<- 1/i_d #Inverse of average incubation period	0.17
r_d <- 0.45 #(REQUIRES INPUT)	#Risk of clinical outcome	0.45
m_d	<- (1/Dog_life_expectancy)/52 #Death rate	0.006410256
mu_d <- (1/10)*7	# (REQUIRES INPUT) Inverse of average infective period, rabid mortality rate	0.7000

beta_d <- R0_dog_to_dog*(((sigma_d)+m_d)*(mu_d+m_d))/(sigma_d*r_d*Sd) #Transmission coefficient (inverse of time between dog contacts)	0.05231
K <- Nd*(1+1/log(Free_roaming_dog_population))*1.05	#Mean carrying capacity	59.73656815
#ge <- 	#Dog density dependent mortality	0.00013

v_d <- 	0.95 #(REQUIRES INPUT) Dog vaccine efficacy	0.95
Vaccination_coverage_per_campaign <- 	0.05
alpha_d1 <-	0.001#Dog vaccination rate	1% ####CHANGED
alpha_d2 <- 0	#Dog vaccination rate	0%

b_h	<-  (Human_birth/52)/1000 #Human birth rate	0.000326923
lambda_h <- 0 	#Human loss of vaccination immunity rate	0
m_h	<- (1/Human_life_expectancy)/52#Human mortality rate	0.000274725
v_h	<- 0.969 # (REQUIRES INPUT)Human vaccine efficacy	0.969
alpha_h	<- 0 #Human prophylactic rate	0
beta_dh <- 0.000016 # (REQUIRES INPUT) Dog human transmission rate	0.0000156
P10 <- 0.25 	#(REQUIRES INPUT)#PEP vaccination rate	25%

mu_h <- 	(1/10)*7 # (REQUIRES INPUT)Inverse of average infective period, rabid human mortality rate	1.0000
gamma_d <- (b_d - m_d)/K 	#Dog density dependent mortality


#times <- seq(0, 100, 1)


initial_run = data.frame(time=0, Sd=Sd,Ed=Ed,Id=Id,Rd=Rd, Nd = Nd)
# alpha_d = alpha_d2 = alpha_d1
# lambda_d =lambda_d1 =lambda_d2
#Loop over step function
for(time in 1:5000){
  
  lambda_d = ifelse(time < 27, lambda_d1, lambda_d2)
  week = ifelse(time %% 52 == 0, 52, time %% 52) 
  alpha_d = ifelse((abs(week-22) + abs(31-week)) <= 10, alpha_d1, alpha_d2)
  percent_immunized <- Rd/Nd
  target_status <- (percent_immunized < Vaccination_coverage_per_campaign) *1 + (percent_immunized > Vaccination_coverage_per_campaign)*0
  
  
  Sd <- Sd + (b_d*Nd) + (lambda_d*Rd) + (sigma_d*(1 - r_d)*Ed) -(m_d*Sd) - (beta_d*Sd*Id) -(gamma_d*Nd*Sd) - (target_status*(v_d*alpha_d*Sd))
  
  Ed <- Ed + (beta_d*Sd*Id) - (m_d*Ed) - (gamma_d*Nd*Ed) - (sigma_d*(1 - r_d)*Ed) - (target_status*(v_d*alpha_d*Ed)) - (sigma_d*r_d*Ed)
  
  Id <- Id + (sigma_d*r_d*Ed) - (m_d*Id) - (gamma_d*Nd*Id) - (mu_d*Id)
  
  Rd  <- Rd + (target_status*(v_d*alpha_d*(Sd + Ed))) - (m_d*Rd) -  (gamma_d * Nd * Rd) - (lambda_d*Rd)
  
  Nd <- Sd+Ed+Id+Rd
  initial_run = rbind(initial_run, data.frame(time, Sd, Ed, Id, Rd, Nd))
}



head(initial_run)
tail(initial_run)

initial_run$week <-  ifelse(initial_run$time %% 52 == 0, 52, initial_run$time %% 52)   #initial_run$time %% 52

# ggplot(data= melt(as.data.frame(initial_run), id = "time"))+
#   aes(x=time, y=value, color=variable)+
#   geom_line()+
#   theme_set(theme_gray(base_size = 15))+
#   xlab("Time Step")+ylab("# Indv.")+
#   
#   scale_color_manual(name="Disease State", values=c("blue", "orange", "green", "red", "black"))
