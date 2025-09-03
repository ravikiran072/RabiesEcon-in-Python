



Program_Area <- Km2_of_program_area	 #(REQUIRES INPUT)  Km2_of_program_area

Sd <- tail(initial_run, n=1)$Sd #Susceptible 
Ed <-	tail(initial_run, n=1)$Ed                                                          #Exposed at  t0	0
Id <- tail(initial_run, n=1)$Id	#Infectious/Rabid at t0	7.2456E-05
C_rd <- tail(initial_run, n=1)$Id

Rd <-	0	#Immune at t0	0
Nd <-	Sd + Ed + Id + Rd #Population at t0	53.0

  


Nh <- Humans_per_km2	#Population at t0	795
Sh <-  Nh #Susceptible at t0	795
Eh <- 0	#Exposed at  t0	0
Ih <- 0	#Infectious/Rabid at t0	0
Rh <- 0	#Immune at t0	0
Dh <- 0
new_expo <- Eh
#Cu_new_expo <- new_expo


b_d<- Dog_birth_rate_per_1000_dogs/52/1000	#Dog birth rate	 0.014 
lambda_d1<- 0	 #(REQUIRES INPUT) 	#Loss of vaccination immunity (first 26 weeks after vaccination)	
lambda_d2<-	0.0096   #(REQUIRES INPUT) Loss of vaccination immunity (last 26 weeks after vaccination)	
i_d<- 6.27	# (REQUIRES INPUT) Dog incubation period	6
sigma_d	<- 1/i_d #Inverse of average incubation period	0.17
r_d <- 0.45 #(REQUIRES INPUT)	#Risk of clinical outcome	0.45
m_d	<- (1/Dog_life_expectancy)/52 #Death rate	0.006410256
mu_d <- (1/10)*7	# (REQUIRES INPUT) Inverse of average infective period, rabid mortality rate	0.7000

beta_d_1 <- beta_d #Transmission coefficient (inverse of time between dog contacts)	0.05231
K_1 <- K	#Mean carrying capacity	59.73656815
#ge <- 	#Dog density dependent mortality	0.00013

v_d <- 	0.95 #(REQUIRES INPUT) Dog vaccine efficacy	0.95
Vaccination_coverage_per_campaign <- 	0.40
alpha_d1 <-	0.05#Dog vaccination rate	1%
alpha_d2 <- 0	#Dog vaccination rate	0%

b_h	<-  (Human_birth/52)/1000 #Human birth rate	0.000326923
lambda_h <- 0 	#Human loss of vaccination immunity rate	0
m_h	<- (1/Human_life_expectancy)/52#Human mortality rate	0.000274725
v_h	<- 0.969 # (REQUIRES INPUT)Human vaccine efficacy	0.969
alpha_h	<- 0 #Human prophylactic rate	0
beta_dh <- 0.0000156 # (REQUIRES INPUT) Dog human transmission rate	0.0000156
P10 <- 0.25 	#(REQUIRES INPUT)#PEP vaccination rate	25%

mu_h <- 	(1/10)*7 # (REQUIRES INPUT)Inverse of average infective period, rabid human mortality rate	1.0000
gamma_d <- (b_d - m_d)/K_1 	#Dog density dependent mortality
R0 <-	(sigma_d*r_d*beta_d*Sd)/((sigma_d+m_d)*(mu_d+m_d)) #Effective reproductive number at t0

#times <- seq(0, 100, 1)


no_GAVI = data.frame(time=0, week = 0,  Sd=Sd,Ed=Ed,Id=Id,C_rd = Id, Rd=Rd, Nd = Nd, Sh = Sh, Eh =Eh, Ih = Ih, Dh = Dh, Rh = Rh, Nh = Nh, new_expo = new_expo)
# alpha_d = alpha_d2 = alpha_d1
# lambda_d =lambda_d1 =lambda_d2
#Loop over step function
for(time in 1:2288){
  
  lambda_d = ifelse(time < 27, lambda_d1, lambda_d2)
  week = ifelse(time %% 52 == 0, 52, time %% 52)
  
  alpha_d = ifelse((abs(week-22) + abs(31-week)) <= 10, alpha_d1, alpha_d2)
  
  
  
  Sd <- Sd + (b_d*Nd) + (lambda_d*Rd) + (sigma_d*(1 - r_d)*Ed) -(m_d*Sd) - (beta_d_1*Sd*Id) -(gamma_d*Nd*Sd) - (v_d*alpha_d*Sd) 
  
  Ed <- Ed + (beta_d_1*Sd*Id) - (m_d*Ed) - (gamma_d*Nd*Ed) - (sigma_d*(1 - r_d)*Ed) - (v_d*alpha_d*Ed) - (sigma_d*r_d*Ed)
  
  Id <- Id + (sigma_d*r_d*Ed) - (m_d*Id) - (gamma_d*Nd*Id) - (mu_d*Id)
  
  C_rd <-  C_rd +(sigma_d*r_d*Ed)
  
  Rd  <- Rd + (v_d*alpha_d*(Sd + Ed)) - (m_d*Rd) -  (gamma_d * Nd * Rd) - (lambda_d*Rd) 
  
  Nd <- Sd+Ed+Id+Rd
  
  Sh <- Sh +(b_h*(Sh+Eh+Rh))+(lambda_h*Rh)+(Eh*p_ExptoNoInf)-(m_h*Sh)-(v_h*alpha_h*Sh)-(beta_dh*Sh*Id) # change Rh
  
  Eh <- Eh+(beta_dh*Sh*Id)-(m_h*Eh)-(Eh*p_ExptoInf*P10*v_h)-(Eh*p_ExptoInf*(1-P10*v_h))-(Eh*p_ExptoNoInf)
  
  Ih <- Ih+(Eh*p_ExptoInf*(1-P10*v_h))-(m_h*Ih)-(mu_h*Ih)
  
  Dh <- Dh+(Eh*p_ExptoInf*(1-P10*v_h)) 
  
  Rh <- Rh + (Eh*p_ExptoInf*P10*v_h)+(v_h*alpha_h*Sh)-(m_h*Rh)-(lambda_h*Rh)
  
  Nh = Sh+Eh+Ih+Dh+Rh
  
  new_expo <- beta_dh*Sh*Id
  
  no_GAVI = rbind(no_GAVI, data.frame(time, week, Sd, Ed, Id,C_rd, Rd, Nd,Sh, Eh, Ih, Dh, Rh, Nh, new_expo))
}



no_GAVI$year <- c(1, rep(seq(1:100), each = 52, length.out = nrow(no_GAVI)-1))

no_GAVI$Cu_new_expo <- cumsum(no_GAVI$new_expo)

head(no_GAVI)
tail(no_GAVI)

#aggregate(no_GAVI$Nd, list(no_GAVI$year), FUN=sum) * Km2_of_program_area


result_no_GAVI = data.frame(year = NA, canine_popn=NA,canine_rabies_cumulative = NA,
                            canine_rabies_annual = NA, hum_popn = NA, hum_rabies_cases_cumulative = NA, 
                            hum_exposure_cumulative = NA, human_rabies_annual = NA)

for(year in 0:30) {
 
  canine_popn <- no_GAVI[no_GAVI['time'] == year * 52][8] * Km2_of_program_area
  canine_rabies_cumulative <- no_GAVI[no_GAVI['time'] == year * 52][6] * Km2_of_program_area
  #canine_rabies_annual
  hum_popn <- no_GAVI[no_GAVI['time'] == year * 52][14] * Km2_of_program_area
  hum_rabies_cases_cumulative <- no_GAVI[no_GAVI['time'] == year * 52][12] * Km2_of_program_area
  hum_exposure_cumulative <- no_GAVI[no_GAVI['time'] == year * 52][17] * Km2_of_program_area
  #human_rabies_annual
  
  
  result_no_GAVI = rbind(result_no_GAVI, data.frame(year, canine_popn,canine_rabies_cumulative,
                                                    canine_rabies_annual = NA, hum_popn, hum_rabies_cases_cumulative,
                                                    hum_exposure_cumulative, human_rabies_annual = NA))
}



result_no_GAVI <- result_no_GAVI[rowSums(is.na(result_no_GAVI)) != ncol(result_no_GAVI), ]  
rownames(result_no_GAVI) <- NULL
head(result_no_GAVI)
result_no_GAVI$canine_rabies_annual <-c(result_no_GAVI$canine_rabies_cumulative[1], diff(result_no_GAVI$canine_rabies_cumulative , lag = 1 ))
result_no_GAVI$human_rabies_annual <-c(result_no_GAVI$hum_rabies_cases_cumulative [1], diff(result_no_GAVI$hum_rabies_cases_cumulative  , lag = 1 ))

head(result_no_GAVI)
tail(result_no_GAVI)



