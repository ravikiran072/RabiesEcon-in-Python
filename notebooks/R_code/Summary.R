setwd("C:/Users/mfl3/OneDrive - CDC/Rabies/Rabies_R0/RabiesEcon_in_R")

start_time <- Sys.time()

source("Defination.R")
source("initial_run.R")
source("GAVI.R")
source("No_GAVI.R")



Rb_an_data <- cbind(year = result_no_GAVI$year, no_GAVI = result_no_GAVI$canine_rabies_annual, GAVI = result_GAVI$canine_rabies_annual)

Rb_an <- ggplot(data= melt(as.data.frame(Rb_an_data), id = "year"))+
  aes(x=year, y=value, color=variable)+
  geom_line()+
  theme_set(theme_gray(base_size = 15))+
  xlab("Year")+ylab("Canine rabies cases")+ ggtitle("Rabid dogs (annual)") +
scale_color_manual(name="", values=c("red", "green"))



Rb_cu_data <- cbind(year = result_no_GAVI$year, no_GAVI = result_no_GAVI$canine_rabies_cumulative, GAVI = result_GAVI$canine_rabies_cumulative)

Rb_cu <- ggplot(data= melt(as.data.frame(Rb_cu_data), id = "year"))+
  aes(x=year, y=value, color=variable)+
  geom_line()+
  theme_set(theme_gray(base_size = 15))+
  xlab("Year")+ylab("Cumulative canine cases")+ ggtitle("Canine rabies cases (cumulative)") +
  scale_color_manual(name="",values=c("red", "green"))

  
Rb_an_data_hu <- cbind(year = result_no_GAVI$year, no_GAVI = result_no_GAVI$human_rabies_annual, GAVI = result_GAVI$human_rabies_annual)
  
Rb_an_hu <- ggplot(data= melt(as.data.frame(Rb_an_data_hu), id = "year"))+
    aes(x=year, y=value, color=variable)+
    geom_line()+
    theme_set(theme_gray(base_size = 15))+
    xlab("Year")+ylab("Human deaths")+ ggtitle("Human deaths due to rabies (annual)") +
  scale_color_manual(name="", values=c("red", "green"))
  

Rb_cu_data_hu <- cbind(year = result_no_GAVI$year, no_GAVI = result_no_GAVI$hum_rabies_cases_cumulative, GAVI = result_GAVI$hum_rabies_cases_cumulative)

Rb_cu_hu <- ggplot(data= melt(as.data.frame(Rb_cu_data_hu), id = "year"))+
  aes(x=year, y=value, color=variable)+
  geom_line()+
  theme_set(theme_gray(base_size = 15))+
  xlab("Year")+ylab("Cumulative human cases")+ ggtitle("Human deaths (cumulative)") +
  scale_color_manual(name="", values=c("red", "green"))


plot_grid(Rb_an, Rb_cu, Rb_an_hu,Rb_cu_hu  , 
         # labels = c("A", "B", "C"),
          ncol = 2, nrow = 2)

end_time <- Sys.time()
end_time - start_time
