setwd("/home/burt/neurospora/data_analysis/xppauto")

### load packages
require(tidyverse)
require(ggthemes)

require(stringr)


strains <- read.csv(file="strains.csv", header=TRUE, sep=",")

strains_tidy <- strains %>% gather(strain, entr, frq:wc2)

### add significance variable

strains_tidy <- strains_tidy %>% mutate(sign = case_when(entr==1.5 ~ 0.5,
                                                         entr == 3.5 ~ 1.0,
                                                         TRUE ~ 1.0))
                                        

entr_crit <- strains_tidy %>% mutate(entr= case_when(entr == 1.0 ~ "1:1",
                                                     entr == 1.5 ~ "1:1",
                                                     entr == 2.0 ~ "1:2",
                                                     entr == 6.0 ~ "2:3",
                                                     entr == 3.0 ~ "NE",
                                                     entr == 3.5 ~ "NE",
                                                     entr == 4.0 ~ "NE",
                                                     entr == 5.0 ~ "NE",
                                                     entr == 0.0 ~ "nan"
                                                     ))

dummy <- entr_crit %>% mutate(fake= case_when(T == 12 ~ 1,
                                              T == 16 ~ 2,
                                              T == 22 ~ 3,
                                              T == 24 ~ 4,
                                              T == 26 ~ 5
                                              ))

dummy <- dummy %>% filter(entr != "nan")

strains_plot <- ggplot(dummy, aes(fake,kappa))

strains_plot+
  geom_point(aes(alpha = sign, size = 1.5, shape = entr))+
  facet_wrap(~strain, ncol =4)+
  scale_y_continuous(breaks=c(0.16,0.25,0.33,0.4,0.5,0.6,0.67,0.75,0.84))+
  scale_x_continuous(labels=c("12","16","22","24","26"))+
  guides(size=FALSE,
         alpha = FALSE,
         shape = guide_legend(title=NULL,override.aes = list(size=5)))+
  ylab(expression(kappa))+
  xlab("T [h]")+
  scale_shape_manual(values = c(19, 17, 15, 4))+
  theme_few()+
  theme(legend.position = c(0.9, 0.25),
        legend.text = element_text(size=15,face="bold"))