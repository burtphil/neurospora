setwd("/home/burt/neurospora/data_analysis/xppauto")

### load packages
require(tidyverse)
require(ggthemes)

require(stringr)
# define header for columns
var.names <- c("par","min","max","type","oscill","useless")

### function to read file names with col names as header
read.files <- function(i){
  read_table2(i, col_names = var.names)
}

### drop the useless column with all zeros
drop_col <- function(df){
  newdf <- df %>% select(-useless)
  return(newdf)
}

### tidy variables collapse min and maxima to key value pair
tidyit <- function(df){
  tidy_df <- gather(df, extrama, val, -par, -type, -oscill)
  return(tidy_df)
}

#### get rid off negative entries
no_neg <- function(df){
  df <- df %>% filter(par >= 0 & val >=0)
}

### read actual data
amp.data <- list.files(pattern = "amp.txt")
amp.bif <- map(amp.data, read.files)

per.data <- list.files(pattern = "per.txt")
per.bif <- map(per.data, read.files)

### tidy amplitude and period data 
### (drop one column, collapse key value variables and get rid off neg values)
amp.bif <- map(amp.bif, drop_col)
amp.bif <- map(amp.bif, tidyit)
amp.bif <- map(amp.bif, no_neg)

per.bif <- map(per.bif, drop_col)
per.bif <- map(per.bif, tidyit)
per.bif <- map(per.bif, no_neg)
### prepare string from loaded file for data analysis
test <- per.bif[1]

amp.names <- str_extract(amp.data, ".\\d+|K")
per.names <- str_extract(per.data, ".\\d+|K")



### test plot

df <- amp.bif[[5]] %>% filter(type == 1 | type == 3)
max <- df %>% filter(extrama == "max")
min <- df %>% filter(extrama == "min")



### define a theme for plotting
size_axis <- element_text(face="bold", size = 30)
size_ticks <- element_text(face="bold", size = 30, color = "black")
size_line <- element_line(color = "black", size = 2)

theme_thesis <- theme_few() +
  theme(axis.title = size_axis,
        axis.title.y = element_text(margin = margin(0, 0.8, 0, 0, "cm")),
        axis.title.x = element_text(margin = margin(0.5, 0, 0, 0, "cm")),
        axis.text = size_ticks,
        axis.ticks = size_line,
        axis.ticks.length=unit(2,"mm"),
        panel.border = element_rect(size = 2, color = "black")
       )

### test plot

bifurcation <- ggplot()

bif.plot <- bifurcation + 
  geom_line(data=max, aes(par,val), color = "black", size = 1)+
  labs(x=amp.names[5], y ="FRQ [a.u.]")+
  geom_line(data=min, aes(par,val), color = "black", size = 1)+
  theme_thesis

bif.plot



### plot and save amplitude bifurcation diagrams

for (i in seq_along(amp.names)){
  
  df <- amp.bif[[i]] %>% filter(type == 1 | type == 3)
  max <- df %>% filter(extrama == "max")
  min <- df %>% filter(extrama == "min")
  
  
  bifurcation <- ggplot()
### I can change this by adding a line with user input for the x axis  
  bif.plot <- bifurcation + 
    geom_line(data=max, aes(par,val), color = "black", size = 1)+
    labs(x=amp.names[i], y ="FRQ [a.u.]")+
    geom_line(data=min, aes(par,val), color = "black", size = 1)+
    theme_thesis
  
  bif.plot
  
  ggsave(paste(amp.names[i],"amp","pdf", sep = "."), dpi = 1200)
}




### plot and save period bifurcation diagrams

for (i in seq_along(per.names)){
  
  df <- per.bif[[i]] %>% filter(oscill == 2)
  per <- df %>% group_by(par) %>% summarize(period = mean(val))
  borders <- per %>% summarize(mean(period))  
  borders <- as.integer(borders)
  
  bifurcation <- ggplot()
  
  if (per.names[i] == "k5" | per.names[i] =="k4"){
    bif.plot <- bifurcation + 
      geom_line(data=per, aes(par,period), color = "black", size = 1)+
      labs(x=per.names[i], y ="Period [h]")+
      theme_thesis
  } else {
    bif.plot <- bifurcation + 
      geom_line(data=per, aes(par,period), color = "black", size = 1)+
      coord_cartesian(ylim = c(borders - 12, borders + 12))+
      labs(x=per.names[i], y ="Period [h]")+
      theme_thesis
  }
  
  bif.plot
  
  ggsave(paste(per.names[i],"per","pdf", sep = "."), dpi = 1200)
}