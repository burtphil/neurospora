setwd("/home/burt/neurospora/strains_bif")

### load packages
require(tidyverse)
require(ggthemes)
require(cowplot)
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

plot_amps <- function(df){
  df <- df %>% filter(type == 1 | type == 3)
  max <- df %>% filter(extrama == "max")
  min <- df %>% filter(extrama == "min")
  
  
  bifurcation <- ggplot()
  ### I can change this by adding a line with user input for the x axis  
  bif.plot <- bifurcation + 
    geom_line(data=max, aes(par,val), color = "black", size = 1)+
    labs(x = expression(bold(k[bold(1)])),y =expression(bold(FRQ[c])))+
    geom_line(data=min, aes(par,val), color = "black", size = 1)+
    theme_thesis
  
  return(bif.plot)
}

plot_per <- function(df){
  df <- df %>% filter(oscill == 2)
  per <- df %>% group_by(par) %>% summarize(period = mean(val))
  borders <- per %>% summarize(mean(period))  
  borders <- as.integer(borders)
  
  bifurcation <- ggplot()
  
  per.plot <- bifurcation + 
    geom_line(data=per, aes(par,period), color = "black", size = 1)+
    coord_cartesian(ylim = c(borders - 12, borders + 12))+
    labs(x=expression(bold(k[bold(1)])),y ="Period (h)")+
    theme_thesis
  
  return(per.plot)
}


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

amp.names <- str_extract(amp.data, ".\\d+|K")
per.names <- str_extract(per.data, ".\\d+|K")

amp.bif <- map(amp.bif,plot_amps)
per.bif <- map(per.bif, plot_per)

labels = c("A","B","C","D","E","F")

amp.bif[[3]] <- amp.bif[[3]]+
  annotate("text", x = 1.56, y = 34, label = "frq+", size = 8)+
  scale_x_continuous(breaks=c(1.6,1.8,2.0))

per.bif[[3]] <- per.bif[[3]]+
  annotate("text", x = 1.67, y = 31, label = "frq+", size = 8)+
  scale_x_continuous(breaks=c(1.6,1.8,2.0))+
  scale_y_continuous(breaks=c(10,20,30))

amp.bif[[1]] <- amp.bif[[1]]+
  annotate("text", x = 0.95, y = 13, label = "frq[1]", size = 8,parse = TRUE)

per.bif[[1]] <- per.bif[[1]]+
  annotate("text", x = 1.2, y = 25, label = "frq[1]", size = 8,parse = TRUE)

amp.bif[[2]] <- amp.bif[[2]]+
  annotate("text", x = 0.95, y = 44, label = "frq[7]", size = 8,parse = TRUE)+
  scale_y_continuous(breaks=c(15,30,45))

per.bif[[2]] <- per.bif[[2]]+
  annotate("text", x = 1.1, y = 35, label = "frq[7]", size = 8,parse = TRUE)+
  scale_y_continuous(breaks=c(20,30,40))


plot_grid(amp.bif[[3]],
          per.bif[[3]],
          amp.bif[[1]],
          per.bif[[1]],
          amp.bif[[2]],
          per.bif[[2]],
          ncol=2,
          labels = "AUTO",
          label_x = 0, label_y = 1,
          hjust = -0.5,
          vjust = c(1.2,1.2,0.5,0.5,0.5,0.5),
          label_size = 30
          )

ggsave("plot.pdf", dpi = 1200, width = 12, height = 12, units = "in")
