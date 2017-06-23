
### purpose: to visualize connies entrainment data
### approach: load data as data frames into R. store dataframes in list

### set working directory
setwd("C:\\Users\\Philipp\\Desktop\\neurospora\\all_raw_data\\csv_files")


require(dplyr)
require(ggplot2)
require(tidyr)



### read files

entrain_names <- list.files(pattern = ".csv")
entrain_data <- lapply(entrain_names, function (i){read.csv(i, sep = ";")})

test0 <- entrain_data[[1]]


rename <- function(df){
  df <- df %>% rename(Time = timepoints..min.)

  return(df)
}

rename_data <- lapply(entrain_data, rename)

test1 <- rename_data[[1]]

test1 <- test1 %>% mutate(strain = substr(strain,4,21))

test1$strain <- gsub('_', '-', data1$strain)