library(tidyverse)

process_path <- function(main_path, label){
  
 
  all_files <- list.files(path=main_path, pattern=".csv", all.files=FALSE,
                      full.names=TRUE)

  full_data  <- read_csv(all_files[1])
  full_data <- full_data %>% mutate(algorithm = label, exp = 0)

  for(i in 2:length(all_files)){
    tmp_data  <- read_csv(all_files[i])
    tmp_data <- tmp_data  %>% mutate(algorithm = label, exp = i-1)
    
    full_data <- full_data %>% 
      bind_rows(tmp_data)
  }
  return(full_data)
    
}

process_models <- function(all_models, all_labels){
  full_df <- process_path(all_models[1], all_labels[1])
  for(i in 2:length(all_models)){
    tmp_df <- process_path(all_models[i], all_labels[i])
    full_df <- full_df %>% 
      bind_rows(tmp_df)
  }
  return(full_df)
}


