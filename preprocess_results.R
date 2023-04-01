library(tidyverse)
source("utils.R")


results_path <- "/Users/roinaveiro/Library/CloudStorage/OneDrive-CUNEF/CUNEF/research/sbbo/results/"
problem <- "BQP"

models <- c("RS", "SA", "MH/BOCS", "MH/GPr", "MH/NGBdec", 
             "MH/NGBlinCV")
all_labels <- c("RS", "SA", "sbbo-BOCS", "sbbo-GPr", "sbbo-NGBdec", 
                "sbbo-NGBlinCV")

#models <- c("RS", "SA", "MH/BOCS", "MH/GPr", "MH/NGBdec", 
#            "MH/NGBlin", "MH/NGBlinCV")
#all_labels <- c("RS", "SA", "sbbo-BOCS", "sbbo-GPr", "sbbo-NGBdec", 
#                "sbbo-NGBlin","sbbo-NGBlinCV")


all_models <- paste0(results_path, problem, "/", models)
full_df <- process_models(all_models, all_labels)

write.csv(full_df, "results/BQP/BQP_full_results.csv", row.names = F)