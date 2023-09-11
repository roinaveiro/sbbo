library(tidyverse)
source("utils.R")


results_path <- "/Users/roinaveiro/Library/CloudStorage/OneDrive-CUNEF/CUNEF/research/sbbo/results/"
problem <- "CON"

#models <- c("RS", "SA", "MH/BOCS", "MH/BNN", "MH/GPr", "MH/NGBdec", 
#             "MH/NGBlinCV")
#all_labels <- c("RS", "SA", "sbbo-BOCS", "sbbo-BNN", "sbbo-GPr", "sbbo-NGBdec", 
#                "sbbo-NGBlinCV")

models <- c("RS", "SA", "MH/BOCS", "MH/GPr", "MH/NGBdec", 
             "MH/NGBlinCV")
all_labels <- c("RS", "SA", "sbbo-BLr", "sbbo-GPr", "sbbo-NGBdec", 
                "sbbo-NGBlinCV")


all_models <- paste0(results_path, problem, "/", models)
full_df <- process_models(all_models, all_labels)

write.csv(full_df, "results/CON/CON_full_results.csv", row.names = F)
