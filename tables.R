library(tidyverse)
library(latex2exp)
library(kableExtra)

dpi <- 700
width <- 8.33
height <- 5.79

################
# Tables
################

make_table <- function(problem, label, iter_lim=499){

  data_path <- paste0("results/", problem, "/", problem, "_full_results.csv")
  fig_path <- paste0("figs/", problem, ".png")
  # data <- read_csv("results/pRNA/pRNA_full_results_noBOCS.csv")
  data <- read_csv(data_path)
  
  pdata <- data %>% select(-current_vals) %>% 
    mutate(Algorithm = algorithm) %>% 
    filter( !(Algorithm %in% c("sbbo-NGBlin")) ) %>% 
    filter(iter == iter_lim) %>% 
    group_by(iter, Algorithm) %>% 
    summarise("Best Value" = mean(best_vals),  "Lower" = mean(best_vals) - 1*sd(best_vals),
              "Upper" = mean(best_vals) + 1*sd(best_vals), "Std. Dev" = sd(best_vals)) %>% 
    ungroup() %>% 
    select(Algorithm, `Best Value`, `Std. Dev`)
  
  tex_out <- pdata %>% 
    mutate(
      `Best Value` = paste("$", as.character(round(`Best Value`,2)), "\\pm", round(`Std. Dev`, 2), "$"),
    ) %>%
    select(Algorithm, `Best Value` ) %>% 
    kbl(caption = label, format = "latex", booktabs=T, escape = F) %>% kable_styling(latex_options = c("hold_position"))
    
  return(tex_out)
}

iter_lim <-  120
label <- paste("Contamination Problem. Best objective function value found after", as.character(iter_lim), "iterations")
make_table("BQP", label, iter_lim = iter_lim)


################
# Tables - BQP
################

opt <- 11.239633938255269
iter_lim <- 120


problem <- "BQP"

data_path <- paste0("results/", problem, "/", problem, "_full_results.csv")
fig_path <- paste0("figs/", problem, ".png")
# data <- read_csv("results/pRNA/pRNA_full_results_noBOCS.csv")
data <- read_csv(data_path)

pdata <- data %>% mutate(best_vals = opt - best_vals) %>% select(-current_vals) %>% 
  mutate(Algorithm = algorithm) %>% 
  filter( !(Algorithm %in% c("sbbo-NGBlin")) ) %>% 
  filter(iter == iter_lim) %>% 
  group_by(iter, Algorithm) %>% 
  summarise("Best Value" = mean(best_vals),  "Lower" = mean(best_vals) - 1*sd(best_vals),
            "Upper" = mean(best_vals) + 1*sd(best_vals), "Std. Dev" = sd(best_vals)) %>% 
  ungroup() %>% 
  select(Algorithm, `Best Value`, `Std. Dev`)

tex_out <- pdata %>% 
  mutate(
    `Best Value` = paste("$", as.character(round(`Best Value`,2)), "\\pm", round(`Std. Dev`, 2), "$"),
  ) %>%
  select(Algorithm, `Best Value` ) %>% 
  kbl(caption = label, format = "latex", booktabs=T, escape = F) %>% kable_styling(latex_options = c("hold_position"))


