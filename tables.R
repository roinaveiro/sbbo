library(tidyverse)
library(latex2exp)
library(kableExtra)


################
# Tables
################

make_table <- function(problem, label, iter_lim=499, nexp=10){
  
  data_path <- paste0("results/", problem, "/", problem, "_full_results.csv")
  fig_path <- paste0("figs/", problem, ".png")
  # data <- read_csv("results/pRNA/pRNA_full_results_noBOCS.csv")
  data <- read_csv(data_path)
  
  pdata <- data %>% 
    mutate(Algorithm = algorithm) %>% 
    filter( !(Algorithm %in% c("sbbo-NGBlin")) ) %>% 
    filter(iter == iter_lim) %>% 
    group_by(iter, Algorithm) %>% 
    summarise("Best Value" = mean(best_vals),  "Lower" = mean(best_vals) - 1*sd(best_vals)/sqrt(nexp),
              "Upper" = mean(best_vals) + 1*sd(best_vals)/sqrt(nexp), "Std. Dev" = sd(best_vals)/sqrt(nexp)) %>% 
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
label <- paste("BQP Problem. Best objective function value found after", as.character(iter_lim), "iterations")
make_table("BQP", label, iter_lim = iter_lim)

iter_lim <-  499
label <- paste("CON Problem. Best objective function value found after", as.character(iter_lim), "iterations")
make_table("CON", label, iter_lim = iter_lim)

iter_lim <-  299
label <- paste("RNA Problem. Best objective function value found after", as.character(iter_lim), "iterations")
make_table("pRNA", label, iter_lim = iter_lim)


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

pdata <- data %>% mutate(best_vals = opt - best_vals)  %>% 
  mutate(Algorithm = algorithm) %>% 
  filter( !(Algorithm %in% c("sbbo-NGBlin")) ) %>% 
  filter(iter == iter_lim) %>% 
  group_by(iter, Algorithm) %>% 
  summarise("Best Value" = mean(best_vals),  "Lower" = mean(best_vals) - 1*sd(best_vals)/sqrt(10),
            "Upper" = mean(best_vals) + 1*sd(best_vals)/sqrt(10), "Std. Dev" = sd(best_vals)/sqrt(10)) %>% 
  ungroup() %>% 
  select(Algorithm, `Best Value`, `Std. Dev`)

tex_out <- pdata %>% 
  mutate(
    `Best Value` = paste("$", as.character(round(`Best Value`,2)), "\\pm", round(`Std. Dev`, 2), "$"),
  ) %>%
  select(Algorithm, `Best Value` ) %>% 
  kbl(caption = label, format = "latex", booktabs=T, escape = F) %>% kable_styling(latex_options = c("hold_position"))

tex_out
