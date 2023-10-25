library(tidyverse)
library(latex2exp)

dpi <- 700
width <- 8.33
height <- 5.79

################
# OPT PLOTS
################

make_comparison_plot <- function(problem, label, iter_lim=500){

  data_path <- paste0("results/", problem, "/", problem, "_full_results.csv")
  fig_path <- paste0("figs/", problem, ".png")
  # data <- read_csv("results/pRNA/pRNA_full_results_noBOCS.csv")
  data <- read_csv(data_path)
  
  pdata <- data %>% select(-current_vals) %>% 
    mutate(Algorithm = algorithm) %>% 
    filter( !(Algorithm %in% c("sbbo-NGBlin")) ) %>% 
    filter(iter <= iter_lim) %>% 
    group_by(iter, Algorithm) %>% 
    summarise("Best Value" = mean(best_vals),  "Lower" = mean(best_vals) - 1*sd(best_vals),
              "Upper" = mean(best_vals) + 1*sd(best_vals), "Std. Dev" = sd(best_vals))
  
  
  pdata %>% 
    ggplot(aes(x=iter, y=`Best Value`, color=Algorithm, fill=Algorithm)) + 
    geom_line(size=0.5) +
    geom_ribbon(aes(ymin=Lower,ymax=Upper), linetype = 0, alpha=0.1) +
    #geom_errorbar(aes(ymin=Lower, ymax=Upper), size=0.5, alpha=0.25,
    #              position=position_dodge(0.05)) +
    # scale_color_viridis_d() + 
    labs(title    = label,
         #subtitle = TeX("|Q| = 20, |T| = 20, |X| = 20"),
         x = "Iteration",
         y = "Best Value") + 
    theme_minimal() +
    #theme(legend.position="bottom", legend.box = "horizontal") +
    theme(axis.text.x=element_text(angle=-90, hjust=0, vjust=1)) +
    theme(plot.title=element_text(size=15, hjust=0.5, face="bold", vjust=-1)) +
    theme(plot.subtitle=element_text(size=12, hjust=0.5, vjust=-1)) +
    theme(text = element_text(size=12)) +
    theme(legend.title = element_text(face = "bold")) 
  
  
  ggsave(filename = fig_path, 
         device = "png", 
         dpi = dpi, width = width, height = height)
}

make_comparison_plot("CON", "Contamination Problem")
make_comparison_plot("pRNA", "MFE RNA Design Problem", iter_lim=300)
make_comparison_plot("BQP", "Binary Quadratic Problem")


###################
# CONVERGENCE PLOTS
###################

conv <- read_csv("results/convergence.csv")
cols <- c("gray", "gray", "gray", "gray", "gray", "gray", "gray", "gray", "gray", "gray", "red")
conv <- conv %>% select(Temperature, EU, Exp)

conv %>% group_by(Temperature) %>%  summarise(EU = mean(EU)) %>% mutate(Exp = 20) %>% 
  bind_rows(conv) %>%
  ggplot(aes(x=Temperature, y=EU, color=factor(Exp) )) + geom_line(alpha=0.8, lwd=0.8) + theme_minimal() + 
  xlim(0,2500) + scale_color_manual(values = cols) + theme(legend.position="none") +
  labs(x="H", y="Expected Utility", title = "Convergence of SBBO-MH")

################
# ACC PLOTS
################

data <- read_csv("results/CON/acc_CON_ss400.csv")
sum_data <- data %>% mutate(Algorithm = fct_recode(Algorithm, "GPr" = "GPr",
                              "BLr" = "BOCS",
                              "NGBlinCV" = "NGBlinCV",
                              "NGBdec" = "NGBdec")) %>% 
  group_by(Algorithm, Quantile) %>% 
  summarise("mean_R2"  = mean(R2),
            "std_R2"   = sd(R2),
            "mean_MAE" = mean(MAE),
            "std_MAE"  = sd(MAE),
            "mean_RMSE"= mean(RMSE),
            "std_RMSE" = sd(RMSE),
            "empQ"     = mean(`E-quantile`),
            "empQ_u"     = mean(`E-quantile`) + sd(`E-quantile`),
            "empQ_l"     = mean(`E-quantile`) - sd(`E-quantile`)
  ) %>% 
  mutate(Algorithm = paste0(Algorithm, " (R²: ", 
                         round(mean_R2,2), " ",
                         "±", round(std_R2,2), ")"))


sum_data %>% ggplot(., aes(x=Quantile, y=empQ, color = Algorithm)) +
  geom_line(size=0.5) +
  geom_abline(intercept = 0.0, slope = 1, linetype = "dashed", color="gray") + 
  geom_errorbar(aes(ymin=empQ_l, ymax=empQ_u), size=0.5, alpha=0.25,
                position=position_dodge(0.05)) +
  # scale_color_viridis_d() + 
  labs(title    = "Contamination Problem",
       subtitle = "Sample size: 400",
       x = "Interval",
       y = "Coverage") + 
  theme_minimal() +
  theme(legend.position="top", legend.box = "horizontal") +
  theme(axis.text.x=element_text(angle=-90, hjust=0, vjust=1)) +
  theme(plot.title=element_text(size=15, hjust=0.5, face="bold", vjust=-1)) +
  theme(plot.subtitle=element_text(size=12, hjust=0.5, vjust=-1)) +
  theme(text = element_text(size=12)) +
  theme(legend.title = element_text(face = "bold")) 

ggsave(filename = "figs/CON_acc400.png", 
       device = "png", 
       dpi = dpi, width = width, height = height)
  