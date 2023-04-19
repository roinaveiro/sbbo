library(tidyverse)
library(latex2exp)

dpi <- 700
width <- 8.33
height <- 5.79

################
# OPT PLOTS
################

data <- read_csv("results/CON/CON_full_results.csv")


pdata <- data %>% select(-current_vals) %>% 
  mutate(Algorithm = algorithm) %>% 
  filter( !(Algorithm %in% c("sbbo-NGBlin")) ) %>% 
  group_by(iter, Algorithm) %>% 
  summarise("Best Value" = mean(best_vals),  "Lower" = mean(best_vals) - 1*sd(best_vals),
            "Upper" = mean(best_vals) + 1*sd(best_vals), "Std. Dev" = sd(best_vals))

branded_colors <- c(
  "RS" = "#00798c",
  "SA" = "#d1495b",
  "sbbo-BOCS" = "#edae49",
  "sbbo-GPr" = "#66a182",
  "sbbo-NGBdec" = "#2e4057",
  "sbbo-NGBlinCV" = "#8d96a3"
)

pdata %>% filter(iter %% 5 == 0) %>% 
  ggplot(aes(x=iter, y=`Best Value`, color=Algorithm, fill=Algorithm)) + 
  geom_line(size=0.5) +
  geom_ribbon(aes(ymin=Lower,ymax=Upper), linetype = 0, alpha=0.1) +
  #geom_errorbar(aes(ymin=Lower, ymax=Upper), size=0.5, alpha=0.25,
  #              position=position_dodge(0.05)) +
  # scale_color_viridis_d() + 
  labs(title    = "Binary Quadratic Problem",
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



ggsave(filename = "figs/BQP.png", 
       device = "png", 
       dpi = dpi, width = width, height = height)

################
# ACC PLOTS
################

data <- read_csv("results/CON/acc_CON_ss50.csv")

sum_data <- data %>% group_by(Algorithm, Quantile) %>% 
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
       subtitle = "Sample size: 50",
       x = "Interval",
       y = "Coverage") + 
  theme_minimal() +
  theme(legend.position="top", legend.box = "horizontal") +
  theme(axis.text.x=element_text(angle=-90, hjust=0, vjust=1)) +
  theme(plot.title=element_text(size=15, hjust=0.5, face="bold", vjust=-1)) +
  theme(plot.subtitle=element_text(size=12, hjust=0.5, vjust=-1)) +
  theme(text = element_text(size=12)) +
  theme(legend.title = element_text(face = "bold")) 

ggsave(filename = "figs/CON_acc50.png", 
       device = "png", 
       dpi = dpi, width = width, height = height)
  