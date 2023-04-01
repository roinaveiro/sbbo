library(tidyverse)
library(latex2exp)

dpi <- 700
width <- 8.33
height <- 5.79


data <- read_csv("results/BQP/BQP_full_results.csv")


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

  