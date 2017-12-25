library(dplyr)
library(ggplot2)
library(reshape2)
library(wesanderson)

rm(list = ls())

# Load data and grids
load("02_results/df_results.RData")

head(sim_results)

# AUC
p1 <- sim_results %>% 
  filter(data_set <= 2) %>% 
  mutate(data_set = factor(data_set, levels = 1:2, labels = paste("Dataset", 1:2))) %>% 
  ggplot(., aes(x = metric_test, fill = booster)) + 
    geom_density(color = "white", alpha = .8) + 
    scale_fill_manual("Booster", values = wes_palette("GrandBudapest")) + 
    facet_wrap(~data_set, scales = "free") + 
    labs(x = "AUC", y = "Density", 
         title = "Fig. 1: Classification Error (Out-of-Sample) by Dataset and Booster") + 
    theme_minimal()

# RMSE
p2 <- sim_results %>% 
  filter(data_set > 2) %>% 
  mutate(data_set = factor(data_set, levels = 3:4, labels = paste("Dataset", 3:4))) %>% 
  ggplot(., aes(x = metric_test, fill = booster)) + 
    geom_density(color = "white", alpha = .8) + 
    scale_fill_manual("Booster", values = wes_palette("GrandBudapest")) + 
    facet_wrap(~data_set, scales = "free") + 
    labs(x = "RMSE", y = "Density", 
         title = "Fig. 2: Regression Error (Out-of-Sample) by Dataset and Booster") + 
    theme_minimal()


# N obs
p3 <- im_results %>% 
  ggplot(., aes(x = n, y = metric_test, color = booster)) + 
    geom_point() + 
    scale_color_manual("Booster", values = wes_palette("GrandBudapest")) + 
    facet_wrap(~data_set, scales = "free") + 
    labs(x = "Evaluation Metric (AUC/RMSE)", y = "Number of simulated data points", 
         title = "Fig. 3: Evaluation Error (Out-of-Sample) by Dataset and Booster") + 
    theme_minimal() + 
    geom_smooth(method = "lm", alpha = .15)


# Save
ggsave("03_docu/plots/result_oos_classification.png", p1, width = 8, height = 6, units = "in")
ggsave("03_docu/plots/result_oos_regression.png", p2, width = 8, height = 6, units = "in")
ggsave("03_docu/plots/result_oos_n_sims", p3, width = 8, height = 6, units = "in")


