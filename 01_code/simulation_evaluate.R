library(dplyr)
library(ggplot2)
library(reshape2)
library(wesanderson)

rm(list = ls())

# Load data and grids
load("02_results/df_results.RData")

head(sim_results)

# STATS -------------------------------------------------------------------

sim_results %>% 
  group_by(booster, data_set) %>% 
  summarise(metric_mean = round(mean(metric_test), 3),
            metric_sd = round(sd(metric_test), 3)) %>% 
  arrange(data_set, booster)


# PLOTS -------------------------------------------------------------------

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

# Performance correlation

df_linear <- filter(sim_results, booster == "linear") %>% transmute(metric_test_linear = metric_test, data_set)
df_tree <- filter(sim_results, booster == "tree") %>% transmute(metric_test_tree = metric_test)
df_cor <- bind_cols(df_linear, df_tree)

p3 <- df_cor %>% 
  mutate(data_set = factor(data_set, levels = 1:4, labels = paste("Dataset", 1:4))) %>% 
  ggplot(., aes(x = metric_test_linear, y = metric_test_tree)) + 
  geom_point(color = wes_palette("GrandBudapest")[1]) + 
  facet_wrap(~data_set, scales = "free") + 
  labs(x = "Performance Linear", y = "Performance Tree", 
       title = "Fig. 3: Evaluation Error (Out-of-Sample) by Correlation") + 
  theme_minimal() + 
  geom_smooth(method = "lm", alpha = .15, color = wes_palette("GrandBudapest")[2])
  
df_cor %>% 
  group_by(data_set) %>% 
  summarise(correlation = cor(metric_test_linear, metric_test_tree))


# N obs
p4 <- sim_results %>% 
  ggplot(., aes(x = n, y = metric_test, color = booster)) + 
    geom_point() + 
    scale_color_manual("Booster", values = wes_palette("GrandBudapest")) + 
    facet_wrap(~data_set, scales = "free") + 
    labs(x = "Number of simulated data points", y = "Evaluation Metric (AUC/RMSE)", 
         title = "Fig. 4: Evaluation Error (Out-of-Sample) by Dataset and Booster") + 
    theme_minimal() + 
    geom_smooth(method = "lm", alpha = .15)


# Save
ggsave("03_docu/plots/result_oos_classification.png", p1, width = 8, height = 6, units = "in")
ggsave("03_docu/plots/result_oos_regression.png", p2, width = 8, height = 6, units = "in")
ggsave("03_docu/plots/result_oos_cor.png", p3, width = 8, height = 6, units = "in")
ggsave("03_docu/plots/result_oos_n_sims.png", p4, width = 8, height = 6, units = "in")


