library(dplyr)
library(ggplot2)
library(reshape2)
library(wesanderson)

rm(list = ls())

# Load data and grids
load("02_results/list_results_cv.RData")
load("02_results/list_results_benchmarks.RData")
load("02_results/param_grids.RData")


# Overall results ---------------------------------------------------------

results_cv <- list_results_cv %>% 
  bind_rows() %>% 
  group_by(booster, data_set) %>% 
  filter(metric_cv == min(metric_cv)) %>% 
  arrange(data_set, booster)

# Classifikation
results_cv %>% 
  dplyr::select(-param_id, -best_iter) %>% 
  filter(data_set <= 2) %>% 
  melt(., id.vars = c("data_set", "booster")) %>% 
  mutate(data_set = factor(data_set, levels = 1:2, labels = paste("Dataset", 1:2))) %>% 
  ggplot(., aes(x = booster, y = value, fill = variable)) + 
    geom_bar(stat = "identity", position = "dodge") + 
    geom_text(aes(label = round(value, 2)), 
              position = position_dodge(width = 1),
              vjust = 1.5,
              color = "white") + 
    facet_wrap(~data_set, scales = "free_y") + 
    scale_fill_manual("",
                      values = wes_palette(n = 2, name = "GrandBudapest"),
                      labels = c("Training", "Crossvalidation")
                      ) + 
    labs(x = "Booster", y = "AUC", title = "Classification Error by Dataset and Booster") + 
    theme_minimal()

# Regression
results_cv %>% 
  dplyr::select(-param_id, -best_iter) %>% 
  filter(data_set > 2) %>% 
  melt(., id.vars = c("data_set", "booster")) %>% 
  mutate(data_set = factor(data_set, levels = 3:4, labels = paste("Dataset", 3:4))) %>% 
  ggplot(., aes(x = booster, y = value, fill = variable)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  geom_text(aes(label = round(value, 2)), 
            position = position_dodge(width = 1),
            vjust = 1.5,
            color = "white") + 
  facet_wrap(~data_set, scales = "free_y") + 
  scale_fill_manual("",
                    values = wes_palette(n = 2, name = "GrandBudapest"),
                    labels = c("Training", "Crossvalidation")
  ) + 
  labs(x = "Booster", y = "RMSE", title = "Regression Error by Dataset and Booster") + 
  theme_minimal()



# Tuning results ----------------------------------------------------------

linear_grid$param_id <- 1:nrow(linear_grid)
linear_grid$booster <- "linear"
tree_grid$param_id <- 1:nrow(tree_grid)
tree_grid$booster <- "tree"

tune_grid <- bind_rows(linear_grid, tree_grid)

results_tuning <- list_results_cv %>% 
  bind_rows() %>% 
  left_join(., 
            tune_grid,
            by = c("booster", "param_id"))
  

results_tuning %>% 
  group_by(booster, data_set) %>% 
  filter(metric_cv == min(metric_cv))


