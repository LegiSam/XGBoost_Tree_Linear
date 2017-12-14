library(plyr)
library(dplyr)
library(purrr)
library(caret)
library(xgboost)

rm(list = ls())

# https://github.com/dmlc/xgboost/issues/332

# DATA SIMULATION ---------------------------------------------------------

# Number of simulated data points
n_sim <- 10e3

# Classification data
df_class_1 <- twoClassSim(n = 1000, linearVars = 10, noiseVars = 5)
df_class_2 <- LPH07_1(n = 1000, class = TRUE, noiseVars = 5, corrVars = 5)

# Build numeric target
df_class_1 <- df_class_1 %>% 
  mutate(y = as.numeric(Class) - 1) %>% 
  dplyr::select(-Class)

df_class_2 <- df_class_2 %>% 
  mutate(y = as.numeric(Class) - 1) %>% 
  dplyr::select(-Class)

# Regression data
df_reg_1 <- LPH07_2(n = 1000, noiseVars = 5)
df_reg_2 <- SLC14_1(n = 1000, noiseVars = 5, corrVars = 5)

# Combine data
df_list <- list(df_class_1, df_class_2, df_reg_1, df_reg_2)


# TUNING SETTINGS ---------------------------------------------------------

# General training settings
n_rounds <- 1000
n_folds <- 10

# Parameter grid (tree booster)
tree_grid <- expand.grid(eta = c(0.3),
                         max_depth = c(1, 3, 6, 9),
                         subsample = c(0.75, 1),
                         colsample_bytree = c(0.75, 1),
                         lambda = c(0, 0.5, 1, 5, 10),
                         alpha = c(0, 0.5, 1, 5, 10))

# Parameter grid (linear booster)
linear_grid <- expand.grid(lambda = c(0, 0.5, 1, 5, 10),
                           alpha = c(0, 0.5, 1, 5, 10))


# RUN MODELS --------------------------------------------------------------

list_results_cv <- list()

# Loop over datasets
for (i in seq_along(df_list)) {
  
  # Status
  print(paste("Dataset", i, "of", length(df_list)))
  
  # Settings
  objective <- ifelse(i <= 2, "binary:logistic", "reg:linear")
  eval_metric <- ifelse(i <= 2, "auc", "rmse")
  
  # Data
  df_model <- df_list[[i]]
  
  # Model data
    
  # Train/Test split
  in_train <- unlist(createDataPartition(y = df_model$y, p = 0.7))
  df_train <- df_model[in_train, ]
  df_test <- df_model[-in_train, ]
  
  # Train/Validation split
  in_val <- unlist(createDataPartition(y = df_train$y, p = 0.3))
  df_val <- df_train[in_val, ]
  df_train <- df_train[-in_val, ]
  
  # Build design matrix
  x_train <- dplyr::select(df_train, -y) %>% as.matrix(.)
  x_test <- dplyr::select(df_test, -y) %>% as.matrix(.)
  x_val <- dplyr::select(df_val, -y) %>% as.matrix(.)
  
  # Build xgb.DMatrix
  xgb_train <- xgb.DMatrix(data = x_train, label = df_train$y)
  xgb_test <- xgb.DMatrix(data = x_test, label = df_test$y)
  xgb_val <- xgb.DMatrix(data = x_val, label = df_val$y)

  # CV result containers
  results_tuning_linear <- data.frame(param_id = 1:nrow(linear_grid),
                                      metric_train = NA,
                                      metric_cv = NA,
                                      best_iter = NA,
                                      booster = "linear",
                                      data_set = i)
  
  results_tuning_tree <- data.frame(param_id = 1:nrow(tree_grid),
                                    metric_train = NA,
                                    metric_cv = NA,
                                    best_iter = NA,
                                    booster = "tree",
                                    data_set = i)

  # Progressbars
  pb_linear <- txtProgressBar(min = 1, max = nrow(linear_grid), style = 3)
  pb_tree <- txtProgressBar(min = 1, max = nrow(tree_grid), style = 3)
  
  # Tuning linear booster
  for (j in 1:nrow(linear_grid)) {
    
    # Training with k-fold cv
    xgb_mod_linear <- xgb.cv(data = xgb_train,
                             params = apply(linear_grid, 1, as.list)[[j]],
                             objective = objective,
                             eval_metric = eval_metric,
                             booster = "gblinear",
                             nrounds = n_rounds,
                             nfold = n_folds,
                             watchlist = list(train = xgb_train, val = xgb_val),
                             early_stopping_rounds = 10,
                             verbose = 0)
    
    # Save results
    best_iter_linear <- xgb_mod_linear$best_iteration
    results_tuning_linear$best_iter[j] <- best_iter_linear
    results_tuning_linear$metric_train[j] <- as.numeric(xgb_mod_linear$evaluation_log[best_iter_linear, 2])
    results_tuning_linear$metric_cv[j] <- as.numeric(xgb_mod_linear$evaluation_log[best_iter_linear, 4])
    
    # Update bar
    setTxtProgressBar(pb_linear, j)
  }
  
  # Close bar
  close(pb_linear)
  
  # Tuning tree booster
  for (j in 1:nrow(tree_grid)) {
    
    # Training with k-fold cv
    xgb_mod_tree <- xgb.cv(data = xgb_train,
                           params = apply(tree_grid, 1, as.list)[[j]],
                           objective = objective,
                           eval_metric = eval_metric,
                           booster = "gbtree",
                           nrounds = n_rounds,
                           nfold = n_folds,
                           watchlist = list(train = xgb_train, val = xgb_val),
                           early_stopping_rounds = 10,
                           verbose = 0)
    
    # Save results
    best_iter_tree <- xgb_mod_tree$best_iteration
    results_tuning_tree$best_iter[j] <- best_iter_tree
    results_tuning_tree$metric_train[j] <- as.numeric(xgb_mod_tree$evaluation_log[best_iter_tree, 2])
    results_tuning_tree$metric_cv[j] <- as.numeric(xgb_mod_tree$evaluation_log[best_iter_tree, 4])
    
    # Update bar
    setTxtProgressBar(pb_tree, j)
  }
  
  # Close bar
  close(pb_tree)
  
  # Save results for current dataset
  list_results_cv[[i]] <- suppressWarnings({
    bind_rows(results_tuning_linear, results_tuning_tree)
    })
  
}

# Save results
save(list_results_cv, file = "~/Intern/Projekte/Blog/XGBoost_Tree_Linear/02_results/list_results_cv.RData")
save(tree_grid, linear_grid, file = "~/Intern/Projekte/Blog/XGBoost_Tree_Linear/02_results/param_grids.RData")

# LM BENCHMARK ------------------------------------------------------------

# Caret trControls
reg_control <- trainControl(method = "cv", number = 10)

class_control <- trainControl(method = "cv", number = 10, 
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

# Models
mod_1 <- df_list[[1]] %>% 
  mutate(y = factor(y, levels = c(0, 1), labels = c("No", "Yes"))) %>% 
  train(y ~ .,
        method = "glm",
        data = .,
        trControl = class_control,
        metric = "ROC",
        family = "binomial")

mod_2 <- df_list[[2]] %>% 
  mutate(y = factor(y, levels = c(0, 1), labels = c("No", "Yes"))) %>% 
  train(y ~ .,
        method = "glm",
        data = .,
        trControl = class_control,
        metric = "ROC",
        family = "binomial")

mod_3 <- df_list[[3]] %>% 
  train(y ~.,
        method = "lm",
        data = .,
        trControl = reg_control)

mod_4 <- df_list[[4]] %>% 
  train(y ~.,
        method = "lm",
        data = .,
        trControl = reg_control)

# Save benchmarks
list_results_benchmarks <- list(data_set_1 = mod_1$results$ROC,
                                data_set_2 = mod_2$results$ROC,
                                data_set_3 = mod_3$results$RMSE,
                                data_set_4 = mod_4$results$RMSE)

save(list_results_benchmarks, file = "02_results/list_results_benchmarks.RData")





