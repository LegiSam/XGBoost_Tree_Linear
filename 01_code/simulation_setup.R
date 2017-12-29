library(plyr)
library(dplyr)
library(purrr)
library(caret)
library(xgboost)
library(pROC)
library(foreach)
library(doParallel)
library(purrr)

rm(list = ls())

# DATA SIMULATION ---------------------------------------------------------

sim_data <- function(n_sim) {
  
  # Classification data
  df_class_1 <- twoClassSim(n = n_sim, linearVars = 10, noiseVars = 5)
  df_class_2 <- LPH07_1(n = n_sim, class = TRUE, noiseVars = 5, corrVars = 5)
  
  # Build numeric target
  df_class_1 <- df_class_1 %>% 
    mutate(y = as.numeric(Class) - 1) %>% 
    dplyr::select(-Class)
  
  df_class_2 <- df_class_2 %>% 
    mutate(y = as.numeric(Class) - 1) %>% 
    dplyr::select(-Class)
  
  # Regression data
  df_reg_1 <- LPH07_2(n = n_sim, noiseVars = 5)
  df_reg_2 <- SLC14_1(n = n_sim, noiseVars = 5, corrVars = 5)
  
  # Combine data
  df_list <- list(df_class_1, df_class_2, df_reg_1, df_reg_2)
  
  # Return
  return(df_list)
  
}


# TUNING SETTINGS ---------------------------------------------------------

sim_params <- list(

  # General training settings
  n_rounds = 1000,
  n_folds = 10,
  
  # Parameter grid (tree booster)
  tree_grid = expand.grid(eta = c(0.3),
                           max_depth = c(1, 3, 6, 9),
                           subsample = 1,
                           colsample_bytree = 1,
                           lambda = c(0, 0.5, 1, 5, 10),
                           alpha = c(0, 0.5, 1, 5, 10)),
  
  # Parameter grid (linear booster)
  linear_grid = expand.grid(lambda = c(0, 0.5, 1, 5, 10),
                            alpha = c(0, 0.5, 1, 5, 10))
  )


# RUN MODELS --------------------------------------------------------------

sim_run <- function(df_list, param_list) {
  
  # Container
  list_results_cv <- list()
  list_results_oos <- list()
  
  # Loop over datasets
  for (i in seq_along(df_list)) {
    
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
    results_tuning_linear <- data.frame(param_id = 1:nrow(param_list$linear_grid),
                                        metric_train = NA,
                                        metric_cv = NA,
                                        best_iter = NA,
                                        booster = "linear",
                                        data_set = i)
    
    results_tuning_tree <- data.frame(param_id = 1:nrow(param_list$tree_grid),
                                      metric_train = NA,
                                      metric_cv = NA,
                                      best_iter = NA,
                                      booster = "tree",
                                      data_set = i)
    
    # OOS result containers
    results_final_linear <- data.frame(metric_test = NA,
                                       booster = "linear",
                                       data_set = i)
    
    results_final_tree <- data.frame(metric_test = NA,
                                     booster = "tree",
                                     data_set = i)
    
    # Tuning linear booster
    for (j in 1:nrow(param_list$linear_grid)) {
      
      # Training with k-fold cv
      xgb_mod_linear <- xgb.cv(data = xgb_train,
                               params = apply(param_list$linear_grid, 1, as.list)[[j]],
                               objective = objective,
                               eval_metric = eval_metric,
                               booster = "gblinear",
                               nrounds = param_list$n_rounds,
                               nfold = param_list$n_folds,
                               watchlist = list(train = xgb_train, val = xgb_val),
                               early_stopping_rounds = 10,
                               verbose = 0)
      
      # Save results
      best_iter_linear <- xgb_mod_linear$best_iteration
      results_tuning_linear$best_iter[j] <- best_iter_linear
      results_tuning_linear$metric_train[j] <- as.numeric(xgb_mod_linear$evaluation_log[best_iter_linear, 2])
      results_tuning_linear$metric_cv[j] <- as.numeric(xgb_mod_linear$evaluation_log[best_iter_linear, 4])

    }
    
    # Best param set by 10-fold cv
    best_param_linear <- param_list$linear_grid[which.min(results_tuning_linear$metric_cv), ]
    
    # Train final model
    xgb_final_linear <- xgb.train(data = xgb_test,
                                  params = as.list(best_param_linear),
                                  objective = objective,
                                  eval_metric = eval_metric,
                                  booster = "gblinear",
                                  nrounds = param_list$n_rounds,
                                  watchlist = list(train = xgb_test, val = xgb_val),
                                  early_stopping_rounds = 10,
                                  verbose = 0)
    
    # Predict test set and eval
    p_linear <- predict(xgb_final_linear, xgb_test)
    
    if (eval_metric == "auc") {
      roc_obj <- pROC::roc(response = df_test$y, predictor = p_linear)
      auc_obj <- pROC::auc(roc_obj)
      metric <- as.numeric(auc_obj)
    } else {
      metric <- sqrt(mean((df_test$y - p_linear)^2))
    }
    
    results_final_linear$metric_test <- metric
    
    # Tuning tree booster
    for (j in 1:nrow(param_list$tree_grid)) {
      
      # Training with k-fold cv
      xgb_mod_tree <- xgb.cv(data = xgb_train,
                             params = apply(param_list$tree_grid, 1, as.list)[[j]],
                             objective = objective,
                             eval_metric = eval_metric,
                             booster = "gbtree",
                             nrounds = param_list$n_rounds,
                             nfold = param_list$n_folds,
                             watchlist = list(train = xgb_train, val = xgb_val),
                             early_stopping_rounds = 10,
                             verbose = 0)
      
      # Save results
      best_iter_tree <- xgb_mod_tree$best_iteration
      results_tuning_tree$best_iter[j] <- best_iter_tree
      results_tuning_tree$metric_train[j] <- as.numeric(xgb_mod_tree$evaluation_log[best_iter_tree, 2])
      results_tuning_tree$metric_cv[j] <- as.numeric(xgb_mod_tree$evaluation_log[best_iter_tree, 4])
      
    }
    
    # Best param set by 10-fold cv
    best_param_tree <- param_list$tree_grid[which.min(results_tuning_tree$metric_cv), ]
    
    # Train final model
    xgb_final_tree <- xgb.train(data = xgb_test,
                                params = as.list(best_param_tree),
                                objective = objective,
                                eval_metric = eval_metric,
                                booster = "gbtree",
                                nrounds = param_list$n_rounds,
                                watchlist = list(train = xgb_test, val = xgb_val),
                                early_stopping_rounds = 10,
                                verbose = 0)
    
    # Predict test set and eval
    p_tree <- predict(xgb_final_tree, xgb_test)
    
    if (eval_metric == "auc") {
      roc_obj <- pROC::roc(response = df_test$y, predictor = p_tree)
      auc_obj <- pROC::auc(roc_obj)
      metric <- as.numeric(auc_obj)
    } else {
      metric <- sqrt(mean((df_test$y - p_tree)^2))
    }
    
    results_final_tree$metric_test <- metric
    
    # Save results for current dataset
    list_results_oos[[i]] <- suppressWarnings({
      bind_rows(results_final_linear, results_final_tree)
    })
    
  }
  
  # Return results
  return(list_results_oos)
  
}




