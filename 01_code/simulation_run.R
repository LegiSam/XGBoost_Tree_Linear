# Source setuo
source("01_code/simulation_setup.R")

# Number of runs
sim_runs <- 100

# Run simulation n times
sim_results <- foreach(n = 1:sim_runs,
                       .combine = "bind_rows") %do% {
  
  # Status
  print(paste(n, "of", sim_runs))
                         
  # Draw number of observations
  n_obs <- base::sample(x = 100:2500, size = 1)
                                                
  # Simulate data                       
  df_list_current <- sim_data(n_sim = n_obs)
  
  # Run simulation
  result_list_current <- sim_run(df_list = df_list_current,
                                 param_list = sim_params)
  
  # Add n to result
  result_list_current <- map(result_list_current, ~ mutate(.x, n = n_obs))
  
  return(result_list_current)
  
}

save(sim_results, file = "02_results/df_results.RData")

