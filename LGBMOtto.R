library(lightgbm)
library(tidyverse)
library(vroom)
library(tidymodels)
library(parsnip)
library(bonsai)
library(dials)

# Load data
otto_sample <- vroom('sampleSubmission.csv')
otto_test <- vroom('test.csv')
otto_train <- vroom('train.csv')

# # Data preparation
# clean_otto <- otto_train %>%
#   select(id, target, everything())

# Parallel backend try running it without parallel processing
# c1 <-makeCluster(parallel::detectCores() - 3)#reduce the number of cores
# registerDoParallel(c1)

# Recipe for preprocessing
otto_recipe <- recipe(target ~ ., data = otto_train) %>%
  update_role(id, new_role = "Id") %>%
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE)

prepped_recipe <- prep(otto_recipe) 
baked <- bake(prepped_recipe, new_data = otto_train)

# LightGBM model specification
lgbm_model <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
) %>%
  set_engine("lightgbm") %>% # prob don't need , bagging_fraction = 0.8use default or fixed value
  set_mode("classification")

  
# Workflow
otto_workflow <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(lgbm_model)

# Grid of hyperparameters
lgbm_grid <- grid_regular(
  trees(),               # Number of trees
  tree_depth(),            # Maximum tree depth
  learn_rate(),      # Learning rate              # Bagging fraction (custom parameter)
  levels = 4                       
)

# Cross-validation
folds <- vfold_cv(otto_train,v = 4, repeats = 1)

#run the CV
CV_results <- otto_workflow %>%
  tune_grid(
    resamples = folds,
    grid = lgbm_grid,
    metrics = metric_set(mn_log_loss),
    control = control_grid(save_pred = TRUE, verbose = TRUE)
  )

# Stop parallel backend
# stopCluster(c1)

# Find the best parameters
bestTune <- select_best(CV_results, metric = "mn_log_loss")
#print(bestTune)
# trees tree_depth learn_rate .config            
#   <int>      <int>      <dbl> <chr>              
# 1    50         10       1.14 Preprocessor1_Mode…
# > print(bestTune)
# trees tree_depth learn_rate .config  with 3 levels and 3 folds          
# <int>      <int>      <dbl> <chr>              
#   1    50         10       1.14 Preprocessor1_Mode…
# > with 5 levels 5 folds, similar predictions just a little lower tree_depth of 8
# Finalize the workflow
final_wf <- finalize_workflow(otto_workflow, bestTune) %>%
  fit(data = otto_train)

# Predict on test data
lgbm_predictions <- predict(final_wf, new_data = otto_test, type = "prob")

# Format predictions for submission
colnames(lgbm_predictions) <- sub(".pred_", "", colnames(lgbm_predictions))
#colnames(lgbm_predictions)

lgbm_submission <- lgbm_predictions %>%
  bind_cols(otto_test %>% select(id)) %>%
  select(id, everything())

# Write the submission file
vroom_write(lgbm_submission, "LGBM_Otto_Preds.csv", delim = ",")
# woah after a zillion hours on the batch server, it finally ran through all the code 13827.873 and a huge score of 8.16612
#with crazy fast result time of 171.087 I got a really bad result of 23.12092 :(
# result time 4847.395 it got lower, in the right direction but still high 6.64180
#result time 149.489, with accuracy as the metric oy ve 24.12851
#result time 1257.871 with accuracy and diff. submission format, went way up to 29.36019
#ok ok, we are going in the way right direction 0.53973 with a reasonable time of 3710.731 on the batch server with 3x3, try 5x5 fc
#threshold is a .487
#1236 start time running the whole flipping thing on my laptop fc 125 processes of 70000 data points woah k it took 6 hrs and went up to .62177 w/accuracy?
#trying mn_log_loss 3x3 grid start time 7:12p fc finished at 8:44p got .56871 try 4x4 aeound 9 p finished around 12:30 a oh my gosh I got it! 0.47655, yay!
