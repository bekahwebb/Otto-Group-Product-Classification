library(lightgbm)
library(tidyverse)
library(vroom)
library(tidymodels)
library(doParallel)
library(parsnip)
library(bonsai)
library(dials)

# Load data
otto_sample <- vroom('sampleSubmission.csv')
otto_test <- vroom('test.csv')
otto_train <- vroom('train.csv')

# Data preparation
clean_otto <- otto_train %>%
  select(id, target, everything())

# Parallel backend try running it without parallel processing
#c1 <-makeCluster(parallel::detectCores() - 2)
#registerDoParallel(c1)

# Recipe for preprocessing
otto_recipe <- recipe(target ~ ., data = clean_otto) %>%
  step_rm(id) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.9)

prepped_recipe <- prep(otto_recipe) 
baked <- bake(prepped_recipe, new_data = clean_otto)

# Define bagging_fraction parameter
bagging_fraction_param <- new_quant_param(
  type = "double",
  range = c(0.1, 1),
  label = c(bagging_fraction = "Bagging Fraction"),
  finalize = NULL,
  inclusive = c(TRUE, TRUE)
)

# LightGBM model specification
lgbm_model <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
) %>%
  set_engine("lightgbm", bagging_fraction = tune()) %>% 
  set_mode("classification")

  
# Workflow
otto_workflow <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(lgbm_model)

# Cross-validation
folds <- vfold_cv(clean_otto, v = 3)

# Grid of hyperparameters
lgbm_grid <- grid_regular(
  trees(range = c(50, 200)),               # Number of trees
  tree_depth(range = c(4, 8)),            # Maximum tree depth
  learn_rate(range = c(0.01, 0.2)),      # Learning rate
  loss_reduction(range = c(0, 1)),      # Minimum loss reduction  
  bagging_fraction_param,              # Bagging fraction (custom parameter)
  levels = 3
)

#run the CV
CV_results <- otto_workflow %>%
  tune_grid(
    resamples = folds,
    grid = lgbm_grid,
    metrics = metric_set(mn_log_loss),
    control = control_grid(save_pred = TRUE)
  )

# Stop parallel backend
#stopCluster(c1)

# Find the best parameters
bestTune <- select_best(CV_results, metric = "mn_log_loss")
print(bestTune)

# Finalize the workflow
final_wf <- finalize_workflow(otto_workflow, bestTune) %>%
  fit(data = clean_otto)

# Predict on test data
lgbm_predictions <- predict(final_wf, new_data = otto_test, type = "prob")

# Format predictions for submission
colnames(lgbm_predictions) <- gsub(".pred_", "", colnames(lgbm_predictions))

lgbm_submission <- lgbm_predictions %>%
  bind_cols(otto_test %>% select(id)) %>%
  select(id, everything())

# Write the submission file
vroom_write(lgbm_submission, "LGBM_Otto_Preds.csv", delim = ",")