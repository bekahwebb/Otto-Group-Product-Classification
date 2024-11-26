library(tidyverse)
library(vroom)
library(tidymodels)
library(doParallel)
library(parsnip)
library(xgboost)

otto_sample <- vroom('sampleSubmission.csv')
#otto_sample
otto_test <- vroom('test.csv') # Rows: 144368 Columns: 94
#otto_test
otto_train <- vroom('train.csv')# Rows: 61878 Columns: 95
#otto_train

clean_otto <-otto_train %>%
  select(id, target, everything()) # This ensures 'target' is included as the first column
#clean_otto
#clean_otto$target <- as.factor(clean_otto$target)
otto_recipe <- recipe(target ~ ., data = clean_otto) %>%
  step_rm(id) %>% 
  step_normalize(all_numeric_predictors()) 
  
prepped_recipe <- prep(otto_recipe)
#Baking
baked <- bake(prepped_recipe, new_data = clean_otto)
#baked
xgb_model <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

#parameters(xgb_model)
# Workflow and Tuning
otto_workflow <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(xgb_model)

# Cross-Validation
folds <- vfold_cv(clean_otto, v = 3)# Reduce folds to save time

# Set up grid of tuning values
grid_of_tuning_params <- grid_regular(
  trees(range = c(50, 150)),
  learn_rate(range = c(0.01, 0.1)),
  tree_depth(range = c(3, 8)),
  min_n(range = c(2, 10)),            # Minimum number of observations in a node
  levels = 3 #smaller grid
)

# Parallel Backend
c1 <- makeCluster(parallel::detectCores() - 2)
registerDoParallel(c1)

# Tuning Control
control <- control_grid(
  verbose = TRUE,
  allow_par = TRUE,   # Enable parallel processing
  save_pred = TRUE
)

# Tune Model with parallel backend
CV_results <- otto_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(mn_log_loss),
            control = control
  )
#control = control_grid(verbose = TRUE)
# Stop parallel backend after tuning
stopCluster(c1)

# Find Best Parameters
bestTune <- select_best(CV_results, metric = "mn_log_loss")
show_notes(CV_results)

# Finalize and Fit the Best Model
final_wf <- finalize_workflow(otto_workflow, bestTune) %>%
  fit(data = clean_otto)

# Make Predictions on Test Data
xg_otto_preds <- predict(final_wf, new_data = otto_test, type = "prob")

# Check column names to see the exact format
colnames(xg_otto_preds)<- gsub(".pred_", "", colnames(xg_otto_preds))

## Format the Predictions for Submission to Kaggle
xg_otto_kaggle_submission <- xg_otto_preds %>%
  bind_cols(otto_test %>% select(id)) %>%  # Add IDs Bind predictions with test data
  select(id, everything())  # Ensure 'id' is the first column

## Write out the file
vroom_write(x=xg_otto_kaggle_submission, file="XgOttoPreds.csv", delim=",") #might have been running since 4:04pm
#ran for 4 hours, trying to run it more efficiently parallel start time 8:37p error at 9:05p
#running on the batch servers for an hour so far fc it's almost done, it says it took 12000 to run on batch, not possible and an error in the submission csv file, forgot to predict on preds my bad :(
#lowest score so far, yikes it's so high 3.64676 and it says it took 10014 to run, it was a little over 3 hours oy ve.
