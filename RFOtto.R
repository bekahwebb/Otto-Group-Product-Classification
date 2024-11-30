library(tidyverse)
library(vroom)
library(tidymodels)

otto_sample <- vroom('sampleSubmission.csv')
#otto_sample
otto_test <- vroom('test.csv') # Rows: 144368 Columns: 94
#otto_test
otto_train <- vroom('train.csv')# Rows: 61878 Columns: 95
#otto_train

clean_otto <-otto_train %>%
  select(id, target, everything()) # This ensures 'target' is included as the first column
#clean_otto

#Random forest classifier

#my recipe
clean_otto <-otto_train %>%
  select(id, target, everything()) # This ensures 'target' is included as the first column
#Feature Engineering
otto_recipe <- recipe(target ~ ., data = clean_otto) %>%
  step_rm(id) %>%
  step_normalize(all_numeric_predictors()) %>%   # Normalize features if numeric
  step_corr(all_numeric_predictors(), threshold = 0.9)


prepped_recipe <- prep(otto_recipe)
#Baking
baked <- bake(prepped_recipe, new_data = clean_otto)
#baked
rf_otto_model <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=850) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

## Create a workflow with model & recipe

rf_wf <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(rf_otto_model)

## Set up grid of tuning values
rf_tunegrid <- grid_regular(
  mtry(range = c(2,94)),
  min_n(),
  levels = 5)

## Set up K-fold CV
folds <- vfold_cv(clean_otto, v = 5, repeats = 1)

## Run the CV
CV_results <- rf_wf %>%
  tune_grid(
    resamples=folds,
    grid=rf_tunegrid,
    metrics=metric_set(roc_auc),
    control = control_grid(save_pred = TRUE)
  )


#started at 9:44p still running at 6:46 am oy ve, it's maybe spinning it's wheels
#started at 10:10 p ended around 11:11 p 
## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")
bestTune
# tibble: 1 × 3
#mtry min_n .config             
#<int> <int> <chr>               
#  1     1    10 Preprocessor1_Model7
## Finalize workflow and predict
final_wf <-
  rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=otto_train)

## Predict
rf_otto_predictions <- final_wf %>%
  predict(new_data = otto_test, type="prob")

# Check column names to see the exact format
colnames(rf_otto_predictions)<- gsub(".pred_", "", colnames(rf_otto_predictions))

## Format the Predictions for Submission to Kaggle
rf_otto_kaggle_submission <- rf_otto_predictions%>%
  bind_cols(otto_test %>% select(id)) %>%  # Add IDs Bind predictions with test data
  select(id, everything())  # Ensure 'id' is the first column

## Write out the file
vroom_write(x=rf_otto_kaggle_submission, file="RfOttoPreds.csv", delim=",")

# Validation
# Use stratified k-fold cross-validation to evaluate the effect of engineered features on log loss.
#Ensure engineered features improve model performance on the validation set. start time 2:25 p
#ran on the server, got .65998 with the rf, it said it took around 13 min. on the server encouraging ... try XG Boost :)
#with 200 trees, got .84042 around 13 min. on the server
#with 750 trees took 55 min..59332! closer to .55 :) try 850
#with 850 trees 2024 seconds (about 34 minutes) 0.59324 prob maxed out trees
#try with 500 trees and a metric of roc_auc