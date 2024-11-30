library(vroom)
library(tidyverse)
library(dplyr)
library(patchwork)
library(tidymodels)
library(glmnet)
library(embed)
library(discrim)
library(naivebayes)
library(themis)
library(bonsai)
library(lightgbm)

otto_sample <- vroom('sampleSubmission.csv')
#otto_sample
otto_test <- vroom('test.csv') # Rows: 144368 Columns: 94
#otto_test
otto_train <- vroom('train.csv')# Rows: 61878 Columns: 95
#otto_train

# #Naive Bayes

otto_recipe <- recipe(target ~ ., data = otto_train) %>%
  step_rm(id) %>%   # Ensure 'id' is not used in training
  step_mutate_at(all_numeric_predictors(), fn = factor)


prepped_recipe <- prep(otto_recipe)
#prepped_recipe
#Baking
baked <- bake(prepped_recipe, new_data = otto_train)
baked
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes

## Create a workflow with model & recipe

nb_wf <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(nb_model)

## Set up grid of tuning values
tuning_grid_nb <- grid_regular(Laplace(), smoothness(),levels = 5)

## Set up K-fold CV
folds <- vfold_cv(otto_train, v = 5, repeats = 1)

## Run the CV
CV_results_nb <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_nb,
            metrics=metric_set(accuracy))

## Find best tuning parameters
bestTune_nb <- CV_results_nb %>%
  select_best(metric="accuracy")

# smoothness Laplace .config
#<dbl>   <dbl> <chr>
#  1          1       0 Preprocessor1_Model3
## Finalize workflow and predict
final_wf_nb <-
  nb_wf %>%
  finalize_workflow(bestTune_nb) %>%
  fit(data=otto_train)

## Predict
otto_nb_predictions <- final_wf_nb %>%
  predict(new_data =otto_test, type="prob")
# Check column names to see the exact format
colnames(otto_nb_predictions)<- gsub(".pred_", "", colnames(otto_nb_predictions))
#colnames(otto_predictions)
## Format the Predictions for Submission to Kaggle
naivebayes_otto_kaggle_submission <- otto_nb_predictions%>%
  bind_cols(otto_test %>% select(id)) %>%  # Add IDs Bind predictions with test data
  select(id, everything())  # Ensure 'id' is the first column

## Write out the file
vroom_write(x=naivebayes_otto_kaggle_submission, file="NbottoPreds.csv", delim = ",")
#public score 1.21478 :(, done, the cutoff is .55
#with just removing id in the recipe the public score is 21.41065 woah-if only I can subtract 21 I'd be there :)

#got it down to 1.21261 without putting a cap on pca
#went up to 1.50123 with step pca