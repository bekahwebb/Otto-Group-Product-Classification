library(tidyverse)
library(vroom)
library(ggplot2)
library(tidyverse)
library(patchwork)
library(dplyr)
library(recipes)
library(tidymodels)
library(GGally)
library(reshape2)
library(yardstick)
library(doParallel)
library(discrim)

otto_sample <- vroom('sampleSubmission.csv')
#otto_sample
otto_test <- vroom('test.csv') # Rows: 144368 Columns: 94
#otto_test
otto_train <- vroom('train.csv')# Rows: 61878 Columns: 95
#otto_train

clean_otto <-otto_train %>%
  select(id, target, everything()) # This ensures 'target' is included as the first column
 clean_otto

# #EDA
# #check for missing data
anyNA(otto_train)    # TRUE if there are missing values, false
# # 
clean_otto %>%
  count(target) %>%
  ggplot(aes(x = target, y = n, fill = target)) +
  geom_bar(stat = "identity") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

pca_result <- prcomp(otto_train %>% select(starts_with("feat_")), center = TRUE, scale. = TRUE)
var_explained <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
qplot(1:length(var_explained), var_explained, geom = "line") +
  labs(title = "Cumulative Variance Explained", x = "Principal Component", y = "Variance Explained")

# Summarize basic statistics for features
summary(clean_otto %>% select(starts_with("feat")))

# Example PCA to reduce dimensions for clustering
pca_result <- prcomp(clean_otto %>% select(starts_with("feat_")), center = TRUE, scale. = TRUE)
# Visualize the first two principal components
pca_data <- as.data.frame(pca_result$x)
pca_data$target <- clean_otto$target
ggplot(pca_data, aes(x = PC1, y = PC2, color = target)) +
  geom_point(alpha = 0.6) +
  labs(title = "PCA of Features with Target Classes")
#

#Random forest classifier
 # Parallel Backend Setup
 cl <- makeCluster(parallel::detectCores() - 1)
 registerDoParallel(cl)
#my recipe
clean_otto <-otto_train %>%
   select(id, target, everything()) # This ensures 'target' is included as the first column
#Feature Engineering
otto_recipe <- recipe(target ~ ., data = clean_otto) %>%
  step_rm(id) %>%
  step_normalize(all_numeric_predictors()) %>%   # Normalize features if numeric
  step_pca(all_numeric_predictors()) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.9)


prepped_recipe <- prep(otto_recipe)
#Baking
baked <- bake(prepped_recipe, new_data = clean_otto)
baked
rf_otto_model <- rand_forest(mtry = tune(),
                            min_n=tune(),
                            trees=200) %>%
  set_engine("ranger", importance = "impurity", num.threads = 1) %>%
  set_mode("classification")

## Create a workflow with model & recipe

rf_wf <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(rf_otto_model)

## Set up grid of tuning values
rf_tunegrid <- grid_regular(
  mtry(range = c(1, 10)),
  min_n(range = c(1, 10)),
  levels = 3)

## Set up K-fold CV
folds <- vfold_cv(clean_otto, v = 5)

## Run the CV
CV_results <- rf_wf %>%
  tune_grid(
    resamples=folds,
    grid=rf_tunegrid,
    metrics=metric_set(mn_log_loss),
    control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# Stop Cluster
stopCluster(cl)
#started at 9:44p still running at 6:46 am oy ve, it's maybe spinning it's wheels
#started at 10:10 p ended around 11:11 p 
## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "mn_log_loss")
bestTune
# tibble: 1 × 3
#mtry min_n .config             
#<int> <int> <chr>               
#  1     1    10 Preprocessor1_Model7
## Finalize workflow and predict
final_wf <-
  rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=clean_otto)

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
#with 200 trees, got .84042
#XG Boost

clean_otto <-otto_train %>%
  select(id, target, everything()) # This ensures 'target' is included as the first column

clean_otto$target <- as.factor(clean_otto$target)
otto_recipe <- recipe(target ~ ., data = clean_otto) %>%
  step_rm(id)
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_numeric_predictors())

prepped_recipe <- prep(otto_recipe)
#Baking
baked <- bake(prepped_recipe, new_data = clean_otto)
#baked
xg_model <- boost_tree(
  trees = 100,
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = 0.8,
  mode = "classification"
) %>%
  set_engine("xgboost", nthread = parallel::detectCores()) %>%
  set_mode("classification")

# Workflow and Tuning
otto_workflow <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(xg_model)

# Cross-Validation
folds <- vfold_cv(clean_otto, v = 3)

# Set up grid of tuning values
grid_of_tuning_params <- grid_regular(
  tree_depth(range = c(4, 8)),
  learn_rate(range = c(0.1, 0.2)),
  loss_reduction(range = c(0, 2)),
  levels = 2 #smaller grid
)
#register parallel backend
registerDoParallel(cores = parallel::detectCores())

# Tune Model with parallel backend
CV_results <- otto_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(mn_log_loss))
            #control = control_grid(verbose = TRUE)
# Stop parallel backend after tuning
stopImplicitCluster()

# Find Best Parameters
bestTune <- select_best(CV_results, metric = "mn_log_loss")
bestTune

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

# #Naive Bayes

otto_recipe <- recipe(target ~ ., data = clean_otto) %>%
  step_rm(id) %>%   # Ensure 'id' is not used in training
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_numeric_predictors()) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.9)


prepped_recipe <- prep(otto_recipe)
#prepped_recipe
#Baking
baked <- bake(prepped_recipe, new_data = clean_otto)
baked
otto_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes

## Create a workflow with model & recipe

otto_wf <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(otto_model)

## Set up grid of tuning values
tuning_grid <- grid_regular(Laplace(range = c(0, 2)), smoothness(range = c(0, 2)),levels = 5)

## Set up K-fold CV
folds <- vfold_cv(clean_otto, v = 10, strata = target)

# Parallel Processing
cl <- makeCluster(parallel::detectCores() - 1)

registerDoParallel(cl)

## Run the CV
CV_results <- otto_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(mn_log_loss))

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric="mn_log_loss")

# smoothness Laplace .config
#<dbl>   <dbl> <chr>
#  1          1       0 Preprocessor1_Model3
## Finalize workflow and predict
final_wf <-
  otto_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=clean_otto)

## Predict
otto_predictions <- final_wf %>%
  predict(new_data =otto_test, type="prob")
# Check column names to see the exact format
colnames(otto_predictions)<- gsub(".pred_", "", colnames(otto_predictions))
#colnames(otto_predictions)
## Format the Predictions for Submission to Kaggle
naivebayes_otto_kaggle_submission <- otto_predictions%>%
  bind_cols(otto_test %>% select(id)) %>%  # Add IDs Bind predictions with test data
  select(id, everything())  # Ensure 'id' is the first column

## Write out the file
vroom_write(x=naivebayes_otto_kaggle_submission, file="NbottoPreds.csv", delim = ",")
#public score 1.21478 :(, done, the cutoff is .55
#with just removing id in the recipe the public score is 21.41065 woah-if only I can subtract 21 I'd be there :)
# Stop the cluster after running
stopCluster(cl)
#got it down to 1.21261 without putting a cap on pca
