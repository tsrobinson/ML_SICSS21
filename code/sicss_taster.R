################################################################################
##                                                                            ##
##            Summer Institute in Computational Social Science (Oxford)       ##
##                                Machine Learning                            ##
##                                                                            ##
##                                                                            ##
##                              Neural Networks in R                          ##
##                                  Walkthrough                               ##
##                                                                            ##
################################################################################


## The data and task for today's workshop is taken from the replication materials
# for "Estimating Treatment Effects with Causal Forests: An Application", by 
# Athey and Wager, and formed part of a wider conference exercise on estimating
# heterogeneous treatment effects.

## The Athey and Wager materials can be found here:
# https://arxiv.org/abs/1902.07409

# More general information about the workshop can be found here:
# https://muse.jhu.edu/article/793355/summary

#### 0. System setup ####

## Set up commands -- follow this to set up on your own machine
# install.packages("remotes")
# remotes::install_github("rstudio/tensorflow")
# install.packages("reticulate")
# tensorflow::install_tensorflow()

## If you get the message "No non-system installation of Python could be found.
# "Would you like to download and install Miniconda?"
# Type 'Y' and press enter

# install.packages("keras")
# keras::install_keras()
# install.packages("recipes")

# If you do not already have tidyverse installed:
# install.packages("tidyverse")

#### 1. Load packages and data ####
library(tensorflow)
library(keras)
library(tidyverse)
library(recipes)

set.seed(89)

## Read in the data
acic18 <- read_csv("https://raw.githubusercontent.com/grf-labs/grf/master/experiments/acic18/synthetic_data.csv") %>% 
  mutate(across(contains("C"), as.factor))

#### 2. Setup the counterfactual data ####

acic18_counterfac <- acic18 %>% 
  mutate(Z = case_when(Z == 1 ~ 0,
                       Z == 0 ~ 1))

#### 3. Build a network ####

# Construct a "recipe"
rec_obj <- recipe(Y ~ ., data = acic18) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>% # One-hot encode columns
  step_center(all_predictors(), -all_outcomes()) %>% # Centre all predictors on 0
  step_scale(all_predictors(), -all_outcomes()) %>% # Scale all predictors with sd=1
  prep(data = acic18)

x_train <- bake(rec_obj, new_data = acic18) %>% select(-Y)
x_test  <- bake(rec_obj, new_data = acic18_counterfac) %>% select(-Y)

y_train <- acic18$Y

## Construct a neural network
model <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 1) %>% 
  
  compile(
    optimizer = 'sgd', # Stochastic gradient descent
    loss      = 'mse', # Determines what is plotted while training occurs
  )

#### 4. Train the network ####

history <- fit(
  object = model,
  x = as.matrix(x_train), 
  y = y_train, 
  batch_size = 50,
  epochs = 50,
  validation_split = 0.30
)

#### 5. Generate counterfactual predictions and plot ITEs ####
ITEs <- acic18 %>% 
  mutate(Y_counterfac = predict(model, as.matrix(x_test))) %>% 
  mutate(ITE = case_when(Z == 1 ~ (Y - Y_counterfac),
                         Z == 0 ~ (Y_counterfac - Y))) %>% 
  arrange(ITE) %>% 
  mutate(ITE_order = 1:nrow(.))

ggplot(ITEs, aes(x = ITE_order, y = ITE)) +
  geom_line() +
  geom_hline(yintercept = 0, linetype = "dashed")

# Check correlations
corr_covars <- c("X1","X2","X3","X4","X5")

sapply(corr_covars, function (x) cor.test(ITEs[["ITE"]], ITEs[[x]]))


kruskal_vars <- c("C1","C2","C3")

sapply(kruskal_vars, function (x) kruskal.test(as.formula(paste0("ITE ~ ",x)), ITEs))

#### 5. Dropout ####

## Add dropout
model_w_dropout <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1) %>% 
  
  compile(
    optimizer = 'sgd', # Stochastic gradient descent -- a variation of what we hand-coded on Monday!
    loss      = 'mse',
  )

history2 <- fit(
  object = model_w_dropout,
  x = as.matrix(x_train), 
  y = y_train, 
  batch_size = 50,
  epochs = 100,
  validation_split = 0.30
)

ITEs2 <- acic18 %>% 
  mutate(Y_counterfac = predict(model_w_dropout, as.matrix(x_test))) %>% 
  mutate(ITE = case_when(Z == 1 ~ (Y - Y_counterfac),
                         Z == 0 ~ (Y_counterfac - Y))) %>% 
  arrange(ITE) %>% 
  mutate(ITE_order = 1:nrow(.))

ggplot(ITEs2, aes(x = ITE_order, y = ITE)) +
  geom_line() +
  geom_hline(yintercept = 0, linetype = "dashed")
