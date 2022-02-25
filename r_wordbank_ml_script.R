# "PH125.9x Data Science: WordBank project"
# author: "Rodrigo Dal Ben de Souza"
# date: "24/02/2022"

#In the present project, we will use machine learning algorithms to generate 
# insights about relationships between demographic/linguistic 
# variables (our predictors) and vocabulary growth, 
# as measured by **productive vocabulary** on the CDI (our outcome measure). 
# All our analyses are exploratory in nature and we don't have any hypotheses 
# or predictions on what we might find. 

# We will start with curating our dataset, moving to descriptive analyses 
# and visualizations, and finally to three machine learning algorithms 
# to the data: regression trees, random forests, and linear regression.

# Libraries

# general info from sessionInfo()
# R version 4.1.2
# Platform: aarch64-apple-darwin20 (64-bit)
# Running under: macOS Monterey 12.1

# install packages if necessary -- code based on: 
# https://statsandr.com/blog/an-efficient-way-to-install-and-load-r-packages/

# packages w/ version
pkgs <- c("wordbankr", # v0.3.1
          "tidyverse", # v1.3.1
          "here", # v1.0.1
          "caret", # v6.0-90
          "rpart", # v4.1-15
          "rpart.plot", # v3.1.0 
          "randomForest" # v4.6-14
          ) 

# if necessary install
installed_pkgs <- pkgs %in% rownames(installed.packages())
if(any(installed_pkgs == F)){install.packages(pkgs[!installed_packages])}

# load packages
invisible(lapply(pkgs, library, character.only = T))

# clean
rm(installed_pkgs, pkgs)

# color-blind friendly palette
color_blind_colors <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

# Data

# load raw data
data_raw <- wordbankr::get_administration_data()

# data overview
head(data_raw)

# drop variables
data_clean01 <- data_raw %>% 
  select(-data_id, -zygosity, -norming, -longitudinal, -source_name, -license)

# convert character to factor
str(data_clean01)

data_clean01 <- data_clean01 %>% 
  mutate(language = as_factor(language),
         form = as_factor(form))

str(data_clean01)

# calculate frequency of languages for each form
freq_form <- data_clean01 %>% 
  group_by(form) %>% 
  summarise(n_language = n_distinct(language), 
            n_obs = n()) %>% 
  arrange(desc(n_obs))

freq_form

# filter forms
data_clean02 <- data_clean01 %>% 
  filter(form %in% c("WS", "WG")) %>% 
  droplevels()

# check NAs
summary(data_clean02)

# proportion of NA in: birth order, ethnicity, caregivers' education, sex
na_prop <- data_clean02 %>% 
  select(birth_order, ethnicity, mom_ed, sex) %>% 
  gather(key = predictor) %>%
  group_by(predictor) %>% 
  summarise(prop_missing = round(sum(is.na(value))/n(),2))

# create table with proportion of missing data
na_prop %>% knitr::kable()

# drop birth order, ethnicity, caregivers' education
data_clean03 <- data_clean02 %>% 
  select(-birth_order, -ethnicity, -mom_ed) 

# glance dataset
summary(data_clean03)

# calculate median vocabulary scores 
data_clean03 %>% 
  group_by(sex, form) %>% 
  summarise(m_prod = median(production))

# classify missing values
data_clean04 <- data_clean03 %>% 
  mutate(sex = case_when((is.na(sex) & form == "WG" & production >= 8) ~ "Female",
                         (is.na(sex) & form == "WG" & production <= 8) ~ "Male",
                         (is.na(sex) & form == "WS" & production >= 318) ~ "Female",
                         (is.na(sex) & form == "WS" & production <= 318) ~ "Male",
                         T ~ as.character(sex)),
         sex = as_factor(sex))

# glance at data and calculate differences
summary(data_clean04)

# set seed for reproducibility
if_else(getRversion() < 3.5, set.seed(123), set.seed(123, sample.kind = "Rounding"))

# set outcome
y <- data_clean04$production
  
# partioning data
# create index: 70% (training) 30% (test)
data_index <- createDataPartition(y, p = 0.7, times = 1, list = F)

# create train and test data
data_train <- data_clean04 %>% slice(data_index)
data_test <- data_clean04 %>% slice(-data_index)

# double check proportions
round(nrow(data_train)/(nrow(data_clean04)), 2)
round(nrow(data_test)/(nrow(data_clean04)), 2)

# summary statistics (training set)
summary(data_train)

# distribution of outcome
data_train %>% 
  ggplot(aes(x = production)) + 
  geom_histogram(binwidth = 10, 
                 fill = color_blind_colors[3], 
                 color = "black",
                 alpha = 0.8) +
  labs(title = "Distribution of produced words",
       x = "Number of produced words", 
       y = "Frequency") +
  theme_bw()

# productive vocab by age, gender, and instrument (form)
data_train %>% 
  ggplot(aes(x = age, y = production, color = sex)) + 
  stat_summary(fun = mean) +
  stat_summary(fun = mean, geom = "line") +
  facet_wrap(~ form) +
  scale_color_manual(values= color_blind_colors) +
  xlim(5, 40) +
  labs(title = "Word production by age, gender, and instrument",
       x = "Age (months)", 
       y = "Number of produced words",
       color = "Gender") +
  theme_bw()

# productive vocab by age and language
data_train %>% 
  ggplot(aes(x = age, y = production)) + 
  stat_summary(fun = mean, geom = "line") +
  facet_wrap(~ language) +
  xlim(5, 40) +
  labs(title = "Word production by age and language",
       x = "Age (months)", 
       y = "Number of produced words") +
  theme_bw() +
  theme(strip.text = element_text(size = 6))

# productive vocab by comprehension and instrument (form)
data_train %>% 
  ggplot(aes(x = comprehension, y = production)) + 
  geom_smooth(se = F, color = color_blind_colors[3]) +
  facet_wrap(~ form) +
  labs(title = "Word production by comprehension and instrument",
       x = "Average number of comprehended words", 
       y = "Average number of produced words") +
  theme_bw()

# Results
## Regression Tree

# set seed for reproducibility
if_else(getRversion() < 3.5, set.seed(234), set.seed(234, sample.kind = "Rounding"))

# fit regression tree to training set
rpart_fit <- rpart(production ~ ., data = data_train, method = "anova")

# plot regression tree
rpart.plot(rpart_fit, digits = 3, fallen.leaves = T)

# calculate RMSE for training
rpart_pred_train <- data_train %>% mutate(pred_prod = predict(rpart_fit))

## plot predicted vs observed vocab
rpart_pred_train %>% 
  ggplot(aes(x = production, y = pred_prod)) +
  geom_point(alpha = 0.2) +
  geom_smooth(color = color_blind_colors[5]) +
  labs(title = "Predicted vs. observed productive vocabulary - Training", 
       x = "Observed productive vocabulary", 
       y = "Predicted productive vocabulary") +
  theme_bw()

# training RMSE
rpart_train_rmse <- caret::RMSE(rpart_pred_train$production, rpart_pred_train$pred_prod)

# create a RMSE table
rmse_scores <- tibble(Model = "Regression Tree - Training", 
                      RMSE = rpart_train_rmse)

rmse_scores %>% knitr::kable()

# rpart accuracy on test set
rpart_test <- predict(rpart_fit, data_test)
rpart_pred_test <- data_test %>% mutate(pred_prod = rpart_test)

# RMSE accuracy
## plot predicted vs observed vocab
rpart_pred_test %>% 
  ggplot(aes(x = production, y = pred_prod)) +
  geom_point(alpha = 0.2) +
  geom_smooth(color = color_blind_colors[5]) +
  labs(title = "Predicted vs. observed productive vocabulary - Test", 
       x = "Observed productive vocabulary", 
       y = "Predicted productive vocabulary") +
  theme_bw()

# test RMSE
rpart_test_rmse <- caret::RMSE(rpart_pred_test$production, rpart_pred_test$pred_prod)

# add test RMSE to table
rmse_scores <- bind_rows(rmse_scores, 
                         tibble(Model = "Regression Tree - Test",
                                RMSE = rpart_test_rmse))

rmse_scores %>% knitr::kable()

## Random Forest

# set seed for reproducibility
if_else(getRversion() < 3.5, set.seed(345), set.seed(345, sample.kind = "Rounding"))

# fit random forest on training set
rf_fit <- randomForest::randomForest(production ~ ., data = data_train) 

# measure variable importance
varImp(rf_fit) %>% arrange(desc(Overall))

# plot random forest
plot(rf_fit) 

# calculate RMSE for training
rf_pred_train <- data_train %>% mutate(pred_prod = predict(rf_fit))

## plot predicted vs observed vocab
rf_pred_train %>% 
  ggplot(aes(x = production, y = pred_prod)) +
  geom_point(alpha = 0.2) +
  geom_smooth(color = color_blind_colors[5]) +
  labs(title = "Predicted vs. observed productive vocabulary - Training", 
       x = "Observed productive vocabulary", 
       y = "Predicted productive vocabulary") +
  theme_bw()

# training RMSE
rf_train_rmse <- caret::RMSE(rf_pred_train$production, rf_pred_train$pred_prod)

# create a RMSE table
rmse_scores <- bind_rows(rmse_scores, 
                         tibble(Model = "Random Forest - Training",
                                RMSE = rf_train_rmse))

rmse_scores %>% knitr::kable()

# rf accuracy on test set
rf_test <- predict(rf_fit, data_test)
rf_pred_test <- data_test %>% mutate(pred_prod = rf_test)

# RMSE accuracy
## plot predicted vs observed vocab
rf_pred_test %>% 
  ggplot(aes(x = production, y = pred_prod)) +
  geom_point(alpha = 0.2) +
  geom_smooth(color = color_blind_colors[5]) +
  labs(title = "Predicted vs. observed productive vocabulary - Test", 
       x = "Observed productive vocabulary", 
       y = "Predicted productive vocabulary") +
  theme_bw()

# test RMSE
rf_test_rmse <- caret::RMSE(rf_pred_test$production, rf_pred_test$pred_prod)

# add test RMSE to table
rmse_scores <- bind_rows(rmse_scores, 
                         tibble(Model = "Random Forest - Test",
                                RMSE = rf_test_rmse))

rmse_scores %>% knitr::kable()

## Linear Regression 

# fit lm on training set
lm_fit <- lm(production ~ ., data = data_train)

# model summary 
summary(lm_fit)

# RMSE accuracy
## add predicted scores to training set
lm_pred_train <- data_train %>% mutate(pred_prod = predict(lm_fit))

## plot predicted vs observed vocab
lm_pred_train %>% 
  ggplot(aes(x = production, y = pred_prod)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = lm, color = color_blind_colors[5]) +
  labs(title = "Predicted vs. observed productive vocabulary - Training", 
       x = "Observed productive vocabulary", 
       y = "Predicted productive vocabulary") +
  theme_bw()

# training RMSE
lm_train_rmse <- caret::RMSE(lm_pred_train$production, lm_pred_train$pred_prod)

# add training RMSE to table
rmse_scores <- bind_rows(rmse_scores, 
                         tibble(Model = "Linear Regression - Training",
                                RMSE = lm_train_rmse))

rmse_scores %>% knitr::kable()

# lm accuracy on test set
lm_test <- predict(lm_fit, data_test)
lm_pred_test <- data_test %>% mutate(pred_prod = lm_test)

# RMSE accuracy
## plot predicted vs observed vocab
lm_pred_test %>% 
  ggplot(aes(x = production, y = pred_prod)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = lm, color = color_blind_colors[5]) +
  labs(title = "Predicted vs. observed productive vocabulary - Test", 
       x = "Observed productive vocabulary", 
       y = "Predicted productive vocabulary") +
  theme_bw()

# test RMSE
lm_test_rmse <- caret::RMSE(lm_pred_test$production, lm_pred_test$pred_prod)

# add test RMSE to table
rmse_scores <- bind_rows(rmse_scores, 
                         tibble(Model = "Linear Regression - Test",
                                RMSE = lm_test_rmse))

rmse_scores %>% knitr::kable()

# THE END