# What is this model?
# Negative binomial regression can be used for over-dispersed count data, that is when the 
# conditional variance exceeds the conditional mean. It can be considered as a generalization 
# of Poisson regression since it has the same mean structure as Poisson regression and it has 
# an extra parameter to model the over-dispersion. If the conditional distribution of the outcome 
# variable is over-dispersed, the confidence intervals for the Negative binomial regression are 
# likely to be narrower as compared to those from a Poisson regression model.

# PART 1 - load required packages and read in the data
# ______________________________________________________________________________________________
# import the required packages
library(readr)
library(broom)
library(dplyr)
library(tidyr)
library(lubridate)
library(graphics)
library(ggplot2)
library(car)
library(Metrics)

# Read in the count and percentge data
data <- read_csv("C:/Users/n0284436/Documents/NYU_Stern_MSBA/Capstone/Master.1.20.2019.csv") 

# Part 1A - Aggeragate the data to build a model at the tract level
# extract the census data and store it in an object called census
census <- data %>%
              select(26, 37:93)

# aggregate the collisions, injuries and deaths by city and census tracts and store it in an object called collisions
collisions <- data %>%
                group_by(City, GEOID) %>%
                summarise(Collisions=n(), TotalInjuries=sum(TotalInjuries),TotalDeaths=sum(TotalDeaths))
              
# now that we have our two main data frames we can join them together based on GEOID
df_count <- left_join(census, collisions, by="GEOID")
df_count <- unique( df_count[ , 1:62 ] )

# remove columns for modeling
modeldata <- df_count[,2:62]

# Create a data set to model for total injuries
m1data <- modeldata
m1data$TotalDeaths <- NULL
m1data$Collisions <- NULL

# Create a data set to model for total deaths
m2data <- modeldata
m2data$TotalInjuries <- NULL
m2data$Collisions <- NULL

# Create a data set to model for total deaths
m3data <- modeldata
m3data$TotalInjuries <- NULL
m3data$TotalDeaths <- NULL

# PART 2 - Build a negative binomial model to predict the number of injuries
# ______________________________________________________________________________________________
# NEGATIVE BINOMIAL REGRESSION
library(MASS)

# copy the data
modeldata2 <- m1data

# variables removed because of NA
dropvars <- c("race_other","female","ge_75plus","trav_carpool","house_group","pct_pop",
              "pct_female","age_75plus")

# remove the dropped vars from the data
modeldata2 <- modeldata2[ , !(names(modeldata2) %in% dropvars)]

# Specify the full model using all of the potential predictors
library(MASS)
m1 <- glm.nb(TotalInjuries~.,data = modeldata2)
summary(m1)

# look at the Variance Inflation Factor for m1
m1_vif <- tidy(vif(m1))
names(m1_vif) <- c("Variable","VIF","DF","GVIF..1..2.Df..")
m1_vif <- arrange(m1_vif, desc(VIF, Variable))
m1_vif

# plot the model
# plot(m1)

# make predictions from this model
modeldata2$predictions <- predict(m1, newdata=modeldata2)

# Calculate residuals
modeldata2$residuals <- modeldata2$TotalInjurie-modeldata2$predictions

# look at the RMSE 
modeldata2 <- modeldata2[complete.cases(modeldata2), ]
rmse(modeldata2$TotalInjuries, modeldata2$predictions)
# RMSE of 122.9003

# PART 3 - Stepwise regression model
# ______________________________________________________________________________________________
# Forward Stepwise regression model
m2 <- step(m1)

fwdstepsummary <- tidy(m2)
arrange(fwdstepsummary, desc(p.value)) # AIC of 34992.12 - 29 vars

m2_vif <- tidy(vif(m2))
names(m2_vif) <- c("Variable","VIF","DF","GVIF..1..2.Df..")
m2_vif <- arrange(m2_vif, desc(VIF, Variable))
m2_vif




# PART 4 - remove variables with VIF of > 20
# ______________________________________________________________________________________________
# Log: removed vars with the highest VIF
# First removed "pop",AIC of 11691
# Second removed "race_white", AIC of 11690
# Third removed "house_family", AIC of 11688
# Fourth removed "never_married", AIC of 11687
# Fifth removed "age_5t17", AIC of 11687
# Sixth removed "trav_cars", AIC of 11685
# Seventh removed "trav_pub", AIC of 11684
# Eighth removed "male", AIC of 11682
# Nineth removed "house_nonfamily", AIC of 11680
# Tenth removed "no_cars", AIC of 11678
# Eleventh removed "edu_bs", AIC of 11676
# Twelveth removed "married", AIC of 11674
# Thirteenth removed "below_pov", AIC of  11673 - this model had VIF for all variables below 20

# variables removed for low p-value
dropvarsNA <- c("race_other","female","ge_75plus","trav_carpool","house_group","pct_pop",
                "pct_female","age_75plus")

# variables removed because of NA
dropvarsVIF <- c("pop","race_white","house_family","never_married","age_5t17","trav_cars",
                 "trav_pub","male","house_nonfamily","no_cars","edu_bs","married","below_pov")

# combine the dropped vars together 
drops <- c(dropvarsVIF, dropvarsNA)

# remove the dropped vars from the data
modeldata3 <- modeldata
modeldata3 <- modeldata3[ , !(names(modeldata3) %in% drops)]

# Specify the full model using all of the potential predictors
m3 <- glm.nb(num_incidents~.,data = modeldata3)
summary(m3)
m3summary <- tidy(m3)
m3summary$p.value <- round(m3summary$p.value, 2)
arrange(m3summary, desc(p.value))

# look at the Variance Inflation Factor for m1
m3_vif <- tidy(vif(m3))
names(m3_vif) <- c("Variable","VIF")
m3_vif <- arrange(m3_vif, desc(VIF, Variable))
m3_vif

# Make predictions from the model
modeldata3$predictions <- predict(m3, newdata=modeldata3)

# look at the RMSE 
modeldata3 <- modeldata3[complete.cases(modeldata3), ]
rmse(modeldata3$num_incidents, modeldata3$predictions)
# RMSE of 631.5229

# PART 5 - remove variables with VIF of > 10
# ______________________________________________________________________________________________
# Fourteenth removed "agg_travel_time", AIC of 23629 (largest jump in AIC from 11673)
# Fifteenth removed "edu_ms", AIC of 23627
# Sixteenth removed "soc_benefits", AIC of 23625

# variables removed for low p-value
dropvarsNA2 <- c("race_other","female","ge_75plus","trav_carpool","house_group","pct_pop",
                 "pct_female","age_75plus")

# variables removed because of NA
dropvarsVIF2 <- c("pop","race_white","house_family","never_married","age_5t17","trav_cars",
                  "trav_pub","male","house_nonfamily","no_cars","edu_bs","married","below_pov",
                  "agg_travel_time","edu_ms","soc_benefits")

# combine the dropped vars together 
drops2 <- c(dropvarsVIF2, dropvarsNA2)

# remove the dropped vars from the data
modeldata4 <- modeldata
modeldata4 <- modeldata4[ , !(names(modeldata4) %in% drops2)]

# Specify the full model using all of the potential predictors
m4 <- glm.nb(num_incidents~.,data = modeldata4)
summary(m4)
m4summary <- tidy(m4)
m4summary$p.value <- round(m4summary$p.value, 2)
arrange(m4summary, desc(p.value))

# look at the Variance Inflation Factor for m1
m4_vif <- tidy(vif(m4))
names(m4_vif) <- c("Variable","VIF")
m4_vif <- arrange(m4_vif, desc(VIF, Variable))
m4_vif

# plot the model
plot(m4)

# Make predictions from the model
modeldata4$predictions <- predict(m4, newdata=modeldata4)

# look at the RMSE 
modeldata4 <- modeldata4[complete.cases(modeldata4), ]
rmse(modeldata4$num_incidents, modeldata4$predictions)
# RMSE of 538.3038

# write the coefficients to a csv
write.csv(m4summary, "negbinomialsummary.csv")

# PART 6 - split the data into train and test sets
# ______________________________________________________________________________________________
# CREATE TRAINING SET
modeldata5 <- modeldata[ , !(names(modeldata) %in% drops2)]

# Use nrow to get the number of rows in mpg (N) and print it
(N <- nrow(modeldata5))

# Calculate how many rows 75% of N should be and print it
# Hint: use round() to get an integer
(target <- round(N * 0.75))

# Create the vector of N uniform random variables: gp
gp <- runif(N)

# Use gp to create the training set: mpg_train (75% of data) and mpg_test (25% of data)
data_train <- modeldata5[gp < 0.75, ]
data_test <- modeldata5[gp >= 0.75, ]

# Use nrow() to examine mpg_train and mpg_test
nrow(data_train)
nrow(data_test)

# build a negative binomial model on training data
negmod <- glm.nb(num_incidents~.,data = data_train)
tidy(negmod)

# look at the summary
summary(negmod)
# AIC of 17819

# build a poisson glml on training data
## Poission Regression
### Regular Poisson Model with BIC Selection
pois_mod <- glm(num_incidents~., data = data_train, family = "poisson")
tidy(pois_mod)

# look at the summary
summary(pois_mod)
# AIC of 108302

# make predictions from the model
data_test$negbinompreds <- predict(negmod, data_test)
data_test$poisspreds <- predict(pois_mod, data_test)

# look at the rmse of each model
data_test <- data_test[complete.cases(data_test), ]
rmse(data_test$num_incidents, data_test$negbinompreds)
rmse(data_test$num_incidents, data_test$poisspreds)
