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
census <- data %>%select(26, 37:96)
census <- unique(census)
census$GEOID <- as.character(census$GEOID)

# aggregate the collisions, injuries and deaths by city and census tracts and store it in an object called collisions
collisions <- data %>%
  group_by(City, GEOID) %>%
  summarise(Collisions=n(),TotalPedestrianInjuries=sum(TotalPedestrianInjuries),TotalDeaths=sum(TotalDeaths))
collisions$GEOID <- as.character(collisions$GEOID) 

# now that we have our two main data frames we can join them together based on GEOID
df_count <- left_join(census, collisions, by="GEOID")
df_count <- unique( df_count[ , 1:65 ] )

# remove columns for modeling
modeldata <- df_count[,2:65]

# Create a data set to model for total injuries
m1data <- subset(modeldata, City=="NYC") # change this to la
m1data$TotalDeaths <- NULL
m1data$Collisions <- NULL

# PART 2 - Build a negative binomial model to predict the number of injuries for NYC
# ______________________________________________________________________________________________
# NEGATIVE BINOMIAL REGRESSION
library(MASS)

# copy the data
modeldata2 <- m1data

# variables removed because of NA
dropvars <- c("City", "race_other","female","ge_75plus","trav_carpool","house_group","pct_pop",
              "pct_female","age_75plus")

# remove the dropped vars from the data
modeldata2 <- modeldata2[ , !(names(modeldata2) %in% dropvars)]

# Specify the full model using all of the potential predictors
m1 <- glm.nb(TotalPedestrianInjuries~.,data = modeldata2)
summary(m1)

# look at the Variance Inflation Factor for m1
m1_vif <- tidy(vif(m1))
names(m1_vif) <- c("Variable","VIF")
m1_vif <- arrange(m1_vif, desc(VIF, Variable))
m1_vif


# make predictions from this model
modeldata2$predictions <- predict(m1, newdata=modeldata2)

# Calculate residuals
modeldata2$residuals <- modeldata2$TotalPedestrianInjuries-modeldata2$predictions

# look at the RMSE 
modeldata2 <- modeldata2[complete.cases(modeldata2), ]
rmse(modeldata2$TotalPedestrianInjuries, modeldata2$predictions)
# RMSE of 28.53027
# AIC of 16487
# this is higher than the mean of 23.11425, sd of 20.49

# PART 3 - Stepwise regression model
# ______________________________________________________________________________________________
# Forward Stepwise regression model
m2 <- step(m1)

fwdstepsummary <- tidy(m2)
arrange(fwdstepsummary, desc(p.value)) # AIC of 34992.12 - 29 vars

m2_vif <- tidy(vif(m2))
names(m2_vif) <- c("Variable","VIF")
m2_vif <- arrange(m2_vif, desc(VIF, Variable))
m2_vif

# make predictions from this model
modeldata2$predictions <- predict(m2, newdata=modeldata2)

# Calculate residuals
modeldata2$residuals <- modeldata2$TotalPedestrianInjuries-modeldata2$predictions

# look at the RMSE 
modeldata2 <- modeldata2[complete.cases(modeldata2), ]
rmse(modeldata2$TotalPedestrianInjuries, modeldata2$predictions)

# RMSE of 28.53095
# AIC of 16438.95
# this is higher than the mean of 23.11425, sd of 20.49

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
dropvarsNA <- c("City", "race_other","female","ge_75plus","trav_carpool","house_group","pct_pop",
                "pct_female","age_75plus")

# variables removed because of NA
dropvarsVIF <- c("pop","race_white","house_family","never_married","age_5t17","trav_cars",
                 "trav_pub","male","house_nonfamily","no_cars","edu_bs","married","below_pov")

# combine the dropped vars together 
drops <- c(dropvarsVIF, dropvarsNA)

# remove the dropped vars from the data
modeldata3 <- modeldata2
modeldata3 <- modeldata3[ , !(names(modeldata3) %in% drops)]

# Specify the full model using all of the potential predictors
m3 <- glm.nb(TotalPedestrianInjuries~.,data = modeldata3)
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
rmse(modeldata3$TotalPedestrianInjuries, modeldata3$predictions)
# RMSE of 28.35092
# AIC of 13715
# this is higher than the mean of 23.11425, sd of 20.49

# PART 5 - remove variables with VIF of > 10
# ______________________________________________________________________________________________
# Fourteenth removed "agg_travel_time", AIC of 23629 (largest jump in AIC from 11673)
# Fifteenth removed "edu_ms", AIC of 23627
# Sixteenth removed "soc_benefits", AIC of 23625

# variables removed for low p-value
dropvarsNA2 <- c("City","race_other","female","ge_75plus","trav_carpool","house_group","pct_pop",
                 "pct_female","age_75plus")

# variables removed because of NA
dropvarsVIF2 <- c("pop","race_white","house_family","never_married","age_5t17","trav_cars",
                  "trav_pub","male","house_nonfamily","no_cars","edu_bs","married","below_pov",
                  "agg_travel_time","edu_ms","soc_benefits")

# combine the dropped vars together 
drops2 <- c(dropvarsVIF2, dropvarsNA2)

# remove the dropped vars from the data
modeldata4 <- modeldata2
modeldata4 <- modeldata4[ , !(names(modeldata4) %in% drops2)]

# Specify the full model using all of the potential predictors
m4 <- glm.nb(TotalPedestrianInjuries~.,data = modeldata4)
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
# plot(m4)

# Make predictions from the model
modeldata4$predictions <- predict(m4, newdata=modeldata4)

# look at the RMSE 
modeldata4 <- modeldata4[complete.cases(modeldata4), ]
rmse(modeldata4$TotalPedestrianInjuries, modeldata4$predictions)
# RMSE of 28.35083
# AIC of 13712
# this is higher than the mean of 23.11425, sd of 20.49

# write the coefficients to a csv
# write.csv(m4summary, "negbinomialsummary.csv")

# PART 6 - Feature Engineering
# ______________________________________________________________________________________________
area <- read_csv("C:/Users/n0284436/Documents/NYU_Stern_MSBA/Capstone/data_2010_area.csv")
area <- area[,-(2:5)]
colnames(area) <- c("GEOID", "sqmi_total", "sqmi_water", "sqmi_land")
area$GEOID <- as.character(area$GEOID)


# Merge tract size to census data
features_0 <- left_join(census, area, by = "GEOID")

# Setup pairwise correlation calculation and view
library(Hmisc)

flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

view_corr <- function(df) {
  corr <- rcorr(as.matrix(df[-1]))
  v_corr <- flattenCorrMatrix(corr$r, corr$P)
  View(v_corr)
}

# Review feature_0
view_corr(features_0)

################################################## Merge, transform, and clean features

############################## Iteration 1
features_1 <- features_0 %>%
  group_by(GEOID) %>%
  transmute(pop_dens = (pop / (sqmi_land)),
            race_white,
            race_minority = (sum(race_black, race_asian, race_hispanic, race_native, 
                                 race_hawaiian, race_other, race_twoplus)),
            female,
            age_genz = (sum(age_under5, age_5t17)),
            age_millenial = (sum(age_18t24, age_25t34)),
            age_genx = (sum(age_35t44, age_45t54)),
            age_boomer = (sum(age_55t59, age_60t61, age_62t64, age_65t74)),
            age_retiree = (age_75plus), 
            divsep = (sum(divorced, separated)), 
            widowed = (widowed),
            median_age, 
            not_us_citizen, 
            median_earnings, 
            trav_cars, 
            trav_trans = (sum(trav_pub, trav_taxi)),
            trav_motorcycle,
            trav_bike, 
            trav_walk, 
            trav_home,
            edu_lowedu = (sum(edu_none, edu_some_hs)),
            edu_hsged = (sum(edu_hs, edu_ged, edu_some_bs)),
            edu_bs,
            edu_grad = (sum(edu_ms, edu_phd)),
            unemp, 
            below_pov
  )
view_corr(features_1)
collisions <- subset(collisions, City=="NYC") # change this out to LA
df_features_1 <- left_join(features_1, collisions, by = "GEOID")

# write.csv(df_features_1, "df_features_1.csv", row.names = FALSE)
drops3 <- c("GEOID","City","Collisions","TotalDeaths")
df_features_1 <- df_features_1[ , !(names(df_features_1) %in% drops3)]

# Specify the full model using all of the potential predictors
m5 <- glm.nb(TotalPedestrianInjuries~.,data = df_features_1) 
summary(m5)
m5summary <- tidy(m5)
m5summary$p.value <- round(m5summary$p.value, 2)
arrange(m5summary, desc(p.value))

# look at the Variance Inflation Factor for m1
m5_vif <- tidy(vif(m5))
names(m5_vif) <- c("Variable","VIF")
m5_vif <- arrange(m5_vif, desc(VIF, Variable))
m5_vif

# Make predictions from the model
df_features_1$predictions <- predict(m5, newdata=df_features_1)

# look at the RMSE 
df_features_1 <- df_features_1[complete.cases(df_features_1), ]
rmse(df_features_1$TotalPedestrianInjuries, df_features_1$predictions)

# RMSE of 28.61043
# AIC of 16584
# this is higher than the mean of 23.11425, sd of 20.49

# Note: This iteration was mostly combining and removing variables without any 
#       normalization or transformation, with the exception of normalizing 
#       population by land square miles.


############################## Iteration 2
features_2 <- features_0 %>%
  group_by(GEOID) %>%
  transmute(pop_dens = (pop / (sqmi_land)),
            perc_minority = (sum(race_black, race_asian, race_hispanic, race_native, 
                                 race_hawaiian, race_other, race_twoplus) / pop),
            perc_female = (female / pop),
            perc_genz = (sum(age_under5, age_5t17) / pop),
            perc_millenial = (sum(age_18t24, age_25t34) / pop),
            perc_genx = (sum(age_35t44, age_45t54) / pop),
            perc_boomer = (sum(age_55t59, age_60t61, age_62t64, age_65t74) / pop),
            perc_retiree = (age_75plus / pop), 
            perc_divsep = (sum(divorced, separated) / pop), 
            perc_widowed = (widowed / pop),
            median_age, 
            perc_foreign = (not_us_citizen / pop), 
            median_earnings, 
            perc_cars = (trav_cars / pop), 
            perc_trans = (sum(trav_pub, trav_taxi) / pop),
            perc_mbike = (trav_motorcycle / pop),
            perc_bike = (trav_bike / pop), 
            perc_walk = (trav_walk / pop), 
            perc_wfm = (trav_home / pop),
            perc_lowedu = (sum(edu_none, edu_some_hs) / pop),
            perc_hsged = (sum(edu_hs, edu_ged, edu_some_bs) / pop),
            perc_bs = (edu_bs / pop),
            perc_grad = (sum(edu_ms, edu_phd) / pop),
            perc_unemp = (unemp / pop), 
            perc_pov = (below_pov / pop)
  )
#view_corr(features_2)
df_features_2 <- left_join(features_2, collisions, by = "GEOID")
# View(df_features_2)

# write.csv(df_features_1, "df_features_1.csv", row.names = FALSE)
drops4 <- c("GEOID","City","Collisions","TotalDeaths")
df_features_2 <- df_features_2[ , !(names(df_features_2) %in% drops4)]

# Specify the full model using all of the potential predictors
m6 <- glm.nb(TotalPedestrianInjuries~.,data = df_features_2)
summary(m6)
m6summary <- tidy(m6)
m6summary$p.value <- round(m6summary$p.value, 2)
arrange(m6summary, desc(p.value))

# look at the Variance Inflation Factor for m1
m6_vif <- tidy(vif(m6))
names(m6_vif) <- c("Variable","VIF")
m6_vif <- arrange(m6_vif, desc(VIF, Variable))
m6_vif

# Make predictions from the model
df_features_2$predictions <- predict(m6, newdata=df_features_2)

# look at the RMSE 
df_features_2 <- df_features_2[complete.cases(df_features_2), ]
rmse(df_features_2$TotalPedestrianInjuries, df_features_2$predictions)

# RMSE of 28.62324
# AIC of 16606
# this is higher than the mean of 23.11425, sd of 20.49

# PART 7 - split the data into train and test sets
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
