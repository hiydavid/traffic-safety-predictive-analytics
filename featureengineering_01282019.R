# David Huang
# Capstone Modeling Preparation & Feature Engineering

##################################################  Change Log
# 2019-01-21  Created file with feature 1 and 2
# 2019-01-23  Updated to include pedestrian data
# 2019-01-28  Updated with feature 3



##################################################  Load libraries and data 
library(readr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(lubridate)
library(scales)
library(tidycensus)



################################################## Read and transform master data

# Set directory
setwd("D:/_dhuang/Work/NYU Stern MSBA Work/Capstone/Data/CapstoneModeling")

# Upload old master data
data <- read_csv("Master.1.20.2019.csv")
data$GEOID <- as.character(data$GEOID)

# Extract the census data and store it in an object called census
census <- data %>% select(26, 37:96)
census <- unique(census)

# Aggregate by city and census tracts
collisions <- data %>%
  group_by(City, GEOID) %>%
  summarise(Collisions = n(),
            PedeInjuries = sum(TotalPedestrianInjuries),
            PedeDeaths = sum(TotalPedestrianDeaths),
            TotalInjuries = sum(TotalInjuries), 
            TotalDeaths = sum(TotalDeaths))

# Export file as csv for later use
write.csv(collisions, "data_collisions.csv", row.names = FALSE)
write.csv(census, "data_census.csv", row.names = FALSE)



################################################## Merge, transform, and clean features

# Bring in the tract squarefootage data
area <- read_csv("data_2010_area.csv")
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
df_features_1 <- left_join(features_1, collisions, by = "GEOID")
View(df_features_1)
write.csv(df_features_1, "df_features_1.csv", row.names = FALSE)

# Note: This iteration was mostly combining and removing variables
#       without any normalization or transformation, with the exception
#       of normalizing population by land square miles.

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
view_corr(features_2)
df_features_2 <- left_join(features_2, collisions, by = "GEOID")
View(df_features_2)
write.csv(df_features_2, "df_features_2.csv", row.names = FALSE)

# Note: This iteration picks up from iteration 1, with the addition of
#       normalization by population where applicable. No transformations
#       were done otherwise.

############################## Iteration 3
features_3 <- features_0 %>%
  group_by(GEOID) %>%
  transmute(pop_dens = (pop / (sqmi_land)),
            minority_dens = (sum(race_black, race_asian, race_hispanic, race_native, 
                                 race_hawaiian, race_other, race_twoplus) / sqmi_land),
            female_dens = (female / sqmi_land),
            genz_dens = (sum(age_under5, age_5t17) / sqmi_land),
            millenial_dens = (sum(age_18t24, age_25t34) / sqmi_land),
            genx_dens = (sum(age_35t44, age_45t54) / sqmi_land),
            boomer_dens = (sum(age_55t59, age_60t61, age_62t64, age_65t74) / sqmi_land),
            retiree_dens = (age_75plus / sqmi_land), 
            divsep_dens = (sum(divorced, separated) / sqmi_land), 
            widowed_dens = (widowed / sqmi_land),
            median_age, 
            foreign_dens = (not_us_citizen / sqmi_land), 
            median_earnings, 
            cars_dens = (trav_cars / sqmi_land), 
            trans_dens = (sum(trav_pub, trav_taxi) / sqmi_land),
            mbike_dens = (trav_motorcycle / sqmi_land),
            bike_dens = (trav_bike / sqmi_land), 
            walk_dens = (trav_walk / sqmi_land), 
            wfm_dens = (trav_home / sqmi_land),
            lowedu_dens = (sum(edu_none, edu_some_hs) / sqmi_land),
            hsged_dens = (sum(edu_hs, edu_ged, edu_some_bs) / sqmi_land),
            bs_dens = (edu_bs / sqmi_land),
            grad_dens = (sum(edu_ms, edu_phd) / sqmi_land),
            unemp_dens = (unemp / sqmi_land), 
            pov_dens = (below_pov / sqmi_land)
  )
view_corr(features_3)
df_features_3 <- left_join(features_3, collisions, by = "GEOID")
View(df_features_3)
write.csv(df_features_3, "df_features_3.csv", row.names = FALSE)

# Note: This iteration differs from iteration 2 in that the variables are
#       normalized by land square miles instead of population. No variables
#       were transformed.

############################## Iteration 4

# TO BE CONTINUED 
