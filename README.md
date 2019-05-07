# Predictive Analytics for Traffic Safety
### NYU Stern MSBA 2019 Capstone Project
**Team Knight Rider**
* Noah Branham
* David Huang
* Tamara Kempf
* Mike Marra
* Aaron Smith

**Executive Summary**

Our Capstone project seeks to help cities who are new to Vision Zero 1.) prioritize which local areas to focus their efforts on and 2.) glean insights from other cities as to how socio-economic and road design data might impact traffic injuries / deaths on a local level. We collected public collision data from New York City, Washington D.C., and Los Angeles from 2013 to 2017 and aggregate injuries and deaths at the census tract level (a unit of analysis about the size of 2–3 city blocks). We then pair each census tract’s collision data with its corresponding socio-economic and road inventory data. 

We then use predictive analytics to assist cities without robust collision data understand where they can have the greatest impact. Rather than build a model that predicts total count of casualties (injuries + deaths) in each census tract, we are interested in how well we can predict the *ranking* of census tracts in a city from highest to lowest casualties over a five year period. With this ranking, city officials can make more informed decisions about where to allocate resources, and learn about the informativeness of the variables that produce it.

We find that, using solely a city’s Census data, we can predict the ranking of census tract casualties, from highest to lowest, better than random chance. Performance is further improved with road inventory data. To a certain extent, we can also transfer a model built on New York City Census data to another city, and achieve ranking results better than random chance. This suggests that there is something predictive about socio-economic and road inventory data, and we use linear regression techniques to further explore their relationship to casualties.
