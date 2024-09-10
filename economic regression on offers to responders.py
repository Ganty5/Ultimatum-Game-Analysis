import pandas as pd
import statsmodels.api as sm

#import my dataset
UG_Thesis = pd.read_csv("C:/Users/Chika/Desktop/MY THESIS/20212020.csv")
UG_Thesis.head(30)

#merging 2019 and 2016 dataset
data2019 = pd.read_csv("C:/Users/Chika/Desktop/MY THESIS/2019.csv")
data2016 = pd.read_csv("C:/Users/Chika/Desktop/MY THESIS/2016.csv")

#Merging the 2019 and 2016 dataset
UG_201916 = pd.concat([data2019, data2016], ignore_index=True)
print(UG_201916)

#merging the 2021, 2020 and 201916 dataset
Thesis = pd.concat([UG_Thesis, UG_201916], ignore_index=True)
print(Thesis)


#creating dummy variable for year
Thesis['Y_2021'] = (Thesis['Year'] == 2021).astype(int)
Thesis['Y_2020'] = (Thesis['Year'] == 2020).astype(int)
Thesis['Y_201916'] = ((Thesis['Year'] == 2019) | (Thesis['Year'] == 2016)).astype(int)


#create a single binary variable for gender (0 for female, 1 for male)
Thesis['Gender'] = (Thesis['Gender'] == 'M').astype(int)


# Aggregating data for final rounds and mean offers
# Assuming final round is the last round number available for each session
final_round = Thesis.groupby('Session Name')['Round'].max().reset_index()
final_round_Thesis = Thesis.merge(final_round, on=['Session Name', 'Round'], how='inner')
print(final_round_Thesis)


mean_offers_Thesis = Thesis.groupby('Rounds').agg({'Offers to Responders': 'mean','Y_2021': 'mean', 'Y_2020': 'mean', 'Y_201916': 'mean', 'Gender': 'mean'}).reset_index()
print(mean_offers_Thesis)


#Extracting Round 1 data
Round1 = Thesis[Thesis['Round'] == 1]
print(Round1)


#ROUND 1
#Performing regression on Round 1 without Gender 

X_round_1_no_gender = Round1[['Y_2021', 'Y_2020', 'Y_201916']]
X_round_1_no_gender = sm.add_constant(X_round_1_no_gender)
y_round_1 = Round1['Offers to Responders']
model_round_1_no_gender = sm.OLS(y_round_1, X_round_1_no_gender).fit()
print(model_round_1_no_gender.summary())

#performing regression on Round 1 (with Gender)
X_round_1_with_gender = Round1[['Y_2021', 'Y_2020', 'Y_201916', 'Gender']]
X_round_1_with_gender = sm.add_constant(X_round_1_with_gender)
model_round_1_with_gender = sm.OLS(y_round_1, X_round_1_with_gender).fit()
print(model_round_1_with_gender.summary())


#FINAL ROUND
#Performing regression on Final round (without gender)

X_final_round_no_gender = final_round_Thesis[['Y_2021', 'Y_2020', 'Y_201916']]
X_final_round_no_gender = sm.add_constant(X_final_round_no_gender)
y_final_round = final_round_Thesis['Offers to Responders']
model_final_round_no_gender = sm.OLS(y_final_round, X_final_round_no_gender).fit()
print(model_final_round_no_gender.summary())

#Performing regression on Final round (with gender)
X_final_round_with_gender = final_round_Thesis[['Y_2021', 'Y_2020', 'Y_201916', 'Gender']]
X_final_round_with_gender = sm.add_constant(X_final_round_with_gender)
model_final_round_with_gender = sm.OLS(y_final_round, X_final_round_with_gender).fit()
print(model_final_round_with_gender.summary())


#MEAN OFFER REGRESSION 
#Performing regression on mean offers (without gender)

X_mean_offers_no_gender = mean_offers_Thesis[['Y_2021', 'Y_2020', 'Y_201916']]
X_mean_offers_no_gender = sm.add_constant(X_mean_offers_no_gender)
y_mean_offers = mean_offers_Thesis['Offers to Responders']
model_mean_offers_no_gender = sm.OLS(y_mean_offers, X_mean_offers_no_gender).fit()
print(model_mean_offers_no_gender.summary())


X_mean_offers_with_gender = mean_offers_Thesis[['Y_2021', 'Y_2020', 'Y_201916','Gender']]
X_mean_offers_with_gender = sm.add_constant(X_mean_offers_with_gender)
model_mean_offers_with_gender = sm.OLS(y_mean_offers, X_mean_offers_with_gender).fit()
print(model_mean_offers_with_gender.summary())




