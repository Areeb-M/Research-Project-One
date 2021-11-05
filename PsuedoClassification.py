from os import name
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import metrics

from numpy import nan

xls1 = pd.ExcelFile("dataset/discourse_analysis_one.xlsx")
xls2 = pd.ExcelFile("dataset/discourse_analysis_two.xlsx")

sims_1 = [pd.read_excel(xls1, sheet, header=None) for sheet in xls1.sheet_names]
sims_1.pop(0) # remove the metadata sheet

sims_2 = [pd.read_excel(xls2, sheet) for sheet in xls2.sheet_names]
sims_2.pop(0) # remote the metadata sheet

sims = sims_1 + sims_2
sim_names = xls1.sheet_names[1:] + xls2.sheet_names[1:]

def clean_sim_data(sim_df):
  sim_df = sim_df.rename(columns={0:"Turn Taking", 1:"Time", 2:"Total", 3:"Simulation", 4:"SA Team"})
  sim_df = sim_df[sim_df["Turn Taking"].notna()]
  sim_df = sim_df.drop(columns=sim_df.columns.values[5:])
  sim_df['SA Team'] = sim_df['SA Team'].apply(lambda x: eval("[" + str(x) + "]"))

  try:
    # If there are multiple SA, split them into two columns
    sa = sim_df.iloc[:,4].apply(lambda x : pd.Series(x).dropna()).merge(sim_df, left_index = True, right_index = True)

    # Drop all the rows that only have one SA rating or are NAN
    sa_two = sa[sa[1].notna()].drop(columns=[0]).rename(columns={1:"SA"})
    sim_df = sa.drop(columns=[1]).rename(columns={0:"SA"})
    
    # Merge the copies back into one DataFrame
    sim_df = pd.concat([sim_df, sa_two])
  except:
    # If there's no item with multiple SA values in the trial, it'll throw an error, so move on
    sim_df['SA'] = sim_df['SA Team'].apply(lambda x: eval(str(x))[0])
  
  sim_df = sim_df.drop(columns=["SA Team"])

  return sim_df


clean_sims = []

for i in range(len(sims)):
  try:
    clean_sims.append(clean_sim_data(sims[i]))
  except Exception as e:
    print(i, sim_names[i], e)
    print(sims[i])

# Median + Mode => 7 - 12
# Low: 7,8
# Med: 9, 10
# High: 11, 12
#sim_classifications = [['low', 'low', 'medium', 'medium', 'high', 'high'][int(sim['SA'].median() + sim['SA'].mode()[0]) - 7] for sim in clean_sims]
sim_classifications = [[1, 1, 2, 2, 3, 3][int(sim['SA'].median() + sim['SA'].mode()[0]) - 7] for sim in clean_sims]
sim_number_turns = [len(sim) for sim in clean_sims]
sim_turn_taking = [[len(sim[sim['Turn Taking']==i+1]) for sim in clean_sims] for i in range(0, 4)]
sim_turn_taking_by_sa = [[len(sim[sim['SA']==i]) for sim in clean_sims] for i in range(3, 7)]

prediction_df = pd.DataFrame.from_dict(
  {
    
    'Name' : sim_names,
    'Num Turns' : sim_number_turns,
    'Player 1 Turn Taking' : sim_turn_taking[0],
    'Player 2 Turn Taking' : sim_turn_taking[1],
    'Player 3 Turn Taking' : sim_turn_taking[2],
    'Player 4 Turn Taking' : sim_turn_taking[3],
    'SA 3 Turn Taking': sim_turn_taking_by_sa[0],
    'SA 4 Turn Taking': sim_turn_taking_by_sa[1],
    'SA 5 Turn Taking': sim_turn_taking_by_sa[2],
    'SA 6 Turn Taking': sim_turn_taking_by_sa[3],
    'Performance Rank': sim_classifications
  }
)

y = prediction_df['Performance Rank']
x = prediction_df.drop(columns=['Performance Rank'])

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=2)

rgr_tree = DecisionTreeRegressor(random_state=22).fit(X_train, Y_train)
Y_pred = rgr_tree.predict(X_test)
print('\n[ANOVA Decision Tree]')
print("R2 Score:", metrics.r2_score(Y_test, Y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(Y_test, Y_pred))
print("Mean Absolute Error:", metrics.mean_absolute_error(Y_test, Y_pred))


clf_tree = DecisionTreeClassifier(random_state=21).fit(X_train, Y_train)
Y_pred = clf_tree.predict(X_test)

print("\n[Decision Tree Classifier]")

print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
print("Confusion Matrix:\n",metrics.confusion_matrix(Y_test, Y_pred))

print("\n")