import enum
from os import name
import matplotlib.pyplot as plt
import numpy as np
from numpy.random.mtrand import random_integers
import pandas as pd
import pickle
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


clean_sims = {}

for i in range(len(sims)):
  try:
    clean_sims[sim_names[i]] = clean_sim_data(sims[i])
  except Exception as e:
    print(i, sim_names[i], e)
    print(sims[i])

print(clean_sims[sim_names[1]])

with open("clean_data.pickle", 'wb') as handle:
    pickle.dump(clean_sims, handle, protocol=pickle.HIGHEST_PROTOCOL)