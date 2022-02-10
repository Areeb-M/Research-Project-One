import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from numpy import nan


def main():
    sns.set()
    sns.set_palette("hls", 8)

    with open('clean_data.pickle', 'rb') as handle:
        data = pickle.load(handle)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    sim_5 = data['5']
    sim_319 = data['319']
    sim_321 = data['321']
    sim_325 = data['325']

    '''
    sns.histplot(x=sim_5['SA'], hue=sim_5['Simulation'], ax=ax[0,0], palette="rocket", discrete=True, multiple="stack")
    sns.histplot(x=sim_319['SA'], hue=sim_319['Simulation'], ax=ax[0,1], palette="rocket", discrete=True, multiple="stack")
    sns.histplot(x=sim_321['SA'], hue=sim_321['Simulation'], ax=ax[1,1], palette="rocket", discrete=True, multiple="stack")
    sns.histplot(x=sim_325['SA'], hue=sim_325['Simulation'], ax=ax[1,0], palette="rocket", discrete=True, multiple="stack")
    '''

    '''
    sns.histplot(x=sim_5['Turn Taking'], hue=sim_5['SA'], ax=ax[0,0], palette="rocket", discrete=True, multiple="stack")
    sns.histplot(x=sim_319['Turn Taking'], hue=sim_319['SA'], ax=ax[0,1], palette="rocket", discrete=True, multiple="stack")
    sns.histplot(x=sim_321['Turn Taking'], hue=sim_321['SA'], ax=ax[1,1], palette="rocket", discrete=True, multiple="stack")
    sns.histplot(x=sim_325['Turn Taking'], hue=sim_325['SA'], ax=ax[1,0], palette="rocket", discrete=True, multiple="stack")
    '''

    vc5 = sim_5['SA'].value_counts(sort=False)
    vc319 = sim_319['SA'].value_counts(sort=False).sort_index()
    vc325 = sim_325['SA'].value_counts(sort=False).sort_index()
    vc321 = sim_321['SA'].value_counts(sort=False).sort_index()

    vc5 = vc5.append(pd.Series([0], index=[5])).sort_index()

    print(vc5)

    plot = ax[0,0].pie(vc5, labels=vc5.index, autopct='%1.0f%%')
    plot = ax[0,1].pie(vc319, labels=vc319.index, autopct='%1.0f%%')
    plot = ax[1,0].pie(vc325, labels=vc325.index, autopct='%1.0f%%')
    plot = ax[1,1].pie(vc321, labels=vc321.index, autopct='%1.0f%%')

    ax[0,0].set_title("Sim 5")
    ax[0,1].set_title("Sim 319")
    ax[1,0].set_title("Sim 325")
    ax[1,1].set_title("Sim 321")

    plt.show()

if __name__ == '__main__':
    main()