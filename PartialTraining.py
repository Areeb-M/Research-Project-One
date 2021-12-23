from datetime import *
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import metrics

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

import timeit

with open('clean_data.pickle', 'rb') as handle:
  data = pickle.load(handle)

sim_names = list(data.keys())
clean_sims = [data[name] for name in sim_names]

sim_classifications = [[1, 1, 2, 2, 3, 3][int(sim['SA'].median() + sim['SA'].mode()[0]) - 7] for sim in clean_sims]
day = date.today()

def calcuate_average(sim, i):
  try:
    return 60 * len(sim[sim['Simulation']==i+1]) / (lambda x: datetime.combine(day, x.iloc[-1]) - datetime.combine(day, x.iloc[0]))(sim[sim['Simulation']==i+1]['Time'].sort_values()).seconds
  except:
    return 0

def calculate_sentiment(df):
  try:
    sia = SentimentIntensityAnalyzer()
    df['Sentiment_Pos'] = df['Total'].apply(lambda x: sia.polarity_scores(str(x))['pos'])
    df['Sentiment_Neg'] = df['Total'].apply(lambda x: sia.polarity_scores(str(x))['neg'])
    df['Sentiment_Neu'] = df['Total'].apply(lambda x: sia.polarity_scores(str(x))['neu'])
    df['Sentiment_Compound'] = df['Total'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    return df['Sentiment_Pos'].mean(), df['Sentiment_Neg'].mean(), df['Sentiment_Neu'].mean(), df['Sentiment_Compound'].mean()
  except:
    return 0, 0, 0, 0

sim_average_turn_taking = [
  sum([
    calcuate_average(sim, i) for i in range(0, 4)
  ]) for sim in clean_sims
]

sim_number_turns = [len(sim) for sim in clean_sims]
sim_turn_taking = [[len(sim[sim['Turn Taking'] == i+1])for sim in clean_sims] for i in range(0, 4)]
sim_player_sa = [[sim[sim['Turn Taking'] == i+1]['SA'].dropna().mean() for sim in clean_sims] for i in range(0, 4)]
sim_player_sa_median = [[sim[sim['Turn Taking'] == i+1]['SA'].median() for sim in clean_sims] for i in range(0, 4)]
sim_turn_taking_by_sa = [[len(sim[sim['SA']==i]) for sim in clean_sims] for i in range(3, 7)]

sim_sentiments = [calculate_sentiment(sim) for sim in clean_sims]
sim_sen_pos = [sim_sentiments[i][0] for i in range(len(clean_sims))]
sim_sen_neg = [sim_sentiments[i][1] for i in range(len(clean_sims))]
sim_sen_neu = [sim_sentiments[i][2] for i in range(len(clean_sims))]
sim_sen_comp = [sim_sentiments[i][3] for i in range(len(clean_sims))]

prediction_df = pd.DataFrame.from_dict(
  {
    
    'Name' : sim_names,
    'Number of Turns' : sim_number_turns,
    'Player 1 Turn Taking' : sim_turn_taking[0],
    'Player 2 Turn Taking' : sim_turn_taking[1],
    'Player 3 Turn Taking' : sim_turn_taking[2],
    'Player 4 Turn Taking' : sim_turn_taking[3],
    'Player 1 SA Average' : sim_player_sa[0],
    'Player 2 SA Average' : sim_player_sa[1],
    'Player 3 SA Average' : sim_player_sa[2],
    'Player 4 SA Average' : sim_player_sa[3],
    'Average Turn Taking' : sim_average_turn_taking,
    'SA 3 Turn Taking': sim_turn_taking_by_sa[0],
    'SA 4 Turn Taking': sim_turn_taking_by_sa[1],
    'SA 5 Turn Taking': sim_turn_taking_by_sa[2],
    'SA 6 Turn Taking': sim_turn_taking_by_sa[3],
    'Positive Sentiment Rating': sim_sen_pos,
    'Negative Sentiment Rating': sim_sen_neg,
    'Neutral Sentiment Rating': sim_sen_neu,
    'Compound Sentiment Rating': sim_sen_comp,
    'Performance Rank': sim_classifications
  }
).fillna(0)

y = prediction_df['Performance Rank']
x = prediction_df.drop(columns=['Performance Rank', 'Name'])
x2 = prediction_df.drop(columns=['Performance Rank', 'Name', 'Average Turn Taking'])

def partial_training_results(x, y, test_range):
  results = {}

  for i in test_range:
    individual_results = {}

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=i, random_state=2)

    #==========================================================================

    rgr_tree = DecisionTreeRegressor(random_state=22).fit(X_train, Y_train)
    Y_pred = rgr_tree.predict(X_test)

    individual_results['ANOVA Decision Tree Regressor'] = {
      "Accuracy": f"R2:{metrics.r2_score(Y_test, Y_pred)}; MSE:{metrics.mean_squared_error(Y_test, Y_pred)}; MAE:{metrics.mean_absolute_error(Y_test, Y_pred)}",
      "Time": timeit.timeit('rgr_tree.predict(x)', globals=locals(), number=100)/100
    }

    importance = rgr_tree.feature_importances_
    for n, v in enumerate(importance):
      individual_results['ANOVA Decision Tree Regressor'][x.columns[n]] = v

    #==========================================================================

    clf_tree = DecisionTreeClassifier(random_state=22).fit(X_train, Y_train)
    Y_pred = clf_tree.predict(X_test)

    individual_results['Decision Tree Classifier'] = {
      "Accuracy": metrics.accuracy_score(Y_test, Y_pred),
      "Time": timeit.timeit('clf_tree.predict(x)', globals=locals(), number=100)/100
    }

    importance = clf_tree.feature_importances_
    for n, v in enumerate(importance):
      individual_results['Decision Tree Classifier'][x.columns[n]] = v

    #==========================================================================

    clf_tree = RandomForestClassifier(random_state=22).fit(X_train, Y_train)
    Y_pred = clf_tree.predict(X_test)

    individual_results['Random Forest Classifier'] = {
      "Accuracy": metrics.accuracy_score(Y_test, Y_pred),
      "Time": timeit.timeit('clf_tree.predict(x)', globals=locals(), number=100)/100
    }

    importance = clf_tree.feature_importances_
    for n, v in enumerate(importance):
      individual_results['Random Forest Classifier'][x.columns[n]] = v

    #==========================================================================

    rgr_tree = RandomForestRegressor(random_state=22).fit(X_train, Y_train)
    Y_pred = rgr_tree.predict(X_test)

    individual_results['Random Forest Regressor'] = {
      "Accuracy": f"R2:{metrics.r2_score(Y_test, Y_pred)}; MSE:{metrics.mean_squared_error(Y_test, Y_pred)}; MAE:{metrics.mean_absolute_error(Y_test, Y_pred)}",
      "Time": timeit.timeit('rgr_tree.predict(x)', globals=locals(), number=100)/100
    }

    importance = rgr_tree.feature_importances_
    for n, v in enumerate(importance):
      individual_results['Random Forest Regressor'][x.columns[n]] = v

    #==========================================================================

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, Y_train)
    individual_results['SVM Classifier'] = {
      "Accuracy": clf.score(X_test, Y_test),
      "Time": timeit.timeit('clf.predict(X_test)', globals=locals(), number=100)/100
    }

    #==========================================================================

    fit = SelectKBest(score_func=f_classif, k=5).fit(X_train, Y_train)
    kx = fit.transform(x)
    kx_train = fit.transform(X_train)
    ky_train = Y_train
    kx_test = fit.transform(X_test)
    ky_test = Y_test

    clf.fit(kx_train, ky_train)
    individual_results['SVM Classifier - K=5 best'] = {
      "Accuracy": clf.score(kx_test, ky_test),
      "Time": timeit.timeit('clf.predict(kx)', globals=locals(), number=100)/100
    }

    #==========================================================================

    clf = MLPClassifier(random_state=2)
    clf.fit(X_train, Y_train)
    individual_results['MLP Classifier - Adam'] = {
      "Accuracy": clf.score(X_test, Y_test),
      "Time": timeit.timeit('clf.predict(x)', globals=locals(), number=100)/100
    }

    #==========================================================================

    clf = MLPClassifier(random_state=2, solver="lbfgs")
    clf.fit(X_train, Y_train)
    individual_results['MLP Classifier - LBFGS'] = {
      "Accuracy": clf.score(X_test, Y_test),
      "Time": timeit.timeit('clf.predict(x)', globals=locals(), number=100)/100
    }

    #==========================================================================

    results[i] = individual_results

  return results

#results_a = partial_training_results(x, y, [0.5, 0.6, 0.7, 0.8, 0.9])
results_b = partial_training_results(x2, y, [0.5, 0.6, 0.7, 0.8, 0.9])

# one csv for each partial training
# rows are labelled name, accuracy, time, feature list
# columns correspond to each classifier/regressor entry

for training_percent, results in results_b.items():
  file = open(f"results/{training_percent}.csv", 'w')
  header = "Name,"
  line = ["Accuracy", "Time"]
  line += [col for col in x2.columns]
  line_data = [f"{l}," for l in line]

  for name, data in results.items():
    header += f"{name},"
    for i in range(len(line)):
      if line[i] in data:
        line_data[i] += str(data[line[i]])
      line_data[i] += ","
  
  file.write(header + "\n")
  for line in line_data:
    file.write(line + "\n")

  file.close()
