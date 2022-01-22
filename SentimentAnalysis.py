from datetime import *
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np

nltk.download('vader_lexicon')

sia = None


def analyze_sentiment(line):
    global sia
    if sia is None:
        sia = SentimentIntensityAnalyzer()

    scores = sia.polarity_scores(str(line['Total']))
    return scores['compound'] if scores['compound'] != 0 else np.NaN


def main():
    with open('clean_data.pickle', 'rb') as handle:
        data = pickle.load(handle)

    sim_names = list(data.keys())
    clean_sims = [data[name] for name in sim_names]
    sim_classifications = [[1, 1, 2, 2, 3, 3][int(sim['SA'].median() + sim['SA'].mode()[0]) - 7] for sim in clean_sims]

    # print(clean_sims[0].to_string())
    # 319, 321, 345 = high performance
    # 5, 325 = low performance
    a = data['321']
    a['Sentiment'] = a.apply(lambda x: analyze_sentiment(x), axis=1)
    #b = data['325']
    #b['Sentiment'] = b.apply(lambda x: analyze_sentiment(x), axis=1)
    test_sim = a.dropna(subset=['Sentiment'])
    #test_sim = test_sim.append(b.dropna(subset=['SA']))
    print(test_sim.to_string())

    #test_sim = pd.DataFrame()

    #sim_sentiments = [sim.apply(lambda x: analyze_sentiment(x), axis=1).dropna().mean() for sim in clean_sims]

    #for name in sim_names:
    #  sim = data[name]
    #  sim['Sentiment'] = sim.apply(lambda x: analyze_sentiment(x), axis=1)
    #  test_sim = test_sim.append(sim.dropna(subset=['SA']))

    #test_sim = test_sim[test_sim['Simulation'] == 2]

    # plot the average sentiment against the positive negative rankings

    #test_sim = test_sim[test_sim['Sentiment'] != 0]

    x, y = test_sim['SA'], test_sim['Sentiment']
    #x, y = sim_sentiments, sim_classifications
    x = np.arange(len(y))
    m, b = np.polyfit(x, y, 1)

    plt.scatter(x, y)
    plt.plot(x, m * x + b, label=f"Lin Reg(Delta Sentiment={m*len(y):.3},Y-Intercept={b:.3})")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
