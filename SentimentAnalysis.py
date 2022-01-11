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
    return scores['compound']


def main():
    with open('clean_data.pickle', 'rb') as handle:
        data = pickle.load(handle)

    sim_names = list(data.keys())
    clean_sims = [data[name] for name in sim_names]

    # print(clean_sims[0].to_string())

    a = data['345']
    a['Sentiment'] = a.apply(lambda x: analyze_sentiment(x), axis=1)
    #b = data['325']
    #b['Sentiment'] = b.apply(lambda x: analyze_sentiment(x), axis=1)
    test_sim = a.dropna(subset=['SA'])
    #test_sim = test_sim.append(b.dropna(subset=['SA']))
    # print(test_sim.to_string())

    # test_sim = pd.DataFrame()

    # for name in sim_names:
    #   sim = data[name]
    #   sim['Sentiment'] = sim.apply(lambda x: analyze_sentiment(x), axis=1)
    #   test_sim = test_sim.append(sim.dropna(subset=['SA']))

    test_sim = test_sim[test_sim['Simulation'] == 3]
    test_sim = test_sim[test_sim['Sentiment'] != 0]

    x, y = test_sim['SA'], test_sim['Sentiment']
    x = np.arange(len(y))
    m, b = np.polyfit(x, y, 1)

    plt.scatter(x, y)
    plt.plot(x, m * x + b, label=f"Lin Reg(m={m*len(y):.3},b={b:.3})")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
