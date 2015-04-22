
# 
# Will Martin
# bag of words
# 4/15
# BSD
# 



# Standard imports
from __future__ import print_function, division

# Third party
from pandas import read_csv, DataFrame
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import numpy as np

# My local imports
from basic_imports import (
    X_train, y_train, 
    X_cv, y_cv,
    X_test,
    vectorizer, training_data, test_data)

# Define a function to take a clasifier ant return training data
def get_train_data(clf, Xtrain, ytrain, Xcv, ycv, Nchunks=50):
    """
    Train the classifier in Nchunks, and return training data.
    """
    # Break training data into chunks
    Xchunks = [X_train[i::Nchunks] for i in range(Nchunks)]
    ychunks = [y_train[i::Nchunks] for i in range(Nchunks)]
    
    # Loop over the chunks and train the classifier
    M = 0
    Mlist = []
    scores = []
    
    for Xchunk, ychunk in zip(Xchunks, ychunks):

        # Update measurement count
        M += ychunk.size
        Mlist.append(M)

        # train the bernoulli model with more data
        clf.partial_fit(Xchunk, ychunk, classes=[0,1])
        scores.append(clf.score(X_cv, y_cv))

    return clf, np.array(Mlist), np.array(scores)


# Make classifiers by trying different values of smoothing parameter alpha
alphas = np.linspace(.7, 1.0, 3)
bernoulli_clfs = []
bernoulli_score_lists = []
multinomial_clfs = []
multinomial_score_lists = []

# Loop over smoothing parameters
for alpha in alphas:
    
    # Make a Bernoulli based nieve base classifier
    bernoulli_clf = BernoulliNB(binarize=.1, alpha=alpha)
    bernoulli_clf, ms, scores = get_train_data(bernoulli_clf, 
                                               X_train, y_train, 
                                               X_cv, y_cv)
    bernoulli_clfs.append(bernoulli_clf)
    bernoulli_score_lists.append(scores)

    # Make a Multinomial based nieve classifier
    multinomial_clf = MultinomialNB(alpha=alpha)
    multinomial_clf, ms, scores = get_train_data(multinomial_clf, 
                                               X_train, y_train, 
                                               X_cv, y_cv)
    multinomial_clfs.append(multinomial_clf)
    multinomial_score_lists.append(scores)

    


# Train on the cv and test set and output a solution 
multinomial_clf.partial_fit(X_cv, y_cv)
y_test_predict = multinomial_clf.predict(X_test)
out_frame = DataFrame(data={'id': test_data['id'], 
                            'sentiment': y_test_predict})
out_frame.to_csv('Bag_of_Words_model.csv', index=False, quoting=3)




# # Choose binary cutoff
# bin_cuts = np.array([.5, 1.5, 2.5, 3.5, 4.5, 5.5])




# # Make a Nieve Bayes Bercoulli classifier
# bernoulli_scores = np.zeros(len(bin_cuts))
# bernoulli_clfs = []


# for i, bin_cut in enumerate(bin_cuts):
#     # Bernoulli
#     bernoulli_clf = BernoulliNB(binarize=bin_cut)
#     bernoulli_clf.fit(X_train, y_train)
#     bernoulli_clfs.append(bernoulli_clf)

#     # score
#     bernoulli_scores[i] = bernoulli_clf.score(X_cv, y_cv)

# multinomial_clf = MultinomialNB()
# multinomial_clf.fit(X_train, y_train)
# multinomial_score = multinomial_clf.score(X_cv, y_cv)

if __name__ == "__main__":

    # 
    import doctest
    doctest.testmod()


    # plot the score curve during training
    import matplotlib.pyplot as plt
    fig = plt.figure(0, (8,8), facecolor='white')
    fig.clf()
    box = [.1, .1, .8, .8]
    ax = fig.add_axes(box)
    ax.plot(Mlist, bernoulli_scores, linewidth=4, label="BernoulliNB")
    ax.plot(Mlist, multinomial_scores, linewidth=4, label="MultinomialNB")
    ax.set_title("Sentiment prediction based on movie review text classification", fontweight='bold')
    ax.legend()
    fig.show()
