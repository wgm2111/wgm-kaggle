
# 
# Will Martin
# bag of words
# 4/15
# BSD
# 



# Standard imports
from __future__ import print_function, division
import re

# Third party
from pandas import read_csv, DataFrame
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer


# Parameters for data prep
# --
STOP_WORD_SET = set(stopwords.words('english'))
MAX_FEATURES = 5000


# Function for parsing reviews
def review_to_words(raw_review, stop_set=STOP_WORD_SET):
    """
    Function for parsing a raw review and returning a space separated 
    value string of words of interest.

    ## examples

    >>> review_to_words("I am beautifully <br/> soupy.")
    u'beautifully soupy'

    >>> review_to_words("The big fat bear has hair.")
    u'big fat bear hair'
    """
    # Strip HTML
    review_text = BeautifulSoup(raw_review).get_text()
    
    # Strip non-letter characters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text).lower()
    
    # split words and keep non-stopwords
    out = " ".join([
            w for w in letters_only.split() if not (w in stop_set)])
    return out
    


# Load data
training_data = read_csv("labeledTrainData.tsv", 
                         header=0, delimiter="\t", quoting=3)

# Clean reviews and make a new coloumn
words = []
for review in training_data['review'].values:
    words.append(review_to_words(review))
training_data['words'] = words


# Get a subset of data
reviews_train, reviews_cv, y_train, y_cv = cross_validation.train_test_split(
    training_data['words'], training_data['sentiment'], test_size=0.7, random_state=0)

# Vectorizer
vectorizer = CountVectorizer(analyzer='word', max_features=MAX_FEATURES)
X_train =  vectorizer.fit_transform(reviews_train).toarray()
X_cv = vectorizer.transform(reviews_cv).toarray()

# Load the test case
test_data = read_csv("testData.tsv", 
                     header=0, delimiter="\t", quoting=3)
test_data['words'] = [
    review_to_words(review) for review in test_data['review'].values]
X_test = vectorizer.transform(test_data['words'])




# Script 
if __name__ == "__main__":

    # 
    import doctest
    doctest.testmod()
