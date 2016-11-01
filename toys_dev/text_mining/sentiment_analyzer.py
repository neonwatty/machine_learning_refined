import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# library for extracting text from HTML files
from bs4 import BeautifulSoup   

# library for stemming, removing stopwords, etc.,
'''
Make sure to run at command line

python -m nltk.downloader all-corpora

to grab all corpus, including stopwords
 
Or run for much faster download just 

nltk.download("stopwords") 

in python once to grab just stopwords

'''
import nltk
nltk.download("stopwords") 
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem.porter import *
stemmer = PorterStemmer()

# import regular expressions for tokenization
import re

# import scikit learn BoW transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import cPickle

# data transformation tools
import pandas as pd       

class sentiment_analyzer():
    
    def __init__self(self):
        self.data = []
        self.train_data = []
        self.train_labels = []
        self.clean_train_reviews = []
        self.train_data_features = []
        self.trained_model = []
        self.train_accuracy = []
        self.train_accuracy = []

    # import training data 
    def load_training_data(self,csvname):
        # load in dataframe
        self.data = pd.read_csv(csvname, header=0, \
                    delimiter="\t", quoting=3)
        
        # grab data
        self.train_data = self.data['review']
        
        # grab labels
        self.train_labels = np.asarray(self.data['sentiment'])
        
    # remove document from HTML page, parse, remove stop words, and stem
    def review_to_words(self,raw_review):
        # Function to convert a raw review to a string of words
        # The input is a single string (a raw movie review), and 
        # the output is a single string (a preprocessed movie review)

        # Remove HTML tags (if present)
        review_text = BeautifulSoup(raw_review).get_text() 

        # Remove non-letters        
        letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

        # Convert to lower case, split into individual words
        words = letters_only.lower().split()                             

        # In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))                  

        # Remove stop words 
        meaningful_words = [w for w in words if not w in stops]   

        # Stem the word list - no built in functionality in scikit, although you can directly import the nlkt stemmer
        stemmed_words = [stemmer.stem(word) for word in meaningful_words]

        # Join the words back into one string separated by space, 
        # and return the result.
        return( " ".join( stemmed_words )) 
    
    # clean training dataset
    def clean_train_data(self):
        # Get the number of reviews based on the dataframe column size
        num_reviews = self.train_data.size

        # Initialize an empty list to hold the clean reviews
        self.clean_train_reviews = []

        # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list 
        for i in xrange( 0, num_reviews ):
            # Call our function for each one, and add the result to the list of
            # clean reviews
            self.clean_train_reviews.append( self.review_to_words( self.train_data[i] ) )
        
        # print update
        print 'done cleaning data, raw text now stored'
        
    ## convert cleaned, stopword removed, stemmed dataset to BoW features
    def transform_to_BoW(self):
        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.  Keep only top 5000 most commonly occuring words
        vectorizer = CountVectorizer(analyzer = "word",   \
                                     tokenizer = None,    \
                                     preprocessor = None, \
                                     stop_words = None,   \
                                     max_features = 5000) 

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of 
        # strings.
        train_data_features = vectorizer.fit_transform(self.clean_train_reviews)

        # Numpy arrays are easy to work with, so convert the result to an 
        # array
        self.train_data_features = []
        self.train_data_features = train_data_features.toarray()
        
        # Take a look at the words in the vocabulary
        # vocab = vectorizer.get_feature_names()
        print 'done transforming raw data into BoW features'


    # perform classification on training set
    def perform_classification(self):
        # split dataset into training and testing sets
        X_train = self.train_data_features[:20000,:]
        y_train = self.train_labels[:20000]

        X_test = self.train_data_features[20000:,:]
        y_test = self.train_labels[20000:]
        
        # load in classifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        
        # fit classifier to training data
        clf.fit(X_train, y_train)
        
        # save model
        self.trained_model = pickle.dumps(clf)
        with open('my_dumped_classifier.pkl', 'wb') as fid:
            cPickle.dump(self.trained_model, fid)  
        
        # print scores on training and testing sets
        self.train_accuracy = clf.score(X_train, y_train)
        self.test_accuracy = clf.score(X_test, y_test)
        
        print 'done training boosted classifier'
        print 'accuracy on training set is ' + str(self.train_accuracy)  
        print 'accuracy on testing set is ' + str(self.test_accuracy)  