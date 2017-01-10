import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib

# library for extracting text from HTML files
from bs4 import BeautifulSoup   

# library for stemming, removing stopwords, etc.,
'''
Make sure to run at command line

python -m nltk.downloader all-corpora

to grab all corpus, including stopwords
 
Or run for much faster download just 

nltk.download("stopwords") 

in python once to grab just stopwords - downloading the entire corpus takes a lot of time!

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
        
        self.test_data = []
        self.test_labels = []
        self.clean_test_reviews = []
        self.test_data_features = []
        self.test_accuracy = []
        self.test_accuracy = []

    # import training data 
    def load_data(self,csvname):
        # load in dataframe
        all_data = pd.read_csv(csvname)
        
        # grab training data and labels
        data = all_data['review']        
        labels = np.asarray(all_data['sentiment'])
        
        return data,labels
        
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
    def clean_data(self,dataset):
        # Get the number of reviews based on the dataframe column size
        num_reviews = dataset.size

        # Initialize an empty list to hold the clean reviews
        cleaned_reviews = []

        # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list 
        for i in xrange( 0,num_reviews):
            # Call our function for each one, and add the result to the list of
            # clean reviews
            cleaned_reviews.append( self.review_to_words(dataset[i]))
        
        # print update
        return cleaned_reviews
        
    ## convert cleaned, stopword removed, stemmed dataset to BoW features
    def make_BoW_transform(self,train_dataset):
        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.  Keep only top 5000 most commonly occuring words
        BoW_transform = CountVectorizer(analyzer = "word",   \
                                     tokenizer = None,    \
                                     preprocessor = None, \
                                     stop_words = None,   \
                                     max_features = 5000) 

        ## Take BoW features from training data - creating a dictionary of words that will also be used on the testing data
        BoW_transform.fit(train_dataset)
        
        ## save our BoW transform (fit to the training data) so we can use it later to transform future data
        joblib.dump(BoW_transform, 'BoW_transform.pkl') 
        
        # to load from file use below
        ## BoW_transform = joblib.load('BoW_transform.pkl') # load the BoW transform from file
                
        # Take a look at the words in the vocabulary
        # vocab = vectorizer.get_feature_names()

        return BoW_transform

    # perform classification on training set
    def perform_classification(self,X_train,y_train,X_test,y_test):        
        # load in classifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        
        # fit classifier to training data
        clf.fit(X_train, y_train)
        
        # save model
        joblib.dump(clf, 'learned_booster.pkl') 
        
        # print scores on training and testing sets
        self.train_accuracy = clf.score(X_train, y_train)
        self.test_accuracy = clf.score(X_test, y_test)
        
        print 'done training boosted classifier'
        print 'accuracy on training set is ' + str(self.train_accuracy)  
        print 'accuracy on testing set is ' + str(self.test_accuracy)  