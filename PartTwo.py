import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import SVC
import contractions
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('wordnet')


# Task 2 (a)
def rename_dataframe(filename):

    # load csv file
    df = pd.read_csv(filename)

    # replace Labour Co op with Labour
    df['party'] = df['party'].replace("Labour (Co-op)", "Labour")

    # exclude Speaker from party count of party column
    partycount= df['party'].value_counts() 
    if 'Speaker' in partycount.index:
        partycount = partycount.drop('Speaker')

    # top 4 parties appearing the most
    most_common= partycount.nlargest(4).index.to_list()

    # dataframe for most common parties
    most_common_party_df = df['party'].isin(most_common)
    
    # dataframe for Speech in cpeech_class column
    speech_class_df = df['speech_class'] == 'Speech'

    # speech texts with length more than or equal to 1500
    speech_class_length_df = df['speech'].str.len() >= 1500

    # renamed dataframe
    renamed_df = df[(most_common_party_df) & (speech_class_df) & (speech_class_length_df)]

    return renamed_df



# Task 2 (b)
def speech_vectorization(df):

    speech_x = df['speech']
    party_y = df['party'].to_numpy()   # predictor y axis

    # vectorizer initiated with parameters
    tfidf_V = TfidfVectorizer(stop_words='english', 
                              max_features=4000,)  
    
    # Perform the train-test split
    train_X, test_X, train_Y, test_Y = train_test_split(speech_x, 
                                                        party_y, 
                                                        test_size=0.3, 
                                                        random_state=42, 
                                                        stratify=party_y)
    
    speech_X_train = tfidf_V.fit_transform(train_X)
    speech_X_test = tfidf_V.transform(test_X)
    return speech_X_train, speech_X_test, train_Y, test_Y



# task2 (d)
def adjust_parameter_vectorization(df):

    speech_x = df['speech']
    party_y = df['party'].to_numpy()   # predictor y axis

    # vectorizer initiated with parameters
    tfidf_V = TfidfVectorizer(stop_words='english', 
                                ngram_range=(1, 3),
                                max_features=4000) 

    # Perform the train-test split
    train_X, test_X, train_Y, test_Y = train_test_split(speech_x, 
                                                        party_y, 
                                                        test_size=0.3, 
                                                        random_state=42, 
                                                        stratify=party_y)

    speech_X_train = tfidf_V.fit_transform(train_X)
    speech_X_test = tfidf_V.transform(test_X)
    return speech_X_train, speech_X_test, train_Y, test_Y



# Task 2 (e)
def custom_token(text):

    # Replace contractions with their expanded forms
    text = contractions.fix(text)

    # Convert text to lowercase
    text = text.lower()

    # remove punctuations
    punctuations = string.punctuation
    text = ''.join([char for char in text if char not in punctuations])

    # Remove stopwords and perform stemming
    stop_words = set(stopwords.words('english'))

    # word tokenizer
    tokens = nltk.word_tokenize(text)

    # stemming and lematizing
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens if token not in stop_words]
    return tokens



# Task 2 (e)
def custom_vectorization(df):

    speech_x = df['speech']
    party_y = df['party'].to_numpy()  

    # vectorizer initiated with parameters
    tfidf_V = TfidfVectorizer(ngram_range=(1, 3),
                              max_features=4000,
                              tokenizer=custom_token,
                               token_pattern=None) 

    # Perform the train-test split
    train_X, test_X, train_Y, test_Y = train_test_split(speech_x, 
                                                        party_y, 
                                                        test_size=0.3, 
                                                        random_state=42, 
                                                        stratify=party_y)
    
    speech_X_train = tfidf_V.fit_transform(train_X)
    speech_X_test = tfidf_V.transform(test_X)
    return speech_X_train, speech_X_test, train_Y, test_Y



# Task 2 (c)
def random_forest_classification(speech_x_train, speech_x_test, party_y_train, party_y_test):

    rf_classifier = RandomForestClassifier(n_estimators=400, random_state=99)
    rf_classifier.fit(speech_x_train, party_y_train)
    rf_party_pred = rf_classifier.predict(speech_x_test)

    rf_f1 = f1_score(party_y_test, rf_party_pred, average='macro')
    rf_report = classification_report(party_y_test, rf_party_pred, zero_division=0)  
    return rf_f1, rf_report



# Task 2 (c)
def svm_classification(speech_x_train, speech_x_test, party_y_train, party_y_test):
#
    svm_classifier= SVC(kernel='linear', random_state=99, C=1.0)
    svm_classifier.fit(speech_x_train, party_y_train)
    svm_party_pred = svm_classifier.predict(speech_x_test)

    svm_f1 = f1_score(party_y_test, svm_party_pred, average='macro')
    svm_report = classification_report(party_y_test, svm_party_pred, zero_division=0)
    return svm_f1, svm_report



# Task 2 (e)
def best_classifier(rf, svm, x_train, x_test, y_train, y_test):
    
    # unpack returned values from classifiers
    rf_f1score, rf_report = rf(x_train, x_test, y_train, y_test)
    svm_f1score, svm_report,  = svm(x_train, x_test, y_train, y_test)

    # compares f1 scores of the classifiers
    if rf_f1score > svm_f1score:
        print(f"RF: {rf_f1score}\n{rf_report}")
    else:
         print(f"SVM: {svm_f1score}\n{svm_report}")

    return " "



if __name__ == "__main__":

    path = "p2-texts/hansard40000.csv"

    # 2 (a)
    renamed_path = rename_dataframe(path)  # leave it uncommented at all time!
    print(renamed_path.shape)   

    """
        The following functions have the same scripts
        except for the parameter requirements from 2(b)-(e)
        in each function, additional parameters are added (or removed in (e))
    """

    # # 2(b)
    # x_train, x_test, y_train, y_test = speech_vectorization(renamed_path)   

    # # 2(d)
    # x_train, x_test, y_train, y_test = adjust_parameter_vectorization(renamed_path)   
    
    # # 2(c) : used for both (c) & (d)
    # # random forest classifier
    # rf_f1, rf_c_report = random_forest_classification(x_train, x_test, y_train, y_test)
    # print(rf_f1)   # f1 score for random forest
    # print(rf_c_report)   # classification report for random forest
    
    # # SVM classifier
    # svm_f1, svm_c_report = svm_classification(x_train, x_test, y_train, y_test)  
    # print(svm_f1)   # f1 score for svm
    # print(svm_c_report)  # classification report for svm
    
    # # 2(e)
    # x_train, x_test, y_train, y_test = custom_vectorization(renamed_path)     
    # best_classifier(random_forest_classification, svm_classification, x_train, x_test, y_train, y_test)  

    


    


