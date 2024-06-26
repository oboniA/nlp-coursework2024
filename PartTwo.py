import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import SVC



# Task 2 (a)
def rename_dataframe(filename):

    # load csv file
    df = pd.read_csv(filename)

    # replace Labour Co op with Labour
    df['party'] = df['party'].replace("Labour (Co-op)", "Labour")
    print(df['party'].value_counts())

    partycount= df['party'].value_counts() 
    print(f"Party counts:", partycount)

    # exclude Speaker from party count of party column
    if 'Speaker' in partycount.index:
        partycount = partycount.drop('Speaker')

    # top 4 parties appearing the most
    most_common= partycount.nlargest(4).index.to_list()
    print(f"most common parties", most_common)

    # dataframe for most common parties
    most_common_party_df = ~df['party'].isin(most_common)
    print(most_common_party_df)
    
    # dataframe for Speech in cpeech_class column
    speech_class_df = df['speech_class'] == 'Speech'

    # speech texts with length more than or equal to 1500
    speech_class_length_df = df['speech'].str.len() >= 1500

    # renamed dataframe
    renamed_df = df[(most_common_party_df) & (speech_class_df) & (speech_class_length_df)]

    rows, columns = renamed_df.shape
    
    # print(f"rows: {rows}")
    # print(f"columns: {columns}")
    return renamed_df




def speech_vectorization(df):

    # drops any None Value
    df_clean = df.dropna(subset=['speech', 'party'])

    speech_x = df_clean['speech']
    party_y = df_clean['party']   # predictor y axis

    # vectorizer initiated with parameters
    tfidf_V = TfidfVectorizer(stop_words='english', max_features=4000)

    # split train test with random seed = 99, test data 20%
    print(f"split train test")
    stratified= StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=99)
    for i,(train_index, test_index) in enumerate(stratified.split(speech_x, party_y)):
        # aplying tfidfvectorizers to extract features
        speech_X_train = tfidf_V.fit_transform(speech_x.iloc[train_index])
        speech_X_test = tfidf_V.fit_transform(speech_x.iloc[test_index])
        party_y_train = party_y.iloc[train_index]
        party_y_test = party_y.iloc[test_index]

        # print(f"\nfold{i}")
        
        # print(f"speech Training data vectorised:\n{speech_X_train[:7]}")
        # print(f"speech Test data vectorised:\n{speech_X_test[:7]}\n")
    print(f"split complete.")

    return speech_X_train, speech_X_test, party_y_train, party_y_test


def random_forest_classification(xtrain, xtest, ytrain, ytest):

    print(f"\nClassifying using random forest.......")
    rf_classifier = RandomForestClassifier(n_estimators=400, random_state=0)
    rf_classifier.fit(xtrain, ytrain)
    rf_party_pred = rf_classifier.predict(xtest)

    rf_f1 = f1_score(ytest, rf_party_pred, average='macro')
    rf_report = classification_report(ytest, rf_party_pred, output_dict=True, zero_division=0)
    print(f"\n---------Random Forest Classifier------\nMarco Average F1 Score: {rf_f1}\nClassification Report:\n{rf_report}")


def svm_classification(xtrain, xtest, ytrain, ytest):
    print(f"\nClassifying using svm.......")
    svm_classifier= SVC(kernel='linear', random_state=99, C=1.0)
    svm_classifier.fit(xtrain, ytrain)
    svm_party_pred = svm_classifier.predict(xtest)

    svm_f1 = f1_score(ytest, svm_party_pred, average='macro')
    svm_report = classification_report(ytest, svm_party_pred, zero_division=0)
    print(f"\n---------SVM Classifier------\nMarco Average F1 Score: {svm_f1}\nClassification Report:\n{svm_report}")



if __name__ == "__main__":

    path = "p2-texts/hansard40000.csv"
    #print(rename_dataframe(path))

    df = pd.read_csv(path)
    print(speech_vectorization(df))


    x_train, x_test, y_train, y_test = speech_vectorization(df)
    print(random_forest_classification(x_train, x_test, y_train, y_test))
    print(svm_classification(x_train, x_test, y_train, y_test))

    


