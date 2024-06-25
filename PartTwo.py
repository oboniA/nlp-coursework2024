import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit


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
    
    print(f"rows: {rows}")
    print(f"columns: {columns}")
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

        print(f"\nfold{i}")
        print(f"train index: {train_index}")
        print(f"test: {test_index}")

        print(f"speech Training data vectorised:\n{speech_X_train[:7]}")
        print(f"speech Test data vectorised:\n{speech_X_test[:7]}\n")
    
    return speech_X_train, speech_X_test




if __name__ == "__main__":

    path = "p2-texts/hansard40000.csv"
    #print(rename_dataframe(path))

    df = pd.read_csv(path)
    print(speech_vectorization(df))

