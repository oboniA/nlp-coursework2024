import pandas as pd
from pathlib import Path

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



if __name__ == "__main__":

    path = "p2-texts/hansard40000.csv"
    print(rename_dataframe(path))

