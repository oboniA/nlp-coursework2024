import nltk
import spacy
from pathlib import Path
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import re


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    pass



def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    pass



def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""

    # initiates empty list for dataframe content
    textfile_data = []

    # iterates each file in directory
    for filepath in path.glob("*.txt"):
        
        # extracts filename 
        file_name = filepath.stem
        # parse the title, author, and year
        title, author, year = file_name.split('-')
        
        # reads texts from the file
        with filepath.open("r", encoding="utf-8") as file:
            file_text= file.read()
        
        # appends texts and parsed data to list
        textfile_data.append({"text": file_text, "title": title, "author": author, "year": year})
    
    # creates dataframe from datalist 
    dataframe = pd.DataFrame(textfile_data)

    # sorts dataframe by year column 
    sorted_dataframe = dataframe.sort_values(by='year')
    # resets by ignoring the order of index
    reset_dataframe = sorted_dataframe.reset_index(drop=True)
    
    return reset_dataframe



def parse(df, store_path=Path.cwd() / "texts" / "novels" / "parsed", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""

    pass


def regex_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using a regular expression."""
    
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""

    # tokenize text with nltk lib
    tokens = word_tokenize(text)
    token_types = set(tokens)
    token_type_ratio= len(token_types) / len(tokens)

    return(token_type_ratio)



def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = {'nltk_ttr' : nltk_ttr(row["text"])
        }
    return f"\nToken Type Ratio: {results}"


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list of tuples."""
    
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list of tuples."""
    pass



def subject_counts(doc):
    """Extracts the most common subjects in a parsed document. Returns a list of tuples."""
    pass





if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    

    path = Path.cwd() / "texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head(10))
    nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "texts" / "novels" / "parsed" / "name.pickle")
    # print(get_subjects(df))


    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "say"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "say"))
        print("\n")
    """

