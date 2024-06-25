import nltk
import spacy
from pathlib import Path
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import pickle
from collections import Counter



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
    words = word_tokenize(text)
    sentence = sent_tokenize(text)
    word_nums = len(words)
    sent_nums = len(sentence)
    syllables = sum(count_syl(word, d) for word in words)

    # prevents zero division errors
    if (word_nums == 0) or (sent_nums == 0):
        fk_grade_level = 0  
    else:
        avg_sent_len = word_nums / sent_nums
        avg_syl_per_word = syllables / word_nums

        # using FK Grade Level formula
        fk_grade_level = 0.39 * avg_sent_len + 11.8 * avg_syl_per_word - 15.59
    
    return fk_grade_level




def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower() # lower-case for case-ensivitiy 

    # checks word existence in CMU dict
    if word in d:
        max_syl=0
        # itirates over word pronunciations in dict
        for pronunciation in d[word]:
            syl_count=0
            # when last char of a phone is digit
            # increment syl count for that pronunciation
            for phone in pronunciation:
                if phone[-1].isdigit():  
                    syl_count += 1  
            # update max syl count if current pronunciation have more syls
            if syl_count > max_syl:
                max_syl = syl_count

        return max_syl
    
    else:
        # if word not in dict, estimate by counting vowels
        return len(re.findall(r'[aeiou]+', word))



def read_novels(path=Path.cwd() / "p1-texts" / "novels"):
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
        textfile_data.append({"text": file_text, 
                              "title": title, 
                              "author": author, 
                              "year": year})
    
    # creates dataframe from datalist 
    dataframe = pd.DataFrame(textfile_data)

    # sorts dataframe by year column 
    sorted_dataframe = dataframe.sort_values(by='year')
    # resets by ignoring the order of index
    reset_dataframe = sorted_dataframe.reset_index(drop=True)
    
    return reset_dataframe



def parse_sections(text):
    """parses texts in section in cases where texts exceed spaCy maximum length"""

    # when text length is more than spacy max len
    if len(text) > nlp.max_length:
        sections=[] # stores texts in chunks

        # gets chunks out of a text by itirating
        for start_pos in range(0, len(text), nlp.max_length):
            # split texts into spacy chunks to parse
            end_pos = start_pos + nlp.max_length
            section = text[start_pos:end_pos]
            sections.append(nlp(section))   
        return sections  
    else:
        return nlp(text)
   


def parse(df, store_path=Path.cwd() / 'pickles', out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    
    print(f"spacy model loading....\nnew collum being added to the dataframe.....")
    df['parsed'] = df['text'].apply(parse_sections)  # create parsed column

    store_path.mkdir(parents=True, exist_ok=True)  # creates path if doesnt exist already
    serialisation_output= (store_path / out_name)  # destination
    print(f"serializing to pickle format...")

    # adds parsed datarfame to pickle file
    with open(serialisation_output, 'wb') as file:
        pickle.dump(df, file)  
    print("DataFrame Parsed to a Pickle File")

    return df


def regex_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using a regular expression."""
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""

    # tokenize text with nltk lib
    tokens = word_tokenize(text.lower())
    token_types = set(tokens)
    token_type_ratio= len(token_types) / len(tokens)

    return(token_type_ratio)



def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return f"\nToken Type Ratio: {results}"


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return f"\nFlech-Kincaid Grade level scores: {results}"



def subjects_by_verb_pmi(doc, target_verb, top_n=10):
    """Extracts the most common subjects of a given verb in a parsed document sorted by PMI."""
    pass




def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list of tuples."""

    print(f"start Subject by Verb count.....")
    subject_counts = Counter()
    for token in doc:
        # dependancy parcing for the main verb 
        if token.lemma_ == "say" and token.dep_ == "ROOT":
            for child in token.children:
                # checks if it is a nominal subject of the verb
                if child.dep_ == 'nsubj':
                    subject_counts[child.text] += 1

    print(f" Subject by Verb count complete..!")
    return subject_counts.most_common(10)  # sorts in decending order


def get_subject_by_verb(df, verb):

    for i, row in df.iterrows():
        novel_title= row["title"]
        parse_doc= row["parsed"]

        print(f"\nNovel {i} Title: {novel_title}")
        print(f" 10 common syntactic subjects of verb 'say':\n", subjects_by_verb_count(parse_doc, verb))



def subject_counts(doc):
    """Extracts the most common subjects in a parsed document. Returns a list of tuples."""
    
    print(f"start Subject count.....")
    subject_counts = Counter()
    for token in doc:
        # dependancy parsing to nominal subject
        if token.dep_ == "nsubj":
            subject_counts[token.text] += 1

    print(f"Subject count complete..!")
    return subject_counts.most_common(10)


def get_subjects(df):
    for i, row in df.iterrows():
        novel_title= row["title"]
        parse_doc= row["parsed"]

        print(f"\nNovel {i} Title: {novel_title}")
        print(f" 10 common syntactic subjects:\n", subject_counts(parse_doc))





if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """


    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head(10))
    #nltk.download("cmudict")
    #parse(df)
    # print(df.head())
    # print(get_ttrs(df))
    # print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" / "parsed.pickle")
    print(df.head())
    #print(get_subjects(df))
    print(get_subject_by_verb(df, 'say'))


    """ 
    for i, row in df.iterrows():
    print(row["title"])
    print(subjects_by_verb_count(row["parsed"], "say"))
    print("\n")
    """
    