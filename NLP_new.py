import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string


def extract_csv_data(file, cols_to_clean=[], exclude=[[]]):
    data = pd.read_csv(file)
    for i, col in enumerate(cols_to_clean):
        exclude_pattern = re.compile('|'.join(exclude[i]))
        data = data[data[col].str.contains(exclude_pattern) == False]
        return data


def remove_duplicates(data):
    processed = set()
    result = []
    pattern = re.compile('X-FileName: .*')
    pattern2 = re.compile('X-FileName: .*?  ')

    for doc in data:
        doc = doc.replace('\n', ' ')
        doc = doc.replace(' .*?nsf', '')
        match = pattern.search(doc).group(0)
        match = re.sub(pattern2, '', match)

        if match not in processed:
            processed.add(match)
            result.append(match)

    return result


def claen(data):
    words_to_exclude = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    word_free = " ".join(
        [i for i in data.lower().split() if i not in words_to_exclude]
    )
    punc_free = ''.join(ch for ch in word_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
path = '/home/saman/python-projects/NLP_data/'
emails = extract_csv_data(
    path+'emails.csv', ['file'], [['notes_inbox', 'discussion_threads']]
)
emails_bodies = emails.message.as_matrix()
unique_emails = remove_duplicates(emails_bodies)
print unique_emails[6]
