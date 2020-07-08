import nltk
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


def pre_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [
        word for word in text.split() if word.lower() not in stopwords.
        words('english')
    ]
    words = ''
    for word in text:
        stemmer = SnowballStemmer('english')
        words += (stemmer.stem(word) + ' ')
    return words


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
path = '/home/saman/python-projects/NLP_data'
data = pd.read_csv(path+'/spam.csv', encoding='latin-1')
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.rename(columns={'v1': 'class', 'v2': 'text'}, inplace=True)
text_features = data['text'].copy()
text_features = text_features.apply(pre_process)
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(text_features)
label = LabelEncoder()
data['class'] = label.fit_transform(data['class'])
# print(vectorizer.get_feature_names())
features_train, features_test, labels_train, labels_test = train_test_split(
    features, data['class'], test_size=0.3, random_state=1
)
svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(features_train, labels_train)
prediction = svc.predict(features_test)
print(
    'the accuracy of this spam classification SVM model is: %.4f' % accuracy_score(
        labels_test, prediction
    )
)
mnb = MultinomialNB(alpha=0.2)
mnb.fit(features_train, labels_train)
prediction = mnb.predict(features_test)
print(
    'the accuracy of this spam classification NaiveB model is: %.4f' %
    roc_auc_score(labels_test, prediction)
)
