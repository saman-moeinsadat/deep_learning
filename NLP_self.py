import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from skfeature.function.statistical_based import CFS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



def clean(row):
    words_to_exclude = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    word_free = " ".join(
        [i for i in row.lower().split() if i not in words_to_exclude]
    )
    punc_free = ''.join(ch for ch in word_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def text_return(path):
    m = open(path)
    return m.read().replace('\n', '')


def prepare_label(item):
    exclude = set(string.punctuation)
    unc_free = ''.join(w for w in item if w not in exclude)
    return unc_free.lower()


def spam_or_not(row):
    if 'spmsg' in row:
        return 1
    else:
        return 0


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# path = '/home/saman/python-projects/NLP_data/ling-spam/train-mails'
# emails_path = [os.path.join(path, f) for f in os.listdir(path)]
# data_emails = pd.DataFrame()
# data_emails['path'] = emails_path
# data_emails['text'] = data_emails['path'].apply(text_return)
# data_emails['class'] = data_emails['path'].apply(spam_or_not)
# path_test = '/home/saman/python-projects/NLP_data/ling-spam/test-mails'
# emails_path_test = [os.path.join(path_test, f) for f in os.listdir(path_test)]
# data_emails_test = pd.DataFrame()
# data_emails_test['path'] = emails_path_test
# data_emails_test['text'] = data_emails_test['path'].apply(text_return)
# data_emails_test['class'] = data_emails_test['path'].apply(spam_or_not)
# text_train_features = data_emails['text'].copy()
# text_train_features = text_train_features.apply(clean)
# vectorizer = TfidfVectorizer()
# train_features = vectorizer.fit_transform(text_train_features)
# # svd = TruncatedSVD(n_components=200, algorithm='arpack', random_state=42)
# # train_features = svd.fit_transform(train_features)
# text_test_features = data_emails_test['text'].copy()
# text_test_features = text_test_features.apply(clean)
# vectorizer1 = TfidfVectorizer()
# test_features = vectorizer1.fit_transform(text_test_features)
# svd1 = TruncatedSVD(n_components=200, algorithm='arpack', random_state=42)
# test_features = svd.fit_transform(test_features)
# # mnb = MultinomialNB(alpha=0.2)
# # mnb.fit(train_features, data_emails['class'])
# # prediction = mnb.predict(train_features, data_emails['class'])
# # print(
# #     'the accuracy of this spam classification NaiveB model is: %.4f' %
# #     roc_auc_score(data_emails_test['class'], prediction)
# # )``
# # param_grid = {
# #     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
# #     'gamma': [0.1, 1, 10],
# #     'C': [0.1, 1, 10],
# #     'tol': [0.001, 0.01, 0.1, 1]
# # }
# # svc_tuned = GridSearchCV(
# #     SVC(), cv=3, scoring='roc_auc',
# #     param_grid=param_grid
# # )
# # svc_tuned.fit(train_features, data_emails['class'])
# # print svc_tuned.best_params_
# svc = SVC(kernel='rbf', C=10, tol=0.1, gamma=1)
# svc.fit(train_features, data_emails['class'])
# prediction = svc.predict(test_features)
# print(
#     'the accuracy of this spam classification SVM model is: %.4f' % roc_auc_score(
#         data_emails_test['class'], prediction
#     )
# )
def text_extract(row):
    headers = ['Message-ID:', 'Date:', 'From:', 'To:', 'Cc:', 'Mime-Version:', 'Content-Type:',
    'Content-Transfer-Encoding:', 'Bcc:', 'X-From:', 'X-To:', 'X-cc:', 'X-bcc:',
    'X-Folder:', 'X-Origin', 'X-FileName:', '-- Forwarded', 'cc:',
    'Sent']
    text_list = []
    for line in row.split('\n'):
        flag = True
        if 'Subject:' in line:
            text_list.append(line)
        for item in headers:
            if item in line:
                flag = False
                break
        if flag:
            text_list.append(line)
    return ' '. join(text_list)


def extract_label(row):
    line_list = row.split('/')
    first = line_list[0].split('-')
    second = line_list[1].split('_')
    if first[0] in second:
        return prepare_label(line_list[2])
    if first[0] not in second:

        return prepare_label(line_list[1])


def label_maker(row):
    if ('project' in row) or ('inbox' in row) or ('document' in row) or ('archive' in row):
        return 'achive'
    elif 'sent' in row:
        return 'sent'
    elif 'delete' in row:
        return 'deleted'
    else:
        return 'personal'


path = '/home/saman/python-projects/NLP_data'
data = pd.read_csv(path+'/emails150MB.csv', encoding='latin-1')
labels = data.iloc[:, 0].copy()
labels = labels.apply(extract_label)
labels = labels.apply(label_maker)
print labels.value_counts()
text = data.iloc[:, 1].copy()
text_body = text.apply(text_extract)
text_body = text.apply(clean)
print text_body[5]
label = LabelEncoder()
labels = label.fit_transform(labels)
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(text_body)
svd = TruncatedSVD(n_components=120, algorithm='arpack', random_state=42)
features = svd.fit_transform(features)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=1
)
svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(features_train, labels_train)
prediction = svc.predict(features_test)
print(
    'the accuracy of this spam classification SVM model is: %.4f' % roc_auc_score(
        labels_test, prediction
    )
)
