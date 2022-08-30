import pandas as pd
import numpy
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

msg = pd.read_csv('C:/Users/RohanK/Documents/prog/SPAM DETECTION PROJECT/SMSSpamCollection', sep = '\t', names = ['label', 'content'])

#Data cleaning, preprocessing
nltk.download('stopwords')
lm = WordNetLemmatizer()
corpus = []
for i in range(0, len(msg)):
    review = re.sub('[^a-zA-Z]', ' ', msg['content'][i])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#Tfidf model
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
#dummy values -> ham=1 spam=0
y=pd.get_dummies(msg['label'])
y=y.iloc[:,1].values

#traintest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)
conmat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(conmat)
print(accuracy)





