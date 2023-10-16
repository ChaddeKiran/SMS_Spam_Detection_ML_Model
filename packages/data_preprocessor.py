from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_data(path_to_data, encoding="latin-1"):
    data=pd.read_csv(path_to_data, encoding=encoding)

    data['label']=data['target'].map({'Not spam' : 0, 'spam': 1})

    x=data['messages']
    y=data['label']

    return {'text':x,'label':y}


def create_train_test_data(x,y,test_size,random_state):
    cv= CountVectorizer()
    x = cv.fit_transform(x)
    x_train, x_test, y_train,y_test = train_test_split(x,y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test':y_test}, cv