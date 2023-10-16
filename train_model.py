import os 
os.chdir("C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml")
import joblib

import packages.data_preprocessor as dp
import packages.model_trainer as mt

path_to_data = 'C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml\\data.csv'


prepared_data = dp.prepare_data(path_to_data,encoding="latin-1")


train_test_data, vectorizer =dp.create_train_test_data(prepared_data['text'],
                                                        prepared_data['label'],
                                                        0.33,2021)


model=mt.run_model_training(train_test_data['x_train'], train_test_data['x_test'],
                            train_test_data['y_train'], train_test_data['y_test'])


joblib.dump(model,'./models/my_spam_model.pkl')

joblib.dump(vectorizer, open("./vectors/my_vectorizer.pickel","wb"))