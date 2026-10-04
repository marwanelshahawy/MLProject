import os
import sys 
import pickle

from sklearn.model_selection import GridSearchCV
from src.exception import customException
from src.logger import logging

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score 

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path ,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise customException(e,sys)


def evaluate_models(X_train,y_train,X_test,y_test,models,param,threshold):

    try:
        model_report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3,scoring='f1',n_jobs=-1)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            proba = model.predict_proba(X_test)[:, 1]
            predicted = (proba >= threshold).astype(int)

            accuracy = accuracy_score(y_test, predicted)
            recall = recall_score(y_test, predicted)
            precision = precision_score(y_test, predicted)
            f1 = f1_score(y_test, predicted)
            model_report[list(models.keys())[i]] = recall
            logging.info(f"{list(models.keys())[i]} model trained with accuracy score: {accuracy}, recall score: {recall}, precision score: {precision}, and F1 score: {f1}")

        return model_report
    except Exception as e:
        raise customException(e,sys)
