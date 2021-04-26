#importing Flask library
from flask import Flask, request, render_template, session, redirect
import pandas as pd
import numpy as np
import scipy.stats as scipy
import matplotlib.pyplot as plt
import io 
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

#Create instance
app = Flask(__name__)

df20_21 = pd.read_csv('20-21.csv')


#Use app as a decorator to create each route/url that is provided by the application
@app.route("/", methods=("POST", "GET"))
def html_table():
    return render_template('simple.html',  
    tables=[this_week.to_html(classes='data', header="true")])
