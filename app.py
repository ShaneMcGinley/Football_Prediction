#importing Flask library
from flask import Flask, request, render_template, session, redirect
# Importing necessary libraries
import pandas as pd
import numpy as np
import scipy.stats as scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.exceptions import DataConversionWarning
import time
warnings.filterwarnings(action="ignore", category=DataConversionWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

#Create instance
app = Flask(__name__)

start = time.time()

shortURL = 'https://github.com/ShaneMcGinley/Football_Prediction/blob/main/'
shortURL_END = '?raw=true'

league1 = pd.read_csv(shortURL + 'German_Results.csv' + shortURL_END)
league2 = pd.read_csv(shortURL + 'French_Results.csv' + shortURL_END)
league3 = pd.read_csv(shortURL + 'Italian_Results.csv' + shortURL_END)
league4 = pd.read_csv(shortURL + 'Spanish_Results.csv' + shortURL_END)
league5 = pd.read_csv('https://github.com/ShaneMcGinley/FYP_Football/blob/main/20-21.csv?raw=true')

leagues = [league1, league2, league3, league4, league5]

pred_this_week = []

count = 1

for x in leagues:
  print(str(count)+"/5 Has Started")
  df1 = x

  pd.options.display.max_columns = None
  pd.options.display.max_rows = None

  df20_21 = df1
  if df1 is league5:
    df20_21 = df20_21.drop(df20_21.iloc[:, 24:139], axis=1)
  else:
    df20_21 = df20_21.drop(df20_21.iloc[:, 23:139], axis=1)
  df20_21 = df20_21.drop(['Div'], axis=1)
  df20_21 = df20_21.drop(['Time'], axis=1)

  feature_table = df20_21.iloc[:,:23]

  # Replacing NaN values with 0 for now
  df20_21.fillna(0, inplace=True)

  avg_home_goals_20_21 = round(df20_21.FTHG.sum()* 1.0 / df20_21.shape[0], 2)
  avg_away_goals_20_21 = round(df20_21.FTAG.sum()* 1.0 / df20_21.shape[0], 2)
  avg_home_conceded_20_21 = avg_away_goals_20_21
  avg_away_conceded_20_21 = avg_home_goals_20_21

  #Team,Home Goals Score,Away Goals Score,Attack Strength,Home Goals Conceded,Away Goals Conceded,Defensive Strength
  table_20 = pd.DataFrame(columns=('Team','HGS','AGS','HAS','AAS','HGC','AGC','HDS','ADS',))
  table_20 = table_20[:-10]

  res_home_20 = df20_21.groupby('HomeTeam').agg(np.sum)
  res_away_20 = df20_21.groupby('AwayTeam').agg(np.sum)

  table_20.Team = res_home_20.index.values
  table_20.HGS = res_home_20.FTHG.values
  table_20.HGC = res_home_20.FTAG.values
  table_20.AGS = res_away_20.FTAG.values
  table_20.AGC = res_away_20.FTHG.values

  if df1 is league1:
    num_games_20 = df20_21.shape[0]/18
  else:
    num_games_20 = df20_21.shape[0]/20

  table_20.HAS = (table_20.HGS / num_games_20) / avg_home_goals_20_21
  table_20.AAS = (table_20.AGS / num_games_20) / avg_away_goals_20_21
  table_20.HDS = (table_20.HGC / num_games_20) / avg_home_conceded_20_21
  table_20.ADS = (table_20.AGC / num_games_20) / avg_away_conceded_20_21

  feature_table = feature_table[['HomeTeam','AwayTeam','FTR','HST','AST']]
  f_HAS = []
  f_HDS = []
  f_AAS = []
  f_ADS = []
  for index,row in feature_table.iterrows():
    f_HAS.append(table_20[table_20['Team'] == row['HomeTeam']]['HAS'].values[0])
    f_HDS.append(table_20[table_20['Team'] == row['HomeTeam']]['HDS'].values[0])
    f_AAS.append(table_20[table_20['Team'] == row['AwayTeam']]['AAS'].values[0])
    f_ADS.append(table_20[table_20['Team'] == row['AwayTeam']]['ADS'].values[0])
      
  feature_table['HAS'] = f_HAS
  feature_table['HDS'] = f_HDS
  feature_table['AAS'] = f_AAS
  feature_table['ADS'] = f_ADS

  def transformResult(row):
      '''Converts results (H,A or D) into numeric values (1, -1 or 0)'''
      if(row.FTR == 'H'):
          return 1
      elif(row.FTR == 'A'):
          return -1
      else:
          return 0

  feature_table["Result"] = feature_table.apply(lambda row: transformResult(row),axis=1)

  x_train = feature_table[['HST','AST','HAS','HDS','AAS','ADS',]]
  y_train = feature_table['Result']

  feat_table = df20_21.sort_index(ascending=False)
  feat_table = feat_table[['HomeTeam','AwayTeam','FTR','FTHG','FTAG','HS','AS','HC','AC']]

  if df1 is league1:
    # Adding next week fixtures
    new_fixtures = pd.DataFrame( [['Bielefeld','Hoffenheim','D',0,0,0,0,0,0],
                                ['Hertha','FC Koln','D',0,0,0,0,0,0],
                                ['Augsburg','Werder Bremen','D',0,0,0,0,0,0],
                                ['Schalke 04','Ein Frankfurt','D',0,0,0,0,0,0],
                                ['M\'gladbach','Stuttgart','D',0,0,0,0,0,0],
                                ['Leverkusen','Union Berlin','D',0,0,0,0,0,0],
                                ['Freiburg','Bayern Munich','D',0,0,0,0,0,0],
                                ['Mainz','Dortmund','D',0,0,0,0,0,0],
                                ['RB Leipzig','Wolfsburg','D',0,0,0,0,0,0]],columns=feat_table.columns)

  elif df1 is league2:
    # Adding next week fixtures
    new_fixtures = pd.DataFrame( [['Nimes','Lyon','D',0,0,0,0,0,0],
                                ['Nice','Strasbourg','D',0,0,0,0,0,0],
                                ['Marseille','Angers','D',0,0,0,0,0,0],
                                ['Lorient','Metz','D',0,0,0,0,0,0],
                                ['Paris SG','Reims','D',0,0,0,0,0,0],
                                ['Lille','St Etienne','D',0,0,0,0,0,0],
                                ['Dijon','Nantes','D',0,0,0,0,0,0],
                                ['Montpellier','Brest','D',0,0,0,0,0,0],
                                ['Bordeaux','Lens','D',0,0,0,0,0,0],
                                ['Monaco','Rennes','D',0,0,0,0,0,0]],columns=feat_table.columns)
    
  elif df1 is league3:
    # Adding next week fixtures
    new_fixtures = pd.DataFrame( [['Genoa','Atalanta','D',0,0,0,0,0,0],
                                ['Spezia','Torino','D',0,0,0,0,0,0],
                                ['Juventus','Inter','D',0,0,0,0,0,0],
                                ['Roma','Lazio','D',0,0,0,0,0,0],
                                ['Fiorentina','Napoli','D',0,0,0,0,0,0],
                                ['Udinese','Sampdoria','D',0,0,0,0,0,0],
                                ['Benevento','Crotone','D',0,0,0,0,0,0],
                                ['Parma','Sassuolo','D',0,0,0,0,0,0],
                                ['Milan','Cagliari','D',0,0,0,0,0,0],
                                ['Verona','Bologna','D',0,0,0,0,0,0]],columns=feat_table.columns)
  elif df1 is league4:
    # Adding next week fixtures
    new_fixtures = pd.DataFrame( [['Sociedad','Valladolid','D',0,0,0,0,0,0],
                                ['Alaves','Granada','D',0,0,0,0,0,0],
                                ['Ath Bilbao','Real Madrid','D',0,0,0,0,0,0],
                                ['Ath Madrid','Osasuna','D',0,0,0,0,0,0],
                                ['Getafe','Levante','D',0,0,0,0,0,0],
                                ['Betis','Huesca','D',0,0,0,0,0,0],
                                ['Barcelona','Celta','D',0,0,0,0,0,0],
                                ['Valencia','Eibar','D',0,0,0,0,0,0],
                                ['Cadiz','Elche','D',0,0,0,0,0,0],
                                ['Villarreal','Sevilla','D',0,0,0,0,0,0]],columns=feat_table.columns)
  elif df1 is league5:
    # Adding next week fixtures
    new_fixtures = pd.DataFrame( [['Southampton','Leicester City','D',0,0,0,0,0,0],
                                ['Crystal Palace','Man City','D',0,0,0,0,0,0],
                                ['Brighton','Leeds United','D',0,0,0,0,0,0],
                                ['Chelsea','Fulham','D',0,0,0,0,0,0],
                                ['Everton','Aston Villa','D',0,0,0,0,0,0],
                                ['Newcastle','Arsenal','D',0,0,0,0,0,0],
                                ['Man United','Liverpool','D',0,0,0,0,0,0],
                                ['Tottenham','Sheffield United','D',0,0,0,0,0,0],
                                ['West Brom','Wolves','D',0,0,0,0,0,0],
                                ['Burnley','West Ham','D',0,0,0,0,0,0]],columns=feat_table.columns)
  else:
    print()

  new_feat_table = new_fixtures.append(feat_table,ignore_index=True)
  new_feat_table = new_feat_table.sort_index(ascending=False)
  new_feat_table = new_feat_table.reset_index().drop(['index'], axis=1)
  new_feat_table = new_feat_table.sort_index(ascending=False)
  feat_table = new_feat_table

  #Adding k recent performance measures
  feat_table["pastHS"] = 0.0
  feat_table["pastHC"] = 0.0
  feat_table["pastAS"] = 0.0
  feat_table["pastAC"] = 0.0
  feat_table["pastHG"] = 0.0
  feat_table["pastAG"] = 0.0

  # Adding k recent performance metrics.
  if df1 is league5:
    k = 5
  else:
    k = 6
  for i in range(feat_table.shape[0]-1,-1,-1):
    row = feat_table.loc[i]
    ht = row.HomeTeam
    at = row.AwayTeam
    ht_stats = feat_table.loc[i-1:-1][(feat_table.HomeTeam == ht) | (feat_table.AwayTeam == ht)].head(k)
    at_stats = feat_table.loc[i-1:-1][(feat_table.HomeTeam == at) | (feat_table.AwayTeam == at)].head(k)

    feat_table.loc[i, 'pastHC'] = (ht_stats[ht_stats["AwayTeam"] == ht].sum().HC + ht_stats[ht_stats["HomeTeam"] == ht].sum().HC)/k
    feat_table.loc[i, 'pastAC'] = (at_stats[at_stats["AwayTeam"] == at].sum().HC + at_stats[at_stats["HomeTeam"] == at].sum().HC)/k
    feat_table.loc[i, 'pastHS'] = (ht_stats[ht_stats["AwayTeam"] == ht].sum().HS + ht_stats[ht_stats["HomeTeam"] == ht].sum().AS)/k
    feat_table.loc[i, 'pastAS'] = (at_stats[at_stats["AwayTeam"] == at].sum().HS + at_stats[at_stats["HomeTeam"] == at].sum().AS)/k
    feat_table.loc[i, 'pastHG'] = (ht_stats[ht_stats["AwayTeam"] == ht].sum().FTAG + ht_stats[ht_stats["HomeTeam"] == ht].sum().FTHG)/k
    feat_table.loc[i, 'pastAG'] = (at_stats[at_stats["AwayTeam"] == at].sum().FTAG + at_stats[at_stats["HomeTeam"] == at].sum().FTHG)/k
  f_HAS = []
  f_HDS = []
  f_AAS = []
  f_ADS = []
  for index,row in feat_table.iterrows():
    #print row
    f_HAS.append(table_20[table_20['Team'] == row['HomeTeam']]['HAS'].values[0])
    f_HDS.append(table_20[table_20['Team'] == row['HomeTeam']]['HDS'].values[0])
    f_AAS.append(table_20[table_20['Team'] == row['HomeTeam']]['AAS'].values[0])
    f_ADS.append(table_20[table_20['Team'] == row['HomeTeam']]['ADS'].values[0])
      
  feat_table['HAS'] = f_HAS
  feat_table['HDS'] = f_HDS
  feat_table['AAS'] = f_AAS
  feat_table['ADS'] = f_ADS


  test_table = feat_table.drop(['FTHG','FTAG','HS','AS','HC','AC'],axis=1)

  test_table["Result"] = test_table.apply(lambda row: transformResult(row),axis=1)
  test_table.sort_index(inplace=True)

  # num_games decides the train-test split
  num_games = feat_table.shape[0]-10

  test_table["pastCornerDiff"] = (test_table["pastHC"] - test_table["pastAC"])/k
  test_table["pastGoalDiff"] = (test_table["pastHG"] - test_table["pastAG"])/k
  test_table["pastShotsDiff"] = (test_table["pastHS"] - test_table["pastAG"])/k

  num_games = feat_table.shape[0]-10
  v_split = 15
  n_games = num_games - v_split

  test_table = test_table.fillna(0)

  test_table.drop(['pastHC','pastAS','pastAC','pastHG','pastAG'],axis=1)
  X_train = test_table[['pastCornerDiff','pastGoalDiff','pastShotsDiff','HAS','HDS','AAS','ADS']].loc[0:n_games]
  y_train = test_table['Result'].loc[0:n_games]
  X_test = test_table[['pastCornerDiff','pastGoalDiff','pastShotsDiff','HAS','HDS','AAS','ADS']].loc[n_games:num_games-1]
  y_test = test_table['Result'].loc[n_games:num_games-1]
  X_predict = test_table[['pastCornerDiff','pastGoalDiff','pastShotsDiff','HAS','HDS','AAS','ADS']].loc[num_games:]

  #KNN
  plot_scores_knn = []
  for b in range(1,50):
      clf_knn = KNeighborsClassifier(n_neighbors=b)
      clf_knn.fit(X_train,y_train)
      scores = accuracy_score(y_test,clf_knn.predict(X_test))
      plot_scores_knn.append(scores)

  #XGBClassifier
  plot_scores_XGB = []
  for i in range(1,100):
      clf_XGB = XGBClassifier(n_estimators=i,max_depth=100,eval_metric='mlogloss')
      clf_XGB.fit(X_train, y_train)
      scores = accuracy_score(y_test,clf_XGB.predict(X_test))
      plot_scores_XGB.append(scores)
      
  #Logistic Regression
  plot_scores_logreg= []
  cs = [0.01,0.02,0.1,0.5,1,3,4,5,10]
  for c in cs:
      clf_logreg = LogisticRegression(C=c,solver='lbfgs',multi_class='ovr')
      clf_logreg.fit(X_train, y_train)
      scores = accuracy_score(y_test,clf_logreg.predict(X_test))
      plot_scores_logreg.append(scores)

  plot_scores_mlp = []
  for h in range(1,100):
    clf_MLP = MLPClassifier(max_iter=h)
    clf_MLP.fit(X_train, y_train)
    scores = accuracy_score(y_test,clf_MLP.predict(X_test))
    plot_scores_mlp.append(scores)

  plot_scores_svlc = []
  for l in range(1,100):
    clf_SVLC = LinearSVC(max_iter=l)
    clf_SVLC.fit(X_train, y_train)
    scores = accuracy_score(y_test,clf_SVLC.predict(X_test))
    plot_scores_svlc.append(scores)

  plot_scores_mnb = []
  for h in range(1,100):
    clf_MNB = MLPClassifier(max_iter=h)
    clf_MNB.fit(X_train, y_train)
    scores = accuracy_score(y_test,clf_MNB.predict(X_test))
    plot_scores_mnb.append(scores)

  max_knn_n = max(plot_scores_knn)
  max_knn_ind = plot_scores_knn.index(max_knn_n)

  max_XGB_e = max(plot_scores_XGB)
  max_XGB_ind = plot_scores_XGB.index(max_XGB_e) if plot_scores_XGB.index(max_XGB_e)!=0 else 1

  max_logreg_c = max(plot_scores_logreg)
  max_logreg_ind = plot_scores_logreg.index(max_logreg_c)

  max_mlp_c = max(plot_scores_mlp)
  max_mlp_ind = plot_scores_mlp.index(max_mlp_c)

  max_svlc_c = max(plot_scores_svlc)
  max_svlc_ind = plot_scores_svlc.index(max_svlc_c)

  max_mnb_c = max(plot_scores_mnb)
  max_mnb_ind = plot_scores_mnb.index(max_mnb_c)

  clf_knn = KNeighborsClassifier(n_neighbors=max_knn_ind).fit(X_train,y_train)
  clf_XGB = XGBClassifier(n_estimators=max_XGB_ind, eval_metric='mlogloss').fit(X_train,y_train)
  clf_logreg = LogisticRegression(C=max_logreg_ind,solver='lbfgs',multi_class='ovr').fit(X_train,y_train)
  clf_MLP = MLPClassifier(max_iter=max_mlp_ind).fit(X_train, y_train)
  clf_LSVC = LinearSVC(max_iter=5000, dual=False).fit(X_train, y_train)

  y_pred_knn = clf_knn.predict(X_predict)
  y_pred_XGB = clf_XGB.predict(X_predict)
  y_pred_logreg = clf_logreg.predict(X_predict)
  y_pred_mpl = clf_MLP.predict(X_predict)
  y_pred_lsvc = clf_LSVC.predict(X_predict)

  this_week = test_table[['HomeTeam','AwayTeam']].loc[num_games:]
  this_week['Result_knn']=y_pred_knn
  this_week['Result_XGB']=y_pred_XGB
  this_week['Result_logreg']=y_pred_logreg
  this_week['Result_MLP']=y_pred_mpl
  this_week['Result_LSVC']=y_pred_lsvc

  def transformResultBack(row,col_name):
      if(row[col_name] == 1):
          return 'Home'
      elif(row[col_name] == -1):
          return 'Away'
      else:
          return 'Draw'

  this_week["Res_knn"] = this_week.apply(lambda row: transformResultBack(row,"Result_knn"),axis=1)
  this_week["Res_XGB"] = this_week.apply(lambda row: transformResultBack(row,"Result_XGB"),axis=1)
  this_week["Res_logreg"] = this_week.apply(lambda row: transformResultBack(row,"Result_logreg"),axis=1)
  this_week["Res_MLP"] = this_week.apply(lambda row: transformResultBack(row,"Result_MLP"),axis=1)
  this_week["Res_LSVC"] = this_week.apply(lambda row: transformResultBack(row,"Result_LSVC"),axis=1)

  this_week.drop(["Result_knn", "Result_XGB","Result_logreg","Result_MLP","Result_LSVC"],axis=1,inplace=True)

  pred_this_week.append(this_week)

  print(str(count)+"/5 Complete")
  count+=1

end = time.time()
print(end - start)


#Use app as a decorator to create each route/url that is provided by the application
@app.route("/", methods=("POST", "GET"))
def html_table():
    return render_template('simple.html')

@app.route('/german_predictions', methods=("POST", "GET"))
def german_predictions():
    return render_template('germany.html',  
    tables=[pred_this_week[0].to_html(classes='data', header="true")])

@app.route('/french_predictions', methods=("POST", "GET"))
def french_predictions():
    return render_template('france.html',  
    tables=[pred_this_week[1].to_html(classes='data', header="true")])

@app.route('/italian_predictions', methods=("POST", "GET"))
def italian_predictions():
    return render_template('italy.html',  
    tables=[pred_this_week[2].to_html(classes='data', header="true")])

@app.route('/spanish_predictions', methods=("POST", "GET"))
def spanish_predictions():
    return render_template('spain.html',  
    tables=[pred_this_week[3].to_html(classes='data', header="true")])

@app.route('/english_predictions', methods=("POST", "GET"))
def english_predictions():
    return render_template('england.html',  
    tables=[pred_this_week[4].to_html(classes='data', header="true")])

