from api import app
import os
import pandas as pd
import json
from flask import jsonify
from flask import render_template
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from flask import url_for
from sklearn import cross_validation as cv
from sklearn import linear_model as lm
from sklearn.metrics import confusion_matrix
import sklearn.metrics as skm
import scipy
def get_abs_path():
    return os.path.abspath(os.path.dirname(__file__))



def get_data():
    f_name= os.path.join(get_abs_path(),'data','breast-cancer-wisconsin.csv')
    columns = ['code', 'clump_thickness', 'size_uniformity', 'shape_uniformity', 'adhesion', 'cell_size', 'bare_nuclei',
               'bland_chromatin',
               'normal_nuclei', 'mitosis', 'class']


    d = pd.read_csv(f_name, sep=',', header=None, names=columns,na_values='?')
    return d.dropna()


@app.route('/')
def index():
    df =get_data()
    x=df.ix[:,(df.columns !='class')&(df.columns!='code')].as_matrix()
    y=df.ix[:,df.columns=='class'].as_matrix()
    #scale
    scaler=preprocessing.StandardScaler().fit(x)
    scaled=scaler.transform(x)
    #PCA

    pcomp=decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components =pcomp.transform(scaled)
    var= pcomp.explained_variance_ratio_.sum()

    #KMeans
    model= KMeans(init='k-means++', n_clusters=2)
    model.fit(components)

    #Plot

    fig=plt.figure()
    plt.scatter(components[:,0],components[:,1],c=model.labels_)
    centers = plt.plot([model.cluster_centers_[0,0],model.cluster_centers_[1,0]],
                       [model.cluster_centers_[1,0], model.cluster_centers_[1,1]],
                       'kx',c='Green')

    #increase size of center points

    plt.setp(centers, ms=11.0)
    plt.setp(centers, mew=1.8)

    #plot axes adjustments

    axes=plt.gca()
    axes.set_xlim([-7.5,3])
    axes.set_ylim([-2,5])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clustering of PCs ({:.2f}% Var. Explained)'.format(var*100))

    #save fig

    fig_path=os.path.join(get_abs_path(),'static','tmp','cluster.png')
    fig.savefig(fig_path)

    return render_template('index.html',
                           fig=url_for('static',filename='tmp/cluster.png'))

@app.route('/d3')
def d3():
    df = get_data()
    x = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    # scale
    scaler = preprocessing.StandardScaler().fit(x)
    scaled = scaler.transform(x)
    # PCA

    pcomp = decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components = pcomp.transform(scaled)
    var = pcomp.explained_variance_ratio_.sum()

    # KMeans
    model = KMeans(init='k-means++', n_clusters=2)
    model.fit(components)

    #Generate CSV
    cluster_data=pd.DataFrame({'pc1':components[:,0],
                               'pc2':components[:,1],
                               'labels': model.labels_
                               })
    csv_path=os.path.join(get_abs_path(),'static','tmp','kmeans.csv')
    cluster_data.to_csv(csv_path)
    return render_template('d3.html', data_file=url_for('static',
                                                      filename='tmp/kmeans.csv'))

@app.route('/head')
def head():
    df=get_data().head()
    data=json.loads(df.to_json())
    return jsonify(data)

@app.route('/prediction')
def prediction():
    df = get_data()
    x = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    scaler = preprocessing.StandardScaler().fit(x)
    scaled = scaler.transform(x)
    X_train, X_test, y_train, y_test = cv.train_test_split(x, y, test_size= 0.4, random_state = 42)
    lgr=lm.LogisticRegression()
    lgr.fit(X_train,y_train)
    y_pred=lgr.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    tn=cm[1,1]
    tp=cm[0,0]
    fp=cm[0,1]
    fn=cm[1,0]
    name="Logistic Regression"
    con_mat={'tn':tn,'tp':tp,'fp':fp,'fn':fn}
    accuracy=skm.accuracy_score(y_pred,y_test)
    for i in range(len(y_test)):
        if(y_test[i]==2):
            y_test[i]=0
        else:
            y_test[i]=1
    for i in range(len(y_pred)):
        if(y_pred[i]==2):
            y_pred[i]=0
        else:
            y_pred[i]=1
            
    false_positive_rate, true_positive_rate, thresholds = skm.roc_curve(y_test, y_pred)

    roc_auc = skm.auc(false_positive_rate, true_positive_rate)

    fig = plt.figure()
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, c='green',
             label=('AUC for model: %s with roc_auc: %0.2f' % (name, roc_auc)))
    plt.legend(loc='lower right', prop={'size': 8})
    plt.plot([0, 1], [0, 1], color='lightgrey', linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'roc_auc.png')
    fig.savefig(fig_path)

    return render_template('prediction.html',
                           fig=url_for('static', filename='tmp/roc_auc.png'))
    
    #pred(con_mat)

@app.route('/api/v1/prediction_confusion_matrix')
def pred():
    df = get_data()
    x = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    scaler = preprocessing.StandardScaler().fit(x)
    scaled = scaler.transform(x)
    X_train, X_test, y_train, y_test = cv.train_test_split(x, y, test_size= 0.4, random_state = 42)
    lgr=lm.LogisticRegression()
    lgr.fit(X_train,y_train)
    y_pred=lgr.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    tn=cm[1,1]
    tp=cm[0,0]
    fp=cm[0,1]
    fn=cm[1,0]
    #name="Logistic Regression"
    con_mat={'logistic regression':{'tn':tn,'tp':tp,'fp':fp,'fn':fn}}
    data=json.loads(json.dumps(con_mat))
    return jsonify(data)

