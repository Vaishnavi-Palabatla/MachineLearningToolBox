from flask import Flask,render_template, request, session, Response
# from werkzeug import secure_filename
# from flask_session import Session
import io
import random
from algorithms import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import json
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, metrics
from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


app = Flask(__name__)
app.secret_key = "any random string"

@app.route('/')
def upload_file():
   return render_template('fileupload.html')
	
@app.route('/uploader', methods = ['GET','POST'])
def uploader():

    fname=request.form.get("filename")
    print(fname.split("."))
    MLtype=request.form.get("algotype")
    if(fname.split(".")[1]!="txt"):
        return render_template('fileupload.html', filetxt=1) 
    df = pd.read_csv(fname,header=None)
    session["df"]=df.to_json()
    # print(df.to_json()) 
    session["MLtype"]=MLtype
    print(fname)
    print(MLtype)
#    if request.method == 'POST':
#       f = request.files['file']
#     #   f.save(secure_filename(f.filename))
    return render_template('SecondPage.html')


# @app.route('/linear',  methods = ['GET','POST'])
# def linear():
#     automate=request.form.get("automate")
#     session["automate"]=automate
#     print(session["automate"])
#     return render_template('linear.html',auto=automate)

@app.route('/linearregression',  methods = ['GET','POST'])
def linearregression():
    automate=request.form.get("automate")
    df=session["df"]
    df=pd.read_json(df)
    df = pd.concat([pd.Series(1,index = df.index,name='00'),df],axis=1)
    drop_col = len(df.columns)-2
    X = df.drop(columns=drop_col)
    y = df.iloc[:,-1]
    print("@@@@@@@@@@@@@ in linear regression")
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    # data = session['testdata']
    test_file=request.form.get("filename")
    data = pd.read_csv(test_file,header=None)
    data_n2 = data.values
    m2 = len(data_n2)
    data = pd.concat([pd.Series(1,index = data.index,name='00'),data],axis=1)
    # data = pd.read_json(data)
    ypred = reg.predict(data)
    data = pd.DataFrame(data)   
    # print(session["automate"])
    if(session["automate"]=="yes"):
        e=0
    else:
        e=1
    if(request.method == ['GET']):
        output = io.BytesIO()
        FigureCanvas(plt).print_png(output)
        img=Response(output.getvalue(), mimetype='image/png')
        return render_template('linear.html',img=img)

    x=data.values.tolist()
    print("@@@@@@@@@@@@@@@@@@@@@ x value = ",x)
    print("@@@@@@@@@@@@@ y value = ",y)
    return render_template('linear.html',auto=automate,xy = zip(x,ypred),flag=1,exitYes=e)

@app.route('/classi', methods = ['GET','POST'])
def classi():
    x=request.form.get("classi")
    y=request.form.get("automate")
    # print(x,y)
    return render_template('fileupload.html')


@app.route('/select', methods = ['GET','POST'])
def select():
    x=request.form.get("select")
    if(x=="shape"):
        df=session["df"]
       
        df=pd.read_json(df)
        # print(":////////////",df.shape)
        y=df.shape
        return render_template('SecondPage.html', tup=str(y),flag=1)
        # return str(y)
    if(x=="desc"):
        df=session["df"]
       
        df=pd.read_json(df)
        x=df.values.tolist()
        
    #     y=df.describe()
    #     print("://////////// describe",y)
    #     print(len(y))
    #     print(len(y.columns))
    #     for i i
        return render_template('SecondPage.html',table1=x, flag=2)
    if(x=="plot"):
        df=session["df"]
       
        df=pd.read_json(df)
        drop_col = len(df.columns)-2
        X = df.drop(columns=drop_col)
        y = df.iloc[:,-1]
        if len(df.columns)==2:
            plt.scatter(X[0],y,c="blue")
            plt.show()
        else:
            return render_template('SecondPage.html',flag=5)

            
    if(x=="predict"):
        return render_template('SecondPage.html',flag=6)
        


    # print(x)
    # return x
@app.route('/plot', methods = ['GET','POST'])
def plot():
    df=session["df"]
    # df = pd.concat([pd.Series(1,index = df.index,name='00'),df],axis=1)  
    df=pd.read_json(df)
    
    drop_col = len(df.columns)-1
    X = df.drop(columns=drop_col)
    y = df.iloc[:,-1]
    col=int(request.form.get("colvalue"))-1
    # col = int(input("Enter the Independent column number to plot = "))
    plt.scatter(X[col],y,c="blue")
    plt.show()
    return render_template('SecondPage.html',flag=5)
@app.route('/predictbutton', methods = ['GET','POST'])
def predictbutton():
    automate=request.form.get("automate")
    session["automate"]=automate
    print(session["automate"])
    if(automate=="yes"):
        return render_template('SecondPage.html',flag=8) 
    else:
        return render_template('SecondPage.html',flag=7)
        
@app.route('/predictNo', methods = ['GET','POST'])   
def predictNo():
    alpha=request.form.get("alpha")
    epoch=request.form.get("epoch")
    test_size = float(request.form.get("tandtY"))
    
    test_name = request.form.get("testfile")
    df1 = pd.read_csv(test_name,header=None)
    print("test_name",test_name)
    print(alpha,epoch)
    print(df1)
    df=session["df"]
    data=pd.read_json(df)
    drop_col = len(data.columns)-1
    X = data.drop(columns=drop_col)
    # print("before = ",X)
    X, mean_X2, std_X2 = featureNormalization(X)
    X = pd.concat([pd.Series(1,index = X.index,name='00'),X],axis=1)
    y = data[drop_col].values.reshape(len(X),1)
    theta = np.zeros((len(X.columns),1))
    theta2, J_history2 = gradientDescent(X,y,theta,float(alpha),int(epoch))
                # print("h(x) ="+str(round(theta2[0,0],2))+" + "+str(round(theta2[1,0],2))+"x1 + "+str(round(theta2[2,0],2))+"x2")
    
    # if test_choose == 1:
    #     test_size = 0.4
    # elif test_choose == 2:
    #     test_size = 0.3
    # elif test_choose == 3:
    #     test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=1)
    predict1=predict(X_test,theta2)
    print("r2 score = ",r2_score(y_test, predict1))
    r22=r2_score(y_test, predict1)

    
   
    x2,mean_x2,std_x2 = featureNormalization(df1)
    x2 = pd.concat([pd.Series(1,index = x2.index,name='00'),x2],axis=1)
    
    predict3=predict(x2,theta2)
    print(predict3)
    x=[]
    length=len(df1.columns)
    df1=df1.values.tolist()
    for i in range(len(df1)):
        b=[]
        b.append(i)
        print("for the values ",end="")
        for j in range(length): 
            # print(df1.iloc[i,j],end=", ")
            b.append(df1[i][j])
        b.append(round(predict3[i],2))
        x.append(b)
        print(" the predicted value is {}".format(round(predict3[i],2)))
    session["automate"]="no"
    return render_template('linear.html',flag=0,xy=x,r2=r22,exitYes=1) 

@app.route('/exitYN', methods = ['GET','POST']) 
def exitYN():
    if(request.form.get("exit")=="yes"):
        return render_template('SecondPage.html')
    else:
        if(session["automate"]=="Yes"):
            return render_template('SecondPage.html')
        else:
            return render_template('SecondPage.html',flag=7)

@app.route('/tandt', methods = ['GET','POST']) 
def tandt():
    session["tandtNo"]=request.form.get("tandt")
    if(request.form.get("tandt")=="yes"):
        # test=0
        # session["testNo"]=1
        return render_template('SecondPage.html', flag=9)
    else:
        # session["testNo"]=0
        return render_template('SecondPage.html',flag=10)

@app.route('/tandtY', methods = ['GET','POST']) 
def tandtY():
    test_name = request.form.get("testfile")
    df1 = pd.read_csv(test_name)
    df1 = pd.concat([pd.Series(1,index = df1.index,name='00'),df1],axis=1)
    # session["test_per"]=testing_per
    if(session["tandtNo"]=="no"):
        
        
    # print("testno",session["testNo"])
        test_size =0.4
    else:
        testing_per=float(request.form.get("tandtY"))
        test_size = testing_per
    train_file=session["df"]
    data=pd.read_json(train_file)
    data_1 = pd.concat([pd.Series(1,index = data.index,name='00'),data],axis=1)
    drop_col_2 = len(data_1.columns)-2
    X_2 = data_1.drop(columns=drop_col_2)
    y_2 = data_1.iloc[:,-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=test_size,random_state=1)
    reg = linear_model.LinearRegression()
    reg.fit(X_train,y_train)
    print("r2 score = ", reg.score(X_test,y_test))
    r22=reg.score(X_test,y_test)
    ypred = reg.predict(df1)
    x=[]
    length=len(df1.columns)
    df1=df1.values.tolist()
    if(session["automate"]=="yes"):
        e=0
    else:
        e=1
    for i in range(len(df1)):
        b=[]
        # b.append(i)
        print("for the values ",end="")
        for j in range(0,length):
            # 
            # print(df1.iloc[i,j],end=", ")
            b.append(df1[i][j])
        b.append(round(ypred[i],2))
        x.append(b)
        print(" the predicted value is {}".format(round(ypred[i],2)))
    return render_template('linear.html',xy=x,flag=0,r2=r22,exitYes=e)

if __name__ == '__main__':
   app.run(debug = True)