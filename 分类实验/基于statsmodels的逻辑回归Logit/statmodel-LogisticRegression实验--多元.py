import pandas as pd
import  statsmodels.api as sma
data=pd.read_csv('Default.csv',header=0)
#print(data.head())
#dummy_default=pd.get_dummies(data['default'],prefix='dafault')
#print(dummy_default.head())
#data_all=data[name].join(dummy_default.loc[:,'dafault_No':])#python2中用.ix
name=['balance','income']
y_default=data['default']
y_default_values=[]
for i in range(len(y_default)):
    if y_default[i]=='No':
        y_default_values.append(0)
    else:
        y_default_values.append(1)
y_student=data['student']
x_student_values=[]
for i in range(len(y_student)):
    if y_student[i]=='No':
        x_student_values.append(0)
    else:
        x_student_values.append(1)
data_all=data[name]
data_all['default_values']=y_default_values#数表中增加1列
data_all['x_student_values']=x_student_values#数表中增加1列
#data_all_ok1=data_all.drop('student',axis=1)
#data_all_ok2=data_all_ok1.drop('income',axis=1)
data_all['intercept']=1.0#需要自行添加逻辑回归所需的intercept变量
x_all=data_all.drop('default_values',axis=1)
logit=sma.Logit(data_all['default_values'],x_all)
result=logit.fit()
print(result.summary())
