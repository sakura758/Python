import pandas as pd
import  statsmodels.api as sma
data=pd.read_csv('E:\data\Default.csv',header=0)
name=['student','balance','income']
y_default=data['default']
y_default_values=[]
for i in range(len(y_default)):
    if y_default[i]=='No':
        y_default_values.append(0)
    else:
        y_default_values.append(1)
data_all=data[name]
data_all['default_values']=y_default_values#数表中增加1列
data_all_ok1=data_all.drop('student',axis=1)
data_all_ok2=data_all_ok1.drop('income',axis=1)
#data_all_ok2=data_all_ok1.drop('income',axis=1)
data_all_ok2['intercept']=1.0#需要自行添加逻辑回归所需的intercept变量
x_all=data_all_ok2.drop('default_values',axis=1)
logit=sma.Logit(data_all['default_values'],x_all)
result=logit.fit()
print(result.summary())
