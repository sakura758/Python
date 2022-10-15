import pandas as pd
from sklearn.linear_model import LogisticRegression
data=pd.read_csv('Default.csv',header=0)
#data.head()
x_balance=data.iloc[:,2]#balance数据
x_balance_ok=x_balance.values.reshape(-1,1)#data['default'].values.reshape(-1,1)
y_default=data.iloc[:,0]#default数据df['Age'].reshape(-1, 1)
#y_default=data['default'].values.reshape(-1,1)
y_default_ok=pd.DataFrame(y_default).values.ravel()
clf=LogisticRegression(random_state=0).fit(x_balance_ok,y_default_ok)#训练模型
Beta_0=clf.intercept_
Beta_all=clf.coef_
print('估计参数Beta_0={},Beta_1={}'.format(clf.intercept_,clf.coef_))
y_default_estamete=clf.predict(x_balance_ok)#估计结果
y_default_estamete_Prob=clf.predict_proba(x_balance_ok)#估计结果所对应的概率
score=clf.score(x_balance_ok, y_default_ok)#准确率
print('预测的准确率为：{}'.format(score))
#pd.DataFrame(housing_labels).values.ravel()
from matplotlib import pyplot as plt
y_default_values=[]
for i in range(len(y_default)):
    if y_default[i]=='No':
        y_default_values.append(0)
    else:
        y_default_values.append(1)
        
plt.scatter(x_balance,y_default_values,c='g',label='True Result')
plt.plot(x_balance,y_default_estamete_Prob[:,1],'r--',label='Estamete Result')
plt.show()
