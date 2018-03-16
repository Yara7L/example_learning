import numpy as numpy
import pandas as pandas

data1=pandas.DataFrame(numpy.random.rand(6,4),columns=list('ABCD'))
print(data1)
date3=pandas.date_range('20170101',periods=6)
data2=pandas.DataFrame(numpy.random.rand(6,4),index=date3,columns=list('ABCD'))
print(data2)
data3=pandas.DataFrame({'A':numpy.random.randn(3)})
print(data3)
data4=pandas.DataFrame({'A':pandas.Timestamp('20170101'),'B':numpy.random.randn(3)})
print(data4)

print(numpy.array([1,2,3,4,5]))
numeric_string=numpy.array(['1.23','2.34','3.45'],dtype=numpy.string_)
print(numeric_string)
print(numeric_string.astype(float))

arr2d=numpy.arange(1,10).reshape(3,3)
print(arr2d)
print(arr2d[2])
print(arr2d[0][2],'=====',arr2d[0,2])

names=numpy.array(['Bob','Job','Bob','will'])
print(names=='Bob')

arr=numpy.arange(10)
numpy.save('some_array',arr)
print(numpy.load('some_array.npy'))

print(numpy.linspace(1,10,20))
print(numpy.zeros((3,4)))
print(numpy.ones((3,4)))
print(numpy.eye(4))

a=numpy.ones((2,2))
b=numpy.eye(2)
print(numpy.vstack((a,b)))
print(numpy.hstack((a,b)))

print(numpy.linalg.eig(a))

'''
#维度
a=numpy.array([[1,2,3,4], [5,6,7,8,9], [9,10,11,12]])
print(a)
row_r1=a[1, :]
row_r2=a[1:2, :]
print(row_r1,row_r1.shape)
print(row_r2,row_r2.shape)

a=numpy.array([[1,2],[3,4],[5,6]])
print(a[[0,1,2],[0,1,0]])
print(numpy.array([a[0,0],a[1,1],a[2,0]]))
'''

x=numpy.array([[1,2],[3,4]])
y=numpy.array([[5,6],[7,8]])
v=numpy.array([9,10])
w=numpy.array([11,12])
print(v.dot(w))
print(numpy.dot(v,w))
print(x.dot(w))
print(numpy.dot(x,w))
print(x.dot(y))
print(numpy.dot(x,y))

x=numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
v=numpy.array([1,0,1])
y=numpy.empty_like(x)
for i in range(4):
    y[i,:]=x[i,:]+v
print(y)
vv=numpy.tile(v,(4,1))
print(vv)
y=x+vv
print(y)
y=x+v
print(y)

v=numpy.array([1,2,3])
w=numpy.array([4,5])
print(numpy.reshape(v,(3,1))*w)
x=numpy.array([[1,2,3],[4,5,6]])
print((x.T+w).T)
print(x+numpy.reshape(w,(2,1)))
print(x*2)

'''
#SciPy库的问题
from scipy.misc import imread,imresize,imsave
img=imread('C:/Users/admin/Desktop/flower.jpg')
print(img.dtype,img.shape)
img_tinted=img*[1,0.9,0.9]
img_tinted=imresize(img_tinted,(300,300))
imsave('C:/Users/admin/Desktop/flower.jpg',img_tinted)
'''

s=pandas.Series([1,2,3,4,5],index=['a','b','c','d','e'])
print(s)
print(pandas.Series({'A':'good','B':'bad'}))

dict={
    'name':['A','B','C','D'],
    'sex':['F','M','M','F'],
    'age':[19,21,18,20]
}
df=pandas.DataFrame(dict)
print(df)
'''
print(df.info())
print(df['age'])
print(df.name)

print(df[0:1])
df.age=22
print(df)
df['score']=[90,91,90,98]
print(df)
df['country']='China'
print(df)
print(df[['name','score']])
df.age=df.age+1
print(df[df.age>=20])
print((df.sex=='M')&(df.age>20))
print(df[(df.sex=='M')&(df.age>20)])
df1=pandas.DataFrame(numpy.arange(4).reshape(2,2),columns=['a','b'])
df2=pandas.DataFrame(numpy.arange(6).reshape(2,3),columns=['a','b','c'])
print(df1)
print(df2)
print(df1+df2)
print(df1.add(df2,fill_value=0))
print(df1*df2)
print(df1.mul(df2,fill_value=2))
'''
print(df.iloc[0])
df.index=['a','b','c','d']
print(df.loc['a'])
print(df.iloc[0:2])
print(df.iloc[0:3,1:3])
print(df.ix[0])
print(df.ix['a','age'])
print(df.ix[['a','c'],'name'])
print(df.ix['a':'c','name'])
print(df.values)
print(df.describe)
print(df.sort_values(by='age'))
df_test=['A','C','M']
print(df['name'].isin(df_test))
group_sex=df.groupby('sex')
print(group_sex.first().head())
print(df.groupby(['name','age']).aggregate(numpy.sum).head())
'''
#'numpy.int32' object is not callable
df6=df[df['age']<22]
print(df6.size().head())
'''
print(group_sex.agg([numpy.sum,numpy.mean,numpy.std]).head(3))
print(df['age'].agg({'总计':numpy.sum,'均值':numpy.mean,'标准差':numpy.std}).head())

df['country']='China'
print(df)
df.insert(3,'grade',5)
print(df)
del df['grade']
print(df)
df_drop=df.drop(['country'],axis=1)
print(df)
print(df_drop)

import matplotlib.pyplot as plt
data=pandas.DataFrame(numpy.random.randn(100,2),columns=list('XY'))
plt=data.plot(kind='scatter',x='X',y='Y').get_figure()
plt.savefig('C:/Users/admin/Desktop/plot/try.png')
data2=pandas.DataFrame(numpy.random.rand(6,4),columns=list('ABCD'))
plt2=data2.plot(kind='bar').get_figure()
plt2.savefig('C:/Users/admin/Desktop/plot/plot2.png')
plt3=data2.plot(kind='bar',stacked=True).get_figure()
plt3.savefig('C:/Users/admin/Desktop/plot/plot3.png')
data2.boxplot()
plt.savefig('C:/Users/admin/Desktop/plot/plot4.png')

df_nan=pandas.DataFrame(numpy.random.randn(4,3),index=list('absd'),columns=['one','two','three'])
print(df_nan)
df_nan.ix[1,:-1]=numpy.nan
df_nan.ix[1:-1,2]=numpy.nan
print(df_nan)
print(df_nan.fillna(0))
print(df_nan.fillna('missing'))
print(df_nan.fillna(method='pad'))
print(df_nan.fillna(method='bfill'))
print(df_nan.fillna(method='bfill',limit=1))
print(df_nan.fillna(df_nan.mean()))
print(df_nan.fillna(df_nan.mean()['one':'two']))
print(df_nan.interpolate())
print(df_nan.dropna(axis=0))

df_1=pandas.DataFrame(numpy.random.rand(3,4),columns=['A','B','C','D'])
df_2=pandas.DataFrame(numpy.random.rand(2,4),columns=['B','D','A','E'])
print(pandas.concat([df_1,df_2]))
print(pandas.concat([df_1,df_2],ignore_index=True))
print(pandas.concat([df_1,df_2],axis=1))
print(pandas.concat([df_1,df_2],axis=1,ignore_index=True))

s=pandas.Series(list('ABCD'))
print(s)
print(s.str.lower())
print(s.str.upper())
print(s.str.len())
print(s.str.split())
print(s.str.get(2))
print(s.str.replace('A','a'))
s2=pandas.Series(['a','b','c',numpy.nan])
pattern=r'[a-z]'
print(s2.str.contains(pattern))
print(s2.str.contains(pattern,na=False))
print(s2.str.match(pattern,na=False))
print(s2.str.startswith('A',na=False))
print(s2.str.endswith('c',na=False))

'''
import sqlite3
con=sqlite3.connect('user_information.sqlite')
sql='select * from user_information LIMIT 3'
df=pandas.read_sql(sql.con)
df=pandas.read_sql(sql,con,index_col='id')
#可以设置多个index，index_col=['id','bank_id']
con.execute('DROP TABLE IF EXISTS user_information')
pandas.io.sql.write_sql(df,'user_information',con)
import MySQLdb
con=MySQLdb.connect(host='',db='')
'''
