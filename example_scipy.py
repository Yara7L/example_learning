import numpy as numpy
import matplotlib.pyplot as pyplot
from scipy import integrate,interpolate

print(integrate.quad(lambda x:numpy.exp(-x**2),-10,10))
def half_circle(x):
    return (1-x**2)**0.5
def half_sphere(x,y):
    return (1-x**2-y**2)**0.5
res=integrate.dblquad(half_sphere,-1,1,lambda x:-half_circle(x),lambda x:half_circle(x))
print(res[0])

func=numpy.poly1d(numpy.array([1,2,3,4]))
func1=func.deriv(m=1)
func2=func.deriv(m=2)
x=numpy.linspace(-10,10,30)
y=func(x)
y1=func1(x)
y2=func2(x)
pyplot.plot(x,y,'ro',x,y1,'g--')
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.show()

pyplot.subplot(3,1,1)
pyplot.plot(x,y,c='r',linestyle='-')
pyplot.title('Polynomial')

pyplot.subplot(3,1,2)
pyplot.plot(x,y1,c='b',linestyle='',marker='^')
pyplot.title('Firest Derivative')

pyplot.subplot(3,1,3)
pyplot.plot(x,y2,c='g',linestyle='',marker='o')
pyplot.title('Second Derivative')

pyplot.semilogx(x,y)
pyplot.semilogy(x,y)
pyplot.loglog(x,y)

fig=pyplot.figure()
ax=fig.add_subplot(211)
ax.fill_between(x,y,y1,facecolor='b')
ax.grid(True)

ax2=fig.add_subplot(212)
ax2.fill(x,y,facecolor='b',alpha=0.3)
ax2.fill(x,y1,facecolor='g',alpha=0.3)
ax2.grid(True)
pyplot.show()

u=numpy.linspace(-1,1,100)
x,y=numpy.meshgrid(u,u)
z=x**2+y**2
fig=pyplot.figure()
ax=fig.add_subplot(111)
ax.contourf(x,y,z)
pyplot.show()

#高斯分布
a=[]
for i in range(50000):
    a.append(numpy.random.normal())
pyplot.hist(a,10000)
pyplot.show()

#矩阵
A=numpy.array([[1,2,3],[4,5,6],[7,8,9]])
print(numpy.linalg.det(A))#行列式
print(numpy.linalg.inv(A)*numpy.linalg.det(A))#伴随矩阵
print(numpy.linalg.inv(A))#逆矩阵
x=numpy.array([[1,0,0],[0,2,0],[0,0,3]])
a,b=numpy.linalg.eig(x)#特征向量
print(a,b)