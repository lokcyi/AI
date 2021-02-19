import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def getData():
    df = pd.read_csv("TestData\\matplot.csv", encoding='utf-8')
    return df
def drawPosition():
    position_data = getData()
    plt.scatter("HEIGHT", "WEIGHT", data=position_data[position_data['POSITION']=='Center'], alpha = 0.2)
    plt.xlabel('Height(")')
    plt.ylabel('Weight(lb)')
    plt.show()
def demoSavepic():
    #data prepare
    df = pd.DataFrame({
        'name':['john','mary','peter','jeff','bill','lisa','jose'],
        'age':[23,78,22,19,45,33,20],
        'gender':['M','F','M','M','M','F','M'],
        'state':['california','dc','california','dc','california','texas','texas'],
        'num_children':[2,0,0,3,2,1,4],
        'num_pets':[5,1,0,5,2,2,3]
    })

    df.groupby('state')['name'].nunique().plot(kind='bar')
    plt.savefig('TestData\\output.png')
    plt.show()    
def demoDrawMutiData():
    x = np.arange(10)
    y1 = np.random.randint(1,10,10)
    y2 = np.random.randint(-10,-1,10)
    plt.bar(x,y1)
    plt.bar(x,y2)
    plt.show()
def demoDrawMutiDataWithX():    
    x1 = np.arange(0,30,3)
    x2 = np.arange(1,30,3)
    y1 = np.random.randint(1,10,10)
    y2 = np.random.randint(1,10,10)
    # plt.bar(x1,y1)
    # plt.bar(x2,y2)
    # #改成散點圖
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.show()
def demoDraw3D():
    x = np.linspace(-3, 3, 1001)
    y = np.linspace(-3, 3, 1001)
    X, Y = np.meshgrid(x, y)
    Z = (2 * np.pi) * np.exp((-1)/2 * (X ** 2 + Y ** 2))

    fig = plt.figure()
    ax = Axes3D(fig) #繪製有3D座標的Axeㄋ物件
    ax.plot_surface(X, Y, Z,cmap=plt.get_cmap('rainbow')) #繪製3D圖形 #rainbow颜色映射
    plt.show()
def ggplot():
    plt.style.use('ggplot')
    x = np.random.rand(50)
    y = np.random.rand(50)
    colors = np.random.rand(50)
    area = np.pi * (15 * np.random.rand(50))**2
    plt.scatter(x,y,s=area,c=colors,alpha=0.5)
    plt.show()

if __name__ == '__main__':
    # demoSavepic()
    # demoDrawMutiData()
    # demoDrawMutiDataWithX()
    demoDraw3D()
    # ggplot()
        





