import numpy as np
# MAD，即median absolute deviation，可译为绝对中位值偏差
def MAD(dataset, n):
    median = np.median(dataset)  # 中位数
    deviations = abs(dataset - median)
    mad = np.median(deviations)
 
    remove_idx = np.where(abs(dataset - median) >  n * mad)
    new_data = np.delete(dataset, remove_idx)
 
    return new_data
#3\sigma法
#3\sigma法又称为标准差法。标准差本身可以体现因子的离散程度，和MAD算法类似，只是3\sigma法用到的不是中位值，而是均值，并且n的取值为3，代码如下    
# 3sigma法
def three_sigma(dataset, n= 3):
    mean = np.mean(dataset)
    sigma = np.std(dataset)
 
    remove_idx = np.where(abs(dataset - mean) > n * sigma)
    new_data = np.delete(dataset, remove_idx)
 
    return new_data
#3.百分位法
#百分位计算的逻辑是将因子值进行升序的排序，对排位百分位高于97.5%或排位百分位低于2.5%的因子值，类似于比赛中”去掉几个最高分，去掉几个最低分“的做法。代码如下：这里参数采用的是20%和80%，具#体取值，还需具体情况具体分析。    

def percent_range(dataset, min= 0.20, max= 0.80):
    range_max = np.percentile(dataset, max * 100)
    range_min = -np.percentile(-dataset, (1 - min) * 100)
 
    # 剔除前20%和后80%的数据
    new_data = []
    for value in dataset:
        if value < range_max and value > range_min:
            new_data.append(value)
#三：使用Box Plot来发现Outliers

#一个典型的Box Plot是基于以下五个值计算而来的
#a. 一组样本的最小值
#b. 一组样本的最大值
#c. 一组样本的中值
#d. 下四分位数（Lower Quartile / Q1）
#e. 上四分位数（Upper Quartile / Q3）
#根据这五个值构建出来基本的Box Plot，某些图形软件还会显示平均值，IQR= Q3 – Q1
#显然超出上下四分位数的值可以看做为Outliers。我们通过眼睛就可以很好的观察到这些Outliers值的点。
#假设一组数据为：2,4,6,8,12,14,16,18,20,25,45
#中值 Median = 14
#Q1-下四分位数（11 * 0.25 = 3） = 7
#Q3-上四分位数（11 * 0.75 = 9） =19
#IQR（Q3 – Q1） = 12
#1.5 * IQR = 18
#最小值（6 – 1.5 * IQR）= 2
#最大值（20 + 1.5 * IQR）= 25
#很显然值45是一个适度Outliers
#对比的一组数据为：2,4,6,8,12,14,16,18,20,25,26
