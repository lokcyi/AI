import numpy as np

#定義一維陣列  
# a = np.array([2, 0, 1, 5, 8, 3])  
# print(a)
# print(a[0])
# print('第一個後面皆取',a[1:])
# print('取第一個',a[:1])
# print('從最後取兩個',a[-2:])
# print('最後兩個不取',a[:-2])
# print(a.min()) #最小值
# print(a.max()) #最大值


# # #定義二維陣列  
# b = np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]])  
# print(b)
# print('取第1行,最後兩個', b[1][-2:])


# 定義 5x4 陣列
# c = np.arange(20).reshape(5, 4)
# print(c)
# # If no axis mentioned, then it works on the entire array
# print(np.argmax(c))
# # If axis=1, then it works on each row
# print(np.argmax(c, axis=1))
# # If axis=0, then it works on each column
# print(np.argmax(c, axis=0))

# # 陣列相加
# A = np.array([2, 4, 6])
# B = np.array([4, 5, 6])
# result_1 = A + B
# print(result_1)

# A2 = np.array([[1, 2, 3], [4, 5, 6]])
# B2 = np.array([[7, 8, 9], [1, 2, 3]])
# result_2 = A2 + B2
# print(result_2)

# 陣列相乘
A1 = np.array([[1, 2, 3], [4, 5, 6]])
B1 = np.array([[7, 8, 9], [1, 2, 3]])
result_1 = A1 * B1
print(result_1)

A2 = np.array([[1, 2, 3], [4, 5, 6]])
B2 = np.array([[7, 8, 9], [1, 2, 3],[1, 2, 3]])
#result_2 = A2 * B2
result_2 = A2.dot(B2) #https://zh.wikipedia.org/zh-tw/%E7%9F%A9%E9%99%A3%E4%B9%98%E6%B3%95
print(result_2)
print(np.inner(A2,B2))
print(np.outer(A2,B2))