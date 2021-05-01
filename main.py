import time
import numpy as np
from scipy import optimize
from Class import ipProblem
from Class import simplexMethod
from Class import dualSimplex

# 读取用户输入并转化为标准形式
print("请输入决策变量个数和约束个数：\n")
n, m = map(int, input().split())

print("请输入目标函数中决策变量的系数：\n")
arr = input()
c = [int(num) for num in arr.split()]

print("请输入约束中决策变量的系数、右端系数和约束类型：\n")
A = [[0] * n for i in range(m)]
b = np.zeros(m, dtype=int)
count = 0
for i in range(m):
    arr = input()
    arr = [int(num) for num in arr.split()]
    for j in range(n):
        A[i][j] = arr[j]
    b[i] = arr[n]
    if arr[n + 1] == 1:  # 若约束类型是大于等于则需要减去额外的决策变量
        for k in range(m):
            A[k].append(0)
        A[i][-1] = -1
        count = count + 1
    elif arr[n + 1] == -1:  # 若约束类型是小于等于则需要加上额外的决策变量
        for k in range(m):
            A[k].append(0)
        A[i][-1] = 1
        count = count + 1
for i in range(count):  # c中需要添加对应数量的决策变量
    c.append(0)
# 若某个约束的右端系数小于0则将等式两端乘以-1
for i in range(m):
    if b[i] < 0:
        b[i] = -b[i]
        for j in range(len(A[i])):
            A[i][j] = -A[i][j]
print("请输入决策变量的约束类型：\n")
arr = input()
arr = [int(num) for num in arr.split()]
for i in range(n):
    if arr[i] == -1:  # 若约束类型是小于等于0则需要对该变量取反
        c[i] = -c[i]
        for j in range(m):
            A[j][i] = -A[j][i]
    elif arr[i] == 0:  # 若约束类型是无约束则需要用两个变量相减替换该变量
        c.insert(i + 1, -c[i])
        for j in range(m):
            A[j].insert(i + 1, -A[j][i])
c = np.array(c)
c = np.negative(c)  # 将min转化为max
A = np.array(A)

pb = ipProblem(c, A, b)
# 单纯形法求解并统计运行时间
time_start = time.time()
s = pb.solve(simplexMethod)
time_end = time.time()
if s.target != None:
    s.target = -s.target  # 因为求解的是max问题所以需要取反
    for i in range(n):
        if arr[i] == -1:  # 若约束类型是小于等于0则需要对该变量取反
            s.solve[i] = -s.solve[i]
        elif arr[i] == 0:  # 若约束类型是无约束则需要用两个变量相减替换该变量
            s.solve[i] = s.solve[i] - s.solve[i + 1]
            np.delete(s.solve, s.solve[i + 1])
    s.solve = np.resize(s.solve, (n))  # 只输出原n个决策变量
print("\n")
print(s)
print("单纯形法的求解时间为：", time_end - time_start)
print("\n")
# 对偶单纯形法求解并统计运行时间
time_start = time.time()
s = pb.solve(dualSimplex)
time_end = time.time()
if s.target != None:
    s.target = -s.target  # 因为求解的是max问题所以需要取反
    for i in range(n):
        if arr[i] == -1:  # 若约束类型是小于等于0则需要对该变量取反
            s.solve[i] = -s.solve[i]
        elif arr[i] == 0:  # 若约束类型是无约束则需要用两个变量相减替换该变量
            s.solve[i] = s.solve[i] - s.solve[i + 1]
            np.delete(s.solve, s.solve[i + 1])
    s.solve = np.resize(s.solve, (n))  # 只输出原n个决策变量
print("\n")
print(s)
print("对偶单纯形法的求解时间为：", time_end - time_start)
print("\n")

print("\n第三方库对比求解：\n")
# 使用第三方库对比求解
c = np.negative(c)  # 将max转化为scipy库接受的标准形式min
res = optimize.linprog(c, None, None, A, b)
print(res)
