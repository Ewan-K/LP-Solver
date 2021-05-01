import numpy as np
import copy


# ipSolution描述了一个整数规划问题的解
class ipSolution(object):
    def __init__(self, state: bool, specification: str, solve: list,
                 target: float):
        self.state = state  # 是否得到最优解
        self.specification = specification  # 最优解的具体描述
        self.solve = solve  # 最优解对应的决策变量
        self.target = target  # 最优目标函数值

    def __str__(self):
        return f'最优化状态\t: {self.state}\n解的描述\t: {self.specification}\n最优解\t\t: {self.solve}\n最优目标函数值\t: {self.target}'


# ipSolve描述了整数规划问题的抽象解法
class ipSolve(object):
    def __init__(self, problem):  # problem: 待求解问题
        pass

    def solve(self, **parameters):  # 调用对传入的ipProblem求解
        pass


# ipProblem 描述一个整数规划问题
class ipProblem(object):
    def __init__(self, c, a, b):
        self.c = np.array(c, dtype='float64')  # 目标函数系数
        self.a = np.array(a, dtype='float64')  # 系数矩阵
        self.b = np.array(b, dtype='float64')  # 右端常数
        # base_index: 基变量的下标集合
    def solve(self, solver: type, **parameters):  # 调用solver的指定求解
        assert issubclass(solver, ipSolve)
        s = solver(self)
        return s.solve(**parameters)


# 单纯形法
class simplexMethod(ipSolve):
    # 单纯形法内部的整数规划问题表示
    class Problem(ipProblem):
        def __init__(self, c, a, b):
            super().__init__(c, a, b)
            self.base_index = np.ones(len(b), dtype='int')
            self.base_index = np.negative(self.base_index)
            self.enter_index = -1
            self.leave_index = -1
            self.backup_c = copy.deepcopy(c)
            self.ratio = []

    def __init__(self, problem: ipProblem):
        super().__init__(problem)
        self.problem = self.Problem(problem.c, problem.a, problem.b)

    # 寻找单位阵作为初始基
    def search_ini_base(self):
        base_index = np.ones(len(self.problem.b), dtype='int')
        base_index = np.negative(base_index)
        aT = self.problem.a.T
        for i in range(len(self.problem.b)):
            ini = np.zeros(len(self.problem.b))
            ini[i] = 1
            for j in range(len(aT)):
                if np.all(aT[j] == ini):
                    base_index[i] = j

        self.problem.base_index = base_index
        return np.all(base_index >= 0)

    # 使用大M法寻找初始基，步骤参考了《运筹学》第四版教材（清华大学出版社）
    def bigM(self, **parameters):
        M = ((max(abs(self.problem.c)) + 1)**2) * 10 + 10

        for i in range(len(self.problem.base_index)):
            if self.problem.base_index[i] < 0:
                self.problem.c = np.insert(self.problem.c, len(self.problem.c),
                                           np.array([-M]))

                semi_p = np.zeros(len(self.problem.b))
                semi_p[i] = 1
                self.problem.a = np.c_[self.problem.a, semi_p]

                self.problem.base_index[i] = len(self.problem.c) - 1

    def solve(self, **parameters) -> ipSolution:
        if not self.search_ini_base():  # 没有找到单位阵作初始基，用大M法
            self.bigM(**parameters)

        pb = copy.deepcopy(self.problem)

        return simplex_method_solve(pb)


# 对给定了初始基的标准型使用进行单纯形法进行求解
def simplex_method_solve(p: simplexMethod.Problem) -> ipSolution:
    # 初始单纯形表的检验数计算
    for i in range(len(p.base_index)):
        p.c -= p.a[i] * p.c[p.base_index[i]]

    p.enter_index = np.argmax(p.c)  # 确定入基变量
    while p.c[p.enter_index] > 0:
        p.ratio = []
        for i in range(len(p.b)):
            if p.a[i][p.enter_index] > 0:
                p.ratio.append(p.b[i] / p.a[i][p.enter_index])
            else:
                p.ratio.append(float("inf"))

        p.leave_index = np.argmin(np.array(p.ratio))  # 确定出基变量

        if p.ratio[p.leave_index] == float("inf"):  # 出基变量=无穷大
            return ipSolution(False, "无界解", [None], None)

        transform_pivot(p)

        p.enter_index = np.argmax(p.c)  # 下一个入基变量

    # 分析解的状态
    variables = np.zeros(len(p.c))
    variables[p.base_index] = p.b

    varia_ini = variables[0:len(p.backup_c)]
    varia_add = variables[len(p.backup_c):]  # 人工变量

    if np.any(varia_add != 0):
        return ipSolution(False, "无可行解", None, None)

    res = np.dot(varia_ini, p.backup_c)

    for i in range(len(p.c)):
        if abs(p.c[i]) < 1e-8 and i not in p.base_index:  # 非基变量检验数为0
            return ipSolution(True, "无有限最优解", varia_ini, res)

    return ipSolution(True, "存在有限最优解", varia_ini, res)


# 对给定原问题进行基变换
def transform_pivot(p: simplexMethod.Problem) -> None:
    domi_num = p.a[p.leave_index][p.enter_index]

    p.a[p.leave_index] /= domi_num
    p.b[p.leave_index] /= domi_num

    p.base_index[p.leave_index] = p.enter_index

    for i in range(len(p.b)):
        if p.a[i][p.enter_index] != 0 and i != p.leave_index:
            p.b[i] -= p.a[i][p.enter_index] * p.b[p.leave_index]
            p.a[i] -= p.a[i][p.enter_index] * p.a[p.leave_index]

    p.c -= p.c[p.enter_index] * p.a[p.leave_index]


# 对偶单纯形法
class dualSimplex(ipSolve):
    # 对偶单纯形法内部的整数规划问题表示
    class Problem(ipProblem):
        def __init__(self, c, a, b):
            super().__init__(c, a, b)
            self.base_index = np.ones(len(b), dtype='int')
            self.base_index = np.negative(self.base_index)
            self.backup_c = copy.deepcopy(c)

    def __init__(self, problem: ipProblem):
        super().__init__(problem)
        self.problem = self.Problem(problem.c, problem.a, problem.b)

    # 找单位阵作初始基
    def search_ini_base(self):
        base_index = np.ones(len(self.problem.b), dtype='int')
        base_index = np.negative(base_index)
        aT = self.problem.a.T
        for i in range(len(self.problem.b)):
            ini = np.zeros(len(self.problem.b))
            ini[i] = 1
            for j in range(len(aT)):
                if np.all(aT[j] == ini):
                    base_index[i] = j

        self.problem.base_index = base_index
        return np.all(base_index >= 0)

    # 使用大M法寻找初始基，步骤参考了《运筹学》第四版教材（清华大学出版社）
    def bigM(self, **parameters):
        M = ((max(abs(self.problem.c)) + 1)**2) * 10 + 10

        for i in range(len(self.problem.base_index)):
            if self.problem.base_index[i] < 0:
                self.problem.c = np.insert(self.problem.c, len(self.problem.c),
                                           np.array([-M]))

                semi_p = np.zeros(len(self.problem.b))
                semi_p[i] = 1
                self.problem.a = np.c_[self.problem.a, semi_p]

                self.problem.base_index[i] = len(self.problem.c) - 1

    def solve(self, **parameters) -> ipSolution:
        if not self.search_ini_base():  # 没有找到单位阵作初始基，用人工变量法（大M法 / 两阶段法）
            self.bigM(**parameters)
        self.problem.base = np.identity(len(self.problem.b))
        self.problem.base_inv = np.identity(len(self.problem.b))

        pb = copy.deepcopy(self.problem)
        return dual_method_solve(pb)


# 对给定了初始基的标准型使用对偶单纯形法进行求解
def dual_method_solve(p: dualSimplex.Problem) -> ipSolution:
    base_index = p.base_index
    not_base_index = get_not_base_index(p, base_index)

    base_inv = np.linalg.inv(p.a.T[base_index].T)

    # 初始的检验数计算
    sigma = p.c - np.dot(np.dot(p.c[base_index], base_inv), p.a)

    if np.any(sigma > 0):
        return ipSolution(False, "对偶问题无可行解，无法使用对偶单纯形法。", None, None)

    variables = np.dot(base_inv, p.b)

    while np.any(variables < 0):
        # 确定换出变量
        leave_base = np.argmin(variables)
        leave_index = base_index[leave_base]

        if np.all(p.a[leave_base] >= 0):
            return ipSolution(False, "对偶问题无有限最优解，原问题无可行解", None, None)

        # 确定换出变量
        ratio_out = search_ratio_out(sigma, p, not_base_index, leave_base,
                                     base_inv)

        eb = np.argmin(ratio_out)
        enter_index = not_base_index[eb]

        # 基变换
        base_index[leave_base] = enter_index
        not_base_index[eb] = leave_index

        base_inv = np.linalg.inv(p.a.T[base_index].T)
        variables = np.dot(base_inv, p.b)

        # 下一个检验数
        sigma = p.c - np.dot(np.dot(p.c[base_index], base_inv), p.a)

    # 迭代结束，计算解的值
    xb = np.dot(base_inv, p.b)
    variables = np.zeros(len(p.c))
    variables[base_index] = xb

    # 分析解的状态
    return analyse_solve(p, variables, sigma, not_base_index)


# 非基变量检验数计算
def search_ratio_out(sigma, p, not_base_index, leave_base, base_inv):
    ratio = np.ones(len(p.c[not_base_index])) * np.inf
    n = np.dot(base_inv, p.a.T[not_base_index].T)  # 基变换后的非基矩阵
    a_leave = n[leave_base]
    for i in range(len(ratio)):
        a_leave_num = a_leave[i]
        if a_leave_num < 0:
            ratio[i] = sigma[not_base_index][i] / a_leave_num
    return ratio


# 初始化非基变量，返回下标
def get_not_base_index(p: dualSimplex.Problem, base_index):
    not_base_index = list(range(len(p.c)))
    for i in base_index:
        try:
            not_base_index.pop(not_base_index.index(i))
        except ValueError:
            pass
    return np.array(not_base_index, 'int')


# 分析解的状态
def analyse_solve(p: dualSimplex.Problem, variables, last_s,
                  last_not_base_index) -> ipSolution:
    varia_ini = variables[0:len(p.backup_c)]
    varia_add = variables[len(p.backup_c):]  # 人工变量

    if np.any(varia_add != 0):
        return ipSolution(False, "无可行解", None, None)

    res = np.dot(varia_ini, p.backup_c)

    for i in last_s[last_not_base_index]:
        if abs(i) < 1e-8:  # 非基变量检验数为0
            return ipSolution(True, "无有限最优解", varia_ini, res)

    return ipSolution(True, "存在有限最优解", varia_ini, res)
