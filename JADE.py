from math import sin, e

import numpy as np
from numpy import cos, pi


class JADE():
    '''
    JADE算法：根据论文JADE: Adaptive Differential Evolution withOptional External Archive复现
    特点：   1.采用自适应策略，自适应缩放因子F和自适应交叉概率CR，将成功的CR和F存储进SCR和SF中，然后根据该集合的均值更新uCR和uF，SCR和SF每代都会重置更新
            2.采用外部存档，存储每次迭代中的遗弃个体
            3.采用变异策略DE/current-to-pbest/1
    输入：   fitness:适应度函数
            constraints:约束条件
            lowwer:下界
            upper:上界
            pop_size:种群大小
            dim:维度
            mut_way:变异方式
            epochs:迭代次数
    输出：   best:最优个体

    本程序在原JADE算法上增加约束处理策略，使其能使用于约束优化问题
    '''

    def __init__(self, fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs):
        self.fitness = fitness  # 适应度函数
        self.constraints = constraints  # 约束条件
        self.lowbound = lowwer  # 下界
        self.upbound = upper  # 上界
        self.pop_size = pop_size  # 种群大小
        self.dim = dim  # 种群大小
        self.population = np.random.rand(self.pop_size, self.dim)  # 种群
        self.best = self.population[0]  # 最优个体
        self.fit=np.random.rand(self.pop_size)#适应度
        self.conv=np.random.rand(self.pop_size)#约束
        self.mut_way = mut_way#变异策略
        self.Archive = []  # 存档
        #*********此部分参数为CR和F相关参数*********
        self.uCR = 0.5  # CR的期望
        self.uF = 0.5  # F的期望
        self.F = self.uF + np.sqrt(0.1) * np.random.standard_cauchy(self.pop_size)  # 缩放因子
        self.CR = np.random.normal(self.uCR, 0.1, self.pop_size)  # 交叉概率
        self.c = 0.02  # 系数
        self.SCR = []  # 存放成功的CR
        self.SF = []  # 存放成功的F
        #***************************************
        self.Epochs = epochs  # 迭代次数
        self.NFE = 0  # 记录函数调用次数
        self.max_NFE = 10000*self.dim  # 最大函数调用次数

    #初始化种群
    def initpop(self):
        self.population = self.lowbound + (self.upbound - self.lowbound) * np.random.rand(self.pop_size,
                                                                                          self.dim)  # 种群初始化
        self.fit = np.array([self.fitness(chrom) for chrom in self.population])  # 适应度初始化,此处复杂度为pop_size
        self.conv = np.array([self.constraints(chrom) for chrom in self.population])  # 约束初始化
        self.NFE += self.pop_size  # 记录函数调用次数
        self.best = self.population[np.argmin(self.fit)]  # 最优个体初始化
    #变异操作
    def mut(self):
        mut_population = []  # 定义新种群
        if self.mut_way == 'DE/current-to-pbest/1':
            for i in range(self.pop_size):
                p = 0.05
                # 选择适应度值前p的个体
                idx_set = np.argsort(self.fit)[:int(self.pop_size * p)]
                xpbest_idx = np.random.choice(idx_set, 1, replace=False)
                xpbest = self.population[xpbest_idx].flatten()  # 因为argsort返回值作索引时会变成二维，所以要flatten
                a = self.population[
                    np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)].flatten()
                # b为外部存档和当前种群的并集中随机选择的个体
                idxb = np.random.choice(np.arange(self.pop_size + len(self.Archive)), 1, replace=False)[0]  # 随机选择一个下标
                if (idxb >= self.pop_size):  # 如果下标大于种群大小，则从存档中选择
                    idxb -= self.pop_size
                    b = self.Archive[idxb]
                else:  # 否则从当前种群中选择
                    b = self.population[idxb].flatten()
                v = self.population[i] + self.F[i] * (xpbest - self.population[i]) + self.F[i] * (a - b)  # 突变运算
                mut_population.append(v)
        return mut_population
    #交叉操作
    def cross(self, mut_population):
        cross_population = self.population.copy()
        #Binomial crossover
        for i in range(self.pop_size):
            j = np.random.randint(0, self.dim)  # 随机选择一个维度
            for k in range(self.dim):  # 对每个维度进行交叉
                if np.random.rand() < self.CR[i] or k == j:  # 如果随机数小于交叉率或者维度为j
                    cross_population[i][k] = mut_population[i][k]  # 交叉
                    #边界处理
                    if cross_population[i][k]>self.upbound:
                        cross_population[i][k]=(self.upbound+self.population[i][k])/2#如果超过上界，则取上界和原来的中间值
                    elif cross_population[i][k]<self.lowbound:
                        cross_population[i][k]=(self.lowbound+self.population[i][k])/2
        return cross_population
    #选择操作
    def select(self, cross_population):
        self.SF = []  # 每一代都要清空
        self.SCR = []
        for i in range(self.pop_size):#此处复杂度为pop_size
            temp=self.fitness(cross_population[i])
            temp_v=self.constraints(cross_population[i])
            self.NFE += 1  # 记录函数调用次数
            if (self.conv[i]==0 and temp_v==0 and temp<self.fit[i]) or (temp_v<self.conv[i]):#如果新个体适应度和约束都优于原来的
                self.population[i] = cross_population[i]
                self.fit[i] = temp
                self.conv[i] = temp_v
                self.SCR.append(self.CR[i])
                self.SF.append(self.F[i])
                self.Archive.append(self.population[i])
                if len(self.Archive) > self.pop_size * 2.6:
                    # 如果存档超过种群大小，则随机删除一些
                    for o in range(int(len(self.Archive) - self.pop_size * 2.6)):
                        self.Archive.pop(np.random.randint(0, len(self.Archive)))
    #迭代
    def run(self):
        pre=np.min(self.fit)
        count=0
        pre=0
        self.initpop()  # 初始化种群
        for i in range(self.Epochs):  # 迭代
            #每一代开始时，生成F和CR
            self.F = np.clip(self.uF + np.sqrt(0.1) * np.random.standard_cauchy(self.pop_size), 0.000001,1)  # 缩放因子更新
            self.CR = np.clip(np.random.normal(self.uCR, 0.1, self.pop_size), 0, 1)  # 交叉概率更新
            #变异操作，使用DE/current-to-pbest/1策略
            mut_population = self.mut()  # 变异
            #交叉操作
            cross_population = self.cross(mut_population)  # 交叉
            #选择操作，并更新历史归档A和SCR，SF
            self.select(cross_population)  # 选择
            self.best = self.population[np.argmin(self.fit)]  # 更新最优个体
            #每一代结束时更新uCR和uF
            self.uCR = (1 - self.c) * self.uCR + self.c * np.mean(self.SCR)  # 更新平均交叉概率
            self.uF = (1 - self.c) * self.uF + self.c * np.sum(np.array(self.SF) ** 2) / np.sum(self.SF)  # 更新平均缩放因子
            # 打印每次迭代的种群最优值，均值，最差值，方差
            print('epoch:', i, 'best:', np.min(self.fit),
                  'mean:',np.mean(self.fit),
                  'worst:',np.max(self.fit),
                  'std:',np.std(self.fit),
                  'NFE:',self.NFE,
                  'conv:',self.constraints(self.best))
            # if self.NFE > self.max_NFE:
            #     break
            if count>100:
                break
            if pre-(np.min(self.fit)+self.constraints(self.best))<1e-6:
                count+=1
            else:
                count=0
            pre=np.min(self.fit)+self.constraints(self.best)
        return self.best  # 返回最优个体


if __name__ == '__main__':
    def fitness(x):
        y = 0
        for i in range(len(x)):
            y = y + x[i] ** 2 - 10 * cos(2 * pi * x[i]) + 10

        return y


    def constraints(x):
        g1 = 0
        g2 = 0
        for i in range(len(x)):
            g1 = g1 + (-x[i] * sin(2 * x[i]))
            g2 = g2 + x[i] * sin(x[i])
        g = max(g1, 0) + max(g2, 0)
        return g/2


    lowwer = -10
    upper = 10
    pop_size = 180
    dim = 10
    mut_way = 'DE/current-to-pbest/1'
    epochs = 1000
    jade = JADE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs)
    best = jade.run()
    print(best)
    print(fitness(best))
