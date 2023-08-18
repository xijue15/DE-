from math import sin, e

from numpy import cos, pi

import numpy as np


class SaDE():
    '''
    SaDE算法：根据论文Self-adaptive Differential Evolution Algorithm for Numerical Optimization复现
    特点：   1.采用两种变异策略，DE/rand/1和DE/current-to-pbest/1
            2.根据两种变异策略的成功次数来选择当前个体使用哪种变异策略
            3.采用自适应策略，自适应交叉概率CR，将成功的CR存储进CRm，然后根据该集合的均值更新uCR，CRm定期重置更新
    输入：   fitness:适应度函数
            constraints:约束条件
            lowwer:下界
            upper:上界
            pop_size:种群大小
            dim:维度
            mut_way:变异方式
            epochs:迭代次数
    输出：   best:最优个体
    本程序在原SaDE算法上增加约束处理策略，使其能使用于约束优化问题
    '''

    def __init__(self, fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs):
        self.fitness = fitness  # 适应度函数
        self.constraints = constraints  # 约束条件
        self.lowbound = lowwer  # 下界
        self.upbound = upper  # 上界
        self.pop_size = pop_size  # 种群大小
        self.dim = dim  # 种群大小
        self.population = np.random.rand(self.pop_size, self.dim)  # 种群
        self.fit = np.random.rand(self.pop_size)  # 适应度
        self.conv=np.random.rand(self.pop_size)
        self.mut_way = [mut_way]*self.pop_size
        self.best = self.population[0]  # 最优个体
        self.F = np.clip(np.random.normal(0.5, 0.3), 0, 2)  # 缩放因子,正态分布生成0-2内的正态分布数，服从（0.5,0.3）
        self.CRm = 0.5  # CR均值
        self.CR = np.random.normal(self.CRm, 0.1, self.pop_size)  # 针对每个个体，正态分布生成交叉概率
        self.CR_set = []  # 成功的CR集合
        self.Epochs = epochs  # 迭代次数
        self.p1 = 0.05  # p1,变异策略1被用到的概率
        self.p2 = 0.05  # p2,变异策略2被用到的概率
        self.ns1 = 0  # 变异策略1被用到的次数
        self.ns2 = 0  # 变异策略2被用到的次数
        self.nf1 = 0  # 变异策略1被丢弃的次数
        self.nf2 = 0  # 变异策略2被丢弃的次数
        self.NFE=0#函数评价次数
        self.max_NFE=10000*self.dim #最大函数评价次数

    def initpop(self):
        self.population = self.lowbound + (self.upbound - self.lowbound) * np.random.rand(self.pop_size,
                                                                                          self.dim)  # 种群初始化
        self.fit = np.array([self.fitness(chrom) for chrom in self.population])  # 适应度初始化,此处复杂度为pop_size
        self.conv=np.array([self.constraints(chrom) for chrom in self.population])
        self.NFE+=self.pop_size
        self.best = self.population[np.argmin(self.fit)]  # 最优个体初始化
    def mut(self):
        mut_population = []  # 定义新种群
        # 随机生成0-1范围内的pop_size个数
        p = np.random.rand(self.pop_size)
        for i in range(self.pop_size):
            if p[i] <= self.p1:
                self.mut_way[i] = 'DE/rand/1'
            else:
                self.mut_way[i] = 'DE/current-to-best/1'
            if self.mut_way[i] == 'DE/rand/1':
                # 随机选择三个不同于当前个体的个体
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = self.population[idxs]
                # print(idxs)
                v = a + self.F * (b - c)  # 突变运算
                mut_population.append(v)
            elif self.mut_way[i] == 'DE/current-to-best/1':
                # 随机选择三个不同于当前个体的个体
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 2, replace=False)
                a, b = self.population[idxs]
                v = self.population[i] + self.F * (self.best - self.population[i]) + self.F * (a - b)
                mut_population.append(v)
        return mut_population

    def cross(self, mut_population):
        cross_population = self.population.copy()
        for i in range(self.pop_size):
            j = np.random.randint(0, self.dim)  # 随机选择一个维度
            for k in range(self.dim):  # 对每个维度进行交叉
                if np.random.rand() < self.CR[i] or k == j:  # 如果随机数小于交叉率或者维度为j
                    cross_population[i][k] = mut_population[i][k]  # 交叉
        return cross_population

    def select(self, cross_population):
        for i in range(self.pop_size):  # 此处复杂度为pop_size
            temp = self.fitness(cross_population[i])
            temp_v=self.constraints(cross_population[i])
            self.NFE += 1
            if (temp_v==0 and self.conv[i]==0 and self.fit[i]>temp) or (temp_v<self.conv[i]) :
                self.population[i] = cross_population[i]
                self.fit[i] = temp
                self.conv[i]=temp_v
                if self.mut_way[i] == 'DE/rand/1':  # 如果变异策略为DE/rand/1
                    self.ns1= self.ns1+ 1  # ns1加1
                    self.nf2 =self.nf2+ 1
                else:  # 如果变异策略为DE/current-to-best/1
                    self.ns2 =self.ns2+ 1  # ns2加1
                    self.nf1 =self.nf1+ 1
                self.CR_set.append(self.CR[i])  # 将成功的CR加入CR集合

    def run(self):
        pre=np.min(self.fit)
        count=0
        pre_d=0
        self.initpop()  # 初始化种群
        for i in range(1, self.Epochs+1 ):  # 迭代
            self.F = np.clip(np.random.normal(0.5, 0.3), 0, 2)  # 缩放因子更新,正态分布生成0-2内的正态分布数，服从（0.5,0.3）
            if (i % 5 == 0):  # 每5代更新一次交叉率
                self.CR = np.random.normal(self.CRm, 0.1, self.pop_size)  # 每5代交叉率更新
            if (i % 25 == 0):
                self.CRm = np.mean(self.CR_set)  # 每25代更新CRm
                self.CR_set = []  # 清空CR集合
            if (i % 50 == 0):  # 每50次迭代更新一次ns,nf,p1,p2
                if(self.ns1==0 and self.ns2==0 and self.nf1==0 and self.nf2==0):
                    self.p1=0.5
                else:
                    self.p1 = self.ns1 * (self.ns2 + self.nf2) / (
                                self.ns2 * (self.ns1 + self.nf1) + self.ns1 * (self.ns2 + self.nf2))  # 计算p1
                self.p2 = 1 - self.p1  # 计算p2
                self.ns1 = 0
                self.ns2 = 0
                self.nf1 = 0
                self.nf2 = 0
            mut_population = self.mut()  # 变异
            cross_population = self.cross(mut_population)  # 交叉
            self.select(cross_population)  # 选择
            self.best = self.population[np.argmin(self.fit)]  # 更新最优个体
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
        g1 = 0
        g2 = 0
        for i in range(len(x)):
            g1 = g1 + (-x[i] * sin(2 * x[i]))
            g2 = g2 + x[i] * sin(x[i])
        g = max(g1, 0) + max(g2, 0)
        return y + g / 2


    def constraints(x):
        return 0


    lowwer = -100
    upper = 100
    pop_size = 18*10
    dim = 10
    mut_way = 'DE/rand/1'
    epochs = 1000
    sade = SaDE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs)
    best = sade.run()
    print(best)
    print(fitness(best))
