from math import e

import numpy as np
from numpy import cos, pi, sin


class LSHADE():
    '''
    LSHADE算法：根据论文Improving_the_search_performance_of_SHADE_using_linear_population_size_reduction复现
    特点：   相比于SHADE，增加种群大小线性减少策略

    输入：   fitness:适应度函数
            constraints:约束条件
            lowwer:下界
            upper:上界
            pop_size:种群大小
            dim:维度
            mut_way:变异方式
            epochs:迭代次数
    输出：   best:最优个体
    本程序在原LSHADE算法的基础上，增加了约束处理策略，使其适用于约束优化问题
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
        self.conv=np.random.rand(self.pop_size)#约束
        self.best = self.population[0]  # 最优个体
        self.mut_way = mut_way  # 变异策略
        self.Archive = []  # 存档
        # *********此部分参数为CR和F相关参数*********
        self.uCR = 0.5  # CR的期望
        self.uF = 0.5  # F的期望
        self.F = 0.5  # 缩放因子
        self.CR = 0.5  # 交叉概率
        self.c = 0.01  # 系数
        self.SCR = []  # 存放成功的CR
        self.SF = []  # 存放成功的F
        self.success_set = []  # 存放成功的索引
        self.fail_set = []  # 存放失败的适应度值
        self.H = 6  # 历史集大小
        self.MCR = np.array([0.5] * self.H)  # 存放H个的uCR
        self.MF = np.array([0.5] * self.H)  # 存放H个的uF
        self.stopvalue = 0  # M终止值
        # ***************************************
        self.Epochs = epochs  # 迭代次数
        self.Ninit=self.pop_size
        self.Nmin=4
        self.NFE=0#函数评价次数
        self.max_NFE=10000*self.dim

    def initpop(self):
        self.population = self.lowbound + (self.upbound - self.lowbound) * np.random.rand(self.pop_size,
                                                                                          self.dim)  # 种群初始化
        self.fit = np.array([self.fitness(chrom) for chrom in self.population])  # 适应度初始化,此处复杂度为pop_size
        self.conv=np.array([self.constraints(chrom) for chrom in self.population])#约束初始化
        self.NFE+=self.pop_size
        self.best = self.population[np.argmin(self.fit)]  # 最优个体初始化
    def mut(self,i):
        mut_population = []  # 定义新种群
        if self.mut_way == 'DE/current-to-pbest/1':
            p = 0.05
            # 选择适应度值前p的个体
            idx_set=np.argsort(self.fit)[:int(self.pop_size*p)]
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
            v = self.population[i] + self.F * (xpbest - self.population[i]) + self.F * (a - b)  # 突变运算
            mut_population.append(v)
        return v

    def cross(self, mut_chorm,i):
        cross_chorm = self.population[i].copy()
        j = np.random.randint(0, self.dim)  # 随机选择一个维度
        for k in range(self.dim):  # 对每个维度进行交叉
            if np.random.rand() < self.CR or k == j:  # 如果随机数小于交叉率或者维度为j
                cross_chorm[k] = mut_chorm[k]  # 交叉
                # 边界处理
                if cross_chorm[k] > self.upbound:
                    cross_chorm[k] = (self.upbound + self.population[i][k]) / 2  # 如果超过上界
                elif cross_chorm[k] < self.lowbound:
                    cross_chorm[k] = (self.lowbound + self.population[i][k]) / 2
        return cross_chorm

    def select(self, cross_chorm,i):
        # self.SCR = []  # 每次迭代清空
        # self.SF = []
        # self.success_set = []
        # self.fail_set = []

        temp=self.fitness(cross_chorm)
        temp_v=self.constraints(cross_chorm)
        self.NFE+=1
        if (self.conv[i] == 0 and temp_v == 0 and self.fit[i] >= temp) or (self.conv[i] > temp_v):
            self.SCR.append(self.CR)
            self.SF.append(self.F)
            self.success_set.append(i)
            self.fail_set.append(self.fit[i])
            self.Archive.append(self.population[i])
            if len(self.Archive) > self.pop_size * 2.6:
                # 如果存档超过种群大小，则随机删除一些
                for o in range(int(len(self.Archive) - self.pop_size * 2.6)):
                    self.Archive.pop(np.random.randint(0, len(self.Archive)))
        # if temp <= self.fit[i]:
            self.population[i]=cross_chorm
            self.fit[i]=temp
            self.conv[i]=temp_v

    def Linear_reduction(self, epoch):
        self.pop_size = int(self.Ninit-self.NFE/(20000*self.dim)*(self.Ninit-self.Nmin))
        # print(self.pop_size)
        # print(self.fit.shape)
        # print(self.population.shape)
        # 去掉种群中适应度值最大的个体
        idx = np.argsort(self.fit)[:self.pop_size]
        self.population = self.population[idx]
        self.fit=self.fit[idx]

    def run(self):
        pre=np.min(self.fit)
        count=0
        self.initpop()  # 初始化种群
        for epoch in range(self.Epochs):  # 迭代
            self.SCR = []  # 每次迭代清空
            self.SF = []
            self.success_set = []
            self.fail_set = []
            # 更新CR和F
            for j in range(self.pop_size):
                # 更新CR
                mcrj = np.random.choice(self.MCR)
                if mcrj == self.stopvalue:
                    self.CR =0
                else:
                    self.CR = np.random.normal(mcrj, 0.1)
                self.CR = np.clip(self.CR, 0, 1)  # 截断
                # 更新F
                mfj = np.random.choice(self.MF)
                self.F = mfj + np.sqrt(0.1) * np.random.standard_cauchy(1)  # 更新F
                while self.F <= 0:  # 如果F小于0则重新生成
                    self.F = mfj + np.sqrt(0.1) * np.random.standard_cauchy(1)
                if self.F > 1:  # 如果F大于1则截断
                    self.F = 1
                # 普通DE操作
                mut_chorm = self.mut(j)  # 变异
                cross_chorm = self.cross(mut_chorm,j)  # 交叉
                self.select(cross_chorm,j)  # 选择
            self.best = self.population[np.argmin(self.fit)]  # 更新最优个体
            # 计算更新MF和MCR
            if (np.any(self.SCR) and np.any(self.SF)):  # 如果成功集合不为空
                d_f = np.abs(self.fail_set -self.fit[self.success_set])  # 计算F的变化量
                w = d_f / np.sum(d_f)  # 计算权重
                self.uCR = np.sum(np.multiply(w, np.array(self.SCR) ** 2)) / np.sum(np.multiply(w, np.array(self.SCR)))  # 更新平均交叉概率
                self.uF = np.sum(np.multiply(w, np.array(self.SF) ** 2)) / np.sum(np.multiply(w, np.array(self.SF)))  # 更新平均缩放因子
                self.MCR[epoch % self.H] =self.stopvalue if (self.MF[epoch % self.H] == self.stopvalue or max(self.SCR) == 0) else self.uCR  # 更新MF中的值
                self.MF[epoch % self.H] = self.uF  # 更新MF中的值
            self.Linear_reduction(epoch)  # 线性减小种群大小
            # 打印每次迭代的种群最优值，均值，最差值，方差
            print('epoch:', epoch, 'best:', np.min(self.fit),
                  'mean:', np.mean(self.fit),
                  'worst:', np.max(self.fit),
                  'std:', np.std(self.fit),
                  'NFE:', self.NFE,
                  'pop_size',self.pop_size,
                  'conv',self.constraints(self.best))
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


    lowwer = -100
    upper = 100
    pop_size = 180
    dim = 10
    mut_way = 'DE/current-to-pbest/1'
    epochs = 1000
    lshade = LSHADE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs)
    best = lshade.run()
    print(best)
    print(fitness(best))
