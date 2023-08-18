import numpy as np



class DE():
    '''
    JADE算法
    输入：   fitness:适应度函数
            constraints:约束条件
            lowwer:下界
            upper:上界
            pop_size:种群大小
            dim:维度
            mut_way:变异方式
            epochs:迭代次数
    输出：   best:最优个体
    '''
    def __init__(self,fitness,constraints,lowwer,upper,pop_size,dim,mut_way,epochs):
        self.fitness=fitness#适应度函数
        self.constraints=constraints#约束条件
        self.lowbound=lowwer#下界
        self.upbound=upper#上界
        self.pop_size=pop_size#种群大小
        self.dim=dim#种群大小
        self.population=np.random.rand(self.pop_size,self.dim)#种群
        self.fit=np.random.rand(self.pop_size)#适应度
        self.mut_way=mut_way
        self.best=self.population[0]#最优个体
        self.F=0.5#缩放因子
        self.CR=0.1#交叉概率
        self.Epochs=epochs#迭代次数
        self.NFE=0#函数评价次数

    def initpop(self):
        self.population = self.lowbound + (self.upbound - self.lowbound) * np.random.rand(self.pop_size,self.dim)  # 种群初始化
        self.fit = np.array([self.fitness(chrom) for chrom in self.population])  # 适应度初始化,此处复杂度为pop_size
        self.NFE += self.pop_size
        self.best = self.population[np.argmin(self.fit)]  # 最优个体初始化
    def mut(self):
        mut_population = []#定义新种群
        if self.mut_way=='DE/rand/1':
            for i in range(self.pop_size):
                # 随机选择三个不同于当前个体的个体
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = self.population[idxs]
                # print(idxs)
                v=a+self.F*(b-c)#突变运算
                mut_population.append(v)
        elif self.mut_way=='DE/best/1':
            for i in range(self.pop_size):
                # 随机选择二个不同于当前个体的个体
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 2, replace=False)
                a, b = self.population[idxs]
                v=self.best+self.F*(a-b)#突变运算
                mut_population.append(v)
        elif self.mut_way=='DE/rand/2':
            for i in range(self.pop_size):
                # 随机选择五个不同于当前个体的个体
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 5, replace=False)
                a, b, c,d,e = self.population[idxs]
                v=a+self.F*(b-c)+self.F*(d-e)#突变运算
                mut_population.append(v)
        elif self.mut_way=='DE/best/2':
            for i in range(self.pop_size):
                # 随机选择四个不同于当前个体的个体
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 4, replace=False)
                a, b, c,d = self.population[idxs]
                v=self.best+self.F*(a-b)+self.F*(c-d)
                mut_population.append(v)
        elif self.mut_way=='DE/current-to-best/1':
            for i in range(self.pop_size):
                # 随机选择三个不同于当前个体的个体
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = self.population[idxs]
                v=self.population[i]+self.F*(self.best-self.population[i])+self.F*(a-b)
                mut_population.append(v)
        return mut_population
    def cross(self,mut_population):
        cross_population=self.population.copy()
        for i in range(self.pop_size):
            j=np.random.randint(0,self.dim)#随机选择一个维度
            for k in range(self.dim):#对每个维度进行交叉
                if np.random.rand()<self.CR or k==j:#如果随机数小于交叉率或者维度为j
                    cross_population[i][k]=mut_population[i][k]#交叉
                if cross_population[i][k]>self.upbound:
                    cross_population[i][k]=self.upbound
                if cross_population[i][k]<self.lowbound:
                    cross_population[i][k]=self.lowbound
        return cross_population
    def select(self,cross_population):
        for i in range(self.pop_size):
            temp=self.fitness(cross_population[i])
            self.NFE+=1
            if temp<=self.fit[i]:
                self.fit[i]=temp
                self.population[i]=cross_population[i]
    def run(self):
        pre=np.min(self.fit)
        count=0
        self.initpop()#初始化种群
        for i in range(self.Epochs):#迭代
            mut_population=self.mut()#变异
            cross_population=self.cross(mut_population)#交叉
            self.select(cross_population)#选择
            self.best=self.population[np.argmin(self.fit)]#更新最优个体

            # 打印每次迭代的种群最优值，均值，最差值，方差
            print('epoch:', i, 'best:', np.min(self.fit),
                  'mean:', np.mean(self.fit),
                  'worst:', np.max(self.fit),
                  'std:', np.std(self.fit))

            if count>50:#如果连续20次最优值没有变化，则停止迭代
                break
            if (abs(np.min(self.fit) - pre) < 1e-6):
                count+=1
            else:
                count=0
            pre = np.min(self.fit)

        return self.best#返回最优个体

if __name__ == '__main__':
    def fitness(x):
        return np.sum(x**2)
    def constraints(x):
        return 0
    lowwer=-10
    upper=10
    pop_size=200
    dim=2
    mut_way='DE/rand/1'
    epochs=1000
    de=DE(fitness,constraints,lowwer,upper,pop_size,dim,mut_way,epochs)
    best=de.run()
    print(best)
    print(fitness(best))


