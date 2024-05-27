import math
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
import plot_util

def FIT_Func(X_mat, P_mat, T_Value):
    pass

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Vehicle():
    pass

class USV(Vehicle):
    def __init__(self, x_i, y_i, Ru_i, type):
        super().__init__()
        self.x_i = x_i
        self.y_i = y_i
        self.Ru_i = Ru_i
        self.type = type
    
    def Pos_i_update():
        pass

class ROI():
    def __init__(self) :
        pass

class Particle:
    def __init__(self, N, M):
        self.position_i = np.random.rand(N, M) # 随机生成一个NxM矩阵
        self.velocity_i = np.random.rand(N, M) # 随机生成一个NxM矩阵

        self.pos_best_i = np.zeros((N, M)) # 初始化一个NxM的零矩阵
        self.err_best_i = -1
        self.err_i = -1        

        # 初始化
        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # 自我更新
    def evaluate(self):
        self.err_i=FIT_Func(self.position_i)

        if self.err_i < self.err_best_i or self.err_best_i==-1:
            #更新个体最优位置
            self.pos_best_i=self.position_i
            #更新个体最优值
            self.err_best_i=self.err_i

    def update_velocity(self, pos_best_g):
        w=0       
        c1=1        
        c2=2        

        for i in range(0, num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    def update_position(self, bounds, r=0.2):
        
        # 将矩阵A中每个元素按照sigmoid函数归一化到0到1之间，得到归一化后的矩阵A'
        A_prime = sigmoid(self.velocity_i)
    
        # 归一化后的矩阵A'中，大于r的位置置1，小于1的位置置0，得到B矩阵
        self.position_i = (A_prime > r).astype(int)

class PSO():
    def __init__(self, x0, bounds, num_particles, maxiter):
        global num_dimensions

        num_dimensions=len(x0)
        err_best_g=-1                   
        pos_best_g=[]                   

        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        i=0
        while i < maxiter:
            for j in range(0,num_particles):
                swarm[j].evaluate()

                # 全局更新
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        print('FINAL:')
        print(pos_best_g)
        print(err_best_g)

if __name__ == "__PSO__":
    initial=[5,5]               
    bounds=[(-10,10),(-10,10)]  
    PSO(initial,bounds,num_particles=15,maxiter=30)