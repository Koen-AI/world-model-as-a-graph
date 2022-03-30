import numpy as np


#this class is a scheduler for the disclipping minimum and maximum!
class Scheduler:
    def __init__(
        self,
        args,
    ):
        self.empty = True
        self.epoch = 0
        self.start_epoch = 0
        self.n = args.n_epochs        #number of epochs in the run
        self.d = -1.0*args.dist_clip  #distance clipping as starting point for scheduler
        self.f_max = args.omega_max   #factor that determines when the scheduler should allow for the agent to plan as far apart as it would like
        self.f_min = args.omega_min   #factor that determines when the scheduler should allow for the agent to plan as far apart as it would like
        self.varpi = args.varpi       #difference in final heigth between the two schedulers
        self.demo = args.video > 0 or args.plot > 0
        
        
        
        if args.env_name == 'PointMaze-v1':
            self.delta_max = 175              #approximate max distance for point maze
            self.h = self.delta_max * self.varpi
        elif args.env_name == 'AntMaze-v1':
            self.delta_max = 230              #approximate max distance for ant maze
            self.h = self.delta_max * self.varpi
        else:
            raise ValueError('Unsupported environment for scheduler: ', args.env_name)
            self.delta_max = 200              #dummy value for max steps in other envs 
            self.h = self.delta_max * self.varpi
        self.type = args.d_scheduler
        
        self.d_max = np.zeros(self.n, dtype=float)
        self.d_min = np.zeros(self.n, dtype=float)
        
        if not(self.type == "linear" or self.type == "exponential" or self.type == "logarithmic"):
            raise ValueError('Unknown scheduler type: ', self.type)
    
    def activate(self, epoch):
        if self.empty == False:
            raise ValueError('Scheduler activated twice!')
        self.empty = False
        self.start_epoch = min(max(epoch, 0), self.n - 1)
        self.epoch = min(max(epoch, self.start_epoch), self.n - 1) - self.start_epoch
        self.n -= self.start_epoch
        if self.type == "linear":
            c_max = (self.delta_max - self.d) / (self.f_max * self.n)
            c_min = (self.h - self.d) / (self.f_min * self.n)
            for x in range(self.n):
                self.d_max[x] = -1.0*(c_max * x + self.d)
                self.d_min[x] = -1.0*(c_min * x)
              
        elif self.type == "exponential":
            b_max = ((self.delta_max - self.d)**(1/(self.f_max * self.n)))
            b_min = ((self.h - self.d)**(1/(self.f_min * self.n)))
            for x in range(self.n):
                self.d_max[x] = -1.0*(b_max**x + self.d)
                self.d_min[x] = -1.0*(b_min**x)
                
        elif self.type == "logarithmic":
            epsilon = 0.000000001#tiny constant to prevent division by 0
            c_max = ((self.delta_max - self.d) / np.log(self.f_max * (self.n + epsilon)))
            c_min = ((self.h - self.d) / np.log(self.f_min * (self.n + epsilon)))
            for x in range(0, self.n):
                self.d_max[x] = -1.0*(c_max * np.log(x+1) + self.d) #+1 to avoid negative infinity
                self.d_min[x] = -1.0*(c_min * np.log(x+1)) #+1 to avoid negative infinity
         
    def set_epoch(self, epoch):
        self.epoch = min(max(epoch, self.start_epoch), self.n - 1) - self.start_epoch
  
    def get_d_max(self):
        if self.demo:
            return self.d_max[self.n - 1]
        else:
            return self.d_max[self.epoch]
        
    def get_d_min(self):
        if self.demo:
            return self.d_min[self.n - 1]
        else:
            return self.d_min[self.epoch]
        
    def plot_schedule(self):
        self.activate(0)
        import numpy as np
        import matplotlib.pyplot as plt
        X = np.arange(self.n)
        m_list = np.zeros(self.n)
        for i in range(self.n):
            m_list[i] = self.delta_max 
        col = 'navy'
        plt.plot(X, np.zeros_like(X), color='black')
        plt.plot(X, m_list, linestyle="--")
        plt.plot(X, -self.d_max, label="max scheduler", color='red')
        plt.plot(X, -self.d_min, label="min scheduler", color=col)
        plt.fill_between(X, y1=-self.d_max, y2=-self.d_min, color=col, alpha=0.5)
        plt.ylim(0, self.delta_max+30)
        plt.xlim(0, 360)
        plt.grid(b=True)
        plt.legend()
        plt.show()
