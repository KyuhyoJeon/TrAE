import numpy as np

class EngagementDynamics(object):
    def __init__(self, missile_state, target_state, dt, desired_impact,*args):
        """
        Missile state = [xm, ym, vm_t, gamma]
        Target state = [xt, yt, vt_t, gamma]
        """
        self.g = 9.81
        self.xm_init = missile_state[0]
        self.ym_init = missile_state[1]
#         self.Vxm_init = missile_state[2]
#         self.Vym_init = missile_state[3]
        self.Vm_init = missile_state[2]
        self.gamma_init = missile_state[3]

        self.xt_init = target_state[0]
        self.yt_init = target_state[1]
#         self.Vxt_init = target_state[2]
#         self.Vyt_init = target_state[3]
        self.Vt_init = target_state[2]
        self.gammat_init = target_state[3]
        
        self.args = args
        self.dt = dt
        self.gamma_d = desired_impact

        
    def reset(self, agent_name):
        if agent_name == 'Missile':
            x_init = np.array([self.xm_init, self.ym_init, self.Vm_init, self.gamma_init])
        elif agent_name == 'Target':
            x_init = np.array([self.xt_init, self.yt_init, self.Vt_init, self.gammat_init])
            
        if self.args:
            if agent_name == 'Missile':
                x_init = np.array([self.xm_init, self.ym_init, self.Vm_init, self.gamma_init, 0])
            elif agent_name == 'Target':
                x_init = np.array([self.xt_init, self.yt_init, self.Vt_init, self.gammat_init, 0])
        self.iter = 0
        
        return x_init
        
    def plant_dyn(self, x, u):
        """
        State x = [xm, ym, Vm, gamma]
        """
        xm = x[0]
        ym = x[1]
        Vm = x[2]
        gamma = x[3]
        
        xm_dot = Vm * np.cos(gamma) 
        ym_dot = Vm * np.sin(gamma) 
        Vm_dot = - self.g * np.sin(gamma)

        if u == 0:
            gamma_dot = 0  # - g * np.cos(gamma) / Vm 
        else:
            gamma_dot = u/Vm - self.g * np.cos(gamma) / Vm   
        
        dx = np.array([xm_dot, ym_dot, Vm_dot, gamma_dot])
        
        return dx
    
    def lag_dyn(self, x, u, k):
        """
        State x = [xm, ym, Vm, gamma, u]
        """
        u_com = u

        xm = x[0]
        ym = x[1]
        Vm = x[2] 
        gamma = x[3]
        u = x[4]
        
        xm_dot = Vm * np.cos(gamma) 
        ym_dot = Vm * np.sin(gamma) 
        Vm_dot = - self.g * np.sin(gamma)
        u_dot = - k * u + k * u_com
        
        if u == 0:
            gamma_dot = 0 #   - self.g * np.cos(gamma) / Vm    
        else:
            gamma_dot = u / Vm - self.g * np.cos(gamma) / Vm 
        
        dx = np.array([xm_dot, ym_dot, Vm_dot, gamma_dot, u_dot])
        
        return dx
    
    def RK4(self, x, u):
        dt = self.dt
        if self.args:
            k = self.args[0]
            lag = self.args[1]
          
        #if lag == True:
            k1 = self.lag_dyn(x, u, k)
            k2 = self.lag_dyn(x + k1/2,u, k)
            k3 = self.lag_dyn(x +k2/2, u, k)
            k4 = self.lag_dyn(x +k3, u, k)
        else:
            k1 = self.plant_dyn(x, u)
            k2 = self.plant_dyn(x + k1/2,u)
            k3 = self.plant_dyn(x +k2/2, u)
            k4 = self.plant_dyn(x +k3, u)
        
        x_new = x +dt/6*(k1 + 2*k2 + 2*k3 + k4)

        self.iter += 1

        return x_new
    
    
    def Biased_PNG(self, N, xm, xt, lamda_dot):

        
        if self.iter == 0:
            self.b_init = (N*self.lamda_init - self.gamma_init -(N-1)*self.gamma_d)/(self.R_init/self.Vm_init)
            b = self.b_init
        else:
            b = self.b_init*Vm/self.Vm_init
        
        ac = Vm *(N * lamda_dot + b)

        
        return ac

    
    def estimate_LOSR(self,xm, xt):
        # Relative range
        rx = xt[0] - xm[0]
        ry = xt[1] - xm[1]
        R = np.sqrt(rx**2 + ry**2)
        Rr  = np.array([rx, ry, 0])
        
        # Relative Velocity
        Vrx = xt[2]*np.cos(xt[3]) - xm[2]*np.cos(xm[3])
        Vry = xt[2]*np.sin(xt[3]) - xm[2]*np.sin(xm[3])
        Vr = np.array([Vrx, Vry, 0])
        
        #Closing Velocity
        Vc = -np.dot([Vrx, Vry],[rx, ry])/R
        
        # LOS angle, LOS rate
        lamda = np.arctan2(ry,rx)*180/np.pi #angle 단위
        lamda_dot = np.cross(Rr, Vr)/ R**2 
        
        return lamda_dot[2], R, Vc
        
    
    def PNG(self, N, Vc, lamda_dot):
        
        ac = Vc * N *lamda_dot

        ne = 3
        if ac< -ne*self.g:
            ac = -ne*self.g
        elif ac > ne*self.g:
            ac = ne*self.g
  
            
        return ac
