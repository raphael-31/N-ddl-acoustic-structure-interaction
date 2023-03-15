import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.linalg import eig
from copy import copy
from matplotlib.widgets import Slider, Button


def normalize(v, vector = None):
    if type(vector) != np.ndarray:
        return v / np.linalg.norm(v)
    return v / np.linalg.norm(vector)

def normalize_family(vecs):
    new_vecs = np.zeros_like(vecs)
    n_ligns, n_cols = np.shape(vecs)
    for col in range(n_cols):
        vec = vecs[:, col]
        new_vecs[:,col] = normalize(vec)
    return new_vecs

def eig_solving(A, B, nb_values = None, born_inf = 0, sorting = True):
    """
    A.V = w B.V
    """
    w, V = eig(A, B)
    # print("eig solving")
    # print(w)
    if nb_values == None:
        n_ligns, = np.shape(w)
        nb_values = n_ligns
    idx, = np.where(w.real >= born_inf)
    w = w[idx]
    V = V[:,idx]
    if sorting:
        idx = np.abs(w.real).argsort()
        w = w[idx][:nb_values]
        V = V[:,idx][:,:nb_values]
    else:
        w = w[:nb_values]
        V = V[:,:nb_values]
    return w, V

@dataclass
class struct:
    k:float
    e:float
    rho:float
    inter:float

@dataclass
class acc:
    k:float
    cel:float
    rho:float
    H:float

def average(bounds):
    return bounds[0] / 2 + bounds[1] / 2

class figs_2:
    def __init__(self, N_points:int, rho_acc_bounds:tuple,
                cel_bounds:tuple, 
                 H_bounds:tuple, 
                 gamma_1_bounds:tuple,
                 ks_1_bounds:tuple, 
                 ka:float, 
                 rho_1:float):
        self.N_points = N_points
        self.ka = ka
        self.rho_1 = rho_1
        self.fig_1, self.axs_1 = plt.subplots(1, 1)
        self.fig_2, self.axs_2 = plt.subplots(1, 2)

        self.fig_slide, self.axs_3 = plt.subplots(4, 1)
        self.rho_slide = self.create_slider("rho acc", self.axs_3[0],
                                       rho_acc_bounds)
        self.gamma_1_slide = self.create_slider("intercorr 1", self.axs_3[1],
                                        gamma_1_bounds)
        

        self.H_slide = self.create_slider("Height", self.axs_3[2],
                                     H_bounds)
        self.ks_1_slide = self.create_slider("ks 1", self.axs_3[3],
                                     ks_1_bounds)
   
        
        self.cel_x = np.linspace(cel_bounds[0], cel_bounds[1], N_points)
        self.V_ligns_dic = {}

        
        self.init_plot()
        def update(val):
            self.update_plot()
            self.fig_1.canvas.draw_idle()
            self.fig_2.canvas.draw_idle()
        
        self.gamma_1_slide.on_changed(update)
        self.rho_slide.on_changed(update)
        self.H_slide.on_changed(update)
        self.ks_1_slide.on_changed(update)

        plt.show()


    def solve(self):
        gamma_1,  rho, H, ks_1 = self.gamma_1_slide.val, \
                              self.rho_slide.val, self.H_slide.val, self.ks_1_slide.val
        st_1 = struct(ks_1, 1, self.rho_1, gamma_1)
        cs = self.cel_x
        om = np.zeros((self.N_points, 2), dtype=float)
        V_vec = np.zeros((self.N_points, 2, 2), dtype=float)
        for i, c in enumerate(cs):
            ac = acc(self.ka, c, rho, H)
            w, V = solve_eigen_prob_2(st_1, ac)
            om[i, :] = np.real(w)
            V_vec[i, :,:] = np.abs(V)
        return om, V_vec
    
    def init_plot_V(self, Vs, om_idx:int):
        self.V_ligns_dic[0, om_idx], = self.axs_2[om_idx].plot(self.cel_x,
                                             Vs[:, 0, om_idx],
                                             label = "struct")

        self.V_ligns_dic[1, om_idx], = self.axs_2[om_idx].plot(self.cel_x,
                                             Vs[:, 1, om_idx],
                                             label = "acc")
        
        self.axs_2[om_idx].set_xlabel('célérité')
        if om_idx==0:
            self.axs_2[om_idx].set_ylabel('énergie')
        
        
        self.axs_2[om_idx].legend()

    def init_plot(self):
        om, V_vec = self.solve()
        self.line_om_1, = self.axs_1.plot(self.cel_x,
                                             om[:, 0],
                                             label = "om 1")
        self.line_om_2, = self.axs_1.plot(self.cel_x,
                                             om[:, 1],
                                             label = "om 2")

        ones = np.ones_like(self.cel_x)
        self.line_om_s_1, = self.axs_1.plot(self.cel_x,
                                             self.ks_1_slide.val ** 2 * ones, "--",
                                             label = "om str 1")

        
        self.axs_1.plot(self.cel_x,
                                        self.cel_x * self.ka, "--",
                                             label = "nat acc")

        self.axs_1.set_xlabel('célérité')
        self.axs_1.set_ylabel('omega')
        
        self.axs_1.legend()

        for i in range(2):
            self.init_plot_V(V_vec, i)

    def update_plot_V(self, Vs, om_idx:int):
        self.V_ligns_dic[0, om_idx].set_ydata(Vs[:, 0, om_idx])
        self.V_ligns_dic[1, om_idx].set_ydata(Vs[:, 1, om_idx])

        

    def update_plot(self):
        om, V_vec = self.solve()
        self.line_om_1.set_ydata(om[:, 0])
        self.line_om_2.set_ydata(om[:, 1])
        ones = np.ones_like(self.cel_x)
        self.line_om_s_1.set_ydata(self.ks_1_slide.val ** 2 * ones)
        for i in range(2):
            self.update_plot_V(V_vec, i)
        

    def create_slider(self, label, ax, bounds):
        x1, x2 = bounds
        # Make a horizontal slider to control the slider.
        return Slider(
            ax=ax,
            label=label,
            valmin=x1,
            valmax=x2,
            valinit=x1 / 2 + x2 / 2,
        )



class figs_3:
    def __init__(self, N_points:int, rho_acc_bounds:tuple,
                cel_bounds:tuple, 
                 H_bounds:tuple, 
                 gamma_1_bounds:tuple,
                 gamma_2_bounds:tuple,
                 ks_1_bounds:tuple, ks_2_bounds:tuple, 
                 ka:float, 
                 rho_1:float,
                 rho_2:float):
        self.N_points = N_points
        self.ka = ka
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.fig_1, self.axs_1 = plt.subplots(1, 1)
        self.fig_2, self.axs_2 = plt.subplots(1, 3)

        self.fig_slide, self.axs_3 = plt.subplots(6, 1)
        self.rho_slide = self.create_slider("rho acc", self.axs_3[0],
                                       rho_acc_bounds)
        self.gamma_1_slide = self.create_slider("intercorr 1", self.axs_3[1],
                                        gamma_1_bounds)
        
        self.gamma_2_slide = self.create_slider("intercorr 2", self.axs_3[2],
                                        gamma_2_bounds)
        self.H_slide = self.create_slider("Height", self.axs_3[3],
                                     H_bounds)
        self.ks_1_slide = self.create_slider("ks 1", self.axs_3[4],
                                     ks_1_bounds)
        self.ks_2_slide = self.create_slider("ks 2", self.axs_3[5],
                                     ks_2_bounds)
        
        self.cel_x = np.linspace(cel_bounds[0], cel_bounds[1], N_points)
        self.V_ligns_dic = {}

        
        self.init_plot()
        def update(val):
            self.update_plot()
            self.fig_1.canvas.draw_idle()
            self.fig_2.canvas.draw_idle()
        
        self.gamma_1_slide.on_changed(update)
        self.gamma_2_slide.on_changed(update)
        self.rho_slide.on_changed(update)
        self.H_slide.on_changed(update)
        self.ks_1_slide.on_changed(update)
        self.ks_2_slide.on_changed(update)

        plt.show()


    def solve(self):
        gamma_1, gamma_2, rho, H, ks_1, ks_2 = self.gamma_1_slide.val, self.gamma_2_slide.val, \
                              self.rho_slide.val, self.H_slide.val, self.ks_1_slide.val,\
                              self.ks_2_slide.val
        st_1 = struct(ks_1, 1, self.rho_1, gamma_1)
        st_2 = struct(ks_2, 1, self.rho_2, gamma_2)
        cs = self.cel_x
        om = np.zeros((self.N_points, 3), dtype=float)
        V_vec = np.zeros((self.N_points, 3, 3), dtype=float)
        for i, c in enumerate(cs):
            ac = acc(self.ka, c, rho, H)
            w, V = solve_eigen_prob(st_1, st_2, ac)
            om[i, :] = np.real(w)
            V_vec[i, :,:] = np.abs(V)
        return om, V_vec
    
    def init_plot_V(self, Vs, om_idx:int):
        self.V_ligns_dic[0, om_idx], = self.axs_2[om_idx].plot(self.cel_x,
                                             Vs[:, 0, om_idx],
                                             label = "struct 1")
        self.V_ligns_dic[1, om_idx], = self.axs_2[om_idx].plot(self.cel_x,
                                             Vs[:, 1, om_idx],
                                             label = "struct 2")
        self.V_ligns_dic[2, om_idx], = self.axs_2[om_idx].plot(self.cel_x,
                                             Vs[:, 2, om_idx],
                                             label = "acc")
        
        self.axs_2[om_idx].set_xlabel('célérité')
        if om_idx==0:
            self.axs_2[om_idx].set_ylabel('énergie')
        
        
        self.axs_2[om_idx].legend()

    def init_plot(self):
        om, V_vec = self.solve()
        self.line_om_1, = self.axs_1.plot(self.cel_x,
                                             om[:, 0],
                                             label = "om 1")
        self.line_om_2, = self.axs_1.plot(self.cel_x,
                                             om[:, 1],
                                             label = "om 2")
        self.line_om_3, = self.axs_1.plot(self.cel_x,
                                             om[:, 2],
                                             label = "om 3")
        ones = np.ones_like(self.cel_x)
        self.line_om_s_1, = self.axs_1.plot(self.cel_x,
                                             self.ks_1_slide.val ** 2 * ones, "--",
                                             label = "om str 1")
        self.line_om_s_2, = self.axs_1.plot(self.cel_x,
                                             self.ks_2_slide.val ** 2 * ones, "--",
                                             label = "om str 2")
        
        self.axs_1.plot(self.cel_x,
                                        self.cel_x * self.ka, "--",
                                             label = "nat acc")

        self.axs_1.set_xlabel('célérité')
        self.axs_1.set_ylabel('omega')
        
        self.axs_1.legend()

        for i in range(3):
            self.init_plot_V(V_vec, i)

    def update_plot_V(self, Vs, om_idx:int):
        self.V_ligns_dic[0, om_idx].set_ydata(Vs[:, 0, om_idx])
        self.V_ligns_dic[1, om_idx].set_ydata(Vs[:, 1, om_idx])
        self.V_ligns_dic[2, om_idx].set_ydata(Vs[:,2, om_idx])
        

    def update_plot(self):
        om, V_vec = self.solve()
        self.line_om_1.set_ydata(om[:, 0])
        self.line_om_2.set_ydata(om[:, 1])
        self.line_om_3.set_ydata(om[:, 2])
        ones = np.ones_like(self.cel_x)
        self.line_om_s_1.set_ydata(self.ks_1_slide.val ** 2 * ones)
        self.line_om_s_2.set_ydata(self.ks_2_slide.val ** 2 * ones)
        for i in range(3):
            self.update_plot_V(V_vec, i)
        

    def create_slider(self, label, ax, bounds):
        x1, x2 = bounds
        # Make a horizontal slider to control the slider.
        return Slider(
            ax=ax,
            label=label,
            valmin=x1,
            valmax=x2,
            valinit=x1 / 2 + x2 / 2,
        )
        



    
    




def m_ratio(st:struct, ac:acc):
    return ac.rho / st.rho


def make_matrixs_2(st_1:struct, acc:acc):
    M = np.identity(2)
    L = np.array(
        [
        [0, st_1.inter * m_ratio(st_1, acc) / st_1.e],
        [-st_1.inter * acc.cel ** 2 / acc.H,  0]
        ]
    )
    K = np.array(
        [
        [st_1.k ** 4, 0],
        [ 0, acc.k ** 2 * acc.cel ** 2]
        ]
    )
    return M, L, K


def solve_eigen_prob_2(st_1:struct, acc:acc):
    M, L, K = make_matrixs_2(st_1, acc)
    K_en = copy(K)
    K_en[1, 1] = K[1,1] / acc.cel ** 2
    M_en = copy(M)
    M_en[1, 1] = M[1,1] / acc.cel ** 2
    
    A, B = order_2_to_order_1(K, L, M)
    w, V = eig_solving(A, -1j * B)
    V = K_en @ V[:2, :] + np.diag(w) ** 2 @ M_en @ V[:2, :]
    return w, normalize_family(V)



def make_matrixs(st_1:struct, st_2:struct, acc:acc):
    M = np.identity(3)
    L = np.array(
        [
        [0, 0, st_1.inter * m_ratio(st_1, acc) / st_1.e],
        [0, 0, st_2.inter * m_ratio(st_2, acc) / st_2.e],
        [-st_1.inter * acc.cel ** 2 / acc.H, -st_2.inter * acc.cel ** 2 / acc.H, 0]
        ]
    )
    K = np.array(
        [
        [st_1.k ** 4, 0, 0],
        [0, st_2.k ** 4, 0],
        [0, 0, acc.k ** 2 * acc.cel ** 2]
        ]
    )
    return M, L, K


def order_2_to_order_1(K, C, M):
    """
    M Xpp + C Xp + K X = 0
    equ. to :
    A Yp + B Y = 0
    """
    zero = np.zeros_like(K)
    A = np.block([
        [zero, - K],
        [K, C]
    ])
    B = np.block([
        [K, zero],
        [zero, M]
    ])
    return A, B

def solve_eigen_prob(st_1:struct, st_2:struct, acc:acc):
    M, L, K = make_matrixs(st_1, st_2, acc)
    K_en = copy(K)
    K_en[2, 2] = K[2,2] / acc.cel ** 2
    M_en = copy(M)
    M_en[2, 2] = M[2,2] / acc.cel ** 2
    
    A, B = order_2_to_order_1(K, L, M)
    w, V = eig_solving(A, -1j * B)
    V = K_en @ V[:3, :] + np.diag(w) ** 2 @ M_en @ V[:3, :]
    return w, normalize_family(V)




if __name__ == "__main__":
    figs_3(100, (0.0001, 10), (0.01, 10), (0.01, 10), (0.001, 1),
         (0.001, 1), (1, 5), (1, 8), 1, 1, 1)
    # figs_2(100, (0.0001, 10), (0.01, 10), (0.01, 10), (0.001, 1),
    #       (1, 5),  1, 1)
    

