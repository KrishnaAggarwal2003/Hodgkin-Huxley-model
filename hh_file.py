import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import yaml


def load_yaml(file_name):
    with open(file_name, 'r') as file:
        data = yaml.safe_load(file)
    return data    


class Hodgkin_Huxley:
    def __init__(self, t_span, steps, time, V_r, C_m, gNa_max, gK_max, gL, E_Na, E_K, E_L, I_total):
        self.t_span = t_span
        self.steps = steps
        self.V_r = V_r
        self.C_m = C_m
        self.gNa_max = gNa_max
        self.gK_max = gK_max
        self.gL = gL
        self.E_K = E_K
        self.E_Na = E_Na
        self.E_L = E_L
        self.I_total = I_total
        self.time = time
        
    def alpha_n(self, V):
        return (0.01 * (10.0 - V)) / (np.exp(1.0 - (0.1 * V)) - 1.0)

    def beta_n(self, V):
        return 0.125 * np.exp(-V / 80.0)

    def alpha_m(self, V):
        return (0.1 * (25.0 - V)) / (np.exp(2.5 - (0.1 * V)) - 1.0)

    def beta_m(self, V):
        return 4.0 * np.exp(-V / 18.0)

    def alpha_h(self, V):
        return 0.07 * np.exp(-V / 20.0)

    def beta_h(self, V):
        return 1.0 / (np.exp(3.0 - (0.1 * V)) + 1.0)

    def derivative(self, y, t):
        V, n, m, h = y
        GNa = self.gNa_max * m**3 * h
        GK = self.gK_max * n**4
        GL = self.gL

        I_Na = GNa * (V - self.E_Na)
        I_K = GK * (V - self.E_K)
        I_L = GL * (V - self.E_L)
        I_ion = I_Na + I_K + I_L

        dVdt = (self.I_total - I_ion) / self.C_m
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        return dVdt, dndt, dmdt, dhdt

    def solver(self):
        m_init = self.alpha_m(self.V_r) / (self.alpha_m(self.V_r) + self.beta_m(self.V_r))
        n_init = self.alpha_n(self.V_r) / (self.alpha_n(self.V_r) + self.beta_n(self.V_r))
        h_init = self.alpha_h(self.V_r) / (self.alpha_h(self.V_r) + self.beta_h(self.V_r))

        initial_conditions = np.array([self.V_r, n_init, m_init, h_init])

        solution = odeint(self.derivative, initial_conditions, self.time)
        return solution



def plot_graphs(time, x1, x2, x3, x4):
    fig,ax=plt.subplots(2,2, figsize=(9,11))

# Volatge vs time
    ax[0,0].plot(time, x1)
    ax[0,0].set_xlabel('Time (ms)')
    ax[0,0].set_ylabel('Voltage (mV)')
    ax[0,0].set_title('Voltage v/s Time')

# Gating variables v/s time
    ax[0,1].plot(time, x2)
    ax[0,1].set_xlabel('Time (ms)')
    ax[0,1].set_ylabel('n variable')
    ax[0,1].set_title('n v/s Time')

    ax[1,0].plot(time, x3)
    ax[1,0].set_xlabel('Time (ms)')
    ax[1,0].set_ylabel('m variable')
    ax[1,0].set_title('m v/s Time')

    ax[1,1].plot(time, x4)
    ax[1,1].set_xlabel('Time (ms)')
    ax[1,1].set_ylabel('h variable')
    ax[1,1].set_title('h v/s Time')

    plt.show()