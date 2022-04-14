import jax.numpy as jnp
from prax import Oscillator
from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

class HodgkinHuxley(Oscillator):
    def __init__(self, input_current, C=1.0, G_Na=120.0, G_K=36.0, G_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.4, dt=0.01, eps=10**-5):
        super().__init__(n_dim=4, dt=dt, eps=eps)
        self.input_current = input_current
        self.C = C
        self.G_Na = G_Na
        self.G_K = G_K
        self.G_L = G_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L

    def alpha_m(self, V):
        return 0.1*(V+40.0)/(1.0 - jnp.exp(-(V+40.0) / 10.0))
    
    def beta_m(self, V):
        return 4.0*jnp.exp(-(V+65.0) / 18.0)
    
    def alpha_h(self, V):
        return 0.07*jnp.exp(-(V+65.0) / 20.0)
    
    def beta_h(self, V):
        return 1.0/(1.0 + jnp.exp(-(V+35.0) / 10.0))
    
    def alpha_n(self, V):
        return 0.01*(V+55.0)/(1.0 - jnp.exp(-(V+55.0) / 10.0))
    
    def beta_n(self, V):
        return 0.125*jnp.exp(-(V+65) / 80.0)

    def forward(self, state):
        V, m, h, n = state
        dVdt = self.G_Na * (m ** 3) * h * (self.E_Na - V) + self.G_K * (n ** 4) * (self.E_K - V) + self.G_L * (self.E_L - V) + self.input_current
        dVdt /= self.C
        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        return jnp.array([dVdt, dmdt, dhdt, dndt])

model = HodgkinHuxley(input_current=30.0)
init_val = jnp.array([-75, 0.6, 0.05, 0.32])
model.find_periodic_orbit(init_val)
model.calc_phase_response()

plt.figure(figsize=[12,8])

plt.subplot(2,2,1)
plt.title("periodic orbit")
plt.xlabel("t")
plt.ylabel("V")
plt.plot(model.ts, model.periodic_orbit[:, 0])

plt.subplot(2,2,2)
plt.title("phase response curve")
plt.plot(model.ts, model.phase_response_curve[:,0])
plt.legend(labels=["$Z_V$"])
plt.xlabel("t")
plt.ylabel("$Z_V$")

plt.subplot(2,2,3)
plt.xlabel("t")
plt.ylabel("m,h,n")
plt.plot(model.ts, model.periodic_orbit[:, 1:])

plt.subplot(2,2,4)
plt.plot(model.ts, model.phase_response_curve[:,1:])
plt.legend(labels=["$Z_m$","$Z_h$","$Z_n$"])
plt.xlabel("t")
plt.ylabel("$Z_m,Z_h,Z_n$")

plt.savefig("hodgkinhuxley.svg")