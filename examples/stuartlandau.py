import jax.numpy as jnp
from prax import Oscillator
from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

class StuartLandau(Oscillator):
    def __init__(self, dt=0.01, eps=10**-5):
        super().__init__(n_dim=2, dt=dt, eps=eps)

    def forward(self, state):
        x, y = state
        vx = x - y - x * (x * x + y * y)
        vy = x + y - y * (x * x + y * y)
        return jnp.array([vx, vy])

model = StuartLandau()
init_val = jnp.array([0.1, 0.2])
model.find_periodic_orbit(init_val)
model.calc_phase_response()

plt.figure(figsize=[12,4])
plt.subplot(1,2,1)
plt.title("periodic orbit")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(model.periodic_orbit[:, 0], model.periodic_orbit[:, 1])
plt.subplot(1,2,2)
plt.title("phase response curve")
plt.plot(model.ts, model.phase_response_curve,)
plt.legend(labels=["$Z_x$", "$Z_y$"])
plt.xlabel("t")
plt.ylabel("$Z_x, Z_y$")

plt.savefig("stuartlandau.svg")