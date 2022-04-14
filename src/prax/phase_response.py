import jax.numpy as jnp
from jax import jit, jacfwd, vmap
from jax.lax import cond, fori_loop, while_loop

from prax.integrator import RungeKutta

class Oscillator:
    def __init__(self, n_dim, dt, eps):
        self.n_dim = n_dim
        self.dt = dt
        self.eps = eps
        self.periodic_orbit = None
        self.phase_response_curve = None

    def forward(self, state):
        raise NotImplementedError()
    
    def update(self, state):
        return RungeKutta(self.forward, state, self.dt)

    def run(self, init_state, t_max):
        ts = jnp.arange(0, t_max, self.dt)
        n_iter = len(ts)
        @jit
        def body_fn(i, orbit):
            state = orbit[i-1]
            next_state = self.update(state)
            orbit = orbit.at[i].set(next_state)
            return orbit
        orbit = jnp.zeros((n_iter, self.n_dim))
        orbit = orbit.at[0].set(init_state)
        orbit = fori_loop(1, n_iter, body_fn, init_val=orbit)
        return ts, orbit

    def find_periodic_orbit(self,
        init_state,
        idx=0,
        section=0.0,
        from_neg_to_pos = True,
        transient=10**5,
        maximum_power_two = 20,
        n_step = 2**12
    ):
        # update init state till state is in perodic orbit
        _body_fn = jit(lambda i, state: self.update(state))
        state = fori_loop(0, transient, _body_fn, init_state)
        
        # get poincate section coordinate
        x_prev, x_now = state, self.update(state)
        if from_neg_to_pos:
            check_cross_fn = lambda x_prev, x_now: jnp.logical_and(x_prev[idx] < section, x_now[idx] > section)
        else:
            check_cross_fn = lambda x_prev, x_now: jnp.logical_and(x_prev[idx] > section, x_now[idx] < section)
        # True if crossing and enough divided
        cond_fn = lambda val: jnp.logical_and(check_cross_fn(*val[0]), val[1] >= maximum_power_two)
        _update_fn = lambda state, dt: RungeKutta(self.forward, state, dt)
        def true_fun(val):
            (x_prev, x_now), power_two = val
            power_two += 1
            x_now = _update_fn(x_prev, self.dt * 0.5 ** power_two)
            return [(x_prev, x_now), power_two]
        def false_fun(val):
            (x_prev, x_now), power_two = val
            x_prev, x_now = x_now, _update_fn(x_now, self.dt * 0.5 ** power_two)
            return [(x_prev, x_now), power_two]
        # x_prev: before crossing poincare section
        # x_now: after crossing poincare section
        (x_prev, x_now), power_two = while_loop(
            cond_fun = jit(lambda val: jnp.logical_not(cond_fn(val))),
            body_fun = jit(lambda val: cond(
                pred = check_cross_fn(*val[0]),
                true_fun = true_fun,
                false_fun = false_fun,
                operand = val
            )),
            init_val=[(x_prev, x_now), 0]
        )

        # calculate period
        x_prev, x_now = x_now, _update_fn(x_now, self.dt * 0.5 ** maximum_power_two)
        def true_fun(val):
            (x_prev, x_now), power_two, period = val
            power_two += 1
            x_now = _update_fn(x_prev, self.dt * 0.5 ** power_two)
            return [(x_prev, x_now), power_two, period]
        def false_fun(val):
            (x_prev, x_now), power_two, period = val
            period += self.dt * 0.5 ** power_two
            x_prev, x_now = x_now, _update_fn(x_now, self.dt * 0.5 ** power_two)
            return [(x_prev, x_now), power_two, period]
        (x_prev, x_now), _, period = while_loop(
            cond_fun = jit(lambda val: jnp.logical_not(cond_fn(val))),
            body_fun = jit(lambda val: cond(
                pred = check_cross_fn(*val[0]),
                true_fun = true_fun,
                false_fun = false_fun,
                operand = val
            )),
            init_val=[(x_prev, x_now), 0, 0.0]
        )
        self.period = period

        # calculate periodic orbit
        ts, delta = jnp.linspace(0, period, num=n_step, retstep=True)
        orbit = jnp.empty((n_step, self.n_dim)); orbit = orbit.at[0].set(x_now)
        def body_fun(i, orbit):
            next_state = _update_fn(orbit[i-1], delta)
            orbit = orbit.at[i].set(next_state)
            return orbit
        orbit = fori_loop(1, n_step, body_fun, init_val=orbit)
        
        self.n_step = n_step
        self.periodic_orbit_dt = delta
        self.ts = ts
        self.periodic_orbit = orbit

    def calc_phase_response(self):
        jacobian_fn = jacfwd(self.forward)
        adjoint_fn = lambda z, x: -jacobian_fn(x).T @ z

        omega = 2.0 * jnp.pi / self.period
        orbit_reverse = jnp.flipud(self.periodic_orbit)

        def _update_fn(i, zs):
            z = zs[i-1]
            z -= self.periodic_orbit_dt * adjoint_fn(z, orbit_reverse[i-1])
            zs = zs.at[i].set(z)
            return zs
        contour_integrate_fn = lambda zs: fori_loop(1, self.n_step, _update_fn, init_val=zs)
        def normalize_omega_fn(zs):
            coeffs = omega / vmap(jnp.vdot, (0, 0))(vmap(self.forward)(orbit_reverse), zs)
            return zs * coeffs.reshape(-1, 1)
        zs_update_fn = jit(lambda zs: normalize_omega_fn(contour_integrate_fn(zs)))

        zs_prev = jnp.zeros(self.periodic_orbit.shape); zs_prev = zs_prev.at[0].set(jnp.ones(self.n_dim))
        zs_now = zs_update_fn(zs_prev)
        
        def body_fn(val):
            zs_prev, zs_now = val
            zs_prev = zs_now
            zs_now = zs_now.at[0].set(zs_prev[-1])
            zs_now = zs_update_fn(zs_now)
            return [zs_prev, zs_now]
        cond_fn = lambda val: jnp.abs(val[0] - val[1]).max() > self.eps
        zs_prev, zs_now = while_loop(cond_fn, body_fn, init_val=[zs_prev, zs_now])
        return jnp.flipud(zs_now)