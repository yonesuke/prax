import jax.numpy as jnp
from jax import jit, jacfwd, vmap
from jax.lax import cond, fori_loop, while_loop

from prax.integrator import RungeKutta

def PeriodicOrbit(func, init_val, idx=0, section=0.0, dt=0.01, transient=10**7, maximum_power_two=20, n_step=2**12):
    n_dim = len(init_val)

    next_state = lambda state, dt: RungeKutta(func, state, 0.0, dt)
    update = jit(lambda i, val: next_state(val, dt))
    state = fori_loop(0, transient, update, init_val)

    x_prev, x_now = state, next_state(state, dt)
    power_two = 0
    # True if prev and now is crossing section
    check_cross = lambda x_prev, x_now: jnp.logical_and(x_prev[idx] < section, x_now[idx] > section)
    # True if crossing and enough divided
    cond_fun = lambda val: jnp.logical_and(check_cross(*val[0]), val[1] > maximum_power_two)
    def true_fun(val):
        (x_prev, x_now), power_two = val
        power_two += 1
        x_now = next_state(x_prev, dt * 0.5 ** power_two)
        return [(x_prev, x_now), power_two]
    def false_fun(val):
        (x_prev, x_now), power_two = val
        x_prev, x_now = x_now, next_state(x_now, dt * 0.5 ** power_two)
        return [(x_prev, x_now), power_two]
        return 0
    # poincate sectionの初期点を得る
    (x_prev, x_now), power_two = while_loop(
        cond_fun = jit(lambda val: jnp.logical_not(cond_fun(val))),
        body_fun = jit(lambda val: cond(
            pred = check_cross(*val[0]),
            true_fun = true_fun,
            false_fun = false_fun,
            operand = val
        )),
        init_val=[(x_prev, x_now), power_two]
    )

    x_prev, x_now = x_now, next_state(x_now, dt * 0.5 ** power_two)
    # orbitの周期を求める
    def true_fun(val):
        (x_prev, x_now), power_two, orbit_period = val
        power_two += 1
        x_now = next_state(x_prev, dt * 0.5 ** power_two)
        return [(x_prev, x_now), power_two, orbit_period]
    def false_fun(val):
        (x_prev, x_now), power_two, orbit_period = val
        orbit_period += dt * 0.5 ** power_two
        x_prev, x_now = x_now, next_state(x_now, dt * 0.5 ** power_two)
        return [(x_prev, x_now), power_two, orbit_period]
    (x_prev, x_now), power_two, orbit_period = while_loop(
        cond_fun = jit(lambda val: jnp.logical_not(cond_fun(val))),
        body_fun = jit(lambda val: cond(
            pred = check_cross(*val[0]),
            true_fun = true_fun,
            false_fun = false_fun,
            operand = val
        )),
        init_val=[(x_prev, x_now), 0, 0.0]
    )

    ts, delta = jnp.linspace(0, orbit_period, num=n_step, retstep=True)
    orbit = jnp.empty((n_step, n_dim))
    def body_fun(i, val):
        state, orbit = val
        state = next_state(state, delta)
        orbit = orbit.at[i].set(state)
        return [state, orbit]
    _, orbit = fori_loop(
        0, n_step,
        body_fun,
        init_val=[x_prev, orbit]
    )

    return orbit_period, ts, orbit

def PhaseSensitivity(func, orbit, orbit_period, eps=10**(-5)):
    jacobian_func = jacfwd(func)
    adjoint_func = lambda z, x: -jacobian_func(x).T @ z

    omega = 2.0 * jnp.pi / orbit_period
    orbit_reverse = jnp.flipud(orbit)
    n_step, n_dim = orbit.shape
    dt = orbit_period / n_step

    def update(i, val):
        z, zs = val
        zs = zs.at[i].set(z)
        z -= dt * adjoint_func(z, orbit_reverse[i])
        return [z, zs]
    contour_integrate = lambda z, zs: fori_loop(0, n_step, update, init_val=[z, zs])[1]
    def normalize_omega(zs):
        coeffs = omega / vmap(jnp.vdot, (0, 0))(vmap(func)(orbit_reverse), zs)
        return zs * coeffs.reshape(-1, 1)
    
    z_prev = normalize_omega(contour_integrate(jnp.ones(n_dim), jnp.empty(orbit.shape)))
    z_now = z_prev

    def body_fun(val):
        z_prev, z_now = val
        z_prev = z_now
        z_now = normalize_omega(contour_integrate(z_prev[-1], z_now))
        return [z_prev, z_now]
    cond_fun = lambda val: jnp.abs(val[0] - val[1]).max() > eps
    z_prev, z_now = while_loop(cond_fun, body_fun, init_val=[z_prev, z_now])
    return jnp.flipud(z_now)

def PhaseCouplingFn(sensitivity_i, orbit_i, orbit_j, coupling_fn):
    steps = orbit_i.shape[0]
    def shifted_coupling(k):
        return vmap(coupling_fn, in_axes=(0, 0))(orbit_i, jnp.roll(orbit_j, -k))
    integral_fn = lambda k: jnp.dot(sensitivity_i, shifted_coupling(k)) / steps
    phase_couplings = vmap(integral_fn)(jnp.arange(steps))
    return phase_couplings