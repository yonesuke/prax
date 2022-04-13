import tree

def RungeKutta(func, x0, dt):
    k1 = func(x0)
    k2 = func(
        tree.map_structure(lambda x, y: x + 0.5 * dt * y, x0, k1)
    )
    k3 = func(
        tree.map_structure(lambda x, y: x + 0.5 * dt * y, x0, k2)
    )
    k4 = func(
        tree.map_structure(lambda x, y: x + dt * y, x0, k3)
    )
    # dx = dt * (k1+2.0*k2+2.0*k3+k4) / 6.0
    # return x0 + dx
    return tree.map_structure(lambda x, y1, y2, y3, y4: x+dt*(y1+2.0*y2+2.0*y3+y4)/6.0, x0, k1, k2, k3, k4)

def Euler(func, x0, dt):
    # dx = dt * func(x0, t)
    # return x0 + dx
    return tree.map_structure(lambda x, y: x + dt * y, x0, func(x0))