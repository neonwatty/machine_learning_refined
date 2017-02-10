import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from JSAnimation import IPython_display

def basic_animation(frames=100, interval=30):
    """Plot a basic sine wave with oscillating amplitude"""
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 10), ylim=(-2, 2))
    line, = ax.plot([], [], lw=2)

    x = np.linspace(0, 10, 1000)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        y = np.cos(i * 0.02 * np.pi) * np.sin(x - i * 0.02 * np.pi)
        line.set_data(x, y)
        return line,

    return animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=interval)


def lorenz_animation(N_trajectories=20, rseed=1, frames=200, interval=30):
    """Plot a 3D visualization of the dynamics of the Lorenz system"""
    from scipy import integrate
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import cnames

    def lorentz_deriv(coords, t0, sigma=10., beta=8./3, rho=28.0):
        """Compute the time-derivative of a Lorentz system."""
        x, y, z = coords
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    # Choose random starting points, uniformly distributed from -15 to 15
    np.random.seed(rseed)
    x0 = -15 + 30 * np.random.random((N_trajectories, 3))

    # Solve for the trajectories
    t = np.linspace(0, 2, 500)
    x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t)
                      for x0i in x0])

    # Set up figure & 3D axis for animation
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')

    # choose a different color for each trajectory
    colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

    # set up lines and points
    lines = sum([ax.plot([], [], [], '-', c=c)
                 for c in colors], [])
    pts = sum([ax.plot([], [], [], 'o', c=c, ms=4)
               for c in colors], [])

    # prepare the axes limits
    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5, 55))

    # set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 0)

    # initialization function: plot the background of each frame
    def init():
        for line, pt in zip(lines, pts):
            line.set_data([], [])
            line.set_3d_properties([])

            pt.set_data([], [])
            pt.set_3d_properties([])
        return lines + pts

    # animation function: called sequentially
    def animate(i):
        # we'll step two time-steps per frame.  This leads to nice results.
        i = (2 * i) % x_t.shape[1]

        for line, pt, xi in zip(lines, pts, x_t):
            x, y, z = xi[:i + 1].T
            line.set_data(x, y)
            line.set_3d_properties(z)

            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])

        ax.view_init(30, 0.3 * i)
        fig.canvas.draw()
        return lines + pts

    return animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=interval)
