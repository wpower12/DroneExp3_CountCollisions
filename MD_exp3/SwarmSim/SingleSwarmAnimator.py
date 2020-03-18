import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class SingleSwarmAnimator():
    def __init__(self, start, destination, run, fig_title):
        plt.ion()
        self.fig_title = fig_title
        self.fig = plt.figure(num=fig_title)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.start = start
        self.destination = destination
        self.run = run
        self.color_str = ""

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        self.var_x_factor = np.outer(np.cos(u), np.sin(v))
        self.var_y_factor = np.outer(np.sin(u), np.sin(v))
        self.var_z_factor = np.outer(np.ones(np.size(u)), np.cos(v))

    def plot_drones(self, sim, in_training=True, plot_errors=False):
        plt.cla()
        self.color_str = sim.color + "."
        for d in sim.drones:
            self.plot_drone(d, in_training)

        # Hardcoding for now, should fix later.
        # read the limits from the waypoints? idk.
        plt.xlim(self.start[0], self.destination[0])
        plt.ylim(self.start[1], self.destination[1])
        self.ax.set_zlim(self.start[2], self.destination[2])

        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])

        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.001)

        # plt.savefig(self.fig_title + '.png')
        model_num = None
        if self.fig_title == 'Dead reckoning':
            model_num = '1'
        elif self.fig_title == 'Unstructured regressor':
            model_num = '2'
        else:
            model_num = '3'

        run_str = str(self.run + 1)
        if len(run_str) == 1:
            run_str = '00' + run_str
        elif len(run_str) == 2:
            run_str = '0' + run_str

        plt.savefig('exp_run_results/' + run_str + '_' + model_num + '.png')

    def plot_drone(self, d, in_training=True):
        x, y, z = d.pos

        self.ax.plot([x], [y], [z], self.color_str)

        if len(d.H_pos) > 0:
            s_hist = np.vstack(d.H_pos)
            self.ax.plot(s_hist[:, 0], s_hist[:, 1], s_hist[:, 2], 'k:')

        if not in_training:  # If in inference:
            x, y, z = d.pos_estimate
            self.ax.plot([x], [y], [z], 'r.')

            if d.pos_variance[0] > 0:
                # Plot the prediction uncertainty (variance)
                var_x = 3 * d.pos_variance[0] * self.var_x_factor
                var_y = 3 * d.pos_variance[1] * self.var_y_factor
                var_z = 3 * d.pos_variance[2] * self.var_z_factor
                # Plot the surface
                self.ax.plot_surface(var_x + x, var_y + y, var_z + z, color='r', alpha=0.1)

            if len(d.H_pos_estimate) > 0:
                s_hist = np.vstack(d.H_pos_estimate)
                self.ax.plot(s_hist[:, 0], s_hist[:, 1], s_hist[:, 2], 'r:')
