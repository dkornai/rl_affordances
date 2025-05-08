import matplotlib.pyplot as plt
import torch
from gridworld import GridWorld

def plot_controllability_matrix(C):
    """
    Plot the controllability matrix C(s,s').
    """
    plt.imshow(C.numpy(), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title("Controllability Matrix C(s,s')")
    plt.xlabel("States s'")
    plt.ylabel("States s")
    plt.show()

def plot_reachability(C, gridworld:GridWorld):

    print("Reachability of states:")
    c_bar = torch.mean(C, dim=0)
    # wrap onto grid to show controllability of states
    c_bar_grid = c_bar.reshape(gridworld.height, gridworld.width)

    plt.imshow(c_bar_grid.numpy(), cmap='viridis', interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("Reachability of states in the 2D gridworld")

    # draw walls from the gridworld
    if gridworld.wall_horiz is not None:
        for wall in gridworld.wall_horiz:
            plt.plot([wall[0]-0.5, wall[1]-0.5], [wall[0]+0.5, wall[0]+0.5], color='red', linewidth=5)
    if gridworld.wall_vert is not None:
        for wall in gridworld.wall_vert:
            plt.plot([wall[0]-0.5, wall[0]-0.5], [wall[1]-0.5, wall[1]+0.5], color='red', linewidth=5)

    plt.show()

def plot_affordance_rate(C, gridworld:GridWorld):

    print("Affordance rate of states:")
    c_bar = torch.mean(C, dim=1)
    # wrap onto grid to show controllability of states
    c_bar_grid = c_bar.reshape(gridworld.height, gridworld.width)

    plt.imshow(c_bar_grid.numpy(), cmap='viridis', interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("Affordance rate of states in the 2D gridworld")

    # draw walls from the gridworld
    if gridworld.wall_horiz is not None:
        for wall in gridworld.wall_horiz:
            plt.plot([wall[0]-0.5, wall[1]-0.5], [wall[0]+0.5, wall[0]+0.5], color='red', linewidth=5)
    if gridworld.wall_vert is not None:
        for wall in gridworld.wall_vert:
            plt.plot([wall[0]-0.5, wall[0]-0.5], [wall[1]-0.5, wall[1]+0.5], color='red', linewidth=5)

    plt.show()