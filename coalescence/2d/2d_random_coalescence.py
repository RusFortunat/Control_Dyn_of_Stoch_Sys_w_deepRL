# 1d coalescence with DQN
import math
import numpy as np
import random
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple, deque
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Main
if __name__ == '__main__':

    runs = 10
    L = 200
    updates = L*L
    Nt = 1000
    particles_left = np.zeros(Nt)
    for run in range(runs):
        print("run ", run)
        big_lattice = np.ones(shape=(L, L))
        perished = 0
        for t in range(Nt):
            for i in range(updates):
                # pick random lattice site
                X = random.randint(0, L-1)
                Y = random.randint(0, L-1)
                if big_lattice[X][Y] != 0:
                    newX = X
                    newY = Y
                    action = random.randint(0,3)
                    if action == 0: # jump to the right
                        newX = X + 1 if X < L - 1 else 0
                    elif action == 1: # jump to the left
                        newX = X - 1 if X > 0 else L - 1
                    elif action == 2: # jump to the top
                        newY = Y + 1 if Y < L - 1 else 0
                    else: # jump to the bottom
                        newY = Y - 1 if Y > 0 else L - 1
                    
                    big_lattice[X][Y] = 0 # particle leaves the original lattice site
                    if big_lattice[newX][newY] == 0:
                        big_lattice[newX][newY] = 1 # diffusion; otherwise coalescence
                    else:
                        perished += 1
            
            # no particles left
            n_sum = 0
            for x in range(L):
                for y in range(L):
                    if big_lattice[x][y] == 1:
                        n_sum +=1
            if(n_sum != L*L - perished): 
                print("Error!")

            particles_left[t] +=  (L*L - perished) / (L*L*runs) # look at how density decreases with time      
            
            # stop the run if only one particle left
            if L*L - perished < 0.001*L*L: # if density becomes too low -- stop the simulation
                break

    filename = "2d_random_decay_L" + str(L) + "_runs" + str(runs) + ".txt"
    with open(filename, 'w') as f:
        for t in range(Nt):
            output_string = str(t) + "\t" + str(particles_left[t]) + "\n"
            f.write(output_string)

    # animation
    '''
    import matplotlib.animation as animation
    fig = plt.figure(figsize=(24, 24))
    im = plt.imshow(memory[:, :, 1], interpolation="none", aspect="auto", vmin=0, vmax=1)

    def animate_func(i):
        im.set_array(memory[:, :, i])
        return [im]

    fps = 60

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=sim_duration,
        interval=1000 / fps,  # in ms
    )

    print("Saving animation...")
    filename_animation = "anim_T" + str(Temp) + "_M=" + str(total_magnetization[sim_duration-1]) + ".mp4"
    anim.save(filename_animation, fps=fps, extra_args=["-vcodec", "libx264"])
    print("Done!")
    '''
