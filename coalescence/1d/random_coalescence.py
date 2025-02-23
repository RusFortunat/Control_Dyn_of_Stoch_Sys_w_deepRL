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
    newL = 1000
    updates = newL
    Nt = 1*newL
    particles_left = np.zeros(Nt)
    for run in range(runs):
        print("run ", run)
        big_lattice = np.ones(newL)
        perished = 0
        for t in range(Nt):
            for i in range(updates):
                # pick random lattice site
                X = random.randint(0, newL-1)
                if big_lattice[X] != 0:
                    newX = -1
                    action = random.randint(0,1)
                    if action == 0: # jump to the right
                        newX = X + 1 if X < newL - 1 else 0
                    else: # jump to the left
                        newX = X - 1 if X > 0 else newL - 1
                    
                    big_lattice[X] = 0 # particle leaves the original lattice site
                    if big_lattice[newX] == 0:
                        big_lattice[newX] = 1 # diffusion; otherwise coalescence
                    else:
                        perished += 1
            
            particles_left[t] +=  (newL - perished) / (newL*runs) # look at how density decreases with time      
            
            # stop the run if only one particle left
            if newL - perished < 0.005*newL: # if density becomes too low -- stop the simulation
                break

    filename = "random_decay_" + str(updates) + "updates_L" + str(newL) + "_runs" + str(runs) + ".txt"
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
