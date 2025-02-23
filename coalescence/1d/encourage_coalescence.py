# 1d coalescence with DQN
import math
import numpy as np
import random
import itertools
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple, deque
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# structure of the Q table
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# class that defines the Q table
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # deque is a more efficient form of a list

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, hidden_size, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)    


# for training phase newL = L; L is a size of an observation vector
# we center the state around the chocen particle and gie the network a sense of distance by preparing the state 
# in the following way:
def get_state(lattice, X, L, newL):
    state = []
    # before X
    for i in range(int(L/2)):
        x = X - int(L/2) + i
        if x < 0:
            x += newL
        value = lattice[x]
        state.append(value)
    # after X
    for i in range(int(L/2)):
        x = X + i + 1
        if x > newL - 1:
            x += -newL
        value = lattice[x]
        state.append(value)
    #print("lattice ", lattice)
    #print("X ", X)
    #print("state ", state, "\n")

    return state

'''
def get_state(system, X, L, newL):
    state = []
    for i in range(int(L/2)):
        index_plus = X + i + 1
        index_minus = X - i - 1
        if index_plus > newL - 1:
            index_plus += -newL
        if index_minus < 0:
            index_minus += newL

        value_plus = system[index_plus] #/ i
        value_minus = system[index_minus] #/ i        

        state.append(value_plus)
        state.append(value_minus)
    
    return state
'''


def select_action_training(state): 
    global steps_done # count total number of steps to go from almost random exploration to more efficient actions
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold: # exploitation
        with torch.no_grad():
            #if predator_or_prey == 1:
            return policy_net(state).max(1)[1].view(1, 1) # view(1,1) changes shape to [[action], dtype]
            #else:
                #return actor_prey(state).max(1)[1].view(1, 1) # view(1,1) changes shape to [[action], dtype]
    else:
        # select a random action; 
        rand_aciton = random.randint(0,1) # left or right
        return torch.tensor([[rand_aciton]], device=device, dtype=torch.long)


def select_action_post_training(state, T): 
    # interpret Q values as probabilities when simulating dynamics of the system 
    # in principle this could be easily extended to make this more general, but i am a lazy boi
    with torch.no_grad():
        #print("state ", state)
        Q_values = trained_net(state)
        #print("Q-values ", Q_values)
        #probs = torch.softmax(Q_values, dim=1) # converts logits to probabilities (torch object)
        #print("probs ", probs)
        meanQ_value = 0
        for i in range(n_actions):
            meanQ_value += Q_values[0][i]/n_actions
        #print("mean Q value", meanQ_value)
        # update Q values
        for i in range(n_actions):
            Q_values[0][i] = (Q_values[0][i] - meanQ_value) / T
        #print("new Q values", Q_values)
        new_probs = torch.softmax(Q_values, dim=1)
        #print("new probs", new_probs)
        dist = Categorical(new_probs) # feeds torch object to generate a list of probs (numpy object ?)
        action = dist.sample().numpy()[0] # sample list of probs and return the action
        #print("action ", action, "\n\n")
        
        return action


def step(lattice, X, L, action):
    # move
    newX = -1
    oppositeX = -1
    if action == 0: # jump to the left
        newX = X - 1 if X > 0 else L - 1
        oppositeX = X + 1 if X < L - 1 else 0
    else: # jump to the right
        newX = X + 1 if X < L - 1 else 0
        oppositeX = X - 1 if X > 0 else L - 1
    
    # here we encourage coalescence
    reward = 0
    lattice[X] = 0 # particle leaves the original lattice site
    if lattice[newX] == 1:
        reward = 1 # coalescence happened
    elif lattice[newX] == 0 and lattice[oppositeX] == 1: 
        lattice[newX] = 1 # diffusion
        reward = -10
    elif lattice[newX] == 0 and lattice[oppositeX] == 0:
        lattice[newX] = 1 # diffusion

    next_state = get_state(lattice, newX, L, L)

    return reward, next_state

def optimize_model():
    if len(memory) < BATCH_SIZE: # execute 'optimize_model' only if #BATCH_SIZE number of updates have happened 
        return
    transitions = memory.sample(BATCH_SIZE) # draws a random set of transitions; the next_state for terminal transition will be NONE
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions)) # turn [transition, (args)] array into [[transitions], [states], [actions], ... ]

    # Compute a mask of non-final states and concatenate the batch elements (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s != None, batch.next_state)), device=device, dtype=torch.bool) # returns a set of booleans
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # creates a list of non-empty next states
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # My understanding (for the original explanation check tutorial): 
    # Policy_net produces [[Q1,...,QN], ...,[]] (BATCH x N)-sized matrix, where N is the size of action space, 
    # and action_batch is BATCH-sized vector whose values are the actions that have been taken. 
    # Gather tells which Q from [Q1,...,QN] row to take, using action_batch vector, and returns BATCH-sized vector of Q(s_t, a) values
    state_action_values = policy_net(state_batch).gather(1, action_batch) # input = policy_net, dim = 1, index = action_batch

    # My understanding (for the original explanation check tutorial): 
    # Compute Q^\pi(s_t,a) values of actions for non_final_next_states by using target_net (old policy_net), from which max_a{Q(s_t, a)} are selected with max(1)[0].
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0] # target_net produces a vector of Q^pi(s_t+1,a)'s and max(1)[0] takes maxQ
    # Compute the expected Q^pi(s_t,a) values for all BATCH_SIZE (default=128) transitions
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def do_training(num_episodes, L, Nt):
    for i_episode in range(num_episodes):
        
        # start with fully filled lattice
        lattice = np.ones(L)
        #lattice = np.random.randint(2,size=L)

        # main update loop; I use Monte Carlo random sequential updates here
        score = 0
        episode_end_time = 0
        for t in range(Nt):
            episode_end_time = t
            # pick random lattice site
            for i in range(L):
                X = random.randint(0, L-1)
                if lattice[X] != 0:
                    state = get_state(lattice, X, L, L) # since the observation state equals to the lattice size
                    #print("state before ", state)
                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = select_action_training(state) # select action
                    #print("action ", action.item())
                    reward, next_state = step(lattice, X, L, action.item()) # update particle's position and do stochastic part
                    #print("reward ", reward)
                    reward = torch.tensor([reward], device=device)
                    #print("state after ", next_state)
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0) 
                    memory.push(state, action, next_state, reward)           
                    score += reward

            optimize_model() # optimize both predators and prey networks
            # Soft update of the target network's weights: θ′ ← τ θ + (1 −τ)θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            n_sum = 0
            for x in range(L):
                if lattice[x] == 1:
                    n_sum +=1
            if n_sum == 1: # one of the species hit the absorbing state
                break

        # sum remaining particles
        n_sum = 0
        for x in range(L):
            if lattice[x] == 1:
                n_sum +=1

        #print("Training episode ", i_episode, " is over. Survived particles ", n_sum, "; episode ended after ", episode_end_time, " steps")
        rewards.append(score) 
        episode_durations.append(episode_end_time)
        plot_score()

    torch.save(policy_net.state_dict(), PATH)
    plot_score(show_result=True)
    plt.ioff()
    #plt.show()
    plt.savefig("./Training_encourage.png", format="png", dpi=600)

# check training efficiency
def count_mistakes(net):
    mistakes = 0
    state_space = 2**n_observations
    for i in range(n_observations + 1):
        grid = [np.bincount(xs, minlength=n_observations) for xs in itertools.combinations(range(n_observations), i)]
        # feed states to the network and check the resulting Q values
        for j in range(len(grid)):
            state = torch.tensor(grid[j], dtype=torch.float32, device=device).unsqueeze(0)
            Q_values = net(state)
            print("state ", state)
            print("Q_values ", Q_values)
            sum_left = 0
            sum_right = 0
            for n in range(n_observations):
                if n < n_observations/2:
                    sum_left += grid[j][n]
                else:
                    sum_right += grid[j][n]
            if sum_left > sum_right and Q_values[0][0] < Q_values[0][1]:
                mistakes += 1
            elif sum_left < sum_right and Q_values[0][0] > Q_values[0][1]:
                mistakes += 1

    return mistakes

# plots
def plot_score(show_result=False):
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf() # clf -- clear current figure
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Episode Duration')
    plt.plot(episode_durations)
    if len(episode_durations) >= 20:
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        means = durations_t.unfold(0, 20, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(19), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            output = "./training_score.png"
            plt.savefig(output, format = "png", dpi = 300)

# Main
if __name__ == '__main__':
    Jesse_we_need_to_train_NN = True
    ############# Model parameters for Machine Learning #############
    num_episodes = 500      # number of training episodes
    BATCH_SIZE = 100        # the number of transitions sampled from the replay buffer
    GAMMA = 0.99            # the discounting factor
    EPS_START = 0.9         # EPS_START is the starting value of epsilon; determines how random our action choises are at the beginning
    EPS_END = 0.001         # EPS_END is the final value of epsilon
    EPS_DECAY = 500        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = 0.005             # TAU is the update rate of the target network
    LR = 1e-3               # LR is the learning rate of the AdamW optimizer
    ############# Lattice simulation parameters #############
    L = 5                  # Lx, we start with fully occupied lattice
    Nt = 20                # episode duration
    n_observations = L - 1      # just give network a difference between positive and negative spins
    n_actions = 2           # the particle can jump on any neighboring lattice sites, or stay put and eat
    hidden_size = 32        # hidden size of the network
    PATH = "./encourage_NN_params_L" + str(L) + ".txt"

    ############# Do the training if needed ##############
    if Jesse_we_need_to_train_NN:
        policy_net = DQN(n_observations, hidden_size, n_actions).to(device)
        target_net = DQN(n_observations, hidden_size, n_actions).to(device)
        mistakes = count_mistakes(policy_net)
        print("Number of mistakes before training: ", mistakes)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(100*Nt) # the overall memory batch size 
        #memory_prey = ReplayMemory(Nt) # the overall memory batch size 
        rewards = []
        episode_durations = []
        steps_done = 0 
        do_training(num_episodes, L, Nt) 

    ############# Training summary ######################
    trained_net = DQN(n_observations, hidden_size, n_actions).to(device)
    trained_net.load_state_dict(torch.load(PATH))
    print("Training results")
    mistakes = count_mistakes(trained_net)
    print("Number of mistakes after training: ", mistakes, "\n")

    ############# Post-training simulation ##############
    runs = 10
    newL = 1000
    Nt = 1*newL
    T = 0.1
    particles_left = np.zeros(Nt)
    for run in range(runs):
        print("run ", run)
        big_lattice = np.ones(newL)
        perished = 0
        for t in range(Nt):
            for i in range(newL):
                # pick random lattice site
                X = random.randint(0, newL-1)
                if big_lattice[X] != 0:
                    state = get_state(big_lattice, X, L, newL)
                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = select_action_post_training(state, T)
                    
                    newX = -1
                    oppositeX = -1
                    if action == 0: # jump to the left
                        newX = X - 1 if X > 0 else newL - 1
                        oppositeX = X + 1 if X < newL - 1 else 0
                    else: # jump to the right
                        newX = X + 1 if X < newL - 1 else 0
                        oppositeX = X - 1 if X > 0 else newL - 1
                    
                    if big_lattice[newX] == 0 and big_lattice[oppositeX] == 1:
                        print("Investigate! Something is wrong!")

                    big_lattice[X] = 0 # particle leaves the original lattice site
                    if big_lattice[newX] == 0:
                        big_lattice[newX] = 1 # diffusion
                    else:
                        perished +=1 # coalescence

            # count particles
            #n_sum = 0
            #for x in range(newL):
            #    if big_lattice[x] == 1:
            #        n_sum +=1
            
            particles_left[t] += (newL - perished) / (newL*runs) # look at how density decreases with time      
            
            # stop the run if only one particle left
            if newL - perished < 0.005*newL: # if density becomes too low -- stop the simulation
                break

    filename = "encourage_data_L" + str(L) + "_bigL" + str(newL) + "_runs" + str(runs) + ".txt"
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
