#Questions:
# Parameters
#  Which hgf parameters? Optimize?
#  Which generative process parameters? Especially gamma parameters.
#  Which noise parameters?
#  How does the rho parameter work?
# Enrionment
#  How do I include a rho drift?
# Active Inference
#  (Explicit) goal prior
#  Inferring own actions
# Where is the break in the perception-action cycle?
# Moving on a circle versus moving on a line

#Extensions
# Infer own position
# Include drift
# Making actions more predictive
# Add a parent to the volatility (or just blocks with varying volatility)
# Make the agent only be able to (prefer to) move some length
# Make the agent's observations depend on the distance to the target
# Make the agent able to affect the hidden states (maybe the target movement's drift?)




#Take modulo of position for the cirle thing





#-- Setup --#
import sys
sys.path.append('ghgf')
import hgf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import math
import imageio
import os


#-- Settings --#
#Simulation
n_timesteps = 200

#Target
gamma_params = [2,0.05]
init_target_position = 0

#Agent
hgf_target_position = hgf.StandardHGF(initial_mu1=0,
                         initial_pi1=1e4,
                         omega1=2,
                         kappa1=1,
                         initial_mu2=1,
                         initial_pi2=1e1,
                         omega2=-2,
                         omega_input=-1,
                         rho1 = 0,
                         rho2 = 0)
agent_position_observation_noise = 0
target_position_observation_noise = 0.5
agent_movement_noise = 0.1
init_agent_position = 0


#-- Functions --#
def save_plot_single_timestep(timepoint, target_position, agent_position):
    #Make filename and append it to the list
    filename = 'p{}'.format(timepoint)
    plots_filenames.append(filename)

    #Create circle for plotting
    circle = plt.Circle((0, 0), 1, fill = False)
    #Make plot with circle
    fig, ax = plt.subplots()
    ax.add_artist(circle)
    #Add point for target
    plt.scatter(math.cos(target_position), math.sin(target_position))
    #Add point for agent
    plt.scatter(math.cos(agent_position), math.sin(agent_position))
    #Set axes
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])

    #Save the figure as the filename
    plt.savefig('gifpics/' + filename)
    plt.close()



#-- Simulation --#
#Make empty arrays for population
agent_positions = np.empty([n_timesteps])
observed_agent_positions = np.empty([n_timesteps])
inferred_agent_positions = np.empty([n_timesteps])
target_variances = np.empty([n_timesteps])
target_positions = np.empty([n_timesteps])
observed_target_positions = np.empty([n_timesteps])
inferred_target_positions = np.empty([n_timesteps])
agent_surprisal = np.empty([n_timesteps])

#Initialize first timestep
agent_positions[0] = init_agent_position

target_variances[0] = np.random.gamma(gamma_params[0], gamma_params[1])
target_positions[0] = np.random.normal(init_target_position, math.sqrt(target_variances[0])) 

observed_agent_positions[0] = np.random.normal(agent_positions[0], agent_position_observation_noise)
inferred_agent_positions[0] = observed_agent_positions[0]

observed_target_positions[0] = np.random.normal(target_positions[0], target_position_observation_noise)
hgf_target_position.input(observed_target_positions[0])
inferred_target_positions[0] = hgf_target_position.x1.mus[-1]

agent_surprisal[0] = (inferred_agent_positions[0] - inferred_target_positions[0])**2

#For each timestep except the first
for timestep in range(1, n_timesteps):

    #The agent moves to it's the inferred position of the target at last timestep
    agent_positions[timestep] = np.random.normal(inferred_target_positions[timestep-1], agent_movement_noise) #Change this to explicit active inference

    #Sample the variance of the target's random walk
    target_variances[timestep] = np.random.gamma(gamma_params[0], gamma_params[1])
    #Sample the position of the target
    target_positions[timestep] = np.random.normal(target_positions[timestep-1], math.sqrt(target_variances[timestep]))

    #The agent observes (noisily) its own position
    observed_agent_positions[timestep] = np.random.normal(agent_positions[timestep], agent_position_observation_noise)
    #The agent assumes its observed own position is its real position 
    inferred_agent_positions[timestep] = observed_agent_positions[timestep] #Change this to add inference on own position
    
    #The agent observes (noisily) the target's position
    observed_target_positions[timestep] = np.random.normal(target_positions[timestep], target_position_observation_noise)
    #Input data to the hgf
    hgf_target_position.input(observed_target_positions[timestep])
    #Extract the inferred position of the target from the hgf
    inferred_target_positions[timestep] = hgf_target_position.x1.mus[-1]

    #Get the surprisal / exact free energy / squared error
    agent_surprisal[timestep] = (inferred_agent_positions[timestep] - inferred_target_positions[timestep])**2 #Change this to refer to goal prior
    


#-- Plots --#
#variance versus inferred variance
pd.Series(target_variances).plot()
pd.Series(np.exp(hgf_target_position.x2.mus)).plot()

#position versus inferred versus observed position
pd.Series(target_positions).plot()
pd.Series(inferred_target_positions).plot()
pd.Series(observed_target_positions).plot()

#own versus target positions with surprisal
pd.Series(inferred_target_positions).plot()
pd.Series(inferred_agent_positions).plot()
pd.Series(agent_surprisal).plot()



#-- GIF --#
#Make GIF single plots
plots_filenames = []

for timestep in range(n_timesteps):
    #Make and save plot of that timepoint
    save_plot_single_timestep(timestep, target_positions[timestep], agent_positions[timestep])

#Build together into a GIF
with imageio.get_writer('chasing_target.gif', mode='I') as writer:
    for filename in plots_filenames:
        image = imageio.imread('gifpics/' + filename + '.png')
        writer.append_data(image)
        os.remove('gifpics/' + filename + '.png')