#Questions:
# Which Gamma parameters?
# Which hgf parameters?
# How do I make the circular thing work ? 
# Should there be a goal prior ? 

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
observation_noise = 0.2
init_target_position = 0

#Agent
stdhgf = hgf.StandardHGF(initial_mu1=0,
                         initial_pi1=1e4,
                         omega1=2,
                         kappa1=1,
                         initial_mu2=1,
                         initial_pi2=1e1,
                         omega2=-2,
                         omega_input=-1)
movement_noise = 0.1
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
agent_observations = np.empty([n_timesteps])
target_positions = np.empty([n_timesteps])
target_variances = np.empty([n_timesteps])
agent_positions = np.empty([n_timesteps])
agent_surprisal = np.empty([n_timesteps])

#Initialization
target_positions[0] = init_target_position
target_variances[0] = np.random.gamma(gamma_params[0], gamma_params[1])
agent_observations[0] = np.random.normal(init_target_position, observation_noise)
agent_positions[0] = init_agent_position

#For each timestep except the first
for timestep in range(1, n_timesteps):
    #Sample the variance of the targets random walk
    target_variances[timestep] = np.random.gamma(gamma_params[0], gamma_params[1]) #You can do this beforehand to save time
    #Sample the position of the target
    target_positions[timestep] = np.random.normal(target_positions[timestep-1], math.sqrt(target_variances[timestep]))
    #Get the noisy observation of the agent
    agent_observations[timestep] = np.random.normal(target_positions[timestep], observation_noise)
    
    #Input data to the hgf
    stdhgf.input(agent_observations[timestep])

    #Find the agent's posterior belief about the current target position
    agent_positions[timestep] = np.random.normal(stdhgf.x1.mus[-1], movement_noise)

    #Get the surprisal / exact free energy / squared error
    agent_surprisal[timestep] = (agent_positions[timestep] - target_positions[timestep])**2
    

#-- Plots --#
pd.Series(target_variances).plot()
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















stdobjf = stdhgf.neg_log_joint_function()

# Set priors
stdhgf.x1.initial_mu.trans_prior_mean = 1.0375
stdhgf.x1.initial_mu.trans_prior_precision = 4.0625e5
stdhgf.x1.initial_pi.trans_prior_mean = -10.1111
stdhgf.x1.initial_pi.trans_prior_precision = 1
stdhgf.x1.omega.trans_prior_mean = -12.1111
stdhgf.x1.omega.trans_prior_precision = 4**-2
stdhgf.x2.initial_pi.trans_prior_mean = -2.3026
stdhgf.x2.initial_pi.trans_prior_precision = 1
stdhgf.x2.omega.trans_prior_mean = -4
stdhgf.x2.omega.trans_prior_precision = 4**-2
stdhgf.xU.omega.trans_prior_mean = -10.1111
stdhgf.xU.omega.trans_prior_precision = 2**-2

stdx0 = [param.value for param in stdhgf.var_params]
stdmin = minimize(stdobjf, stdx0)



pd.Series(np.random.gamma(3, 0.3, 1000)).plot.density()