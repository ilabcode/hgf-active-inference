#Questions:
# Parameters
#  Which hgf parameters?
#     Should I optimize on each timestep? (parameter learning)
#     Should Rho be learnt?
#  Which generative process parameters? Especially gamma parameters.
#  Which noise parameters?
# Environment
#  How do I include a rho drift?
# Active Inference
#  (Explicit) goal prior
#  Inferring own actions

#Extensions
# Include drift
# Enable explicit Active Inference
# Enable sufficient statistics filtering
# Infer own position
# Make things happen on an actual circle (modulo or radians)
# Add a parent to the volatility (or just blocks with varying volatility)
# Make the agent only be able to (prefer to) move some length
# Make the agent's observations depend on the distance to the target
# Make the agent able to affect the hidden states (maybe the target's drift?)



#-- Setup --#
import sys
sys.path.append('ghgf')
import hgf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from scipy.optimize import minimize




#-- Settings --#
#Simulation
n_timesteps = 200

#Target
gamma_params = [2,0.05]
init_target_position = 0

#Agent
agent_position_observation_noise = 0
target_position_observation_noise = 0.5
agent_movement_noise = 0.1
init_agent_position = 0

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
                         
hgf_target_noise = hgf.StandardHGF(initial_mu1=0,
                         initial_pi1=1e4,
                         omega1=2,
                         kappa1=1,
                         initial_mu2=1,
                         initial_pi2=1e1,
                         omega2=-2,
                         omega_input=-1,
                         rho1 = 0,
                         rho2 = 0)



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
    plt.scatter(np.cos(target_position), np.sin(target_position))
    #Add point for agent
    plt.scatter(np.cos(agent_position), np.sin(agent_position))
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
target_positions[0] = np.random.normal(init_target_position, np.sqrt(target_variances[0])) 

observed_agent_positions[0] = np.random.normal(agent_positions[0], agent_position_observation_noise)
inferred_agent_positions[0] = observed_agent_positions[0]

observed_target_positions[0] = np.random.normal(target_positions[0], target_position_observation_noise)
hgf_target_position.input(observed_target_positions[0])
hgf_target_noise.input(observed_target_positions[0]**2)
inferred_target_positions[0] = hgf_target_position.x1.mus[-1]

agent_surprisal[0] = (inferred_agent_positions[0] - inferred_target_positions[0])**2

#For each timestep except the first
for timestep in range(1, n_timesteps):

    #The agent moves to it's the inferred position of the target at last timestep
    agent_positions[timestep] = np.random.normal(inferred_target_positions[timestep-1], agent_movement_noise) #Change this to make explicit the active inference

    #Sample the variance of the target's random walk
    target_variances[timestep] = np.random.gamma(gamma_params[0], gamma_params[1])
    #Sample the position of the target
    target_positions[timestep] = np.random.normal(target_positions[timestep-1], np.sqrt(target_variances[timestep]))

    #The agent observes (noisily) its own position
    observed_agent_positions[timestep] = np.random.normal(agent_positions[timestep], agent_position_observation_noise)
    #The agent assumes its observed own position is its real position 
    inferred_agent_positions[timestep] = observed_agent_positions[timestep] #Change this to add inference on own position
    
    #The agent observes (noisily) the target's position
    observed_target_positions[timestep] = np.random.normal(target_positions[timestep], target_position_observation_noise)
    #Input data to the hgf to infer position
    hgf_target_position.input(observed_target_positions[timestep])
    #Input squared data to the hgf to infer noise
    hgf_target_noise.input(observed_target_positions[0]**2)
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