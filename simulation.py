#Questions:
# Parameters
#  Which hgf parameters?
#  Which generative process parameters? Especially gamma parameters.
#  Which noise parameters?
# Active Inference
#  (Explicit) goal prior
#  Inferring own actions
# Is all the sufficient statistics now filtered ? 

#Extensions
# Plot predictive distribution on circle
# Infer own position
# Use width of predictive posterior t-distribution to inform the learning_rate / lambda in the action
# Make things happen on an actual circle (modulo or radians)
# Add a parent to the volatility (or just blocks with varying volatility)
# Learn model parameters
# Make the agent only be able to (prefer to) move some length
# Make the agent's observations depend on the distance to the target (or another kind of epistemic action component)
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
from scipy.stats import t as students_t


#-- Settings --#
#Plotting
gif_frame_duration = 0.3

#Simulation
n_timesteps = 200

#Environment
gamma_params = [2,0.05]
init_target_position = 0
target_position_drift = 0.5 #should be equal to rho1 for the agent to perform well

#Agent
agent_position_observation_noise = 0
target_position_observation_noise = 0.5
agent_movement_noise = 0.1
init_agent_position = 0

action_learning_rate = 0.9

hgf_input_noise = -1
hgf_input_precision = np.exp(-hgf_input_noise)
hgf_target_position = hgf.StandardHGF(  initial_mu1=0,
                                        initial_pi1=1e4,
                                        omega1=2,
                                        kappa1=1,
                                        initial_mu2=1,
                                        initial_pi2=1e1,
                                        omega2=-2,
                                        omega_input=hgf_input_noise,
                                        rho1 = 0.5, #should be equal to target_position_drift for the agent to perform well
                                        rho2 = 0)


#-- Functions --#
def save_plot_single_timestep(  timepoint,
                                agent_position,
                                observed_agent_position,
                                inferred_agent_position,
                                target_variance,
                                target_position,
                                observed_target_position,
                                inferred_target_position,
                                agent_surprisal,
                                predictive_posterior):

    #Make filename and append it to the list
    filename = 'p{}'.format(timepoint)
    plots_filenames.append(filename)

    #Create circle for plotting
    circle = plt.Circle((0, 0), 1, fill = False)
    #Make plot with circle
    fig, ax = plt.subplots()
    ax.add_artist(circle)
    #Add point for target position
    plt.scatter(np.cos(target_position), np.sin(target_position),
                c='green', marker = 'X')
    #Add point for observed target position
    plt.scatter(np.cos(observed_target_position), np.sin(observed_target_position),
                c='lightgreen', marker = 'X', alpha=0.4)
    #Add point for agent position
    plt.scatter(np.cos(agent_position), np.sin(agent_position),
                c='blue')
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
predictive_posteriors = [None] * n_timesteps
agent_surprisal = np.empty([n_timesteps])


#For each timestep
for timestep in range(n_timesteps):
    
    #-- Movement step --#
    #On the first timestep
    if timestep == 0:
        #Set agent to its initial position
        agent_positions[timestep] = init_agent_position
        #Set the target to its initial position
        target_positions[timestep] = init_target_position
    
    #On other timesteps
    else:
        #The agent moves to somewhere between its last position and the expected position of the target, weighted by a learning rate
        #This implicitly minimizes surprise over being far from the target
        agent_positions[timestep] = inferred_agent_positions[timestep-1] + action_learning_rate * (predictive_posteriors[timestep-1].stats('m') - inferred_agent_positions[timestep-1])

        #Sample the variance of the target's random walk
        target_variances[timestep] = np.random.gamma(gamma_params[0], gamma_params[1])
        #Sample the position of the target from a normal distribution with a drift on the mean
        target_positions[timestep] = np.random.normal(target_positions[timestep-1] + target_position_drift, np.sqrt(target_variances[timestep]))


    #-- Inference step --#
    #The agent observes (possibly noisily) its own position
    observed_agent_positions[timestep] = np.random.normal(agent_positions[timestep], agent_position_observation_noise)
    #The agent assumes its observed own position is its real position 
    inferred_agent_positions[timestep] = observed_agent_positions[timestep] #Change this to add inference on own position

    #The agent observes (noisily) the target's position
    observed_target_positions[timestep] = np.random.normal(target_positions[timestep], target_position_observation_noise)
    #Input the observation to the hgf
    hgf_target_position.input(observed_target_positions[timestep])
    #Extract the inferred position of the target from the hgf
    inferred_target_positions[timestep] = hgf_target_position.x1.mus[-1]

    #Get the surprisal / exact free energy / squared error
    agent_surprisal[timestep] = (inferred_agent_positions[timestep] - inferred_target_positions[timestep])**2 #Change this to refer explicitly to goal prior


    #-- Prediction step --#
    #Get the prediction of the target's position on next trial
    prediction_target_position = hgf_target_position.x1.muhats[-1]
    #And the precision of that prediction
    prediction_precision_target_position = hgf_target_position.x1.pihats[-1]

    #Get the implied nu
    nu = prediction_precision_target_position / hgf_input_precision

    #Get the predictive posterior for the position of the target
    predictive_posteriors[timestep] = students_t(   df = nu + 1,
                                                    loc = prediction_target_position,
                                                    scale = 1 / prediction_precision_target_position)



#-- Plots --#
#variance versus inferred variance [This might be wrong]
pd.Series(target_variances).plot()
pd.Series(np.exp(hgf_target_position.x2.mus)).plot()

#position versus inferred versus observed target position
pd.Series(target_positions).plot()
pd.Series(inferred_target_positions).plot()
pd.Series(observed_target_positions).plot()

#position versus inferred versus observed agent position
pd.Series(agent_positions).plot()
pd.Series(inferred_agent_positions).plot()
pd.Series(observed_agent_positions).plot()

#inferred own versus target positions with surprisal
pd.Series(inferred_target_positions).plot()
pd.Series(inferred_agent_positions).plot()
pd.Series(agent_surprisal).plot()



#-- GIF --#
#Make GIF single plots
plots_filenames = []

for timestep in range(n_timesteps):
    #Make and save plot of that timepoint
    save_plot_single_timestep(  timestep,
                                agent_positions[timestep],
                                observed_agent_positions[timestep],
                                inferred_agent_positions[timestep],
                                target_variances[timestep],
                                target_positions[timestep],
                                observed_target_positions[timestep],
                                inferred_target_positions[timestep],
                                agent_surprisal[timestep],
                                predictive_posteriors[timestep])


#Build together into a GIF
with imageio.get_writer('chasing_target.gif', mode='I', duration = gif_frame_duration) as writer:
    for filename in plots_filenames:
        image = imageio.imread('gifpics/' + filename + '.png')
        writer.append_data(image)
        os.remove('gifpics/' + filename + '.png')
