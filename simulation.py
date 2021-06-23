#Questions:
# Is surprise only scored relative to the goal prior ? 
#   Ie is there no epistemic component ? - REMARK ON THIS
# Same with expected surprise
#   Compared to the MDP version
# Should the agent move with a softmax or to the lowest epxected surprise ?
# Would it be nice to compare epxlicitly to MDP's ?
# Goal prior:
#   On one hand: range of places to move (x_coordinates);
#   on other hand: range of places where the target might be;
#   on third hand: goal prior over distances between those two;
#   On fourth hand: predictive posterior distribution over placea the target might go;
#   On fifth hand: expected surprisal distribution over places the agent might go
#   Should we make it into a proper 2D distribution?

#Extensions:
# Plot predictive distribution on circle
# Add a parent to the volatility (or just blocks with varying volatility)
# Add preferences for movement length
# Infer own position
# Make things happen on an actual circle (modulo or radians)
# Make the agent's observations depend on the distance to the target (or another kind of epistemic action component)
# Make the agent able to affect the hidden states (maybe the target's drift?)
# Learn hgf parameters (training period or online)
# Select prior preferences through evolution
# Making the target an agent that tries to run away

#Others:
# Make a way of running the ghgf forward
# Instead of drawing volatility from gamma, draw from gaussian with variance = np.exp(kappa2*x2+omega2)



#############
#-- Setup --#
#############

import sys
sys.path.append('ghgf')
import hgf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from scipy.stats import t as students_t
from scipy.stats import gamma
from scipy.stats import norm
from scipy.signal import convolve



################
#-- Settings --#
################

#-- Plotting --#
#Set whether to generate a gif
make_gif = True
#Speed of the gif
gif_frame_duration = 0.3

#-- Simulation --#
#number of timesteps
n_timesteps = 200

#-- Environment --#
#The distribution which the target volatility is sampled from
volatility_parent = gamma(a = 2, scale = 0.05)
#The drift of the target (corresponding to rho1 in the hgf)
target_position_drift = 0.5
#The initial position of agent and target (correpsonding to initial_mu1 in the hgf)
init_target_position = 0
init_agent_position = 0


#-- Agent --#

#Noise when the agent observes the target's and its own position (root of the exponentiated omega_input in the hgf)
target_position_observation_noise = np.sqrt(np.exp(-1))
agent_position_observation_noise = 0

#Parameters for the hgf
hgf_input_noise = -1
hgf_input_precision = np.exp(-hgf_input_noise)

hgf_target_position = hgf.StandardHGF(  initial_mu1=0,
                                        initial_pi1=1e4,
                                        omega1=0.5,
                                        kappa1=1,
                                        initial_mu2=1,
                                        initial_pi2=1e1,
                                        omega2=-2,
                                        omega_input=hgf_input_noise,
                                        rho1 = 0.5,
                                        rho2 = 0)

#The agent's goal prior about inferred distance to target
gp_mean, gp_standard_deviation = [0, #mean
                                  1] #standard deviation

#################
#-- Functions --#
#################

#-- plotting function for the GIF --#
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


#Convolution function
def get_expected_surprisal(predictive_posterior, goal_prior):
    
    #Get discretized coordinates that cover both distributions
    x_coord = np.linspace(
        min(predictive_posterior.ppf(0.001), goal_prior.ppf(0.001)),
        max(predictive_posterior.ppf(0.999), goal_prior.ppf(0.999)),
        10000
        ) #change this when we have finite state space (i.e. agent moves on a circle) to cover the whole state space
    
    #Get probability densities for each x coordinate
    dens1 = predictive_posterior.pdf(x_coord)
    dens2 = goal_prior.pdf(x_coord)

    #Convolve the two functions
    convolved_probabilities = convolve(dens1, dens2, mode='same') / sum(dens2)

    #Transform probabilities to surprisal
    expected_surprisal = -np.log(convolved_probabilities)

    #Return x coordinates and convolved probability densities
    return expected_surprisal, x_coord



##################
#-- Simulation --#
##################

#-- Setup --#
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

        #Define the goal prior relative to the expectation of the predictive posterior
        goal_prior = norm(gp_mean + predictive_posteriors[timestep-1].stats('m'), gp_standard_deviation)
        #Convolve the goal prior and the predictive posterior to get expected probability for each potential place to move
        expected_surprisal, x_coord = get_expected_surprisal(predictive_posteriors[timestep-1], goal_prior)
        #The agent moves deterministically to the place with the lowest expected surprisal
        agent_positions[timestep] = x_coord[ np.argmin(expected_surprisal) ] #sample action from expected surprisal

        #Sample the variance of the target's random walk
        target_variances[timestep] = volatility_parent.rvs(1)
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

    #Get the inferred distance to the target
    inferred_distance = inferred_agent_positions[timestep] - inferred_target_positions[timestep]
    #Get the surprisal relative to the goal prior
    agent_surprisal[timestep] = - np.log( norm(gp_mean, gp_standard_deviation).pdf(inferred_distance))


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




################
#-- Plotting --#
################

#-- Simple plots --#
#variance versus inferred variance [This might be wrong]
pd.Series(target_variances).plot()
pd.Series(np.exp(hgf_target_position.x2.mus)).plot()

#position versus inferred versus observed target position
pd.Series(target_positions).plot()
pd.Series(inferred_target_positions).plot()
pd.Series(observed_target_positions).plot()

#inferred own versus target positions with surprisal
pd.Series(inferred_target_positions).plot()
pd.Series(inferred_agent_positions).plot()
pd.Series(agent_surprisal).plot()



#-- GIF --#
if make_gif:
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











############
#-- Misc --#
############

if False:
    #Function for plotting scipy distributions
    def plot_distribution(dist):
        fig, ax = plt.subplots(1, 1)

        x_coord = np.linspace(dist.ppf(0.001),
                        dist.ppf(0.999), 10000)

        density = dist.pdf(x_coord)

        ax.plot(x_coord, density, 'b-', lw=3, alpha=0.6, label='pdf')

    gp_mean, gp_standard_deviation = goal_prior['distance_to_target']

    #Get the densities and x coordinates for each of the functions
    predictive_posterior = predictive_posteriors[timestep]
    pp_x_coord = np.linspace(predictive_posterior.ppf(0.001), predictive_posterior.ppf(0.999), 10000)
    pp_density = predictive_posterior.pdf(pp_x_coord)

    goal_prior = norm(gp_mean + predictive_posterior.stats('m'), gp_standard_deviation)
    gp_x_coord = np.linspace(goal_prior.ppf(0.001), goal_prior.ppf(0.999), 10000)
    gp_density = goal_prior.pdf(gp_x_coord)


    #Convolve them
    expected_probability, x_coord = get_expected_surprisal(predictive_posterior, goal_prior)

    #Find the place with the lowest expected surprise
    x_coord[np.argmin(-np.log(expected_probability))]


    #Plot them
    fig, ax = plt.subplots(1, 1)

    ax.plot(pp_x_coord, pp_density, 'b-', lw=3, alpha=0.3, label='pdf')
    ax.plot(gp_x_coord, gp_density, 'y-', lw=3, alpha=0.3, label='pdf')
    ax.plot(x_coord, expected_probability,'r-', lw=3, alpha=0.6, label='pdf')

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_coord, expected_probability,'r-', lw=3, alpha=0.6, label='pdf')
    ax.plot(x_coord, -np.log(expected_probability),'r-', lw=3, alpha=0.6, label='pdf')


    #Find the place with the lowest expected surprise
    x_coord[np.argmin(-np.log(expected_probability))]
