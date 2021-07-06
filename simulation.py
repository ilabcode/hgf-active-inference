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
# Think further ahead (time horizon)
# Add softmax or sampling to the decision
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
# Exploring parameter space
#    Noise
#    Drift
#    Goal priors

#To Do:
### Be explicit: predictive posterior for all observation modalities given control state


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
from math import ceil
from scipy.stats import t as students_t
from scipy.stats import norm
from scipy.signal import convolve

#observed position: as points
#Size up a bit

#Three plots in total
#Remove generating mean from circle plot
#Have gif on the line plot with moving dots
#Green is observed target position, blue is observed agent position

################
#-- Settings --#
################

#-- Plotting --#
#Set whether to generate a gif
make_gif = False
#Speed of the gif
gif_frame_duration = 0.6
#Scaling parameter for plotting 
circle_scale = 10

#Predictive posterior plotted confidence intervals
pp_ci_lower = 0.16
pp_ci_upper = 0.84

#Probability interval for the agent's action
action_probability_interval_lower = 0.45
action_probability_interval_upper = 0.55

#-- Simulation --#
#number of timesteps
n_timesteps = 200

#-- Environment --#
#The distribution which the target volatility is sampled from
volatility_vector = [10,120,10,120,10]
#The drift of the target (corresponding to rho1 in the hgf)
target_position_drift = 0.5
#The initial position of agent and target (correpsonding to initial_mu1 in the hgf)
init_target_position = 0
init_agent_position = 0

#-- Agent --#
#Noise when the agent observes the target's and its own position (root of the exponentiated omega_input in the hgf)
target_position_observation_noise = np.sqrt(np.exp(4))
agent_position_observation_noise = 0

#Parameters for the hgf
hgf_input_noise = 4
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


#The agent's goal prior about distance to target
gp_mean, gp_standard_deviation = [0, #mean
                                  5] #standard deviation

#Sets whether the goal prior is over inferred positions or observations
preference_modality = 'observation'
#preference_modality = 'position'

#################
#-- Functions --#
#################

#-- Helper function for plotting action probabilities --#
def plot_action_prob_interval_upper(predictive_posterior):
    
    convolved_probabilities, possible_actions = get_expected_surprisal(predictive_posterior, gp_mean, gp_standard_deviation)

    mean = possible_actions[ np.argmax(convolved_probabilities) ]
    lower = possible_actions[ np.cumsum( np.log( convolved_probabilities)  ) / sum( np.log( convolved_probabilities) ) > action_probability_interval_lower][0]
    upper = possible_actions[ np.cumsum( np.log( convolved_probabilities)  ) / sum( np.log( convolved_probabilities) ) > action_probability_interval_upper][0]

    return {'mean':mean, 'lower':lower, 'upper':upper}


#-- plotting function for the GIF --#
def save_plot_single_timestep(  timepoint,
                                agent_position,
                                observed_agent_position,
                                inferred_agent_position,
                                target_volatilities,
                                target_position,
                                observed_target_position,
                                inferred_target_position,
                                agent_surprisal,
                                predictive_posterior,
                                circle_scale):

    #Make filename and append it to the list
    filename = 'p{}'.format(timepoint)
    plots_filenames.append(filename)

    #Create circle for plotting
    circle = plt.Circle((0, 0), 1, fill = False)
    #Make plot with circle
    fig, ax = plt.subplots()
    ax.add_artist(circle)
    #Add point for observed target position
    plt.scatter(np.cos(observed_target_position/circle_scale), np.sin(observed_target_position/circle_scale),
                c='green', marker = 'X')
    #Add point for observed agent position
    plt.scatter(np.cos(observed_agent_position/circle_scale), np.sin(observed_agent_position/circle_scale),
                c='blue')
    #Set axes
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])

    #Save the figure as the filename
    plt.savefig('gifpics/' + filename)
    plt.close()


#Convolution function
def get_expected_surprisal(predictive_posterior, gp_mean, gp_standard_deviation):
    
    #Define the goal prior relative to the expectation of the predictive posterior
    goal_prior_absolute = norm(gp_mean + predictive_posterior.stats('m'), gp_standard_deviation)
    
    #Get discretized coordinates that cover both distributions
    possible_actions = np.linspace(
        min(predictive_posterior.ppf(0.001), goal_prior_absolute.ppf(0.001)),
        max(predictive_posterior.ppf(0.999), goal_prior_absolute.ppf(0.999)),
        10000
        ) #change this when we have finite state space (i.e. agent moves on a circle) to cover the whole state space
    
    #Get probability densities for each x coordinate
    predictive_posterior_discrete = predictive_posterior.pdf(possible_actions)
    goal_prior_absolute_discrete = goal_prior_absolute.pdf(possible_actions)

    #Convolve the two functions
    convolved_probabilities = convolve(predictive_posterior_discrete, goal_prior_absolute_discrete, mode='same') / sum(goal_prior_absolute_discrete)

    #Return x coordinates and convolved probability densities
    return convolved_probabilities, possible_actions



##################
#-- Simulation --#
##################

#-- Setup --#
#Make empty arrays for population
agent_actions = np.empty([n_timesteps]) * np.nan
agent_positions = np.empty([n_timesteps]) * np.nan
observed_agent_positions = np.empty([n_timesteps]) * np.nan
inferred_agent_positions = np.empty([n_timesteps]) * np.nan
target_volatilities = np.empty([n_timesteps]) * np.nan
target_positions = np.empty([n_timesteps]) * np.nan
observed_target_positions = np.empty([n_timesteps]) * np.nan
inferred_target_positions = np.empty([n_timesteps]) * np.nan
agent_surprisal = np.empty([n_timesteps]) * np.nan
predictive_posteriors = [None] * n_timesteps

current_volatility = volatility_vector[0]

#For each timestep
for timestep in range(n_timesteps):
    
    #On the first timestep
    if timestep == 0:
        #Set agent to its initial position
        agent_positions[timestep] = init_agent_position
        #Set the target to its initial position
        target_positions[timestep] = init_target_position
    
    #On other timesteps
    else:
        #-- Action step --#
        #Convolve the goal prior and the predictive posterior to get expected probability for each potential place to move
        convolved_probabilities, possible_actions = get_expected_surprisal(predictive_posteriors[timestep-1], gp_mean, gp_standard_deviation)
        #The agent moves deterministically to the place with the lowest expected surprisal
        agent_actions[timestep] = possible_actions[ np.argmin( -np.log( convolved_probabilities ) ) ] #Change this to sample action from expected surprisal (or make softmax)

        #-- Environment step --#
        #The agent's posoition is fully determined by its action
        agent_positions[timestep] = agent_actions[timestep] #Change this to make movements non-determinstic

        #This if at evenly spaced times throughout the timeseries
        if n_timesteps % round(n_timesteps / len(volatility_vector)) == 0:
            #So that the volatility goes through the volatility_vector
            current_volatility = volatility_vector[ ceil( timestep/round( n_timesteps / len(volatility_vector) ) ) -1 ]
        
        #Save volatility trial-by-trial
        target_volatilities[timestep] = current_volatility

        #Sample the position of the target from a normal distribution with a drift on the mean
        target_positions[timestep] = np.random.normal(target_positions[timestep-1] + target_position_drift, np.sqrt(target_volatilities[timestep]))
        
        
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

    #Get the distance to the target in the modality that the goal prior is over
    if preference_modality == 'position':
        distance_target_agent = inferred_agent_positions[timestep] - inferred_target_positions[timestep]
    elif preference_modality == 'observation':
        distance_target_agent = observed_agent_positions[timestep] - observed_target_positions[timestep]
    
    #Get the surprisal relative to the goal prior
    agent_surprisal[timestep] = - np.log( norm(gp_mean, gp_standard_deviation).pdf(distance_target_agent))


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
#Combine into dataframe for easy plotting
plotting_df =   pd.DataFrame(list(zip(
                    agent_actions,
                    [plot_action_prob_interval_upper(x)['mean'] for x in predictive_posteriors],
                    [plot_action_prob_interval_upper(x)['lower'] for x in predictive_posteriors],
                    [plot_action_prob_interval_upper(x)['upper'] for x in predictive_posteriors],
                    agent_positions,
                    inferred_agent_positions,
                    observed_agent_positions,
                    target_volatilities,
                    [np.exp(x) for x in hgf_target_position.x2.mus],
                    [np.exp(1/x) for x in hgf_target_position.x2.pis],
                    target_positions,
                    inferred_target_positions,
                    [1/x for x in hgf_target_position.x1.pis],
                    observed_target_positions,
                    [float(pp.stats('m')) for pp in predictive_posteriors],
                    [pp.ppf(pp_ci_lower) for pp in predictive_posteriors],
                    [pp.ppf(pp_ci_upper) for pp in predictive_posteriors],   
                    agent_surprisal
                )), 
                columns=[  
                    'agent_actions',
                    'agent_action_mean',
                    'agent_action_lower',
                    'agent_action_upper',
                    'agent_positions',
                    'inferred_agent_positions',
                    'observed_agent_positions',
                    'target_volatilities',
                    'inferred_target_volatilities',
                    'uncertainty_target_volatilities',
                    'target_positions',
                    'inferred_target_positions',
                    'uncertainty_target_positions', 
                    'observed_target_positions',
                    'pp_mean',
                    'pp_ci_lower',
                    'pp_ci_upper',
                    'agent_surprisal'
                ])

#Shift all the predictive measures to the next timestep
for predictive_measure in ['pp_mean', 'pp_ci_lower', 'pp_ci_upper', 'agent_action_mean', 'agent_action_lower', 'agent_action_upper']:
    plotting_df[predictive_measure] = np.roll(plotting_df[predictive_measure], shift=1)
    plotting_df[predictive_measure][0] = np.nan

#Add timestep column
plotting_df['timestep'] = plotting_df.index + 1



#-- HGF inference of position --#
plt.figure(figsize=(12,5))
plt.title('Prediction and following of target position')
plt.ylabel('Position')
plt.xlabel('Timestep')

ax1 = plotting_df.observed_agent_positions.plot(color='blue', grid=True, label = 'Observed Agent Position', zorder = 1)
plt.scatter(color = 'green', s = 4, x = plotting_df.timestep, y = plotting_df.observed_target_positions, label = 'Observed Target Position', zorder = 20)

plt.fill_between(   plotting_df.timestep,
                    plotting_df.inferred_target_positions - plotting_df.uncertainty_target_positions,
                    plotting_df.inferred_target_positions + plotting_df.uncertainty_target_positions,
                    color = 'teal', alpha = .2)

plt.fill_between(   plotting_df.timestep,
                    plotting_df.pp_ci_lower,
                    plotting_df.pp_ci_upper,
                    color = 'lightgreen', alpha = .2)

h1, l1 = ax1.get_legend_handles_labels()
plt.legend(l1+['Inference uncertainty'] + ['Predictive uncertainty'], loc=2)
plt.show()




#-- HGF inference of volatility --#
plt.figure(figsize=(12,5))
plt.title('HGF inference on target volatility')
plt.ylabel('Volatility')
plt.xlabel('Timestep')

ax1 = plotting_df.target_volatilities.plot(color='indigo', grid=True, label='Actual volatility')
ax1 = plotting_df.inferred_target_volatilities.plot(color='darkviolet', grid=True, label='Inferred volatility')

plt.fill_between(   plotting_df.timestep,
                    plotting_df.inferred_target_volatilities - plotting_df.uncertainty_target_volatilities,
                    plotting_df.inferred_target_volatilities + plotting_df.uncertainty_target_volatilities,
                    color = 'darkviolet', alpha = .2)

h1, l1 = ax1.get_legend_handles_labels()
plt.legend(l1+['Inference uncertainty'], loc=2)
plt.show()




#-- Goal Prior surprisal --#
plt.figure(figsize=(12,5))
plt.title('Goal prior surprisal')
plt.ylabel('Surprisal')
plt.xlabel('Timestep')

ax1 = plotting_df.agent_surprisal.plot(color='pink', grid=True, label='Agent Surprisal')
ax1 = plotting_df.agent_surprisal.rolling(5).mean().plot(color='darkred', grid=True, label='Moving Average')

h1, l1 = ax1.get_legend_handles_labels()
plt.legend(l1, loc=2)
plt.show()




# #-- HGF inference of position --#
# plt.figure(figsize=(12,5))
# plt.title('HGF inference on target position')
# plt.ylabel('Position')
# plt.xlabel('Timestep')

# ax1 = plotting_df.observed_target_positions.plot(color='darkgrey', grid=True, label='Observed Position')
# ax1 = plotting_df.target_positions.plot(color='darkblue', grid=True, label='Actual Position')
# ax1 = plotting_df.inferred_target_positions.plot(color='teal', grid=True, label='Inferred Position')

# plt.fill_between(   plotting_df.observed_target_positions.index,
#                     plotting_df.inferred_target_positions - plotting_df.uncertainty_target_positions,
#                     plotting_df.inferred_target_positions + plotting_df.uncertainty_target_positions,
#                     color = 'teal', alpha = .2)

# h1, l1 = ax1.get_legend_handles_labels()
# plt.legend(l1+['Inference uncertainty'], loc=2)
# plt.show()


# #-- Predictive posterior --#
# plt.figure(figsize=(12,5))
# plt.title('Predictive posterior over observations of target')
# plt.ylabel('Position')
# plt.xlabel('Timestep')

# ax1 = plotting_df.observed_target_positions.plot(color='darkgrey', grid=True, label='Observed Position')
# ax1 = plotting_df.pp_mean.plot(color='teal', grid=True, label='Predictive Posterior Mean')

# plt.fill_between(   plotting_df.observed_target_positions.index,
#                     plotting_df.pp_ci_lower,
#                     plotting_df.pp_ci_upper,
#                     color = 'teal', alpha = .2)

# h1, l1 = ax1.get_legend_handles_labels()
# plt.legend(l1+['{}% Confidence Intervals'.format(
#                 round((pp_ci_upper-pp_ci_lower)*100)
#                 )], loc=2)
# plt.show()


# #-- Agent actions --#
# plt.figure(figsize=(12,5))
# plt.title('Agent actions')
# plt.ylabel('Position')
# plt.xlabel('Timestep')

# ax1 = plotting_df.observed_target_positions.plot(color='darkgrey', grid=True, label='Observed Target Position')
# ax1 = plotting_df.agent_action_mean.plot(color='teal', grid=True, label='Highest Probability Action')
# ax1 = plotting_df.agent_actions.plot(color='teal', grid=True, label='Agent Action')

# plt.fill_between(   plotting_df.observed_target_positions.index,
#                     plotting_df.agent_action_lower,
#                     plotting_df.agent_action_upper,
#                     color = 'teal', alpha = .2)

# h1, l1 = ax1.get_legend_handles_labels()

# plt.legend(l1 
# + ['Action {}% probability interval'.format(
#                 round((action_probability_interval_upper-action_probability_interval_lower)*100)
#                 )]
#                 , loc=2)
# plt.show()


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
                                    target_volatilities[timestep],
                                    target_positions[timestep],
                                    observed_target_positions[timestep],
                                    inferred_target_positions[timestep],
                                    agent_surprisal[timestep],
                                    predictive_posteriors[timestep],
                                    circle_scale)

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
