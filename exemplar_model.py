#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:37:36 2025

@author: joncparamore
"""
import random
import numpy as np
import scipy.optimize
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import copy

# Random seeds to recall same values
#random.seed(89)
#np.random.seed(113)

start_time = time.perf_counter()
####Model Input: Dictionary of lists with four word forms and their starting nasalance values.

def seed_clouds():
    #Nasalance values must be between 0 and 1.
    #four categories correspond to oral (ORAL), nasal (NAS), oral with nasal suffix (ORAL-N), and nasal with nasal suffix (NAS-N)
    #each category has 2 or more semantic forms. 
    exemplars = {'ORAL': {'tɑɑ': [0.2], 'kii': [0.2]}, 
                       'NAS': {'tɑ̃ɑ̃': [0.6], 'kĩĩ': [0.6]}, 
                       'ORAL-N': {'tɑɑ-n': [0.2], 'kii-n': [0.2]}, 
                       'NAS-N':{'tɑ̃ɑ̃-n': [0.6], 'kĩĩ-n': [0.6]}}
    # Make clouds for each form, each cloud has 5 initial samples
    # The values of the 5 samples are decided by adding uniform noise
    sd = 0.025
    num_samples = 1
    exemplar_clouds = {}
    for cat, forms in exemplars.items():
        exemplar_clouds[cat] = {}
        for form, val in forms.items():
            samples = np.random.normal(loc=val, scale=sd, size = num_samples)
            samples = np.clip(samples, 0, 1)
            exemplar_clouds[cat][form] = samples.tolist()
    return exemplar_clouds
exemplar_clouds = seed_clouds()

#define morphological relationships between forms in exemplar_clouds and add them to dict semantic_pairs for quick lookup of relations
semantic_relations = [('tɑɑ', 'tɑɑ-n'), ('kii', 'kii-n'), ('tɑ̃ɑ̃', 'tɑ̃ɑ̃-n'), ('kĩĩ', 'kĩĩ-n')]
semantic_pairs = {word: (word, related) for word, related in semantic_relations}
semantic_pairs.update({related: (related, word) for word, related in semantic_relations})

#define category (same case) relationships between forms in exemplar_clouds and add them to dict category_pairs for quick lookup of relations
category_relations = [('tɑɑ', 'kii'),
                      ('tɑ̃ɑ̃', 'kĩĩ'),
                      ('tɑɑ-n', 'kii-n'),
                      ('tɑɑ-n', 'tɑ̃ɑ̃-n'),
                      ('tɑɑ-n', 'kĩĩ-n'),
                      ('kii-n', 'tɑ̃ɑ̃-n'),
                      ('kii-n', 'kĩĩ-n'),
                      ('tɑ̃ɑ̃-n', 'kĩĩ-n')]
category_pairs = {}
#iterate through each unique category in category_relations
for form1, form2 in category_relations:
    #Add form2 to form1's list
    category_pairs.setdefault(form1, []).append(form2)
    #Add form1 to form2's list
    category_pairs.setdefault(form2, []).append(form1)

# Precompute a direct lookup dictionary to find a form's category efficiently
form_to_category = {word: cat for cat, forms in exemplar_clouds.items() for word in forms}

####Random noise function - ensures no two exemplars are the same
def random_noise(exemplar_val, exemplar_form, exemplar_cat, clouds):
    nas_mean = exemplar_val
    sd = 0.025     #standard deviation so that +- 2 sd is .035
    #generate a random noise value to add to the exemplar value
    exemplar_val_with_rand_noise = np.random.normal(loc=nas_mean, scale=sd)
    #set bounds of new_exemplar_val at (0, 1)
    return np.clip(exemplar_val_with_rand_noise, 0, 1)

####constraints - functions to generate penalties for different constraints
#phonetic nasalization constraint: penalize pre-N vowels that aren't very nasalized
def channel_constraint(exemplar_val, exemplar_cat, clouds):
    nas_mean = sum(np.mean(clouds['NAS'][form]) for form in clouds['NAS']) / len(clouds['NAS']) #determine mean value of nasal cloud
    if exemplar_cat.endswith('-N'):
        return (nas_mean - exemplar_val)**2
    else:
        return 0.0

#morphological bias:penalize morphologically related forms that are far from each other in nasalance
def morphological_constraint(exemplar_val, exemplar_form, exemplar_cat, clouds):
    if not exemplar_cat.endswith('-N'): #ensuring base forms are not drawn toward genitive forms by assigning a 0.0 penalty
        return 0.0
    related_form = semantic_pairs[exemplar_form][1] #assign morphologically related form to variable
    related_cat = form_to_category[related_form] 
    sem_base_val = np.mean(clouds[related_cat][related_form])
    return (sem_base_val - exemplar_val)**2

#category constraint: penalize forms with same inflectional category that are far from each other in nasalance
#But see Burzio (2000, p.269) for reasons to believe ORAL and NAS don't attract each other. 
def category_constraint(exemplar_val, exemplar_form, clouds):
    category_penalty = 0.0
    related_forms = category_pairs[exemplar_form]
    for form in related_forms:
        for cat, forms in clouds.items():
            #find category the related form belongs to and calculate exemplar mean of that form
            if form in forms:
                related_form_mean = np.mean(clouds[cat][form])
                break
        category_penalty += (related_form_mean - exemplar_val)**2
    return category_penalty

#Define the frequency of each form. What approximate proportion of exemplars do you want from each form?
category_frequencies = [1, 1, 1, 1] #there are four categories in exemplar_clouds: [oral, nasal, oral-n, nasal-n]
form_frequencies = [1, 1] #there are two forms from each category in exemplar_clouds

#function to calculate constraint scalars based on current frequency of exemplar categories
def get_constraint_scalars(form, category, clouds):
    #Step 1: Determine Frequency of morphologically related form (e.g., tɑɑ and tɑɑ-n)
    morph_base_form = semantic_pairs[form][1]
    morph_base_cat = form_to_category[morph_base_form]
    morph_base_form_freq = len(clouds[morph_base_cat][morph_base_form])
    
    #Step 2: Determine total Frequency of inflectionally related forms (e.g., for tɑɑ-n: kii-n, kII-n, etc. are inflectionally related with genitive suffix)
    cat_relation_freq = 0
    for word in category_pairs[form]:
        form_cat = form_to_category[word]
        cat_relation_freq += len(clouds[form_cat][word])
    
    #Step 3: Determine frequency of the current exemplar_form
    exemplar_form_freq = len(clouds[category][form])
    
    #Step 4: Determine the penalty scalars based on frequency (static for channel_constraint at this point)
    channel_constraint_scalar = 1
    morphological_constraint_scalar = morph_base_form_freq / exemplar_form_freq * 1 #multiplied by 2 to indicate the stronger influence of paradigmatically related forms
    category_constraint_scalar = cat_relation_freq / exemplar_form_freq
    
    return {
        'morph_base_form': morph_base_form,
        'morph_base_cat': morph_base_cat,
        'channel_scalar': channel_constraint_scalar,
        'morph_scalar': morphological_constraint_scalar,
        'category_scalar': category_constraint_scalar
    }

####Objective function - calculates penalty of current exemplar_val in relation to constraints and frequency of forms
def objective(exemplar_val, exemplar_form, exemplar_cat, clouds, constraint_pen_log):
    #Step 1: calculate constraint scalars using get_constraint_scalars function
    scalars = get_constraint_scalars(exemplar_form, exemplar_cat, clouds)
    channel_constraint_scalar = scalars['channel_scalar']
    morphological_constraint_scalar = scalars['morph_scalar']
    category_constraint_scalar = scalars['category_scalar']
    
    #Step 2: Calculate the penalties for the current exemplar
    channel_pen = channel_constraint(exemplar_val[0], exemplar_cat, clouds)
    morph_pen = morphological_constraint(exemplar_val[0], exemplar_form, exemplar_cat, clouds)
    category_pen = category_constraint(exemplar_val[0], exemplar_form, clouds)

    return (channel_pen*channel_constraint_scalar + 
            morph_pen*morphological_constraint_scalar) #+ 
            #category_pen*category_constraint_scalar)

####Model Process
def exemplar_accumulation(clouds, num_exemplars):
    
    #Initialize a list to log important info about each new exemplar: category, form, nasalance, constraint penalties
    constraint_penalty_log = []    
    # Initialize an empty list to store (iteration, nasalance, word_form) tuples for plotting
    data_points = []
    for cat, words in clouds.items():
        for word, values in words.items():
            for value in values:
                #add initial values to data_points
                data_points.append((0, value, word))
    #store categories in clouds dictionary as a list for fast lookup
    categories = list(clouds.keys())
    #store individual words in dict of values associated with the category as the key
    semantic_forms = {}
    for cat in categories:
        semantic_forms[cat] = []
        for form in clouds[cat]:
            semantic_forms[cat].append(form) 
    
    #develop num_exemplars new exemplars
    for i in range(num_exemplars):
        
        ##step 1: create a new exemplar
        #choose one exemplar category for production from clouds dictionary based on the prespecified category frequencies
        new_exemplar_cat = random.choices(categories, weights = category_frequencies, k = 1)[0]
        #choose one form from that category for production based on the frequency of each form in that category
        new_exemplar_form = random.choices(semantic_forms[new_exemplar_cat], weights = form_frequencies, k = 1)[0]
        ##make the mean value of that form the starting value for the new exemplar (entrenchment)
        new_exemplar_val = np.mean(clouds[new_exemplar_cat][new_exemplar_form])
        
        ##Step 2: Optimize the new exemplar
        #restricting how much the exemplar val can change from biases to +/-.01 to minimize objective
        lower_bound = max(0, new_exemplar_val - .075) #max movement of +/- .075 is arbitrary at this point but meant to avoid one fell swoop changes
        upper_bound = min(1, new_exemplar_val + .075)
        bounds = [(lower_bound, upper_bound)]
        #define objective parameters
        optimized_exemplar_val = scipy.optimize.minimize(fun = lambda x: objective(x, new_exemplar_form, new_exemplar_cat, clouds, constraint_penalty_log),
                                                         x0 = [new_exemplar_val],
                                                         method = 'SLSQP',
                                                         bounds = bounds,
                                                         options={'maxiter':1000, #limit the maximum number of iterations to 1000
                                                                  'ftol':1e-4, #Sets the function tolerance to 1e-4, meaning the optimizer will stop when changes in the function value are smaller than this threshold.
                                                                  'eps':1e-5, #sets step size of the Jacobian (step_size = gradient * Learning_Rate)
                                                                  })
        
        final_exemplar_val = optimized_exemplar_val.x[0]
        
        #compute and log penalty constraints for the optimized exemplar
        scalars = get_constraint_scalars(new_exemplar_form, new_exemplar_cat, clouds)
        channel_pen = channel_constraint(final_exemplar_val, new_exemplar_cat, clouds)
        morph_pen = morphological_constraint(final_exemplar_val, new_exemplar_form, new_exemplar_cat, clouds)
        category_pen = category_constraint(final_exemplar_val, new_exemplar_form, clouds)
        
        constraint_penalty_log.append({
            'category': new_exemplar_cat,
            'form': new_exemplar_form,
            'nasalance': final_exemplar_val,
            'iteration': i,
            'channel_pen_raw': channel_pen,
            'channel_pen_weighted': channel_pen * scalars['channel_scalar'],
            'morph_pen_raw': morph_pen,
            'morph_pen_weighted': morph_pen * scalars['morph_scalar'],
            'category_pen_raw': category_pen,
            'category_pen_weighted': category_pen * scalars['category_scalar'],
            })
        
        ##Step 3: add random noise to optimized exemplar
        final_exemplar_val = random_noise(final_exemplar_val, new_exemplar_form, new_exemplar_cat, clouds)
        
        ##Step 4: Categorize new exemplar into appropriate category
        clouds[new_exemplar_cat][new_exemplar_form].append(final_exemplar_val)
        
        #append new exemplar to data_points list for plotting
        data_points.append((i, final_exemplar_val, new_exemplar_form))
    ##Final Step: calculate mean of each form's exemplar cloud
    final_exemplar_means = {
    cat: {
        form: np.round(np.mean(clouds[cat][form][-50:]), 3)  # Compute mean & round, only for final 50 exemplars in each category. This models decay kind of
        for form in clouds[cat]  # Iterate over words in each category
    }
    for cat in categories  # Iterate over categories
    }
    #output is a dictionary of nested dictionaries for each form as key and the mean of that exemplar cloud as the value. 
    return final_exemplar_means, data_points, pd.DataFrame(constraint_penalty_log)

#function to plot the value that each constraint in the objective function is adding to the loss at each iteration. 
def plot_constraint_penalties():
    plt.show()
#run the model n times using the multiple_trials function: num_trials specifies how many iterations of exemplar_accumulation() you want to run.
def multiple_trials(clouds, num_exemplars, num_trials):
    
    #Step 1: Create an empty list to store the means for each form after each iteration of exemplar_accumulation()
    trial_means = []
    
    #Step 2: Run n trials of exemplar_accumulation, plotting each trial and appending the results to trial_means
    trial_num = 0
    while trial_num < num_trials:
        trial_num += 1
        
        #deep copy original exemplar_clouds to avoid modification of external cloud values
        trial_clouds = copy.deepcopy(clouds)
        
        final_means, data_points, constraint_penalties_df = exemplar_accumulation(trial_clouds, num_exemplars)
        for cat, forms in final_means.items():
            for form, mean_nasalance in forms.items():
                trial_means.append({
                    'category': cat,
                    'form': form,
                    'mean_nasalance': mean_nasalance,
                    'iteration': trial_num})
                
        #plot the exemplar nasalance values over time
        #updating label colors and markers
        color_map = {
            'tɑɑ': '#fdc700', 'kii': '#fdc700', 
            'tɑ̃ɑ̃': '#708090', 'kĩĩ': '#708090',
            'tɑɑ-n': '#006aad', 'kii-n': '#006aad',
            'tɑ̃ɑ̃-n': '#961E25', 'kĩĩ-n': '#961E25'}
        marker_map = {
            'tɑɑ': 'o', 'kii': 'o', 
            'tɑ̃ɑ̃': 'o', 'kĩĩ': 'o',
            'tɑɑ-n': '^', 'kii-n': '^',
            'tɑ̃ɑ̃-n': '^', 'kĩĩ-n': '^'
            }

        plt.figure(figsize = (16, 10), facecolor = '#E9E5DC')
        #preparing legend labels with mean values from final_means
        legend_labels = []

        for cat, words in final_means.items():
            for word, mean_val in words.items():
                #extract mean value and round to two decimals
                mean_rounded = round(mean_val, 2)
                legend_labels.append(f"{word} (μ: {mean_rounded})")
                
                #get iteration and nasalance values for this word
                iteration = [entry[0] for entry in data_points if entry[2] == word]
                nasalance = [entry[1] for entry in data_points if entry[2] == word]

                #Generating scatterplot
                plt.scatter(iteration, nasalance, label = word, alpha = 1, s = 100, color = color_map[word], marker = marker_map[word])

        plt.xlabel("Exemplar Production Iteration", fontweight = 'bold', fontsize = 32, color = "#003c6c", labelpad = 20)
        plt.ylabel("Nasalance", fontweight = 'bold', fontsize = 32, color = "#003c6c", labelpad = 20)
        plt.tick_params(axis='both', labelsize=24, colors="#003c6c")
        plt.title(f"Exemplar Changes in Nasalance Over Time, trial {str(trial_num)}", fontweight = 'bold', fontsize = 42, color = "#003c6c", pad = 25)
        plt.tight_layout()
        plt.gca().set_facecolor('#E9E5DC') #for changing background color of subplot
        legend = plt.legend(legend_labels, title = "words (μ nasalance)", 
                   title_fontproperties=font_manager.FontProperties(weight="bold", size=24),
                   bbox_to_anchor = (1, 1), loc = 'upper left', edgecolor = "#003c6c", fontsize = 24, 
                   labelcolor = '#003c6c', facecolor = '#E9E5DC')
        legend.get_title().set_color("#003c6c")
        plt.grid(False)
        ax = plt.gca()  # get current axes
        for spine in ax.spines.values():
            spine.set_edgecolor("#003c6c")
            spine.set_linewidth(2.5)
        plt.show()
        
        #add a new penalty that is the total penalty
        constraint_penalties_df['Total_Penalty'] = constraint_penalties_df['channel_pen_weighted'] + constraint_penalties_df['morph_pen_weighted'] + constraint_penalties_df['category_pen_weighted']
        #plot each constraint penalty value over time
        '''penalty_cols = [
            #'channel_pen_raw', 'morph_pen_raw', 'category_pen_raw',
            'channel_pen_weighted', 'morph_pen_weighted', 'category_pen_weighted', 'Total_Penalty'
            ]
        
        for penalty_type in penalty_cols:  
            plt.figure(figsize = (16, 10))#, facecolor = '#E9E5DC')
            plotted_categories = set()
            for _, row in constraint_penalties_df.iterrows():
                category = row['category']
                label = category if category not in plotted_categories else None
                plt.scatter(row['iteration'], row[penalty_type],
                            color=color_map[row['form']],
                            marker=marker_map[row['form']],
                            label = label,
                            alpha=0.7, s=50)
                plotted_categories.add(category)

            plt.xlabel('Iteration')
            plt.ylabel(f'{penalty_type}')
            plt.legend(title = 'Category')
            plt.title(f"Exemplar Changes in {penalty_type} Over Time, trial {str(trial_num)}", fontweight = 'bold', fontsize = 18)
            plt.tight_layout()
            plt.grid(False)
            plt.show()
        
        #plot all constraint penalties for ORAL-N
        oral_N_constraint_penalties = constraint_penalties_df[constraint_penalties_df['category'] == 'ORAL-N']
        plt.figure(figsize = (16, 10))#, facecolor = '#E9E5DC')
        for penalty_type in penalty_cols:
            plt.plot(oral_N_constraint_penalties['iteration'], oral_N_constraint_penalties[penalty_type],
                            label = penalty_type.replace('_', ' ').title(),
                            alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Penalty')
        plt.legend(title = 'Penalty Type')
        plt.title(f"Exemplar Changes in penalty values Over Time, trial {str(trial_num)}", fontweight = 'bold', fontsize = 18)
        plt.tight_layout()
        plt.grid(False)
        plt.show()'''
    #Step 3: Convert trial_means to df and return the results
    exemplar_trials = pd.DataFrame(trial_means)
    return exemplar_trials


exemplar_trials = multiple_trials(exemplar_clouds, num_exemplars=1000, num_trials=1)

end_time = time.perf_counter()
print(f"Execution time: {round(end_time - start_time, 2)} seconds/{round((end_time - start_time)/60, 2)} minutes")

exemplar_trials.to_csv("equal_frequencies.csv", index = False)
