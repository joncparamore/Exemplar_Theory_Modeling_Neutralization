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
import pandas as pd
import copy

start_time = time.perf_counter()
####Model Input: Dictionary of lists with four word forms and their starting nasalance values.
#Nasalance values must be between 0 and 1, and for the simple model, they are set at either 0 or 1.
#four categories correspond to oral (ORAL), nasal (NAS), oral with nasal suffix (ORAL-N), and nasal with nasal suffix (NAS-N)
#each category has 2 or more semantic forms. 
exemplar_clouds = {'ORAL': {'taa': [0.2], 'kii': [0.2]}, 
                   'NAS': {'tAA': [0.6], 'kII': [0.6]}, 
                   'ORAL-N': {'taa-n': [0.2], 'kii-n': [0.2]}, 
                   'NAS-N':{'tAA-n': [0.6], 'kII-n': [0.6]}}

#define morphological relationships between forms in exemplar_clouds and add them to dict semantic_pairs for quick lookup of relations
semantic_relations = [('taa', 'taa-n'), ('kii', 'kii-n'), ('tAA', 'tAA-n'), ('kII', 'kII-n')]
semantic_pairs = {word: (word, related) for word, related in semantic_relations}
semantic_pairs.update({related: (related, word) for word, related in semantic_relations})

#define category (same case) relationships between forms in exemplar_clouds and add them to dict category_pairs for quick lookup of relations
category_relations = [('taa', 'kii'),
                      ('tAA', 'kII'),
                      ('taa-n', 'kii-n'),
                      ('taa-n', 'tAA-n'),
                      ('taa-n', 'kII-n'),
                      ('kii-n', 'tAA-n'),
                      ('kii-n', 'kII-n'),
                      ('tAA-n', 'kII-n')]
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
    sd = 0.015     #standard deviation so that +- 2 sd is .03
    #generate a random noise value to add to the exemplar value
    exemplar_val_with_rand_noise = np.random.normal(loc=nas_mean, scale=sd)
    #set bounds of new_exemplar_val at (0, 1)
    return np.clip(exemplar_val_with_rand_noise, 0, 1)

####constraints - functions to generate penalties for different constraints
#channel constraint: penalize pre-N vowels that aren't very nasalized
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
        category_penalty += (related_form_mean - exemplar_val)**2/len(related_forms)
    return category_penalty

#Define the frequency of each form. What approximate proportion of exemplars do you want from each form?
category_frequencies = [.25, .8, .8, .75] #there are four categories in exemplar_clouds: [oral, nasal, oral-n, nasal-n]
form_frequencies = [.5, .5] #there are two forms from each category in exemplar_clouds
      
####Objective function - calculates penalty of current exemplar_val in relation to constraints and frequency of forms
def objective(exemplar_val, exemplar_form, exemplar_cat, clouds):
    
    #Step 1: Determine Frequency of morphologically related form (e.g., taa and taa-n)
    morph_base_form = semantic_pairs[exemplar_form][1]
    morph_base_cat = form_to_category[morph_base_form]
    morph_base_form_freq = len(clouds[morph_base_cat][morph_base_form])
    
    #Step 2: Determine total Frequency of inflectionally related forms (e.g., for taa-n: kii-n, kII-n, etc. are inflectionally related with genitive suffix)
    cat_relation_freq = 0
    for form in category_pairs[exemplar_form]:
        form_cat = form_to_category[form]
        cat_relation_freq += len(clouds[form_cat][form])
    
    #Step 3: Determine total number of exemplars produced in the entire exemplar system
    total_exemplar_freq = 0
    for cat, forms in clouds.items():
        for form in forms:
            total_exemplar_freq += len(forms[form])
            
    #Step 4: Determine the penalty scalars based on frequency (static for channel_constraint at this point)
    channel_constraint_scalar = .75
    morphological_constraint_scalar = morph_base_form_freq / total_exemplar_freq
    category_constraint_scalar = cat_relation_freq / total_exemplar_freq
    
    #Step 5: Calculate the penalty for the current exemplar
    return (channel_constraint(exemplar_val[0], exemplar_cat, clouds)*channel_constraint_scalar + 
            morphological_constraint(exemplar_val[0], exemplar_form, exemplar_cat, clouds)*morphological_constraint_scalar + 
            category_constraint(exemplar_val[0], exemplar_form, clouds)*category_constraint_scalar)

####Model Process
def exemplar_accumulation(clouds, num_exemplars):
    
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
        optimized_exemplar_val = scipy.optimize.minimize(fun = lambda x: objective(x, new_exemplar_form, new_exemplar_cat, clouds),
                                                         x0 = [new_exemplar_val],
                                                         method = 'SLSQP',
                                                         bounds = bounds,
                                                         options={'maxiter':1000, #limit the maximum number of iterations to 1000
                                                                  'ftol':1e-4, #Sets the function tolerance to 1e-4, meaning the optimizer will stop when changes in the function value are smaller than this threshold.
                                                                  'eps':1e-5, #sets step size of the Jacobian (step_size = gradient * Learning_Rate)
                                                                  })
        
        final_exemplar_val = optimized_exemplar_val.x[0]
        
        ##Step 3: add random noise to optimized exemplar
        final_exemplar_val = random_noise(final_exemplar_val, new_exemplar_form, new_exemplar_cat, clouds)
        
        ##Step 4: Categorize new exemplar into appropriate category
        clouds[new_exemplar_cat][new_exemplar_form].append(final_exemplar_val)
        
        #append new exemplar to data_points list for plotting
        data_points.append((i, final_exemplar_val, new_exemplar_form))
    ##Final Step: calculate mean of each form's exemplar cloud
    final_exemplar_means = {
    cat: {
        form: np.round(np.mean(clouds[cat][form][-250:]), 3)  # Compute mean & round, only for final 250 exemplars in each category. This models decay kind of
        for form in clouds[cat]  # Iterate over words in each category
    }
    for cat in categories  # Iterate over categories
    }
    
    #output is a dictionary of nested dictionaries for each form as key and the mean of that exemplar cloud as the value. 
    return final_exemplar_means, data_points

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
        
        final_means, data_points = exemplar_accumulation(trial_clouds, num_exemplars)
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
            'taa': '#003c6c', 'kii': '#fdc700', 
            'tAA': '#da216d', 'kII': '#93c02d',
            'taa-n': '#003c6c', 'kii-n': '#fdc700',
            'tAA-n': '#da216d', 'kII-n': '#93c02d'}
        marker_map = {
            'taa': 'o', 'kii': 'o', 
            'tAA': 'o', 'kII': 'o',
            'taa-n': 'x', 'kii-n': 'x',
            'tAA-n': 'x', 'kII-n': 'x'
            }

        plt.figure(figsize = (10, 6))
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
                plt.scatter(iteration, nasalance, label = word, alpha = 0.5, s = 10, color = color_map[word], marker = marker_map[word])

        plt.xlabel("Exemplar Production Iteration", fontweight = 'bold')
        plt.ylabel("Nasalance", fontweight = 'bold')
        plt.title("Exemplar Changes in Nasalance Over Time", fontweight = 'bold')
        plt.legend(legend_labels, title = "words (μ nasalance)", bbox_to_anchor = (1, 1), loc = 'upper left')
        plt.grid(False)
        plt.show()
                
    #Step 3: Convert trial_means to df and return the results
    exemplar_trials = pd.DataFrame(trial_means)
    return exemplar_trials


exemplar_trials = multiple_trials(exemplar_clouds, num_exemplars=5000, num_trials=2)

end_time = time.perf_counter()
print(f"Execution time: {round(end_time - start_time, 2)} seconds/{round((end_time - start_time)/60, 2)} minutes")