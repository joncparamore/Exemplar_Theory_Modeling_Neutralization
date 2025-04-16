#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:37:36 2025

@author: joncparamore
"""
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#for calculating how long the model needs to run for.
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

####Objective function - calculates penalty of current exemplar_val (filler for now)
def objective(exemplar_val):
    exemplar_val = 0
    return exemplar_val

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
    #store individual words in list of values associated with the category as the key
    semantic_forms = {}
    for cat in categories:
        semantic_forms[cat] = []
        for form in clouds[cat]:
            semantic_forms[cat].append(form) 
    
    #develop num_exemplars new exemplars
    for i in range(num_exemplars):
        
        ##step 1: create a new exemplar
        #randomly choose one exemplar category for production from clouds dictionary
        new_exemplar_cat = random.choice(categories)
        #randomly choose one form from that category for production
        new_exemplar_form = random.choice(semantic_forms[new_exemplar_cat])

        #make the mean value of that form the starting value for the new exemplar (entrenchment)
        new_exemplar_val = np.mean(clouds[new_exemplar_cat][new_exemplar_form])
        
        
        ##step 2: add random noise
        new_exemplar_val = random_noise(new_exemplar_val, new_exemplar_form, new_exemplar_cat, clouds)
        
        ##Step 4: Categorize new exemplar into appropriate category\
        final_exemplar_val = new_exemplar_val
        clouds[new_exemplar_cat][new_exemplar_form].append(final_exemplar_val)
        
        #append new exemplar to data_points list for plotting
        data_points.append((i, final_exemplar_val, new_exemplar_form))
    ##Final Step: calculate mean of each form's exemplar cloud
    final_exemplar_means = {
    cat: {
        form: np.round(np.mean(clouds[cat][form]), 3)  # Compute mean & round
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
        final_means, data_points = exemplar_accumulation(clouds, num_exemplars)
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


exemplar_trials = multiple_trials(exemplar_clouds, num_exemplars=5000, num_trials=1)

end_time = time.perf_counter()
print(f"Execution time: {round(end_time - start_time, 2)} seconds/{round((end_time - start_time)/60, 2)} minutes")