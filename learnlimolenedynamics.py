import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r"C:\Users\Zach\Documents\GitHub\kinetic-learning")  # setting CWD, else file paths break
from KineticLearning import learn_dynamics, read_timeseries_data, simulate_dynamics
from IPython.display import display
import pandas as pd

# Import DataFrame from CSV & Define Important Variables
controls = ['AtoB', 'GPPS', 'HMGR', 'HMGS', 'Idi', 'Limonene Synthase', 'MK', 'PMD', 'PMK']
states = ['Acetyl-CoA', 'HMG-CoA', 'Mevalonate', 'Mev-P', 'IPP/DMAPP', 'Limonene']

init_max = 3
gens = 1
range_aug = [10, 500]  # default 200
range_win = [4, 20]  # must be less than x, default 7
range_pol = [1, 6]  # must be less than window, default 2

aug = round(min(range_aug)+(1/3)*(max(range_aug)-min(range_aug)))
sigma = (max(range_aug)-min(range_aug))/3
scores = [[None, None]*init_max]

inits = 0
while inits < init_max:
    limonene_df = read_timeseries_data("data/limonene_data.csv", states, controls, time='Hour', strain='Strain', augment=aug)  #default augment=200

    model = learn_dynamics(limonene_df, generations=gens, population_size=30, verbose=True)  # original gen=50,pop=30

    strain_df = limonene_df.loc[limonene_df.index.get_level_values(0) == 'L2']
    trajectory_df = simulate_dynamics(model, strain_df, verbose=True)
    """
    for metabolite in limonene_df['states'].columns:
        plt.figure()
        ax = plt.gca()
        strain_df['states'].loc[strain_df.index.get_level_values(0) == 'L2'].reset_index().plot(x='Time', y=metabolite,
                                                                                                ax=ax, label=metabolite)
        metabolite_name = metabolite + " traj."
        trajectory_df.plot(x='Time', y=metabolite, ax=ax, label=metabolite_name)
        plt.show()"""

    # SCORE IT
    score = inits
    scores[inits] = [aug, score]

    # CHOOSE NEXT
    mu = scores[np.argmax([row[1] for row in scores])][0]  # best score position
    if inits == 0:
        aug = round((2/3)*sum(range_aug)/2)
    else:
        while aug > max(range_aug) or aug < min(range_aug) or aug == scores[inits][0]:
            print("Old var: "+str(aug))
            aug = round(sigma*np.random.randn() + mu)  # samples next point from normal dist around current best
            print("New var picked: "+str(aug))

    print(scores)
    inits += 1

