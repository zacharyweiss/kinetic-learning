import os
import matplotlib.pyplot as plt
import numpy as np
import math

os.chdir(r"C:\Users\Zach\Documents\GitHub\kinetic-learning")  # setting CWD, else file paths break
from KineticLearning import learn_dynamics, read_timeseries_data, simulate_dynamics
import pandas as pd

# Import DataFrame from CSV & Define Important Variables
controls = ['AtoB', 'GPPS', 'HMGR', 'HMGS', 'Idi', 'Limonene Synthase', 'MK', 'PMD', 'PMK']
states = ['Acetyl-CoA', 'HMG-CoA', 'Mevalonate', 'Mev-P', 'IPP/DMAPP', 'Limonene']

init_max = 10
gens = 100
range_aug = [10, 500]  # default 200
range_win = [3, 19]  # must be less than x, default 7
range_pol = [1, 6]  # must be less than window, default 2


def square_error(actual, predicted):
    sq_sum = 0
    for i in range(len(actual)):
        sq_sum = sq_sum + (actual[i] - predicted[i]) ** 2
    return sq_sum


def picker(rng):
    val = round(min(rng) + (1 / 3) * (max(rng) - min(rng)))
    variance = (max(rng) - min(rng)) / 3

    return val, variance


def sampler(val, val_rng, sigma, aug=False, win=False, pol=False):
    scores = np.ones((init_max, 2))*math.inf  # not initializing with zeros else argmin() will always pick empty rows
    inits = 0
    while inits < init_max:
        if aug and ~(win or pol):
            limonene_df = read_timeseries_data("data/limonene_data.csv", states, controls, time='Hour', strain='Strain',
                                               augment=200) #val)  # default augment=200
        elif win and ~(aug or pol):
            limonene_df = read_timeseries_data("data/limonene_data.csv", states, controls, time='Hour', strain='Strain',
                                               window_size=val)
        elif pol and ~(win or aug):
            limonene_df = read_timeseries_data("data/limonene_data.csv", states, controls, time='Hour', strain='Strain',
                                               poly_order=val)
        else:
            raise Exception("You absolute fool, you passed the wrong args")

        model = learn_dynamics(limonene_df, generations=gens, population_size=30,
                               verbose=True)  # original gen=50,pop=30

        strain_df = limonene_df.loc[limonene_df.index.get_level_values(0) == 'L2']
        trajectory_df = simulate_dynamics(model, strain_df, verbose=True)

        score = 0
        for metabolite in limonene_df['states'].columns:
            plt.figure()
            ax = plt.gca()
            control = strain_df['states'].loc[strain_df.index.get_level_values(0) == 'L2'].reset_index()
            model = trajectory_df
            control.plot(x='Time', y=metabolite, ax=ax, label=metabolite)
            metabolite_name = str(metabolite) + " model"
            model.plot(x='Time', y=metabolite, ax=ax, label=metabolite_name)
            print("Metabolite:", metabolite)
            err = square_error(np.array(control[metabolite]), np.array(model[metabolite]))
            print("Square Error:", err)
            score += err

            plt.savefig(str(inits)+str(metabolite).replace('/', '_')+str(val))
        plt.close('all')


        # SCORE IT
        scores[inits] = [val, score]

        # CHOOSE NEXT
        mu = scores[np.argmin([row[1] for row in scores])][0]  # best score position
        if inits == 0:
            val = int(round(min(val_rng) + (2 / 3) * (max(val_rng) - min(val_rng))))
            if win and (val % 2 == 0):
                val += 1
        else:
            while val > max(val_rng) or val < min(val_rng) or (val in [row[0] for row in scores]):
                print("Old var: " + str(mu))
                val = int(round(sigma * np.random.randn() + mu))  # samples next point from normal dist around current best
                if win and val % 2 == 0:
                    val += 1
                print("New var picked: " + str(val))

        print(scores)
        inits += 1


value, var = picker(range_aug)
# if value % 2 == 0:  # only for use with win, as values must be odd
#     value += 1
print("Start val: "+str(value))
sampler(value, range_aug, var, aug=True)

# range_aug
# range_win
# range_pol
