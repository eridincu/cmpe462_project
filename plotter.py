import matplotlib.pyplot as plt
import json
from collections import defaultdict

from numpy.lib.function_base import average

data = []
with open('results/best_mnb.json') as f:
    data = json.load(f)

min_dfs = []
max_dfs = []
features = []
results = []

for result in data:
    key = result[0]
    if key.startswith('mutual info'):
        continue

    tokens = key.split()

    for i in range(len(tokens)):
        tokens[i] = float(tokens[i])
    
    max_dfs.append(tokens[0])
    min_dfs.append(tokens[1])
    features.append(tokens[2])

    results.append(result[1])
count = 1
for param_list in [min_dfs, max_dfs, features]:
    totals = defaultdict(float)
    averages = defaultdict(float) 
    for i in range(len(results)):
        totals[param_list[i]] += results[i]

    for key in totals:
        if count == 3:
            averages[key] = totals[key] / 100
        else:
            averages[key] = totals[key] / (len(results) / 10)


    plt.figure()
    plt.scatter(list(reversed(averages.keys())), list(reversed(averages.values())))
    plt.savefig('plots/' + str(count) + '.png')
    plt.close()
    count += 1
