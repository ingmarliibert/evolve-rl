import numpy as np
from matplotlib import pyplot as plt

datas = {}
n = 2000
plt.xlim(0, 4000)
for filename in ["dqn", "original", "crossover", "annealing", "crossover_annealing"]:
    with open(f"{filename}_data.txt") as file:
        scores = np.array(list(map(float, file.readline().split(','))))
        times = np.array(list(map(float, file.readline().split(','))))
        datas[filename] = {'scores': scores, 'times': 'times'}
    if filename == "dqn":
        s = 10
        scores_new, times_new = list(), list()
        for i in range(0, len(scores), s):
            max_i = np.argmax(scores[i:i + s])
            scores_new.append(scores[i + max_i])
            times_new.append(times[i + max_i])
        scores = np.array(scores_new)
        times = np.array(times_new)

    plt.plot(times, scores, label=" ".join(filename.split("_")))
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Score')
plt.savefig("comparison.png")
