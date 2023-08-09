import json

import matplotlib.pyplot as plt
import numpy as np
from my_dataset import plot_solution


def parse_original_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    rrt = []
    irrt = []
    bit = []
    abit = []
    for i, line in enumerate(content):
        # print('mark', line[:15])
        problem = json.loads(line)
        solutions = problem["Solution"]
        for s in solutions:
            if s["Waypoint"] is None or s["Time"] >= 60:
                print('Planning failed', problem["Index"])
                break
            if s["Time"] >= 1:
                print('time consumption:', problem["Index"], s["Planner"], s["Time"])
            if s["Planner"] == "RRTstar":
                rrt.append(float(s["Time"]))
            if s["Planner"] == "InformedRRTstar":
                irrt.append(float(s["Time"]))
            if s["Planner"] == "BITstar":
                bit.append(float(s["Time"]))
            if s["Planner"] == "ABITstar":
                abit.append(float(s["Time"]))
        if problem["Index"] == 587:
            plt.legend(loc=0)
            for s in solutions:
                print(problem["Index"], s["Planner"], s["Time"])
                plot_solution(size=(224, 224), obstacles=problem["Obstacles"], solution=s["Waypoint"])

    # print(max(rrt), rrt.index(max(rrt)),
    #       max(irrt), rrt.index(max(irrt)),
    #       max(bit), rrt.index(max(bit)),
    #       max(abit), rrt.index(max(abit)),
    #       )

    rrt.sort()
    irrt.sort()
    bit.sort()
    abit.sort()

    print('RRT*:', 'Median:', np.median(np.array(rrt)),
                   'IQR', np.percentile(np.array(rrt), 75)-np.percentile(np.array(rrt), 25),
                   'Average:', sum(rrt) / len(rrt),
                   'Std', np.std(np.array(rrt)),
                   'Sum', sum(rrt),
                   'Num', len(rrt),
                   'Max', max(rrt)
          )
    print('IRRT*:', 'Median:', np.median(np.array(irrt)),
                    'IQR', np.percentile(np.array(irrt), 75)-np.percentile(np.array(irrt), 25),
                    'Average:', sum(irrt) / len(irrt),
                    'Std', np.std(np.array(irrt)),
                    'Sum', sum(irrt),
                    'Num', len(irrt),
                    'Max', max(irrt)
          )
    print('BIT*:', 'Median:', np.median(np.array(bit)),
                   'IQR', np.percentile(np.array(bit), 75)-np.percentile(np.array(bit), 25),
                   'Average:', sum(bit) / len(bit),
                   'Std', np.std(np.array(bit)),
                   'Sum', sum(bit),
                   'Num', len(bit),
                   'Max', max(bit)
          )
    print('ABIT*:', 'Median:', np.median(np.array(abit)),
                    'IQR', np.percentile(np.array(abit), 75)-np.percentile(np.array(abit), 25),
                    'Average:', sum(abit) / len(abit),
                    'Std', np.std(np.array(abit)),
                    'Sum', sum(abit),
                    'Num', len(abit),
                    'Max', max(abit)
          )

    plt.subplot(2, 2, 1)
    plt.hist(rrt, bins=500, range=None, weights=None,
             cumulative=False, bottom=None, histtype='bar',
             align='mid', orientation='vertical', rwidth=None,
             log=False, color=None, label=None, stacked=False)
    plt.subplot(2, 2, 2)
    plt.hist(irrt, bins=500, range=None, weights=None,
             cumulative=False, bottom=None, histtype='bar',
             align='mid', orientation='vertical', rwidth=None,
             log=False, color=None, label=None, stacked=False)
    plt.subplot(2, 2, 3)
    plt.hist(bit, bins=500, range=None, weights=None,
             cumulative=False, bottom=None, histtype='bar',
             align='mid', orientation='vertical', rwidth=None,
             log=False, color=None, label=None, stacked=False)
    plt.subplot(2, 2, 4)
    plt.hist(abit, bins=500, range=None, weights=None,
             cumulative=False, bottom=None, histtype='bar',
             align='mid', orientation='vertical', rwidth=None,
             log=False, color=None, label=None, stacked=False)
    plt.show()


def parse_loss_txt(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    loss = []
    for i, line in enumerate(content):
        if int(i%3) == 2 and i > 7:
            loss.append(float(line))
    loss = loss[-10000:]
    plt.plot(np.arange(len(loss)), loss)
    plt.show()


if __name__ == "__main__":
    clearance = 1
    num_obs = 20
    result_root = "/home/long/experiment/result_c{}_{}".format(clearance, num_obs)
    solved_problems_comparison_txt = "./solved_problems_comparison.txt".format(result_root)
    # parse_original_data(solved_problems_comparison_txt)
    loss_txt = "./rrt_50000.txt"
    parse_loss_txt(loss_txt)
