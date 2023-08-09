import json, os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import distance

from my_dataset import plot_solution


def parse_original_data(data_txt):
    print("parse data from text:", data_txt)
    with open(data_txt, 'r', encoding='utf-8') as f:
        content = f.readlines()
    rrt_t = []
    irrt_t = []
    bit_t = []
    abit_t = []
    mpnet_t = []
    rrt_c = []
    irrt_c = []
    bit_c = []
    abit_c = []
    mpnet_c = []
    mpnet_i = []
    ppnet_c = []
    ppnet_i = []
    cost_comparison = []
    for i, line in enumerate(content):
        # print('mark', line[:15])
        problem = json.loads(line)
        ppnet_c.append(float(problem["Length"])/224*50)
        ppnet_i.append(int(problem["Index"]))

        if "Solution" not in problem.keys():
            continue
        solutions = problem["Solution"]
        if not isinstance(solutions, list):
            solutions = [solutions]
        for s in solutions:
            if s["Waypoint"] is None or s["Time"] >= 60:
                print('Planning failed', problem["Index"])
                break
            # if s["Time"] >= 1:
            #     print('time consumption:', problem["Index"], s["Planner"], s["Time"])
            if s["Planner"] == "RRTstar":
                rrt_t.append(float(s["Time"]))
            if s["Planner"] == "InformedRRTstar":
                irrt_t.append(float(s["Time"]))
            if s["Planner"] == "BITstar":
                bit_t.append(float(s["Time"]))
            if s["Planner"] == "ABITstar":
                abit_t.append(float(s["Time"]))
            if s["Planner"] == "MPNet":
                mpnet_t.append(float(s["Time"]))
                mpnet_c.append(float(s["Length"])/224*50)
                mpnet_i.append(int(problem["Index"]))
        # if i == 587:
        #     for s in solutions:
        #         result_root = data_root + '/result_{}'.format(s["Planner"])
        #         if not os.path.exists(result_root):
        #             os.mkdir(result_root)
        #         print(problem["Index"], s["Planner"], s["Time"])
        #         plot_solution(size=(224, 224), obstacles=problem["Obstacles"], solution=torch.cat([p.unsqueeze(dim=1) for p in s["Waypoint"]], dim=1).T, index=problem["Index"], result_root=result_root)

    if len(rrt_t):
        print('RRT*:', 'Median:', np.median(np.array(rrt_t)),
                       'IQR', np.percentile(np.array(rrt_t), 75)-np.percentile(np.array(rrt_t), 25),
                       'Average:', sum(rrt_t) / len(rrt_t),
                       'Std', np.std(np.array(rrt_t)),
                       'Sum', sum(rrt_t),
                       'Num', len(rrt_t),
                       'Max', max(rrt_t)
              )
    if len(irrt_t):
        print('IRRT*:', 'Median:', np.median(np.array(irrt_t)),
                        'IQR', np.percentile(np.array(irrt_t), 75)-np.percentile(np.array(irrt_t), 25),
                        'Average:', sum(irrt_t) / len(irrt_t),
                        'Std', np.std(np.array(irrt_t)),
                        'Sum', sum(irrt_t),
                        'Num', len(irrt_t),
                        'Max', max(irrt_t)
              )
    if len(bit_t):
        print('BIT*:', 'Median:', np.median(np.array(bit_t)),
                       'IQR', np.percentile(np.array(bit_t), 75)-np.percentile(np.array(bit_t), 25),
                       'Average:', sum(bit_t) / len(bit_t),
                       'Std', np.std(np.array(bit_t)),
                       'Sum', sum(bit_t),
                       'Num', len(bit_t),
                       'Max', max(bit_t)
              )
    if len(abit_t):
        print('ABIT*:', 'Median:', np.median(np.array(abit_t)),
                        'IQR', np.percentile(np.array(abit_t), 75)-np.percentile(np.array(abit_t), 25),
                        'Average:', sum(abit_t) / len(abit_t),
                        'Std', np.std(np.array(abit_t)),
                        'Sum', sum(abit_t),
                        'Num', len(abit_t),
                        'Max', max(abit_t)
              )
    if len(mpnet_t):
        print('MPNet:', 'Median:', np.median(np.array(mpnet_t)),
                        'IQR', np.percentile(np.array(mpnet_t), 75)-np.percentile(np.array(mpnet_t), 25),
                        'Average:', sum(mpnet_t) / len(mpnet_t),
                        'Std', np.std(np.array(mpnet_t)),
                        'Sum', sum(mpnet_t),
                        'Num', len(mpnet_t),
                        'Max', max(mpnet_t)
              )

    if len(mpnet_c):
        print('MPNet cost:', 'Median:', np.median(np.array(mpnet_c)),
                        'IQR', np.percentile(np.array(mpnet_c), 75)-np.percentile(np.array(mpnet_c), 25),
                        'Average:', sum(mpnet_c) / len(mpnet_c),
                        'Std', np.std(np.array(mpnet_c)),
                        'Sum', sum(mpnet_c),
                        'Num', len(mpnet_c),
                        'Max', max(mpnet_c)
              )
    if len(ppnet_c):
        ppnet_c = ppnet_c[:3787]
        print('PPNet cost:', 'Median:', np.median(np.array(ppnet_c)),
                        'IQR', np.percentile(np.array(ppnet_c), 75)-np.percentile(np.array(ppnet_c), 25),
                        'Average:', sum(ppnet_c) / len(ppnet_c),
                        'Std', np.std(np.array(ppnet_c)),
                        'Sum', sum(ppnet_c),
                        'Num', len(ppnet_c),
                        'Max', max(ppnet_c)
              )
    # plt.subplot(2, 2, 1)
    # plt.hist(rrt, bins=500, range=None, weights=None,
    #          cumulative=False, bottom=None, histtype='bar',
    #          align='mid', orientation='vertical', rwidth=None,
    #          log=False, color=None, label=None, stacked=False)
    # plt.subplot(2, 2, 2)
    # plt.hist(irrt, bins=500, range=None, weights=None,
    #          cumulative=False, bottom=None, histtype='bar',
    #          align='mid', orientation='vertical', rwidth=None,
    #          log=False, color=None, label=None, stacked=False)
    # plt.subplot(2, 2, 3)
    # plt.hist(bit, bins=500, range=None, weights=None,
    #          cumulative=False, bottom=None, histtype='bar',
    #          align='mid', orientation='vertical', rwidth=None,
    #          log=False, color=None, label=None, stacked=False)
    # plt.subplot(2, 2, 4)
    # plt.hist(abit, bins=500, range=None, weights=None,
    #          cumulative=False, bottom=None, histtype='bar',
    #          align='mid', orientation='vertical', rwidth=None,
    #          log=False, color=None, label=None, stacked=False)
    # plt.show()
    return ppnet_i, ppnet_c, mpnet_i, mpnet_c


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
    result_root = "/home/long/dataset/exp_diff_algo"
    result_root = r"/home/long/dataset/exp_diff_data_generation"
    data_txt = result_root + '/solved_problems.txt'
    ppnet_i, ppnet_c, _, _ = parse_original_data(data_txt)

    for planner in ['RRTstar', 'InformedRRTstar', 'BITstar', 'ABITstar']:
        data_txt = result_root + '/solved_problems_{}.txt'.format(planner)
        _, _, mpnet_i, mpnet_c = parse_original_data(data_txt)
        print(len(mpnet_i), len(ppnet_i))
        cost_diff = []
        for i, index in enumerate(mpnet_i):
            if index in ppnet_i:
                cost_diff.append(mpnet_c[i]-ppnet_c[ppnet_i.index(index)])
        if len(cost_diff):
            print('cost diff ({}):'.format(planner), 'Median:', np.median(np.array(cost_diff)),
                            'IQR', np.percentile(np.array(cost_diff), 75)-np.percentile(np.array(cost_diff), 25),
                            'Average:', sum(cost_diff) / len(cost_diff),
                            'Std', np.std(np.array(cost_diff)),
                            'Sum', sum(cost_diff),
                            'Num', len(cost_diff),
                            'Max', max(cost_diff)
                  )
    data_txt = result_root + '/solved_problems_data_5%.txt'
    ppnet_i, ppnet_c, _, _ = parse_original_data(data_txt)
    # loss_txt = "./rrt_50000.txt"
    # parse_loss_txt(loss_txt)
