import json
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as T
from scipy.spatial import distance


def load_data(data_path, planner, subset):
    assert os.path.exists(data_path), "path '{}' does not exist.".format(data_path)
    assert (planner in {'BITstar', 'ABITstar', 'InformedRRTstar', 'RRTstar', 'PPNet', "All"})
    assert (subset in {'train', 'val', 'test'})

    envs = []
    envs_fold = []
    paths = []
    path_length = []
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    for i, line in enumerate(content):
        problem = json.loads(line)
        e = []
        for o in problem["Obstacles"]:
            e = e + [o[0], o[1], o[2] + 1/50*224/2]
        # print(len(problem["Obstacles"]), len(e))
        e = e + list(np.zeros(150-len(e)))
        solutions = problem["Solution"]
        if planner == 'PPNet':
            path = [[problem["SegPoints"][i][1], problem["SegPoints"][i][0]] for i in range(int(len(problem["SegPoints"])))]
            # if path[-1] != problem["SegPoints"][-1]:
            #     path = path + [[problem["SegPoints"][-1][1], problem["SegPoints"][-1][0]]]
            paths.append(path)
            path_length.append(len(paths[-1]))
            envs.append(e)
            envs_fold.append(problem["Obstacles"])
            # image = plot_obstacles((224, 224), problem["Obstacles"], resolution=(224, 224))
            # plt.imshow(T.ToPILImage()(image))
            # plt.plot(np.array(path).T[0],
            #          np.array(path).T[1])
            # plt.show()
        else:
            for s in solutions:
                if s["Planner"] == planner and s["Waypoint"] and s["Time"] < 59:
                    paths.append(s["Waypoint"])
                    path_length.append(len(paths[-1]))
                    envs.append(e)
                    envs_fold.append(problem["Obstacles"])
                    # image = plot_obstacles((224, 224), problem["Obstacles"], resolution=(224, 224))
                    # plt.imshow(T.ToPILImage()(image))
                    # plt.plot(np.array(s["Waypoint"]).T[0],
                    #          np.array(s["Waypoint"]).T[1])
                    # plt.show()

    if subset == 'train':
        envs = envs[:4000]
        envs_fold = envs_fold[:4000]
        paths = paths[:4000]
        path_length = path_length[:4000]
    if subset == 'val':
        envs = envs[:100]
        envs_fold = envs_fold[:100]
        paths = paths[:100]
        path_length = path_length[:100]
    if subset == 'test':
        envs = envs[:4000]
        envs_fold = envs_fold[:4000]
        paths = paths[:4000]
        path_length = path_length[:4000]

    print('{} set :'.format(subset), len(envs), 'its')
    assert len(envs) == len(paths) and len(envs) == len(envs_fold) and len(envs) == len(path_length)

    if subset == 'train':
        inputs = []
        outputs = []
        for e, p in zip(envs, paths):
            for i, pp in enumerate(p):
                inputs.append(e+p[-1]+pp)
                outputs.append(p[i+1])
                if p[i+1] == p[-1]:
                    break
        print("train dataset", np.array(inputs).shape, np.array(outputs).shape)
        return np.array(inputs), np.array(outputs)
    else:
        return envs_fold, envs, paths, path_length


def plot_obstacles(size: tuple, obstacles, resolution: tuple = (224, 224)):
    ax = plt.gca()
    ax.axis(xmin=0, xmax=size[0], ymin=size[1], ymax=0)
    for obstacle in obstacles:
        [coord_x, coord_y, radius] = obstacle
        ax.add_patch(plt.Circle(xy=(coord_x, coord_y), radius=radius, fc='black', ec='black'))
    plt.savefig(r'./map.jpg', dpi=90)
    plt.clf()
    img = Image.open(r'./map.jpg').convert('1').convert('RGB')
    img = T.ToTensor()(img)
    img = img[:, 53:383, 73:517]
    img = T.Resize(resolution)(img)
    return img


def plot_solution(size, obstacles, solution, planning_time, planner, index, is_replanning=False):
    data_root = '/home/long/dataset/exp_diff_data_generation/'
    solved_problems_txt = data_root + "/solved_problems.txt"
    with open(solved_problems_txt, 'r', encoding='utf-8') as f:
        content = f.readlines()
    solved_problems = []
    for line in content:
        solved_problems.append(json.loads(line))
    problem = solved_problems[index]
    solved_problems_mpnet_txt = data_root + "/solved_problems_mpnet_{}.txt".format(planner)
    Waypoint = [list(p) for p in np.array(solution, dtype=np.float64)]
    length = sum([distance.euclidean(Waypoint[i], Waypoint[i+1]) for i in range(len(Waypoint)-1)])
    problem["Solution"] = {"Planner": "MPNet", "Waypoint": Waypoint, "Length": length, "Time": planning_time}
    with open(solved_problems_mpnet_txt, "a") as f:
        f.write(json.dumps(problem) + "\n")
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.axis(xmin=0, xmax=size[0], ymin=0, ymax=size[1])
    for obstacle in obstacles:
        [coord_x, coord_y, radius] = obstacle
        ax.add_patch(plt.Circle(xy=(coord_x, coord_y), radius=radius, fc='black', ec='black'))
    plt.plot(solution.T[0], solution.T[1], 'b--')
    ax.plot(solution[0][0], solution[0][1], 'rs')
    ax.plot(solution[-1][0], solution[-1][1], 'rs')
    # plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    result_root = './result_{}'.format(planner)
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    plt.savefig(r'{}/map-{}-{}.jpg'.format(result_root, index, "replanning" if is_replanning else "mpnet"), dpi=1000)
    # img = Image.open(r'./result/map-{}_{}.jpg'.format([x.item() for x in solution[0]], [x.item() for x in solution[-1]]))
    # img = T.ToTensor()(img)#T.CenterCrop((224, 224))(T.ToTensor()(img))
    # img = img[:, 53:383, 73:517]
    # T.ToPILImage()(img).save(r'./result/map-cropped-{}_{}.jpg'.format([x.item() for x in solution[0]], [x.item() for x in solution[-1]]))
    # plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    data_path = "./solved_problems_comparison.txt"
    load_data(data_path, planner="BITstar", subset='train')
