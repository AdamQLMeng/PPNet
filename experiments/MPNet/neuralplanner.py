import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import load_test_dataset 
from model import MLP 
from torch.autograd import Variable 
import math
import time
from scipy.spatial import distance
from MPNet.my_dataset import load_data, plot_obstacles, plot_solution
import torchvision.transforms as T

clearance=1/50*224

# Load trained model for path generation
mlp = MLP(154, 2) # simple @D
planner = "ABITstar"
mlp.load_state_dict(torch.load('models/mpnet_obs50_{}_final_500_100.pkl'.format(planner)))

if torch.cuda.is_available():
	mlp.cuda()

#load test dataset
# obc,obstacles, paths, path_lengths= load_test_dataset()
data_root = '/home/long/dataset/exp_diff_data_generation/'
data_path = data_root + "/solved_problems_data_5%.txt"
obc,obstacles, paths, path_lengths= load_data(data_path, planner=planner, subset='test')
print(len(obc), np.array(obstacles).shape, len(paths), len(path_lengths))


def IsInCollision_circle(x,idx):
	for ox, oy, s in obc[idx]:
		if distance.euclidean([ox, oy], x) < s:  # clearance requirement
			return True
	return False


def collision_check_circle_edge(s, e, idx):
    if s[0] < 0 or s[1]>224:
        return True
    if e[0] < 0 or e[1]>224:
        return True
    # s = torch.tensor([s[1], s[0]])
    # e = torch.tensor([e[1], e[0]])
    dir = e - s
    dir = torch.tensor([dir[1], -dir[0]])/np.linalg.norm(dir)
    for ox, oy, size in obc[idx]:
        o = torch.tensor([ox, oy])
        if distance.euclidean(e, o) < size + clearance/2 or distance.euclidean(e, o) < size + clearance/2:
            # print("collision(vertex)", e, s, o, size)
            return True
        dis = np.dot(dir, o - s)
        if dis>0:
            dir = -dir
        dis=abs(dis)
        projection = o + dis * dir
        dir1 = projection - s
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = projection - e
        dir2 = dir2 / np.linalg.norm(dir2)
        if dis < size + clearance/2 and np.dot(dir1, dir2) < 0:  # clearance requirement
            # print("collision(edge)", dir, dis, dir1, dir2, o, size)
            return True
    return False

# def IsInCollision_circle_edge(s, e, idx):
# 	if s[0] < 0 or s[1]>224:
# 		return True
# 	if e[0] < 0 or e[1]>224:
# 		return True
# 	dir = e - s
# 	dir = torch.tensor([dir[1], -dir[0]])
# 	for ox, oy, size in obc[idx]:
# 		o = torch.tensor([ox, oy])
# 		if abs(np.dot(dir, o - s)/np.linalg.norm(dir)) < size:  # clearance requirement
# 			# print("collision", abs(np.dot(dir, o - s) / np.linalg.norm(dir)), o, size + 1)
# 			return True
# 	return False


def steerTo (start, end, idx):
	dist = distance.euclidean(start, end)
	# print("steerTo", dist, start, end)
	if dist>0:
		if collision_check_circle_edge(start, end, idx):
			return 0
	return 1


# checks the feasibility of entire path including the path edges
def feasibility_check(path,idx):

	for i in range(0,len(path)-1):
		ind=steerTo(path[i],path[i+1],idx)
		if ind==0:
			return 0
	return 1


def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)


def get_input(i,dataset,targets,seq,bs):
	bi=np.zeros((bs,18),dtype=np.float32)
	bt=np.zeros((bs,2),dtype=np.float32)
	k=0	
	for b in range(i,i+bs):
		bi[k]=dataset[seq[i]].flatten()
		bt[k]=targets[seq[i]].flatten()
		k=k+1
	return torch.from_numpy(bi),torch.from_numpy(bt)


#lazy vertex contraction 
def lvc(path,idx):

	for i in range(0,len(path)-1):
		for j in range(len(path)-1,i+1,-1):
			ind=0
			ind=steerTo(path[i],path[j],idx)
			if ind==1:
				pc=[]
				for k in range(0,i+1):
					pc.append(path[k])
				for k in range(j,len(path)):
					pc.append(path[k])

				return lvc(pc,idx)
				
	return path


def replan_path(p,g,idx,obs):
	itr=0
	path=[]
	path.append(p[0])
	for i in range(1,len(p)-1):
		if steerTo(p[i],p[i+1],idx):
			path.append(p[i])
	path.append(g)			
	new_path=[]
	# print(path)
	for i in range(0,len(path)-1):
		target_reached=False

		st=path[i]
		gl=path[i+1]
		# print("re st gl 1 ", st, gl)
		steer=steerTo(st, gl, idx)
		if steer==1:
			new_path.append(st)
			new_path.append(gl)
		else:
			itr=0
			pA=[]
			pA.append(st)
			pB=[]
			pB.append(gl)
			target_reached=0
			tree=0
			while target_reached==0 and itr<50 :
				itr=itr+1
				if tree==0:
					ip1=torch.cat((obs,st,gl))
					ip1=to_var(ip1)
					st=mlp(ip1.float())
					# print("re st gl 2 ", st, gl)
					st=st.data.cpu()
					pA.append(st)
					tree=1
				else:
					ip2=torch.cat((obs,gl,st))
					ip2=to_var(ip2)
					gl=mlp(ip2.float())
					# print("re st gl 3 ", st, gl)
					gl=gl.data.cpu()
					pB.append(gl)
					tree=0
				target_reached=steerTo(st, gl, idx)
			if target_reached==0:
				return 0, itr
			else:
				for p1 in range(0,len(pA)):
					new_path.append(pA[p1])
				for p2 in range(len(pB)-1,-1,-1):
					new_path.append(pB[p2])

	return new_path, itr


def main(args):
	# Create model directory
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	tp=0
	fp=0
	tot=[]
	num_iter = []
	step = 0
	step_replan = 0
	for i in range(len(paths)):
		t = 0
		for j in range(1):
			print ("step: i="+str(i)+" j="+str(j))
			p1_ind=0
			p2_ind=0
			p_ind=0
			if path_lengths[i]>0:
				start=np.zeros(2,dtype=np.float32)
				goal=np.zeros(2,dtype=np.float32)
				for l in range(0,2):
					start[l]=paths[i][0][l]
				
				for l in range(0,2):
					goal[l]=paths[i][path_lengths[i]-1][l]
				#start and goal for bidirectional generation
				## starting point
				start1=torch.from_numpy(start)
				goal2=torch.from_numpy(start)
				##goal point
				goal1=torch.from_numpy(goal)
				start2=torch.from_numpy(goal)
				##obstacles
				obs=obstacles[i]
				obs=torch.tensor(obs)
				##generated paths
				path1=[]
				path1.append(start1)
				path2=[]
				path2.append(start2)
				path=[]
				target_reached=0
				step=0
				path=[] # stores end2end path by concatenating path1 and path2
				tree=0
				tic = time.time()
				while target_reached==0 and step<80 :
					step=step+1
					if tree==0:
						inp1=torch.cat((obs,start1,start2))
						inp1=to_var(inp1)
						start1=mlp(inp1.float())
						# print('start1', start1)
						start1=start1.data.cpu()
						# print('start1', start1)
						path1.append(start1)
						tree=1
					else:
						inp2=torch.cat((obs,start2,start1))
						inp2=to_var(inp2)
						start2=mlp(inp2.float())
						# print('start2', start1)
						start2=start2.data.cpu()
						# print('start2', start1)
						path2.append(start2)
						tree=0
					# print(start1, start2)
					target_reached=steerTo(start1,start2,i)
				tp=tp+1

				if target_reached==1:
					path = path1 + list(reversed(path2))
					print("Get initial solution from mpnet!")
					print("Number of waypoint:", len(path), path)
					indicator=feasibility_check(path,i)

					if indicator==1:
						toc = time.time()
						t=toc-tic
						print("Planning successed without replanning in", t, 's')
						plot_solution(size=(224, 224), obstacles=obc[i],
									  solution=torch.cat([p.unsqueeze(dim=1) for p in path], dim=1).T, planning_time=t, planner=planner, index=i,
									  is_replanning=False)
						fp=fp+1
						# print ("path[0]:")
						# for p in range(0,len(path)):
						# 	print (path[p][0])
						# print ("path[1]:")
						# for p in range(0,len(path)):
						# 	print (path[p][1])
						# print ("Actual path[0]:")
						# for p in range(0,path_lengths[i]):
						# 	print (paths[i][p][0])
						# print ("Actual path[1]:")
						# for p in range(0,path_lengths[i]):
						# 	print (paths[i][p][1])
					else:
						sp=0
						indicator=0
						print("Replanning")
						while indicator==0 and sp<10 and path !=0:
							sp=sp+1
							g=torch.tensor(paths[i][-1])
							path, step_replan=replan_path(path,g,i,obs) #replanning at coarse level
							if path !=0:
								path=lvc(path,i)
								indicator=feasibility_check(path,i)
					
							if indicator==1:
								toc = time.time()
								t=toc-tic
								print("Planning successed with replanning in", t, 's')
								plot_solution(size=(224, 224), obstacles=obc[i], solution=torch.cat([p.unsqueeze(dim=1) for p in path], dim=1).T, planning_time=t, planner=planner, index=i, is_replanning=True)
								fp=fp+1
								# if len(path)<20:
								# 	print ("new_path[0]:")
								# 	for p in range(0,len(path)):
								# 		print (path[p][0])
								# 	print ("new_path[1]:")
								# 	for p in range(0,len(path)):
								# 		print (path[p][1])
								# 	print ("Actual path[0]:")
								# 	for p in range(0,path_lengths[i]):
								# 		print (paths[i][p][0])
								# 	print ("Actual path[1]:")
								# 	for p in range(0,path_lengths[i]):
								# 		print (paths[i][p][1])
								# else:
								# 	print ("path found, dont worry"	)
					if not indicator:
						print("Planning failed")
		if t:
			tot.append(t)
			num_iter.append(step+step_replan)
			print("time:", t, "step:", step, "step replan:", step_replan)
	pickle.dump(tot, open("time_s2D_unseen_mlp.p", "wb" ))

	print ("total paths")
	print (tp)
	print ("feasible paths")
	print (fp)
	print('MPNet:', 'Median:', np.median(np.array(tot)),
		  			'IQR', np.percentile(np.array(tot), 75) - np.percentile(np.array(tot), 25),
		  			'Average:', sum(tot) / len(tot),
		  			'Std', np.std(np.array(tot)),
		  			'Sum', sum(tot),
		  			"Number", len(tot)
		)
	print('MPNet:', 'Median:', np.median(np.array(num_iter)),
		  			'IQR', np.percentile(np.array(num_iter), 75) - np.percentile(np.array(num_iter), 25),
		  			'Average:', sum(num_iter) / len(num_iter),
		  			'Std', np.std(np.array(num_iter)),
		  			'Sum', sum(num_iter),
		  			"Number", len(num_iter)
		)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
	parser.add_argument('--no_env', type=int, default=50,help='directory for obstacle images')
	parser.add_argument('--no_motion_paths', type=int,default=2000,help='number of optimal paths in each environment')
	parser.add_argument('--log_step', type=int , default=10,help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=1000,help='step size for saving trained models')

	# Model parameters
	parser.add_argument('--input_size', type=int , default=68, help='dimension of the input vector')
	parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
	parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=4, help='number of layers in lstm')

	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=100)
	parser.add_argument('--learning_rate', type=float, default=0.001)
	args = parser.parse_args()
	print(args)
	main(args)


