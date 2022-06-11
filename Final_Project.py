# -*- coding: utf-8 -*-
"""
Created on Thu May 1 21:33:52 2022

@author: Shawn Hu
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import sys
import pandas as pd
import os

#reading data
path = r'/Users/huchangguo/Desktop'
df = pd.read_csv(os.path.join(path,'60Activities-poseData.csv'),index_col='name')
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        df.iloc[i,j] = list(map(eval,df.iloc[i,j][1:-2].split(',')))
        #解析字符串，这样的话可以达到调用的效果，把字符串变成list
#以上对data做一个process

#小人儿的构成
Rjoints = ['right_wrist','right_elbow','right_shoulder','right_hip','right_knee','right_ankle']
Ljoints = ['left_wrist','left_elbow','left_shoulder','left_hip','left_knee','left_ankle']
Shoulder = ['left_shoulder','right_shoulder']
Hip = ['left_hip','right_hip']
edge = [Rjoints,Ljoints,Shoulder,Hip]

ExerciseList = df.index.to_list() #所有的exercise的名字都在这里
print(ExerciseList)
Exercise = 'dumbbell flys' #change展示动作
df1 = df.loc[Exercise]
length = len(df1[0])

#小人移动
for n in range(length):
    #plot each joint
    fig = plt.figure()
    ax1 = Axes3D(fig)
    ax1.cla()
    for joints in edge:
        x,y,z=[],[],[]
        for joint in joints:
            node_name = joint
            x.append(df1['xX_'+joint][n])
            y.append(df1['yY_'+joint][n])
            z.append(df1['zZ_'+joint][n])
        y = np.dot(y,-1) #吧y里所有的值都乘以负一
        ax1.plot3D(z,x,y,'green') #change color
        #set 坐标轴
        plt.xlim((0,1))
        plt.ylim((-1,1))
        ax1.set_zlim(0,1)
        #plt.zlim((-1,1))
    plt.pause(0.01)


#Q2&Q3
#获取我指定两个关节连接的vector的坐标
def vec(joints,n,df1):
    j1,j2 = joints
    x1 = df1['xX_'+j1][n]
    y1 = df1['yY_'+j1][n]
    z1 = df1['zZ_'+j1][n]
    x2 = df1['xX_'+j2][n]
    y2 = df1['yY_'+j2][n]
    z2 = df1['zZ_'+j2][n]
    vec = (x1-x2,y1-y2,z1-z2)
    return vec

#我指定两个vec，算出夹角
def angle(vec1,vec2):
    vec1_mod = np.linalg.norm(vec1)
    vec2_mod = np.linalg.norm(vec2)
    product = np.dot(vec1,vec2)
    cos = product/(vec1_mod*vec2_mod)
    ang = np.degrees(np.arccos(cos))
    return ang

#指定每一个关节的夹角是由哪个向量组成的
def get_the_angles(df1):
    angles = pd.DataFrame()
    for n in range(length):
        temp = pd.DataFrame()
        #calculate absolute value for each joint
        temp['right_knee'] = [angle(vec(['right_ankle','right_knee'],n,df1),vec(['right_hip','right_knee'],n,df1))]
        temp['left_knee'] = [angle(vec(['left_ankle','left_knee'],n,df1),vec(['left_hip','left_knee'],n,df1))]
        temp['right_elbow'] = [angle(vec(['right_wrist','right_elbow'],n,df1),vec(['right_shoulder','right_elbow'],n,df1))]
        temp['left_elbow'] = [angle(vec(['left_wrist','left_elbow'],n,df1),vec(['left_shoulder','left_elbow'],n,df1))]
        temp['right_hip'] = [angle(vec(['right_knee','right_hip'],n,df1),vec(['right_shoulder','right_hip'],n,df1))] 
        temp['left_hip'] = [angle(vec(['left_knee','left_hip'],n,df1),vec(['left_shoulder','left_hip'],n,df1))] 
        temp['right_shoulder'] = [angle(vec(['right_elbow','right_shoulder'],n,df1),vec(['right_hip','right_shoulder'],n,df1))]
        temp['left_shoulder'] = [angle(vec(['left_elbow','left_shoulder'],n,df1),vec(['left_hip','left_shoulder'],n,df1))]
        angles = pd.concat([angles,temp],ignore_index = True)
    return angles

#调用以上function，做判定的过程
angles = get_the_angles(df1)
print(' moving in the exercise:\n')
moving_angles = []
for joint in angles.columns:
    angle_change = []
    for i in range(len(angles[joint])-1):
        angle_change.append(angles[joint][i+1] - angles[joint][i])
    SD = np.std(angles[joint])
    Mean= np.mean(angles[joint])
    SDMean = SD/Mean
    # 设定threshold，判定是否属于动了
    if SDMean > 0.25:
        print(joint,'\n',angle_change,'\n')
        moving_angles.append(joint)
print(' not moving:\n')
for joint in angles.columns:
    if joint not in moving_angles:
        print(joint)

        
#Q4
#做cluster，对60个运动做聚类，里面把所有的运动都做过详细的process
features = []
for exercise in ExerciseList:
    df1 = df.loc[exercise]
    length = len(df1[0])
    angles = get_the_angles(df1)
    t = []
    for joint in angles.columns:
        SD = np.std(angles[joint])
        Mean = np.mean(angles[joint])
        SDmean = SD/Mean
        t.append(SDmean)
    features.append(t)
features = pd.DataFrame(data = features, columns = angles.columns,index = ExerciseList)
#所有的运动都存在features里


#把feature放到kmeans
from sklearn.cluster import KMeans
K = KMeans(n_clusters=4, random_state=0).fit(features) # 可以改变n_cluster里面的数，定义一共把所有运动分成几类
f = features.copy()
f['cluster'] = K.labels_
print(f)


# Print the cluster result with Kmeans
cluster = set(K.labels_)
for i in cluster:
    print('in the ',i,'th cluster, ',f[f['cluster'] == i].index.tolist())
    
#分类






