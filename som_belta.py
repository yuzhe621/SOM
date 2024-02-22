# coding=utf-8

import cv2
import time
import random
import numpy as np
from math import *
from concurrent.futures import ThreadPoolExecutor

class SOM(object):
    def __init__(self, image_path, mask_size, t1=100000, lr_stop=0.0005, neighbourhood_redius_stop=0.05, eta0=0.1):
        self.PATH = image_path
        self.MASK_SIZE = mask_size
        self.SIGMA0 = max(self.MASK_SIZE)
        self.T1 = t1
        self.LR_STOP = lr_stop
        self.NEIGHBOURHOOD_RADIUS_STOP = neighbourhood_redius_stop
        self.ETA0 = eta0

    def read_data(self, enhance, thresh=100, maxval=100, reSize=(300,150)):
        '''
        读取图片数据
        :param enhance: 是否对图像进行增强与处理
        :param thresh: 二值化阈值
        :param maxval: 达到阈值后最大值
        :param reSize: resize大小
        :return: 返回预处理后归一化的图片
        '''
        try:
            image_path = self.PATH
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            if enhance == "binary":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                retVal, image_binary = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)
                image_out = cv2.cvtColor(image_binary, cv2.COLOR_GRAY2BGR)
                image = image_out / 255

            elif enhance == "resize":
                width, height = reSize
                image_out = cv2.resize(image, (width, height))
                image = image_out / 255

            elif enhance == "binary&resize":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                retVal, image_binary = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)
                image = cv2.cvtColor(image_binary, cv2.COLOR_GRAY2BGR)
                width, height = reSize
                image_out = cv2.resize(image, (width, height))
                image = image_out / 255

            elif enhance == "none":
                image = image / 255

            return image
        except Exception as e:
            print("the error of read_data is : ", e)

    def init_mask(self):
        '''
        初始化竞争层网络，网大小为【H,W,5】,数组中每个元素内部shape为[W1,W2,W3,(坐标)]，
        W为不同通道的像素值，坐标为mask上的点对应到图像上点的坐标
        :return: 初始化后的网络
        '''
        try:
            net_size = (self.MASK_SIZE[0], self.MASK_SIZE[1], 5)
            mask = np.zeros(net_size)

            rate = 0.1
            for i in range(net_size[0]):
                for j in range(net_size[1]):
                    mask[i][j][:3] = [random.random() * rate, random.random() * rate, random.random() * rate]

            return mask

        except Exception as e:
            print("the error of init_mask is : ", e)
    def euclidean_distance(self, point_A, point_B):
        '''
        计算欧式距离，对向量的长度敏感
        :param point_A: 向量A（vector A）
        :param point_B: 向量B（vector B）
        :return: 返回欧式空间距离
        '''
        try:
            distance = 0
            for i in range(len(point_A)):
                distance += (point_A[i] - point_B[i]) ** 2
                distance = sqrt(distance)
            return distance
        except Exception as e:
            print("the error of euclidean_distance is : ", e)

    def h_neighbourhood(self, radius, distance):
        '''
        激活函数
        :param radius: 邻域域半径
        :param distance: 欧式距离，普通节点到优胜节点(BMU)的欧式距离
        :return: 激活函数系数
        '''
        try:
            h = exp(-distance / (radius ** 2))
            return h
        except Exception as e:
            print("the error of h_neighbourhood is : ", e)

    def neighbourhood_ratio_of_active(self, step):
        '''
        激活半径函数，随迭代次数增加而减小
        :param step: 迭代次数
        :return: 返回随迭代次数增加而改变的邻域半径
        '''
        try:
            sigma0 = self.SIGMA0
            t1 = self.T1 # t1越大，半径随时间减小越慢
            temp = exp(-step / t1)
            radius = sigma0 * temp
            return radius

        except Exception as e:
            print("the error of neighbourhood_ratio_of_active is : ", e)

    def learning_rate(self, step):
        '''
        学习率函数，学习率随着迭代次数增加而逐渐下降
        :param step: 迭代次数
        :return: 返回学习率lr
        '''
        try:
            sigma0 = self.SIGMA0
            eta0 = self.ETA0 # 初始学习率
            t = (1000 / log(sigma0)) ** 2
            x = exp(-step / t)
            lr = eta0 * x
            return lr

        except Exception as e:
            print("the error of learning_rate is : ", e)

    def SOI(self, data, mask, flag="True"):
        '''
        网络层竞争，并输出完全竞争后的mask网络
        :param data: 归一化后的图像矩阵
        :param mask: 初始化后的网络层
        :param flag: 是否使用循环计算
        :return: 返回完全竞争后的网络层
        '''
        try:
            # get the shape of data and mask
            Height, Width = data.shape[:2]
            height, width = mask.shape[:2]

            # initiate the step
            step = 0

            #i nitiate the lr
            lr = self.ETA0

            # initiate the neighbourhood radius
            neighbourhood_radius = 200

            # competition
            while(lr > self.LR_STOP and neighbourhood_radius > self.NEIGHBOURHOOD_RADIUS_STOP):
                step += 1
                # refresh neighbourhood radius
                neighbourhood_radius = self.neighbourhood_ratio_of_active(step)

                # refresh learning rate
                lr = self.learning_rate(step)

                # extract the pixel randomly
                x = random.randint(0, Height - 1)
                y = random.randint(0, Width - 1)
                extract_data = data[x][y]

                # find the BMU(best matching unit or winner node) by euclidean or cos similarity
                BMU = [99, 0 , 0]

                # find the BMU
                if flag == "True":
                ######################
                # 单纯使用for循环(代码逻辑便于理解，耗时最长)
                ######################
                    for i in range(height):
                        for j in range(width): # 将MASK层所有节点遍历一遍
                            W = mask[i][j][:3] # 提取每个节点的系数，每个point的数据格式为[W1,W2,W3,(坐标)]，W为不同通道的像素值
                            # 计算竞争层节点与像素点间的相似度
                            # print("extract_data; ",extract_data, "W: ", W)
                            similarity = self.euclidean_distance(extract_data, W)

                            if similarity < BMU[0]: # find the minimun euclidean distance node or the maximum cos similarity
                                BMU[0] = similarity # 更新目前优胜节点的相似度
                                BMU[1:3] = [i, j]
                                mask[i][j][3:5] = [y, x]
                ######################
                #  不使用for循环(耗时短)
                ######################
                else:
                    mask_v = mask[:,:,:3] # 提取mask中的三通道像素值
                    point = np.array(extract_data) # 将从图中随机抽取的点数组化
                    # 计算mask中每个点到被抽取点的欧氏距离
                    differences = mask_v - point
                    distances = np.sqrt(np.sum(differences**2, axis=-1))
                    # 最小距离
                    min_distance = np.min(distances)
                    similarity = min_distance
                    # 最小距离的扁平化索引
                    min_distance_flat_index = np.argmin(distances)
                    # 将扁平化索引转化为二维索引
                    min_distance_index = np.unravel_index(min_distance_flat_index, distances.shape)
                    # 更新BMU
                    BMU[0] = similarity  # 更新目前优胜节点的相似度
                    BMU[1:3] = min_distance_index # 更新优胜节点中与抽取点最近的mask最优点的坐标
                    mask[min_distance_index[0]][min_distance_index[1]][3:5] = [y, x] # 更新当前最优mask点中存储的距离最近的图像点坐标(y,x)
                    # print("min_distance_index : ", min_distance_index,
                    #       "\nmin_distance : ", min_distance,
                    #       "\nsimilarity : ", BMU[0],
                    #       "\nBMU : ", BMU)
                ######################
                ######################

                # refresh the node in BMU's neighbourhood
                winner_point = BMU[1:3]
                for i in range(height):
                    for j in range(width):
                        distance_between_centerNode_and_commonNode = self.euclidean_distance((i, j), winner_point)
                        # 更新所有在邻域范围内的点
                        if distance_between_centerNode_and_commonNode < neighbourhood_radius:
                            # calculate the h_neighbourhood
                            h = self.h_neighbourhood(neighbourhood_radius, distance_between_centerNode_and_commonNode)
                            # refresh the status of Nodes (更新优胜节点和邻域内节点的状态)
                            mask[i][j][0:3] = mask[i][j][0:3] + lr * h * (extract_data - mask[i][j][0:3])
                print("---------------------------------------------------",
                      "\nNo.", step,
                      "\nextract point : ", extract_data,
                      "\nsimilarity : ", similarity,
                      "\nneighbourhood_radius : ", neighbourhood_radius,
                      "\nBMU : ", BMU,
                      "\nstep : ", step,
                      "\nlr : ", lr,
                      "\n---------------------------------------------------")

            return mask
        except Exception as e:
            print("the error of SOI is : ", e)

if __name__ == '__main__':
    try:
        path = "/home/dylanyoung/Desktop/img/6.png"

        # init SOM
        maskSize = (20, 20) # 为了方便进行并行计算，将MASK的边大小设置为5的倍数
        som = SOM(path, maskSize, 10000, 0.03, 0.05)  # 图像路径，竞争层大小，邻域半径衰减常数T1(常数越大，衰减越慢），学习率衰减截止点， 邻域半径衰减截止点
        # read image
        data = som.read_data("none", 210, 100, (100, 100)) # 参数为：是否需要对图像进行预处理(binary/resize/binary&resize/none)，二值化阈值，阈值后最大值， resize大小
        # init mask
        mask = som.init_mask()
        # 网络层竞争
        start_time = time.time()
        FLAG = "Ture"
        result_mask = som.SOI(data, mask, FLAG)
        end_time = time.time()
        print("Full time : ", end_time - start_time)

        height, width = maskSize
        maskSize = (height, width, 3)
        mask_value = np.zeros(maskSize)

        for i in range(height):
            for j in range(width):
                cv2.circle(data, (int(result_mask[i][j][3]), int(result_mask[i][j][4])), 2, (0, 0, 255), 1)
                mask_value[i][j][:3] = result_mask[i][j][:3]
        cv2.imshow("iamge", data)

        # 将mask保存为图片
        mask_value *= 255

        cv2.waitKey(0)
    except Exception as e:
        print("the error of main is : ",e)





