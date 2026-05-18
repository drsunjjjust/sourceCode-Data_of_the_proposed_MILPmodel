# 靡不有初，鲜克有终
# zhangchenhaoseu@foxmail.com
# Southeast University
# 2025/4/24 21:17
import numpy as np
import copy
import pandas as pd
from gurobipy import *
import matplotlib.pyplot as plt
import time


"""——————————————— 2-opt函数 ———————————————"""
def calculate_total_distance(path, cost_matrix, origin_index):  # path不含起终点
    """计算路径的总距离，包括出发和返回origin_index"""
    total = cost_matrix[origin_index][path[0]]  # 出发到第一个客户
    for i in range(len(path) - 1):
        total += cost_matrix[path[i]][path[i + 1]]
    total += cost_matrix[path[-1]][origin_index]  # 最后一个客户返回origin_index
    return total  # 距离含起终点


def two_opt(serviceLst, cost_matrix, origin_index):  # serviceLst不含起终点
    """使用2-opt算法优化访问顺序"""
    best_path = serviceLst[:]
    best_cost = calculate_total_distance(best_path, cost_matrix, origin_index)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best_path) - 1):
            for j in range(i + 1, len(best_path)):
                # 生成新的路径
                new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                new_cost = calculate_total_distance(new_path, cost_matrix, origin_index)
                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost
                    improved = True
                    break  # 可以加速收敛
            if improved:
                break

    return best_path  # 不含起终点


"""——————————————— Step 1 建立类 ———————————————"""
class Data:
    def __init__(self):  # 建立类属性
        self.nodeLst = []  # 存放所有节点实例的列表
        self.depotLst = []  # 车场集合
        self.customerLst = []  # 客户节点集合
        self.clusterLst = []  # 聚类点集合
        self.truckLst = []  # 卡车集合
        self.costMatrix = None  # 节点与结点之间的距离矩阵
        self.earlyPunish = None  # 早到单位时间惩罚
        self.latePunish = None  # 晚到单位时间惩罚
        self.totalTruckOutboundCost = 0  # 所有车辆出库成本
        self.totalTruckTravelingCost = 0  # 所有卡车行驶成本
        self.totalDroneOutboundCost = 0  # 所有无人机出库成本
        self.totalDroneTravelingCost = 0  # 所有无人机飞行成本
        self.totalPunishCost = 0  # 总时间窗惩罚成本=早到惩罚+晚到惩罚
        self.dronePathMtx = None
        self.truckPathMtx = None


class Customer:  # 实例化客户点
    def __init__(self):  # 建立类属性
        self.index = None  # 客户索引
        self.demand = None  # 客户需求
        self.cluster = None  # 客户属于哪个聚类点
        self.timeWindow = None  # 客户时间窗
        self.serviceTime = None  # 客户服务时长
        self.serviceFlag = 0  # 客户服务指示，0为未被服务，1为被服务（无人机是否服务该客户）
        self.corX = None  # 节点横坐标
        self.corY = None  # 节点纵坐标


class Cluster:  # 实例化聚类点
    def __init__(self):  # 建立类属性
        self.index = None  # 聚类点索引
        self.totalDemand = None  # 聚类点总需求
        self.customerLst = None  # 聚类点客户集合
        self.arrivalTime = None  # 卡车在当前聚类点的到达时刻
        self.waitingTime = None  # 卡车在当前聚类点的等待时间
        self.leavingTime = None  # 卡车在当前聚类点的离开时刻
        self.serviceFlag = 0   # 客户服务指示，0为未被服务，1为被服务（卡车是否访问该聚类点）
        self.corX = None  # 节点横坐标
        self.corY = None  # 节点纵坐标


class Depot:  # 实例化车场
    def __init__(self):  # 建立类属性
        self.index = None  # 车场索引
        self.corX = None   # 节点横坐标
        self.corY = None   # 节点纵坐标


class Truck:  # 实例化卡车
    def __init__(self):  # 建立类属性
        self.index = None  # 卡车索引
        self.outBoundCost = None  # 出库成本
        self.drivingCost = None  # 卡车单位时间行驶成本
        self.carryDronesLst = []  # 所携带的无人机实例列表
        self.maxRange = None  # 卡车最大行驶距离
        self.capacity = None  # 卡车容量
        self.speed = None  # 卡车速度
        self.path = []  # 卡车路径，经过的节点集合


class Drone:  # 实例化无人机
    def __init__(self):  # 建立类属性
        self.index = None  # 无人机索引
        self.truckBelong = None  # 无人机所属的卡车索引
        self.outBoundCost = None  # 出库成本
        self.flyingCost = None  # 无人机单位时间飞行成本
        self.maxRange = None  # 无人机最大飞行距离
        self.capacity = None  # 无人机容量
        self.speed = None  # 无人机速度
        self.path = None  # 无人机路径，经过的节点字典（键是聚类点索引，值是访问的客户节点）


"""——————————————— Step 2 实例化并填补实例属性 ———————————————"""
def readData(file_path, coeffLst):  # 文件路径，外部参数列表
    data = Data()
    data_df = pd.read_csv(file_path)
    clusterKeys = []  # 为了建立drone.dronePath初始路径，需要基于聚类点去建立对应的字典
    for i in range(0,len(data_df)):
        if data_df.loc[i, 'type'] == 0:  # 车场O
            depot = Depot()
            depot.index = data_df.loc[i, 'node_id']
            depot.corX = data_df.loc[i, 'x_coord']
            depot.corY = data_df.loc[i, 'y_coord']
            data.depotLst.append(depot)
            data.nodeLst.append(depot)

        if data_df.loc[i, 'type'] == 1:  # 客户点S
            customer = Customer()
            customer.index = data_df.loc[i, 'node_id']
            customer.demand = data_df.loc[i, 'demand']
            customer.cluster = data_df.loc[i, 'cluster']  # 客户属于哪个聚类点
            customer.timeWindow = [data_df.loc[i, 'timeWindow_a'],data_df.loc[i, 'timeWindow_b']]
            customer.serviceTime = coeffLst[0]  # 客户服务时长
            customer.corX = data_df.loc[i, 'x_coord']  # 节点横坐标
            customer.corY = data_df.loc[i, 'y_coord']  # 节点纵坐标
            data.customerLst.append(customer)
            data.nodeLst.append(customer)

        if data_df.loc[i, 'type'] == 2:  # 聚类点C，当进入到该结构体内部时，前面两个列表data.depot、data.customerLst，都已经填完了（由文件和节点命名规则决定的）
            cluster = Cluster()
            cluster.index = data_df.loc[i, 'node_id']  # 聚类点索引
            cluster.totalDemand = data_df.loc[i, 'demand']  # 聚类点总需求
            cluster.corX = data_df.loc[i, 'x_coord']  # 节点横坐标
            cluster.corY = data_df.loc[i, 'y_coord']  # 节点纵坐标
            cluster.customerLst = []  # 聚类点客户实例集合
            cluster.waitingTime = 0

            for _index in range(0,len(data.customerLst)):
                if data.customerLst[_index].cluster == cluster.index:
                    cluster.customerLst.append(data.customerLst[_index])
            data.clusterLst.append(cluster)
            data.nodeLst.append(cluster)
            clusterKeys.append(cluster.index)

    for truckIndex in range(0, coeffLst[1]):  # 卡车数量
        truck = Truck()
        truck.index = truckIndex  # 卡车索引
        truck.outBoundCost = coeffLst[3]  # 卡车出库成本
        truck.drivingCost = coeffLst[5]  # 卡车单位时间行驶成本
        truck.maxRange = coeffLst[7]  # 卡车最大行驶距离
        truck.capacity = coeffLst[9]  # 卡车容量
        truck.speed = coeffLst[11]  # 卡车速度
        truck.path = []  # 卡车路径，经过的节点集合
        for droneIndex in range(0,coeffLst[2]):   # 所携带的无人机
            drone = Drone()
            drone.index = droneIndex  # 无人机索引
            drone.truckBelong = truckIndex  # 无人机所属的卡车索引
            drone.outBoundCost = coeffLst[4]  # 无人机出库成本
            drone.flyingCost = coeffLst[6]  # 无人机单位时间飞行成本
            drone.maxRange = coeffLst[8]  # 无人机最大飞行距离
            drone.capacity = coeffLst[10]  # 无人机容量
            drone.speed = coeffLst[12]  # 无人机速度
            drone.path = {key: [] for key in clusterKeys}  # 无人机路径，经过的节点字典（键是聚类点索引，值是访问的客户节点）
            truck.carryDronesLst.append(drone)
        data.truckLst.append(truck)
    data.earlyPunish = coeffLst[13]  # 早到单位时间惩罚
    data.latePunish = coeffLst[14]  # 晚到单位时间惩罚

    # 填补成本矩阵(距离，米)
    data.costMatrix = np.zeros((len(data.nodeLst), len(data.nodeLst)))
    for i in range(0, len(data.nodeLst)):
        for j in range(0, len(data.nodeLst)):
            if i != j:
                data.costMatrix[i][j] = ((data.nodeLst[i].corX-data.nodeLst[j].corX)**2+(data.nodeLst[i].corY-data.nodeLst[j].corY)**2)**0.5
            else:
                pass

    # print("costMatrix:")
    # print(data.costMatrix)
    # print("costMatrix[4][1]:")
    # print(data.costMatrix[4][1])
    return data


def TWCRCGRH(data,coeffLst):
    # 使用TWCRCGRH+maxSavings确定二阶段各个无人机的解
    """___________（1）确定每个无人机在聚类点出发的路线___________"""
    # 分配/匹配问题，确定每个聚类点，每个无人机需要服务哪些客户
    droneServiceMtx = []  # 三维列表，用于记录每个聚类点中，不同无人机服务客户情况：droneServiceMtx[聚类点索引][无人机索引] = [所服务客户的索引]
    droneDistanceMtx = []  # 二维列表，用于记录每个聚类点中，不同无人机执行任务后的累计飞行距离 droneDistanceMtx[聚类点索引][无人机索引] = 该无人机的累计服务距离
    dronePathMtx = []  # 三维列表，用于记录每个聚类点中，不同无人机服务客户的路径（通过2-opt调整后的）：dronePathMtx[聚类点索引][无人机索引] = [所服务客户的索引]  表示路径
    for cluster in data.clusterLst:  # 对于一个聚类点
        cluster.customerLst.sort(key=lambda customer: customer.demand)  # 按照需求从小到大，排序原客户实例列表
        droneLoadLst = [0 for _droneIndex in range(0,coeffLst[2])]  # 记录无人机的累计负载
        dronedistanceLst = [0 for _droneIndex in range(0,coeffLst[2])]  # 记录无人机的累计飞行距离
        droneServiceLst = [[] for _droneIndex in range(0,coeffLst[2])]  # 记录当前无人机所服务的客户
        dronePathMtx.append(droneServiceLst)  # 空的路径，用于记录后续2-opt调整后的路径

        for _droneIndex in range(0,coeffLst[2]):  # 对于每架无人机
            for customer in cluster.customerLst:  # 对于排序后列表中的每个客户
                if customer.serviceFlag == 0 and (droneLoadLst[_droneIndex]+customer.demand) <= coeffLst[10]:  # 若没有被服务，且服务这一个客户不会导致当前无人机的累计荷载超过无人机容量
                    if len(droneServiceLst[_droneIndex]) >= 1:
                        formerNode = droneServiceLst[_droneIndex][-1]  # 无人机的前一个起飞节点索引
                    else:
                        formerNode = cluster.index  # 无人机的前一个起飞节点索引

                    if (dronedistanceLst[_droneIndex]+data.costMatrix[formerNode][customer.index]) <= coeffLst[8]:  # 也不会超过无人机飞行距离限制
                        droneServiceLst[_droneIndex].append(customer.index)  # 将该客户加入到该无人机
                        customer.serviceFlag = 1  # 将该客户设置为已服务
                        droneLoadLst[_droneIndex] += customer.demand  # 更新无人机的负载
                        dronedistanceLst[_droneIndex] += data.costMatrix[formerNode][customer.index]  # 更新无人机的飞行距离
                else:
                    pass
        droneServiceMtx.append(droneServiceLst)
        droneDistanceMtx.append(dronedistanceLst)
    # 路由问题，确定每个聚类点，所执行服务的无人机，应该以什么样的路线（次序）去服务上面分配好的客户（使用2-opt进行优化）
    for _clusterIndex in range(0,len(droneServiceMtx)): # 对于当前的cluster
        # 使用2-opt需要距离矩阵，距离矩阵中需要知道起点（cluster）的索引clusterIndex
        clusterIndex = data.clusterLst[_clusterIndex].index
        for _droneIndex in range(0,len(droneServiceMtx[_clusterIndex])):  # 对于无人机_droneIndex而言
            serviceCustomerLst = droneServiceMtx[_clusterIndex][_droneIndex]
            if len(serviceCustomerLst) >= 3:  # 只有这个无人机访问3个及以上客户的时候，2-opt才具有优化效果
               # 2-opt optimization
               dronePathMtx[_clusterIndex][_droneIndex] = two_opt(serviceCustomerLst, data.costMatrix, clusterIndex)
            else:
                dronePathMtx[_clusterIndex][_droneIndex] = droneServiceMtx[_clusterIndex][_droneIndex]

    # 确定每个聚类点的等待时间(距离/无人机速度)
    for _clusterIndex in range(0, len(data.clusterLst)):  # 对于一个聚类点
        cluster = data.clusterLst[_clusterIndex]
        clusterIndex = cluster.index
        timeSpend = [0 for _droneIndex in range(0,coeffLst[2])]  # 记录在当前cluster下不同无人机所花费的时间
        for _droneIndex in range(0,coeffLst[2]):  # 对于每一个无人机的路径
            serviceNum = len(dronePathMtx[_clusterIndex][_droneIndex])  #这个路径的客户服务数量
            if serviceNum != 0:
                total_distance = calculate_total_distance(dronePathMtx[_clusterIndex][_droneIndex], data.costMatrix, clusterIndex)
                timeSpend[_droneIndex] = total_distance/coeffLst[12] + serviceNum*coeffLst[0]
            else:
                timeSpend[_droneIndex] = 0
        cluster.waitingTime = max(timeSpend)
    data.dronePathMtx = dronePathMtx

    """___________（2）确定每个卡车在仓库的出发路线___________"""
    # 分配/匹配问题，确定每个卡车需要服务哪些客户
    data.clusterLst.sort(key=lambda cluster: cluster.totalDemand)  # 按照需求从小到大，排序原聚类点实例列表
    truckLoadLst = [0 for _truckIndex in range(0, coeffLst[1])]  # 记录不同卡车的累计负载
    truckDistanceLst = [0 for _truckIndex in range(0, coeffLst[1])]  # 记录不同卡车的累计行驶距离
    truckServiceLst = [[] for _truckIndex in range(0, coeffLst[1])]  # 记录不同卡车所服务的客户
    truckPathLst= [[] for _truckIndex in range(0, coeffLst[1])]   # 空的路径，用于记录后续2-opt调整后的路径
    for _truckIndex in range(0, coeffLst[1]):  # 对于每辆卡车
        for cluster in data.clusterLst:  # 对于排序后列表中的每个聚类点
            if cluster.serviceFlag == 0 and truckLoadLst[_truckIndex] + cluster.totalDemand <= coeffLst[9]:  # 若没有被服务，且服务这一个聚类点不会导致当前卡车的累计荷载超过卡车容量
                if len(truckServiceLst[_truckIndex]) >= 1:
                    formerNode = truckServiceLst[_truckIndex][-1]  # 卡车的前一个行驶过的聚类节点索引
                else:
                    formerNode = data.depotLst[0].index  # 无人机的前一个起飞节点索引

                if (truckDistanceLst[_truckIndex] + data.costMatrix[formerNode][cluster.index]) <= coeffLst[7]:  # 也不会超过卡车行驶距离限制

                    truckServiceLst[_truckIndex].append(cluster.index) # 将该聚类点加入到该卡车访问的列表中
                    cluster.serviceFlag = 1  # 将该聚类点设置为已访问
                    truckLoadLst[_truckIndex] += cluster.totalDemand  # 更新卡车的负载
                    truckDistanceLst[_truckIndex] += data.costMatrix[formerNode][cluster.index]  # 更新卡车行驶距离
            else:
                pass

    # 路由问题，确定所执行服务的卡车，应该以什么样的路线（次序）去服务上面分配好的聚类点（使用2-opt+聚类点的等待时间进行优化）
    for _truckIndex in range(0, len(truckServiceLst)):  # 对于卡车_truckIndex而言
        serviceClusterLst = truckServiceLst[_truckIndex]
        if len(serviceClusterLst) >= 3:  # 只有这个无人机访问3个及以上客户的时候，2-opt才具有优化效果
            # 2-opt optimization considering waiting time
            truckPathLst[_truckIndex] = two_opt(serviceClusterLst, data.costMatrix,data.depotLst[0].index)
        else:
            truckPathLst[_truckIndex] = truckServiceLst[_truckIndex]

    data.truckPathMtx = truckPathLst

    """_______（3）【Ⅱ解+Ⅰ解=输出初始解，得到目标函数值】_______"""
    timeStampMtx = []  # 三维列表，用来记录多辆车、多个路线的到达时间：timeStampMtx[车辆索引][路线索引] = [这条路线上到达各个聚类点的时刻]
    print("data.truckPathMtx",data.truckPathMtx)
    for _truckIndex in range(0, len(data.truckPathMtx)):
        if len(data.truckPathMtx[_truckIndex]) >= 1: # 存在路线
            data.totalTruckOutboundCost += coeffLst[3] # 该卡车出库成本
            currentTruckPath = data.truckPathMtx[_truckIndex]  # 当前的卡车路线，起点默认是0
            timeStampLst = [0 for _ in range(0,len(currentTruckPath))]  # 记录这条卡车路线上到达各个聚类点的时间戳
            timeStamp = 0 # 一个会累计增加的时间戳，记录车辆行驶时间
            endNode = data.depotLst[0].index
            formerTruckNode = data.depotLst[0].index

            for _clusterIndex in range(0,len(currentTruckPath)):  # 从当前卡车路线中依次从0出发，读取聚类点
                distance = data.costMatrix[formerTruckNode][currentTruckPath[_clusterIndex]]  # 车辆行驶距离
                timeCost = distance / coeffLst[11]
                data.totalTruckTravelingCost += timeCost*coeffLst[5]  # 计算卡车的行驶成本
                waitingTime = 0
                for cluster in data.clusterLst:  # 调取waitingTime
                    print(cluster.index,currentTruckPath[_clusterIndex])
                    if cluster.index == currentTruckPath[_clusterIndex]:
                        waitingTime = cluster.waitingTime
                timeStamp += timeCost  # 卡车时间戳
                print(f"当前卡车从{formerTruckNode}出发，到达{currentTruckPath[_clusterIndex]}，距离是{distance}m，用时{timeCost}s，到达时刻是{timeStamp}s，等待时间是{waitingTime}s")
                clusterArrivalTime = 0 # 记录到达这一聚类点的时刻
                for cluster in data.clusterLst:  # 设置到达时刻和离开时hike
                    if cluster.index == currentTruckPath[_clusterIndex]:
                        cluster.arrivalTime = timeStamp
                        clusterArrivalTime = timeStamp
                        cluster.leavingTime = timeStamp + waitingTime
                # 计算无人机的出库成本、飞行成本、时间窗惩罚成本
                formerDroneNode = data.clusterLst[_clusterIndex].index  # 从当前聚类点出发
                droneEndNode = data.clusterLst[_clusterIndex].index  # 回到当前聚类点
                for _droneIndex in range(0,len(data.dronePathMtx[_clusterIndex])):  # 读取当前聚类点对应的客户访问情况，对于当前无人机而言
                    if len(data.dronePathMtx[_clusterIndex][_droneIndex]) >= 1:  # 这个无人机有路线
                        currentdronePath = data.dronePathMtx[_clusterIndex][_droneIndex]  # 当前无人机路径，起点默认是0当前聚类点
                        data.totalDroneOutboundCost += coeffLst[4]  # 计算无人机出库成本
                        DronetimeCost = 0  # 无人机累计用时
                        for _customerIndex in range(0,len(currentdronePath)):  #对于从聚类点_clusterIndex出发的、访问了客户的_droneIndex，所访问的客户_customerIndex而言
                            distance = data.costMatrix[formerDroneNode][currentdronePath[_customerIndex]]  # 无人机飞行距离
                            DronetimeCost += distance / coeffLst[12]  # 无人机飞行用时
                            data.totalDroneTravelingCost += DronetimeCost * coeffLst[6]  # 计算无人机飞行成本
                            DronetimeStamp = clusterArrivalTime + DronetimeCost  # 无人机到达这一节点的时间
                            for customer in data.customerLst:
                                if currentdronePath[_customerIndex] == customer.index:
                                    TW = customer.timeWindow
                                    # 计算时间窗惩罚
                                    if DronetimeStamp <= TW[0]:  # 早到
                                        data.totalPunishCost += data.earlyPunish*(TW[0]-DronetimeStamp)
                                    if DronetimeStamp >= TW[0]:  # 晚到
                                        data.totalPunishCost += data.latePunish*(DronetimeStamp-TW[0])

                            DronetimeCost += waitingTime
                            formerDroneNode = data.dronePathMtx[_clusterIndex][_droneIndex][_customerIndex]

                        distance = data.costMatrix[currentdronePath[-1]][droneEndNode]  # 无人机返程行驶距离
                        timeCost = distance / coeffLst[12]  # 无人机飞行用时
                        data.totalDroneTravelingCost += timeCost * coeffLst[6]   # 计算无人机的飞行成本

                timeStampLst[_clusterIndex] = timeStamp
                timeStamp += waitingTime
                formerTruckNode = currentTruckPath[_clusterIndex]

            distance = data.costMatrix[currentTruckPath[-1]][endNode]  # 车辆行驶距离
            timeCost = distance / coeffLst[11]
            data.totalTruckTravelingCost += timeCost * coeffLst[5]  # 计算卡车的行驶成本

            timeStampMtx.append(timeStampLst)

    return timeStampMtx,data


'''主函数，调用函数实现问题的求解'''
if __name__ =="__main__":
    # 数据集路径
    data_path = r'C:\Users\张晨皓\Desktop\孙哥论文\code\data\nodeForHeuristic.txt'  # 这里是节点文件
    coeffLst = [60,2,2,50,10,0.083,0.0017,100000,10000,15,5,4.17,12.5,0.001,0.1]
    data = readData(data_path, coeffLst)
    # 检验数据读取结果
    print(f"网络节点总数量为:{len(data.nodeLst)}")
    print(f"网络中车场第一个实例索引为:{data.depotLst[0].index}")
    print(f"网络中客户第一个实例索引为:{data.customerLst[0].index}")
    print(f"虚拟聚类点第一个实例索引为:{data.clusterLst[0].index}")
    print(f"第一个聚类点对应的客户索引为:{data.clusterLst[0].customerLst}")

    print(f"第一个客户需求为:{data.customerLst[0].demand}")
    print(f"第一个聚类点需求:{data.clusterLst[0].totalDemand}")

    print(f"第一个客户的节点服务时长为:{data.customerLst[0].serviceTime}秒")
    print(f"车辆数量为:{len(data.truckLst)}辆")
    print(f"第一辆车携带的无人机数量为:{len(data.truckLst[0].carryDronesLst)}架")
    print(f"第一辆卡车出库成本为:{data.truckLst[0].outBoundCost}元/次")
    print(f"第一辆卡车第一架无人机出库成本为:{data.truckLst[0].carryDronesLst[0].outBoundCost}元/次")
    print(f"第一辆卡车行驶成本为:{data.truckLst[0].drivingCost}元/秒")
    print(f"第一辆卡车第一架无人机单位时间飞行成本为:{data.truckLst[0].carryDronesLst[0].flyingCost}元/秒")

    print(f"第一辆卡车最大行驶距离为:{data.truckLst[0].maxRange}米")
    print(f"第一辆卡车第一架无人机最大飞行距离为:{data.truckLst[0].carryDronesLst[0].maxRange}米")

    print(f"第一辆卡车容量为:{data.truckLst[0].capacity}")
    print(f"第一辆卡车第一架无人机容量为:{data.truckLst[0].carryDronesLst[0].capacity}")
    print(f"第一辆卡车速度为:{data.truckLst[0].speed}米/秒")
    print(f"第一辆卡车第一架无人机速度为:{data.truckLst[0].carryDronesLst[0].speed}米/秒")

    print(f"第一辆卡车路径为:{data.truckLst[0].path}")
    print(f"第一辆卡车第一架无人机路径为:{data.truckLst[0].carryDronesLst[0].path}")

    print(f"早到单位时间惩罚为:{data.earlyPunish}元/秒")
    print(f"迟到单位惩罚为:{data.latePunish}元/秒")

    # 求解
    timeStampMtx,data = TWCRCGRH(data,coeffLst)
    print(data.dronePathMtx)
    print(data.truckPathMtx)
    # print(timeStampMtx)
    # print(data.costMatrix)
    print(f"聚类点{data.clusterLst[0].index}的等待时间是{data.clusterLst[0].waitingTime}")
    print(f"聚类点{data.clusterLst[1].index}的等待时间是{data.clusterLst[1].waitingTime}")
    print(f"聚类点{data.clusterLst[2].index}的等待时间是{data.clusterLst[2].waitingTime}")
    print(f"【卡车】总出库成本{data.totalTruckOutboundCost},行驶成本{data.totalTruckTravelingCost}")
    print(f"【无人机】总出库成本{data.totalDroneOutboundCost},飞行成本{data.totalDroneTravelingCost}")
    print(f"【时间窗】惩罚成本{data.totalPunishCost}")