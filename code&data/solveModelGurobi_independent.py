# 靡不有初，鲜克有终
# zhangchenhaoseu@foxmail.com
# Southeast University
# 2025/8/16 22:34
import numpy as np
import pandas as pd
from gurobipy import *
import time

# 所有单位为秒、米、元、个
'''定义数据类。建立用于存放和调用输入data的数据结构框架'''
class Data:
    def __init__(self):  # 建立类属性
        self.nodeNum = 0  # 节点总数量（起点车场O、客户点S、虚拟聚类点C-、虚拟聚类点C+、终点车场D）
        self.N_lst = []  # 存放所有节点的id的列表,N={0,1,2,...,c+1}
        self.dummyDepot = []  # 虚拟起终点集合｛O,D｝
        self.S = []  # 客户节点集合
        self.O = None  # 起点车场
        self.D = None  # 终点车场
        self.flowOutNode = []  # {O}∪S
        self.flowInNode = []  # {D}∪S
        self.corX = []  # 节点横坐标
        self.corY = []  # 节点纵坐标
        self.costMatrix = None  # 节点与结点之间的距离矩阵
        self.customerDemand = []  # 客户S对应的需求
        self.readyTime = []  # 节点时间窗的最早时间
        self.dueTime = []  # 节点时间窗的最晚时间

        self.serviceTime = None  # 客户S的节点服务时长
        self.truckNum = None  # 车辆数量
        self.droneNum = None  # 无人机数量
        self.truckOutboundCost = None  # 卡车出库成本
        self.droneOutboundCost = None  # 无人机出库成本
        self.truckdrivingCost = None  # 卡车单位时间行驶成本
        self.droneflyingCost = None  # 无人机单位时间飞行成本
        self.truckMaxRange = None  # 卡车最大行驶距离
        self.droneMaxRange = None  # 无人机最大飞行距离
        self.truckCapacity = None  # 卡车容量
        self.droneCapacity = None  # 无人机容量
        self.truckSpeed = None  # 卡车速度
        self.droneSpeed = None  # 无人机速度
        self.earlyPunish = None  # 早到单位时间惩罚
        self.latePunish = None  # 晚到单位时间惩罚

        self.M = 100
        self.factor = 1000  # 用于在时间资源函数中扩大M的值（因为时间远大于节点规模）


'''读取数据。将对应的输入数据存放至对应的数据结构框架中'''
def readData(file_path, coeffLst):  # 文件路径，参数列表
    data = Data()
    data.serviceTime = coeffLst[0]  # 客户S的节点服务时长
    data.truckNum = coeffLst[1]  # 车辆数量
    data.droneNum = coeffLst[2]  # 无人机数量
    data.truckOutboundCost = coeffLst[3]  # 卡车出库成本
    data.droneOutboundCost = coeffLst[4]  # 无人机出库成本
    data.truckdrivingCost = coeffLst[5]  # 卡车单位时间行驶成本
    data.droneflyingCost = coeffLst[6]  # 无人机单位时间飞行成本
    data.truckMaxRange = coeffLst[7]  # 卡车最大行驶距离
    data.droneMaxRange = coeffLst[8]  # 无人机最大飞行距离
    data.truckCapacity = coeffLst[9]  # 卡车容量
    data.droneCapacity = coeffLst[10]  # 无人机容量
    data.truckSpeed = coeffLst[11]  # 卡车速度
    data.droneSpeed = coeffLst[12]  # 无人机速度
    data.earlyPunish = coeffLst[13]  # 早到单位时间惩罚
    data.latePunish = coeffLst[14]  # 晚到单位时间惩罚
    data_df = pd.read_csv(file_path)
    # 将1个起始车场（文本数据中的第一个）+ customerNum 个客户点的信息存放在对应数据结构中,此外也会建立一个虚拟终点车场
    data.nodeNum = len(data_df)  # 节点总数量（起点车场O、客户点S、虚拟聚类点C-、虚拟聚类点C+、终点车场D）
    data.O = 0
    data.D = data.nodeNum - 1

    for i in range(0, data.nodeNum):
        data.N_lst.append(data_df.loc[i, 'node_id'])  # 从0开始的所有节点，N={0,1,...,c+1}
        data.corX.append(data_df.loc[i, 'x_coord'])
        data.corY.append(data_df.loc[i, 'y_coord'])
        if data_df.loc[i, 'type'] == 0:  # 虚拟起点
            data.dummyDepot.append(data_df.loc[i, 'node_id'])
            data.flowOutNode.append(data_df.loc[i, 'node_id'])  # {O}∪S

        if data_df.loc[i, 'type'] == 4:  # 虚拟终点
            data.dummyDepot.append(data_df.loc[i, 'node_id'])
            data.flowInNode.append(data_df.loc[i, 'node_id'])  # {D}∪S

        if data_df.loc[i, 'type'] == 1:  # 客户点S
            data.S.append(data_df.loc[i, 'node_id'])
            data.customerDemand.append(data_df.loc[i, 'demand'])
            data.readyTime.append(data_df.loc[i, 'timeWindow_a'])
            data.dueTime.append(data_df.loc[i, 'timeWindow_b'])
            data.flowOutNode.append(data_df.loc[i, 'node_id'])  # {O}∪S
            data.flowInNode.append(data_df.loc[i, 'node_id'])  # {D}∪S

    # 填补成本矩阵(距离，米)
    data.costMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(0, data.nodeNum):
        for j in range(0, data.nodeNum):
            if i != j:
                data.costMatrix[i][j] = ((data.corX[i]-data.corX[j])**2+(data.corY[i]-data.corY[j])**2)**0.5
            else:
                pass

    # print("costMatrix:")
    # print(data.costMatrix)
    # print("costMatrix[4][1]:")
    # print(data.costMatrix[4][1])
    return data

def modelingAndSolve(data):
    # 建立模型
    m = Model('model')

    # 模型设置：由于存在函数printSolution，因此关闭输出;以及容许误差
    m.setParam('MIPGap', 0.01)
    # m.setParam('OutputFlag', 0)

    # 定义变量：
    # Step1.建立合适的数据结构建立变量的索引
    Z = [[[[] for _ in range(0, data.nodeNum)] for _ in range(0, data.nodeNum)] for _ in range(0,data.droneNum)]  # Z^l_ij
    T = [[[] for _ in range(0,data.nodeNum)] for _ in range(0,data.droneNum)]  # t^l_i
    U = [[[] for _ in range(0,data.nodeNum)] for _ in range(0,data.droneNum)]  # u^l_i
    P_a = [[[] for _ in range(0,data.nodeNum)] for _ in range(0,data.droneNum)]   # pa^l_i
    P_b = [[[] for _ in range(0,data.nodeNum)] for _ in range(0,data.droneNum)]   # pb^l_i

    # Step2.根据数据结构向模型中添加对应下标的变量
    '______添加决策变量Z______'
    for l in range(0,data.droneNum):
        for i in range(0, data.nodeNum):
            for j in range(0,data.nodeNum):
                Z[l][i][j] = m.addVar(vtype=GRB.BINARY, name=f"Z_{l}_{i}_{j}")

    '______添加决策变量T______'
    for l in range(0, data.droneNum):
        for i in range(0, data.nodeNum):
            T[l][i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0,name=f"T_{l}_{i}")

    '______添加决策变量U______'
    for l in range(0, data.droneNum):
        for i in range(0, data.nodeNum):
            U[l][i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U_{l}_{i}")

    '______添加决策变量Pa______'
    for l in range(0, data.droneNum):
        for i in range(0, data.nodeNum):
            P_a[l][i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"P_a_{l}_{i}")

    '______添加决策变量Pb______'
    for l in range(0, data.droneNum):
        for i in range(0, data.nodeNum):
            P_b[l][i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"P_b_{l}_{i}")

    m.update()
    # var_lst = m.getVars()
    # print(var_lst)

    # Step3.定义目标函数
    obj1 = 0
    obj2 = 0
    obj3 = 0

    # 目标函数第一项：出车成本
    for l in range(0,data.droneNum):
        for j in data.S:
            obj1 += data.droneOutboundCost*Z[l][data.O][j]
    # 目标函数第二项：行驶成本
    for l in range(0,data.droneNum):
        for i in data.flowOutNode:
            for j in data.flowInNode:
                if i != j:
                    dij = data.costMatrix[i][j]
                    timeCostDrone = data.droneflyingCost * dij / data.droneSpeed
                    obj2 += timeCostDrone* Z[l][i][j]
    # 目标函数第三项：时间窗成本
    for l in range(0,data.droneNum):
        for i in data.S:
            obj3 += data.earlyPunish*P_a[l][i]
            obj3 += data.latePunish*P_b[l][i]
    obj = obj1 + obj2 + obj3
    m.setObjective(obj, sense=GRB.MINIMIZE)

    # Step4.定义约束条件:
    # 1.无人机客户服务约束
    num = 0
    for c in data.S:
        expr = LinExpr(0)
        for l in range(0,data.droneNum):
            for i in data.flowOutNode:
                if i != c:
                    expr.addTerms(1,Z[l][i][c])
        num += 1
        m.addConstr(expr == 1, f'C1_{num}')

    # 2.无人机起点流平衡约束
    num = 0
    for l in range(0,data.droneNum):
        expr = LinExpr(0)
        for j in data.flowInNode:
            expr.addTerms(1, Z[l][data.O][j])
        num += 1
        m.addConstr(expr == 1,f'C2_{num}')

    # 3.无人机终点流平衡约束
    num = 0
    for l in range(0, data.droneNum):
        expr = LinExpr(0)
        for i in data.flowOutNode:
            expr.addTerms(1, Z[l][i][data.D])
        num += 1
        m.addConstr(expr == 1, f'C3_{num}')

    # 4.无人机中间点流平衡约束
    num = 0
    for l in range(0, data.droneNum):
        for c in data.S:
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            for i in data.flowOutNode:
                expr1.addTerms(1, Z[l][i][c])
            for j in data.flowInNode:
                expr2.addTerms(1, Z[l][c][j])
            num += 1
            m.addConstr(expr1 - expr2 == 0, f'C4_{num}')

    # 5.无人机行驶里程约束
    num = 0
    for l in range(0,data.droneNum):
        expr = LinExpr(0)
        for i in data.flowOutNode:
            for j in data.flowInNode:
                if i != j:
                    dij = data.costMatrix[i][j]
                    expr.addTerms(dij,Z[l][i][j])
        num += 1
        m.addConstr(expr <= data.droneMaxRange, f'C5_{num}')

    # 6.无人机容量约束
    num = 0
    for l in range(0, data.droneNum):
        expr = LinExpr(0)
        for i in data.flowOutNode:
            for j in data.flowInNode:
                if i != j:
                    if j != data.D:
                        j_index = data.S.index(j)
                        qj = data.customerDemand[j_index]
                        expr += qj * Z[l][i][j]
        num += 1
        m.addConstr(expr <= data.droneCapacity, f'C6_{num}')

    # 7.无人机破子圈约束
    num = 0
    for l in range(0,data.droneNum):
        for i in data.flowOutNode:
            for j in data.flowInNode:
                if i !=j:
                    expr = U[l][i]- U[l][j] + data.M*Z[l][i][j] - data.M + 1
                    num += 1
                    m.addConstr(expr <= 0, f'C7_{num}')

    # 8.无人机行驶时间资源拓展约束
    num = 0
    for l in range(0, data.droneNum):
        for i in data.flowOutNode:
            for j in data.flowInNode:
                if i != j:
                    if j != data.D:
                        dij = data.costMatrix[i][j]
                        timeCost = dij/data.droneSpeed
                        expr = T[l][i] + timeCost + data.serviceTime + data.factor*data.M*Z[l][i][j] - data.factor*data.M - T[l][j]
                        num += 1
                        m.addConstr(expr <= 0, f'C8_{num}')

    # 9.早到时间逻辑约束
    for l in range(0,data.droneNum):
        for i in data.S:
            i_index = data.S.index(i)
            a_c = data.readyTime[i_index]  # 时间窗左侧
            sum_ = 0
            for j in data.flowInNode:
                if j !=i :
                    sum_ += Z[l][i][j]
            expr = P_a[l][i] - a_c + T[l][i] +data.factor*data.M - data.factor*data.M*sum_
            num += 1
            m.addConstr(expr >= 0, f'C9_{num}')

    # 10.晚到时间逻辑约束
    for l in range(0,data.droneNum):
        for i in data.S:
            i_index = data.S.index(i)
            b_c = data.dueTime[i_index]  # 时间窗右侧
            expr = P_b[l][i] - T[l][i] + b_c
            num += 1
            m.addConstr(expr >= 0, f'C10_{num}')

    # 记录求解开始时间
    start_time = time.time()
    # 求解
    m.optimize()
    m.write('model.lp')
    if m.status == GRB.OPTIMAL:
        print("-" * 20, "求解成功", '-' * 20)
        # 输出求解总用时
        print(f"求解时间: {time.time() - start_time} s")
        print(f"独立无人机的目标函数为（成本）: {m.ObjVal}")
        var_lst = m.getVars()
        print('var_lst:')
        for v in m.getVars():
            split_arr = re.split(r"_", v.VarName)  # 将gurobi形式的变量进行拆解，便于利用数据结构实现实现存储
            if split_arr[0] == 'Z' and round(v.x) != 0:
                print(v)
            if split_arr[0] == 'U' and round(v.x) != 0:
                print(v)
            if split_arr[0] == 'T' and round(v.x) != 0:
                print(v)

    else:
        print("无解")
    return m


'''主函数，调用函数实现问题的求解'''
if __name__ =="__main__":
    # 数据集路径
    data_path = r'C:\Users\张晨皓\Desktop\孙哥论文\code\data\nodeIndependent.txt'  # 这里是节点文件
    coeffLst = [60,2,3,50,10,0.083,0.0017,100000,10000,15,5,4.17,12.5,0.001,0.1]   # 客户S的节点服务时长 # 卡车数量 # 无人机数量
    data = readData(data_path, coeffLst)
    # 检验数据读取结果
    print(f"网络节点总数量为:{data.nodeNum}")
    print(f"网络节点的索引为:{data.N_lst}")
    print(f"起点索引为:{data.O}元/秒")
    print(f"终点索引为:{data.D}元/秒")
    print(f"网络中虚拟起终点索引为:{data.dummyDepot}")
    print(f"网络中的客户索引为:{data.S}")

    print(f"客户需求分别为:{data.customerDemand}")
    print(f"最早时间窗为:{data.readyTime}")
    print(f"最晚时间窗为:{data.dueTime}")

    print(f"客户S的节点服务时长为:{data.serviceTime}秒")
    print(f"车辆数量为:{data.truckNum}个")
    print(f"无人机数量为:{data.droneNum}个")
    print(f"卡车出库成本为:{data.truckOutboundCost}元/次")
    print(f"无人机出库成本为:{data.droneOutboundCost}元/次")
    print(f"卡车单位时间行驶成本为:{data.truckdrivingCost}元/秒")
    print(f"无人机单位时间飞行成本为:{data.droneflyingCost}元/秒")
    print(f"车最大行驶距离为:{data.truckMaxRange}米")
    print(f"无人机最大飞行距离为:{data.droneMaxRange}米")
    print(f"卡车容量为:{data.truckCapacity}")
    print(f"无人机容量为:{data.droneCapacity}")
    print(f"卡车速度为:{data.truckSpeed}米/秒")
    print(f"无人机速度为:{data.droneSpeed}米/秒")
    print(f"早到单位时间惩罚为:{data.earlyPunish}元/秒")
    print(f"迟到单位惩罚为:{data.latePunish}元/秒")


    # 求解
    modelingAndSolve(data)

