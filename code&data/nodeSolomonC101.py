# 靡不有初，鲜克有终
# zhangchenhaoseu@foxmail.com
# Southeast University
# 2025/8/26 10:21
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"


ori_dataX = [
        40,45,45,42,42,42,40,40,38,38,35,35,25,22,22,20,20,18,15,15,
        30,30,28,28,25,25,25,23,23,20,20,10,10,8,8,5,5,2,0,0,
        35,35,33,33,32,30,30,30,28,28,26,25,25,44,42,42,40,40,38,38,
        35,50,50,50,48,48,47,47,45,45,95,95,53,92,53,45,90,88,88,87,
        85,85,75,72,70,68,66,65,65,63,60,60,67,65,65,62,60,60,58,55,55
    ]

ori_dataY = [
        50,68,70,66,68,65,69,66,68,70,66,69,85,75,85,80,85,75,75,80,
        50,52,52,55,50,52,55,52,55,50,55,35,40,40,45,35,45,40,40,45,
        30,32,32,35,30,30,32,35,30,35,32,30,35,5,10,15,5,15,5,15,
        5,30,35,40,30,40,35,40,30,35,30,35,30,30,35,65,35,30,35,30,
        25,35,55,55,58,60,55,55,60,58,55,60,85,85,82,80,80,85,75,80,85
    ]

dataX = []
dataY = []
for i in range(0,len(ori_dataX)):
    dataX.append(ori_dataX[i]*100)

for i in range(0,len(ori_dataY)):
    dataY.append(ori_dataY[i]*100)
# 将数据存入DataFrame
data = {
    "CUST NO.": list(range(0, 101)),
    "XCOORD.": dataX,
    "YCOORD.": dataY
}

df = pd.DataFrame(data)

# 绘图
# 坐标轴范围限制在 0~100

plt.figure(figsize=(10,10))
plt.grid(True, zorder=0)
plt.xlim(0, 10000)
plt.ylim(0, 10000)
plt.scatter(df["XCOORD."], df["YCOORD."], c="blue", s=30, label="Customers",zorder=2)
plt.scatter(df["XCOORD."][0], df["YCOORD."][0], c="red", s=80, marker="s", label="Depot (0)",zorder=2)

# 添加节点编号
for i, row in df.iterrows():
    plt.text(row["XCOORD."], row["YCOORD."], str(row["CUST NO."]), fontsize=9, ha="left", va="bottom",zorder=3)

plt.title("Customer Node Distribution")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
# plt.legend()

plt.show()
