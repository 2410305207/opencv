使用Keras进行深度学习

端到端无人驾驶

所谓 end-to-end 无人驾驶模型，指的是由传感器的输入，直接决定车的行为，例如油门，刹车，方向等。简单来讲，可以利用机器学习的算法直接学习人类司机的驾驶行为：首先，人类司机驾驶安装有各种传感器 (例如摄像头) 的汽车来收集数据；然后，用传感器收集的数据作为输入，相应的人类行为数据作为输出label，训练一个机器学习模型，例如，如果摄像头发现前方有行人，司机行为应当是刹车；最后，将训练好的模型应用于无人车上。这种方法又叫做行为复制，其原理直观，避免了无人车系统中各种复杂的问题，是无人车中很有趣的一个topic。

模拟器：
Self-Driving Simulator Platform：

Linux：https://link.zhihu.com/?target=https%3A//d17h27t6h515a5.cloudfront.net/topher/2017/February/58983558_beta-simulator-linux/beta-simulator-linux.zip

Mac：https://link.zhihu.com/?target=https%3A//d17h27t6h515a5.cloudfront.net/topher/2017/February/58983385_beta-simulator-mac/beta-simulator-mac.zip

Windows：https://link.zhihu.com/?target=https%3A//d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip

模拟器Github地址：https://github.com/udacity/self-driving-car-sim

在训练模式下，键盘按上箭头，可以使小车加速，按下箭头使小车减速。按住鼠标左键，左移/右移鼠标控制方向。按下R可开启记录模式，在记录模式下，模拟器会记录小车摄像头看到的图像，以及当前（鼠标给出的）方向大小。

在运行时，模拟器会给 drive.py 传输小车前置摄像头的图像，然后 drive.py 需要返回一个方向控制量，模拟器根据方向控制量调整方向。

退出训练模式后，它会将刚刚记录的数据，输出到 IMG 目录和 driving_log.csv 文件中。前者是小车左中右三个摄像头记录的图像，后者记录图像对应的控制量。
参考网址：
https://github.com/feixia586/zhihu_material
https://github.com/naokishibuya/car-behavioral-cloning
深度学习源码网址：
https://github.com/udacity/CarND-Behavioral-Cloning-P3
https://github.com/naokishibuya/car-behavioral-cloning

主要内容：
1. Simulator安装与使用
2. 数据收集和处理
3. 深度卷积网络 (CNN) 训练端到端 (end-to-end) 模型

实现步骤：
环境配置：
conda env create -f environments.xml

1. 安装依赖库
不使用GPU
conda env create -f environment.yml 
使用GPU
conda env create -f environment-gpu.yml

2. 采集训练数据
在训练之前，首先要有训练数据。
(1)采集正常行驶的数据：
- 以匀速前行，尽可能地靠近中线行驶，尽可能驾驶平稳（不要抖）。
- 正常行驶的数据应该多采集，因为小车在多数情况下是正常行驶的。
采集非中线行驶的数据：
- 如果在自动驾驶时，小车拐到了车到边缘，或者卡在拐弯位的某处，我们需要教会它自己走出来。
- 暂停记录模式，缓缓把车开到车道边缘，开启记录模式，使劲拐弯使小车走回中线。
- 暂停记录模式，缓缓把车开到拐弯处，模拟拐不出来的情况，开启记录模式，使劲拐弯使之回到正常路上。
(2)数据预处理
在将数据喂给模型之前，要先预处理数据。
- 首先，可以复制出左右对称的数据。例如，将图形左右翻转，对应的方向控制量也左右翻转（从左变成右）。
这样，训练数据量就加倍了，而且使得左右控制量的分布是一样的。
- 然后是 Normalize (归一化)。将图像的像素值从 0-255 映射到 -0.5 ~ 0.5，使之中心为0.
- 图像裁剪。图像的上方大多是天空，不涉及道路，可以裁剪掉，减少数据大小。
- 把左右摄像头得到的图像转变成“中间摄像头”的图像。记录模式会记录左中右三个前置摄像头的图像。
在实际应用中，只会使用中间的摄像头作为输入。左右两边的摄像头看到的图像，相当于中间摄像头在道路偏左和道路偏右的情况下看到的图像，所以我们可以把左右摄像头的图像，用作输入，然后把方向控制量人为地添加偏移即可。例如，左边摄像头的图像，将方向控制量向右偏移0.2,可以看作是中间摄像头在道路偏左的时候对应的控制量。
最终处理完的数据集分布如图。横轴是控制量(负是左，正是右)，纵轴是对应图片的数量。

2. 验证模型
(1) 启动仿真模拟器，运行Autonomous 
(2) 命令行执行以下命令：
python drive.py model.h5

3. 训练模型
python model.py


代码说明：
model.py 创建训练模型
drive.py 使用模型进行自动驾驶
utils.py 工具函数
model.h5 模型文件
environments.yml 环境配置文件不使用GPU
environments-gpu.yml 环境配置文件使用GPU

# 模拟器源码

https://github.com/udacity/self-driving-car-sim
