import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model

import utils

"""
服务端程序
"""

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # 汽车当前的转向角
        steering_angle = float(data["steering_angle"])
        # 汽车当前的油门
        throttle = float(data["throttle"])
        # 汽车的当前速度
        speed = float(data["speed"])
        # 汽车中央摄像头的当前图像
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        # 保存帧
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
            
        try:
            image = np.asarray(image)       #从pil图像到numpy数组
            image = utils.preprocess(image) #应用预处理
            image = np.array([image])       # 模型需要4d数组

            # 预测图像的转向角
            steering_angle = float(model.predict(image, batch_size=1))
            # 随着车速的增加，降低油门
            # 如果速度超过了当前的速度限制，我们就要降速了。
            # 确保我们先减速，然后再回到原来的最大速度。
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # 减速
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)
        
    else:
        # 注意：不要编辑此。
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='图像文件夹的路径。这是保存运行中的图像的位置。'
    )
    args = parser.parse_args()

    # model = load_model(args.model)
    model = load_model("dataout/model-000.h5")

    if args.image_folder != '':
        print("正在创建图像文件夹 {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("记录这些运行结果  ...")
    else:
        print("不记录这些运行结果 ...")

    # 用Engineio的中间件运行flask应用程序
    app = socketio.Middleware(sio, app)

    #部署为eventlet wsgi服务器
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
