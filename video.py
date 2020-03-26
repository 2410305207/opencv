# -*- coding: utf-8 -*-
# 作    者：侯建军
# 创建时间：2019/4/16-16:39
# 文    件：video.py.py
# IDE 名称：PyCharm

from moviepy.editor import ImageSequenceClip
import argparse
import os

IMAGE_EXT = ['jpeg', 'gif', 'png', 'jpg']


def main():
    parser = argparse.ArgumentParser(description='创建驾驶视频.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='图像文件夹的路径。将从这些图像创建视频.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    # 将文件文件夹转换为图像文件类型的列表
    image_list = sorted([os.path.join(args.image_folder, image_file)
                         for image_file in os.listdir(args.image_folder)])

    image_list = [image_file for image_file in image_list if os.path.splitext(image_file)[1][1:].lower() in IMAGE_EXT]

    #  两种处理不同环境的输出视频命名方法
    video_file_1 = args.image_folder + '.mp4'
    video_file_2 = args.image_folder + 'output_video.mp4'

    print("创建视频文件 {}, FPS={}".format(args.image_folder, args.fps))
    clip = ImageSequenceClip(image_list, fps=args.fps)

    try:
        clip.write_videofile(video_file_1)
    except:
        clip.write_videofile(video_file_2)


if __name__ == '__main__':
    main()