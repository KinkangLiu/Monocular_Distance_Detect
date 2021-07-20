'''
Create July 27, 2020
@author ClearTorch
'''

import pandas as pd
import numpy as np
import cv2
import sys
import math
import os
import argparse
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import plotly.graph_objects as go   #Plotly是一个基于Javascript的绘图库,绘图工具一般是graph_objects工具

start_time = time.time()  #起始时间
print('Pandas Version:', pd.__version__)
print('Nunpy Version:', np.__version__)

@torch.no_grad()

class DistanceEstimation:
    '''
    DistanceEstimation类采用yolov5进行目标检测，同时对检测到的目标进行距离估计
    '''
    def __init__(self):
        '''
        初始化图片分辨率及相机标定数据
        '''
        self.W = 640
        self.H = 480
        self.excel_path = r'./camera_parameters.xlsx'

    def camera_parameters(self, excel_path):
        '''
        camera_parameters()函数对标定的相机参数Excel表进行读取及处理
        :param excel_path: Excel表的绝对路径(相机参数标定Excel由Matlab得到)
        :return: 返回相机标定的外参矩阵,和内参矩阵
        '''
        df_intrinsic = pd.read_excel(excel_path, sheet_name='内参矩阵', header=None)
        df_p = pd.read_excel(excel_path, sheet_name='外参矩阵', header=None)

        print('外参矩阵形状：', df_p.values.shape)
        print('内参矩阵形状：', df_intrinsic.values.shape)

        return df_p.values, df_intrinsic.values

    def object_point_world_position(self, u, v, w, h, p, k):
        '''
        object_point_world_position()函数根据物体的检测检测框及相机未知尺度因子等参数求出测距关键点的世界坐标
        测距关键点为框的底边中点
        :param u: 检测目标矩形框的对角线中点x坐标像素值
        :param v: 检测目标矩形框的对角线中点y纵标像素值
        :param w: 检测目标矩形框的宽度像素值
        :param h: 检测目标矩形框的高度像素值
        :param p: 外参矩阵
        :param k: 内参矩阵
        :return: 测距关键点的世界坐标
        '''
        u1 = u  # 图像坐标系中 关键点x坐标的像素值
        v1 = v + h / 2  # 图像坐标系中 关键点y坐标的像素值
        print('关键点坐标：', u1, v1)
        '''
        # 检测C点是否在整张图片内部
        if 0 < u1 and u1 <= self.W and 0 < v1 and v1 <= self.H:
            print('=' * 50)
        else:
            print('目标测距关键点C坐标不合法')
            sys.exit()
        '''
        # 外参矩阵描述世界坐标系(Xw,Yw,Zw)与相机坐标系(Xc,Yc,Zc)的关系
        alpha = -(90 + 0) / (2 * math.pi)  # 绕X旋转角度 由z到y方向为正
        peta = 0  # 绕Y轴旋转角度
        gama = -90 / (2 * math.pi)  # 绕Z轴旋转角度

        fx = k[0, 0]  # x方向焦距 单位为像素
        fy = k[1, 1]
        H = 1  # 相机离地高度
        angle_a = 0  # 相机z轴与水平线夹角
        angle_b = math.atan((v1 - self.H / 2) / fy)  # 关键点与相机光心轴线夹角
        angle_c = angle_b + angle_a
        print('angle_b', angle_b)

        depth = (H / np.sin(angle_c)) * math.cos(angle_b)  # 相机坐标系沿着相机z轴的长度
        print('depth', depth)

        k_inv = np.linalg.inv(k)  # 内参k矩阵的逆
        p_inv = np.linalg.inv(p)  # 外参p矩阵的逆
        # print(p_inv)
        point_c = np.array([u1, v1, 1])
        point_c = np.transpose(point_c)  # 转换成列向量
        print('point_c', point_c)
        print('k_inv', k_inv)
        c_position = np.matmul(k_inv, depth * point_c)  # 矩阵计算
        print('c_position', c_position)
        c_position = np.append(c_position, 1)
        c_position = np.transpose(c_position)
        c_position = np.matmul(p_inv, c_position)  # 得到世界坐标系下关键点的xw,yw,zw
        d1 = np.array((c_position[0], c_position[1]), dtype=float)  # 取xw, yw
        return d1


    def distance(self, kuang, xw=5, yw=0.1):
        print('=' * 50)
        print('开始测距')
        fig = go.Figure()  # 创建绘图对象
        p, k = self.camera_parameters(self.excel_path)
        if len(kuang):  # 存在物体框
            obj_position = []
            u, v, w, h = kuang[1] * self.W, kuang[2] * self.H, kuang[3] * self.W, kuang[4] * self.H
            print('目标框', u, v, w, h)
            d1 = self.object_point_world_position(u, v, w, h, p, k)
        distance = 0
        print('距离', d1)
        if d1[0] <= 0:
            d1[:] = 0
        else:
            distance = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))
        return distance, d1

    # 目标检测主函数 调用yolov5模型检测输入图像/视频 框出物体并标注类别
    def Detect(self, weights='yolov5s.pt',  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='inference/output',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           ):
        '''
        目标检测主函数 调用yolov5模型检测输入图像/视频 框出物体并标注类别
        '''
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories每次试验输出地址新建一个文件夹exp+次数
        #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        save_dir = Path(project)

        # Initialize
        set_logging()
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA 仅在使用CUDA时采用半精度

        # Load model 加载模型
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        #imgsz = check_img_size(imgsz, s=stride)  # check image size  测距不要缩放图片
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        # Second-stage classifier 分类器模型
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader 加载神经网络的输入数据
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)        #在LoadStreams函数中调用cv.VedioCapture()函数打开了摄像头
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]  #得到神经网络输出

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections 检测过程
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count  #path[i]为source 即为0
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path  p为inference/images/demo_distance.mp4
                save_path = str(save_dir / p.name)  # img.jpg  inference/output/demo_distance.mp4
                txt_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt   inference/output/demo_distance_frame
                #print('txt', txt_path)
                s += '%gx%g ' % img.shape[2:]  # print string 图片形状 eg.640X480
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                if len(det):    #det形式为tensor([[313.90039, 241.97386, 325.85822, 272.36917,   0.54204,   0.00000]], device='cuda:0')
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        if not names[int(c)] in ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus', 'traffic light', 'stop sign']:
                            continue
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if not names[int(cls)] in ['person','chair', 'car', 'truck', 'bicycle', 'motorcycle', 'bus', 'traffic light', 'stop sign']:
                            continue
                        n = (det[:, -1] == c).sum()  # detections per class记录每种类别数量
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        kuang = [int(cls), xywh[0], xywh[1], xywh[2], xywh[3]]     #框的信息
                        if save_txt:  # Write to file 将框的形状数据写入txt文件
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (int(cls), *xywh)) # 写入一行：类别标签，框对角线中点的横向坐标占图像横向的比例，框中点纵向坐标比例，框的宽度，框的高度
       
                        distance, d = self.distance(kuang)
    
                        if save_img or save_crop or view_img:  # Add bbox to image 在图像中绘制目标框
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            if label != None and distance!=0:
                                label = label + ' ' + str('%.1f' % d[0]) + 'm'+ str('%.1f' % d[1]) + 'm'
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    #print('框', kuang)
    
                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
    
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond 画面显示时间为1ms
    
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")
    
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    
        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1440, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save_txt',default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='inference/output', help='save results to project/name')  #保存地址
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'thop'))

    print('开始进行目标检测和单目测距！')
    DE = DistanceEstimation()  #构建距离测量类
    DE.Detect(**vars(opt))
    #os.system('python detect.py --device 0 --source 0') #os.system()将字符串转化成命令,创建一个子进程在系统上执行命令行
    if time.time()>(start_time + 10):
        cv2.waitKey (0)
        cv2.destroyAllWindows()









