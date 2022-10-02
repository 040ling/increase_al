import sys
import time

import cv2
import numpy as np
import ImgProcess as pro
import Geometry2D as geo2D
import VisualAnalyzer as visuals
import ContourClassifier as cntr
import VideoAnalyze as va

class Hit:
    def __init__(self,x,y,d,score,len,bullseye):
        self.point = (x,y)
        self.score = score
        self.d = d
        self.bullseye = bullseye
        self.reputation = 1
        self.len = len

    def increase_rep(self):
        self.reputation+=1

    def descrease_rep(self):
        self.reputation-=1



def find_real_hit(proj_contours):
    bullseye = (378, 386)
    hits = []

    for cont in proj_contours:
        contPts = [(cont[m][0][0], cont[m][0][1]) for m in range(len(cont))]
        point_A = contPts[0]  # some random point on the contour

        # find the two furthest points on the contour
        point_B = cntr.contour_distances_from(contPts, point_A)[::-1][0]
        point_A = cntr.contour_distances_from(contPts, point_B)[::-1][0]
        len_line = geo2D.euclidean_dist(point_A, point_B)
        # 这两步找到直线的两个端点

        A_dist = geo2D.euclidean_dist(point_A, bullseye)
        B_dist = geo2D.euclidean_dist(point_B, bullseye)
        hit = point_A if point_A[1]>point_B[1] else point_B
        dist = A_dist if point_A[1]>point_B[1] else B_dist
        real_hit = (hit[0], hit[1], dist, bullseye, len_line)
        hits.append(real_hit)

    return hits


def create_scoreboard(hits,ringsAmount,innerDiam,model):
    scoreboard=[]
    model_w,model_h,_ = model.shape
    for hit in hits:
        if hit[1]<model_h/2:
            score = 10 - int(hit[2] / innerDiam-0.05)
        else:
            score = 10 - int(hit[2] / innerDiam - 0.2)
        if score < 10 - ringsAmount + 1:
            score = 0
        elif score > 10:
            score = 10
        hit_obj = Hit(hit[0],hit[1],hit[2],score,hit[4],hit[3])
        scoreboard.append(hit_obj)
        scoreboard.append(hit_obj)
    return scoreboard


def compare_scoreboard(video_hit,scoreboard):
    if len(scoreboard) > 1:
        hit_choose = scoreboard[0]
        hit_d = video_hit.check_hit(hit_choose)
        for hit in scoreboard:
            d = video_hit.check_hit(hit)
            if d>hit_d:
                hit_choose = hit
                hit_d = d
        hit_choose.increase_rep()

        hit_choose = scoreboard[0]
        hit_len = hit_choose.len
        for hit in scoreboard:
            len_ = hit.len
            if len_ > hit_len:
                hit_choose = hit
                hit_len = len_
        hit_choose.increase_rep()
        hit_choose.increase_rep()
        hit_choose = scoreboard[0]
        hit_rep = hit_choose.reputation
        for hit in scoreboard:
            if hit_rep<hit.reputation:
                hit_rep = hit.reputation
                hit_choose = hit
        video_hit.add_hit(hit_choose)
        return video_hit
    elif len(scoreboard) == 1:
        video_hit.add_hit(scoreboard[0])
        return video_hit



def video_process(model,frame_a,frame_b,frame_c,video_hit,bullseye,innerdist,num_target,point_a,point_b,t1):
    # 第一步，先求两个差值
    start =time.time()
    point_c = pro.img_pts(model,frame_c,bullseye)

    diff1 = pro.img_diff(frame_a,frame_b)
    diff2 = pro.img_diff(frame_b,frame_c)
    #pro.show_photo("diff1",diff1)
    # pro.show_photo("diff2",diff2)

    # 转换图像
    model_w,model_h,_ = model.shape
    pts1 = pro.get_mean_points(point_a,point_b)
    pts2 = pro.get_mean_points(point_b,point_c)
    pts = np.float32([[0, 0], [0, model_w - 2], [model_h - 2, model_w - 2], [model_h - 2, 0]])


    if len(pts1)==4 and len(pts2)==4:
        img1 = pro.img_transform(diff1, model, pts1, pts)
        img2 = pro.img_transform(diff2, model, pts1, pts)
    else:
        print("error:cant find the right points")
        sys.exit(1)

    distances = geo2D.calc_distances_from(model.shape, bullseye)
    line1, radius = pro.img_processing(img1, 350, distances)
    line1 = pro.img_line(line1)
    line2, _ = pro.img_processing(img2, 350, distances)
    line3, _ = pro.img_processing(img1, 350, distances)
    w,h,_ = model.shape
    img_quanhei = np.zeros([w, h])

    #pro.show_photo("line1",line1)
    #pro.show_photo("line3",line3)
    # print((img_quanhei != line1).any())
    if (img_quanhei == line1).all() and (img_quanhei == line3).all():
        out1 = frame_b
        out2 = frame_c
        print(0)
        end = time.time()
        print("日常时间：{}s".format(end-start))
        return out1, out2, video_hit,point_b,point_c
    elif ((img_quanhei == line1).all() and (img_quanhei != line3).any())and(start-t1<5):
        out1 = frame_a
        out2 = frame_c
        print(1)
        return out1, out2, video_hit,point_b,point_c
    elif ((img_quanhei == line1).all() and (img_quanhei != line3).any())and(start-t1>=5):
        out1 = frame_b
        out2 = frame_c
        print(6)
        pic = pro.img_transform(frame_c, model, point_c, pts)
        line1,radius = pro.img_end(img1,350,distances)
        if (img_quanhei == line1).all():
            print("fail")
            return out1, out2, video_hit, point_b, point_c
        proj_contours = visuals.reproduce_proj_contours(line1, distances,
                                                        bullseye, radius, model)  # 边缘检测
        model_ = pic.copy()
        draw = cv2.drawContours(model_, proj_contours, -1, 127, 2)
        cv2.namedWindow('a', 0)
        cv2.imshow("a", model_)
        cv2.waitKey()

        hits = find_real_hit(proj_contours)
        scoreboard = create_scoreboard(hits, num_target, innerdist, model)
        video_hit = compare_scoreboard(video_hit, scoreboard)
        end = time.time()
        print("关键帧判别时长：{}s".format(end - start))
        return out1, out2, video_hit,point_b,point_c
    elif (img_quanhei != line1).any() and (img_quanhei != line2).any():
        out1 = frame_a
        out2 = frame_c
        print(2)
        return out1, out2, video_hit,point_b,point_c
    elif (img_quanhei != line1).any() and (img_quanhei == line2).all():
        out1 = frame_b
        out2 = frame_c
        print(3)
        pic = pro.img_transform(frame_c, model, point_c, pts)
        proj_contours = visuals.reproduce_proj_contours(line1, distances,
                                                        bullseye, radius,model)  # 边缘检测
        model_ = pic.copy()
        draw = cv2.drawContours(model_, proj_contours, -1, 127, 2)
        cv2.namedWindow('a', 0)
        cv2.imshow("a", model_)
        cv2.waitKey()

        hits = find_real_hit(proj_contours)
        scoreboard = create_scoreboard(hits,num_target,innerdist,model)
        video_hit = compare_scoreboard(video_hit,scoreboard)
        end = time.time()
        print("关键帧判别时长：{}s".format(end-start))
        return out1,out2,video_hit,point_b,point_c

    else:
        out1 = frame_b
        out2 = frame_c
        print(4)
        return out1, out2, video_hit,point_b,point_c




