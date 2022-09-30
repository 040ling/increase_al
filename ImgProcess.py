"""
图像处理的各个函数
"""
import cv2
import Geometry2D as geo2D
import HomographicMatcher as matcher
import numpy as np

import ImgProcess


def show_photo(name,img):
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_mean_points(pts1,pts2):
    z = []
    for i in range(len(pts1)):
        x = (pts1[i][0] + pts2[i][0]) / 2
        y = (pts1[i][1] + pts2[i][1]) / 2
        point = (x,y)
        z.append(point)
    z = np.float32(z)
    return z



def img_preprocessing(img):
    """
    Args:
        img:输入图像

    Returns:预处理后的图像

    这个函数对输入图像预处理
    步骤包含灰化和高斯滤波
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = (3, 3)
    img_out = cv2.GaussianBlur(gray, kernel, 0)
    return img_out


def img_diff(img1, img2):
    """

    Args:
        img1: 输入图像1
        img2: 输入图像2

    Returns: 两者之间的图像差

    """
    gray1 = img_preprocessing(img1)
    gray2 = img_preprocessing(img2)
    diff = cv2.absdiff(gray1, gray2)
    return diff


def jiaozheng():
    return 0


def img_match(model,pad_model,img):
    """

    Args:
        model: 模板图像
        pad_model: 填充的模板图像
        img: 输入图像

    Returns:匹配点，输入图像特征点，模板图像特征点

    """
    sift = cv2.xfeatures2d.SIFT_create()
    img_size = img.shape
    model_h, model_w,_ = model.shape
    model_keys, model_desc = sift.detectAndCompute(pad_model, None)
    img_keys, img_desc = sift.detectAndCompute(img, None)
    # 返回特征点和特征描述符
    bf = cv2.BFMatcher(crossCheck=True)
    best_match = []

    if type(img_desc) != type(None):
        matches = bf.match(model_desc, img_desc)
        best_match = sorted(matches, key=lambda x: x.distance)
        # apply ratio test
        """
        matches = bf.knnMatch(queryDesc, train_desc, k=2)

        try:
            for m1, m2 in matches:
                if m1.distance < ratio * m2.distance:
                    best_match.append(m1)
        except ValueError:
            return [], ([], [])

    """

    return best_match, (img_keys, img_desc), (model_keys, model_desc)


def img_pts(model,img,bullseye):
    frame_size = img.shape
    model_h, model_w, _ = model.shape
    anchor_points, pad_model = geo2D.zero_pad_as(model, frame_size)
    # 居中放置了
    anchor_a = anchor_points[0]
    bullseye_anchor = (anchor_a[0] + bullseye[0], anchor_a[1] + bullseye[1])
    anchor_points.append(bullseye_anchor)
    anchor_points = np.float32(anchor_points).reshape(-1, 1, 2)
    model_ = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
    pad_model_ = cv2.cvtColor(pad_model, cv2.COLOR_BGR2GRAY)
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matches, (img_keys, img_desc), (model_keys, model_desc) = img_match(model, pad_model, img)
    if len(matches) >= 4:
        homography = matcher.calc_homography(model_keys, img_keys, matches)
        if type(homography) != type(None):
            warped_transform = cv2.perspectiveTransform(anchor_points, homography)  # 找框
            warped_vertices, warped_edges = geo2D.calc_vertices_and_edges(warped_transform)  # 找点
            bullseye_point = warped_vertices[5]  # 圆心
            pts1 = warped_vertices[0:4]
            pts1 = np.float32(pts1)
            return pts1
        else:
            return None
    else:
        return None


def img_perspective(model,img,bullseye):
    """

    Args:
        model: 模板图像
        img: 待转换的图像
        bullseye: 靶心坐标

    Returns:透视转换的图像

    """
    frame_size = img.shape
    model_h, model_w, _ = model.shape
    anchor_points, pad_model = geo2D.zero_pad_as(model, frame_size)
    anchor_a = anchor_points[0]
    bullseye_anchor = (anchor_a[0] + bullseye[0], anchor_a[1] + bullseye[1])
    anchor_points.append(bullseye_anchor)
    anchor_points = np.float32(anchor_points).reshape(-1, 1, 2)
    model_ = cv2.cvtColor(model,cv2.COLOR_BGR2GRAY)
    pad_model_ = cv2.cvtColor(pad_model,cv2.COLOR_BGR2GRAY)
    img_ = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    matches, (img_keys, img_desc), (model_keys, model_desc) = img_match(model,pad_model,img)
    if len(matches) >= 4:
        homography = matcher.calc_homography(model_keys, img_keys, matches)
        if type(homography) != type(None):
            warped_transform = cv2.perspectiveTransform(anchor_points, homography)  # 找框
            warped_vertices, warped_edges = geo2D.calc_vertices_and_edges(warped_transform)  # 找点
            bullseye_point = warped_vertices[5]  # 圆心
            pts1 = warped_vertices[0:4]
            pts2 = np.float32([[0, 0], [0, model_w - 2], [model_h - 2, model_w - 2], [model_h - 2, 0]])
            pts1 = np.float32(pts1)
            # print(pts1)
            #pts1 = match(pts1, pts_jiaozheng)(后续进行矫正)
            # print(pts1)
            #pts1 = np.float32(pts1)
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(img, matrix, (model_h, model_w))
            result = np.rot90(result)
            result = cv2.flip(result, 0)
            return result
        else:
            return None
    else:
        return None

def img_transform(img,model,pts1,pts2):
    model_h,model_w,_=model.shape
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (model_h, model_w))
    result = np.rot90(result)
    result = cv2.flip(result, 0)
    return result


def img_line(img):
    lines = cv2.HoughLinesP(img, 2, np.pi / 180, 120, minLineLength=30, maxLineGap=0)
    img_line_ = np.zeros(img.shape, dtype=img.dtype)
    if type(lines) != type(None):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_line_, (x1, y1), (x2, y2), (0xff, 0xff, 0xff), 5)
    lines = cv2.HoughLinesP(img_line_, 2, np.pi / 180, 120, minLineLength=30, maxLineGap=30)
    img_line = np.zeros(img.shape, dtype=img.dtype)
    if type(lines) != type(None):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_line, (x1, y1), (x2, y2), (0xff, 0xff, 0xff), 5)

    # show_photo("img_line",img_line)
    return img_line

    #img_quanhei = np.zeros([753, 753])
    #if (img_quanhei == img_line).all():
    #    # print("全黑")
    #    return img_line, 0
    #else:
    #    # print("有目标")
    #    return img_line


def img_processing(img,estimatedRadius,distances):

    radius = estimatedRadius
    img[distances[1] > radius] = 0

    # 二值化
    _, img_b = cv2.threshold(img, 20, 0xff, cv2.THRESH_BINARY)

    img_b = cv2.morphologyEx(img_b, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    return img_b,radius