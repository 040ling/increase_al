# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import ImgProcess as pro
import numpy as np
def circle(model):
    model_h,model_w,_ = model.shape
    canny = cv2.Canny(model,200,80)
    pro.show_photo("canny",canny)
    gray = cv2.cvtColor(model,cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=100, param2=100, minRadius=10, maxRadius=300)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(model, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(model, (i[0], i[1]), 2, (0, 0, 255), 3)
    pro.show_photo("yuan",model)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
    model = cv2.imread("input_img/model1.jpg")
    circle(model)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
