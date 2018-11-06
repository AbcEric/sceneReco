import os
import cv2
import numpy as np

base_dir = "./img/"
dst_dir = "./output/"
#min_val = 10
#min_range = 30

# 字体颜色？
min_val = 30000
# 每行的最小高度
min_range = 24

count = 0

# 判断有多少行：向右投影？必须水平，没有干扰！
def extract_peek(array_vals, minimun_val, minimun_range):
    start_i = None
    #end_i = None
    peek_ranges = []
    print(array_vals)

    for i, val in enumerate(array_vals):
        #print(i, val)
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            if i - start_i >= minimun_range:
                #print(val)
                end_i = i
                #print(end_i - start_i)
                # 代表行数：有几行
                peek_ranges.append((start_i, end_i))
                start_i = None
                #end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")

    return peek_ranges

# 提取每行的文字
def cutImage(img, peek_ranges):
    global count
    for i, peek_range in enumerate(peek_ranges):
        print(peek_range, vertical_peek_ranges2d[i])
        for vertical_range in vertical_peek_ranges2d[i]:
            x = vertical_range[0]
            y = peek_range[0]
            w = vertical_range[1] - x
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            count += 1
            img1 = img[y:peek_range[1], x:vertical_range[1]]
            new_shape = (150, 150)
            img1 = cv2.resize(img1, new_shape)
            cv2.imwrite(dst_dir + str(count) + ".png", img1)
            cv2.rectangle(img, pt1, pt2, (55,255,155))

for fileName in os.listdir(base_dir):
    print(fileName)
    #if fileName != "japen.png":
    if fileName != "6408.bmp":
        continue

    img = cv2.imread(base_dir + fileName)
    # 转为灰度图像：
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 自适应阈值(cv2.adaptiveThreshold):一种局部性的阈值，通过规定一个区域大小，比较这个点与区域大小里面像素点
    # 的平均值（或者其他特征）的大小关系确定这个像素点是属于黑或者白（如果是二值情况）
    # 简单阈值cv2.threshold，简单阈值是一种全局性的阈值，只需要规定一个阈值值，整个图像都和这个阈值比较。
    #
    # 第一个原始图像
    # 第二个像素值上限
    # 第三个cv2.ADAPTIVE_THRESH_GAUSSIAN_C ：领域内像素点加权和，权重为一个高斯窗口
    # 第四个值的赋值方法：只有cv2.THRESH_BINARY 和cv2.THRESH_BINARY_INV
    # 第五个Block size:规定领域大小（一个正方形的领域）
    # 第六个常数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值，就是求得领域内均值或者加权值）
    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                               cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite("g_" + fileName, gray)
    print("THRESH=", adaptive_threshold[2])
    print("THRESH=", adaptive_threshold[5])

    # axis=1以后就是将一个矩阵的每一行向量相加: 相当于向右投影
    horizontal_sum = np.sum(adaptive_threshold, axis=1)

    #print("H_SUM=", horizontal_sum)

    peek_ranges = extract_peek(horizontal_sum, min_val, min_range)
    line_seg_adaptive_threshold = np.copy(adaptive_threshold)

    print("RANGES=", peek_ranges)
    for i, peek_range in enumerate(peek_ranges):
        x = 0
        y = peek_range[0]
        w = line_seg_adaptive_threshold.shape[1]
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)

    vertical_peek_ranges2d = []

    # 根据垂直和水平方向像素与空间阈值确定方格?
    for peek_range in peek_ranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = adaptive_threshold[start_y:end_y, :]
        vertical_sum = np.sum(line_img, axis=0)
        vertical_peek_ranges = extract_peek(vertical_sum, min_val, min_range)
        vertical_peek_ranges2d.append(vertical_peek_ranges)

    #print(vertical_peek_ranges2d)
    cutImage(img, peek_ranges)
    cv2.imwrite("k_" + fileName, img)

# https://www.jianshu.com/p/64808391285e
