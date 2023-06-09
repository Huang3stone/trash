import cv2

def isLowLightImage(imgPath):
    img = cv2.imread(imgPath)  # 读取图像
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    avgBrightness = cv2.mean(imgGray)[0]  # 计算平均亮度
    if avgBrightness < 50:  # 设置阈值为100 (0~255)，如果平均亮度小于100，认为是弱光图像
        return True
    else:
        return False

if __name__ == '__main__':
    path = '220309.jpg'
    print(isLowLightImage(path))