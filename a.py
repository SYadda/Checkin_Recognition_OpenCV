import cv2
import numpy as np

# 读取大图片和模板图片
large_image = cv2.imread('2.jpg')
template = cv2.imread('0.jpg')

# 获取模板图片的宽度和高度
w, h = template.shape[1], template.shape[0]

# 使用模板匹配
result = cv2.matchTemplate(large_image, template, cv2.TM_CCOEFF_NORMED)

# 设置匹配阈值
threshold = 0.8
loc = np.where(result >= threshold)

# 过滤匹配结果，确保每个匹配的图形与之前匹配的图形至少间隔50像素
filtered_points = []
for pt in zip(*loc[::-1]):
    if all(np.linalg.norm(np.array(pt) - np.array(fp)) >= 50 for fp in filtered_points):
        filtered_points.append(pt)

# 统计匹配的图案数量
count = len(filtered_points)
for pt in filtered_points:
    # 在大图片上绘制矩形框
    cv2.rectangle(large_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    
# 在大图片上显示匹配的图案数量
cv2.putText(large_image, f'Count: {count}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)

#保存图片
cv2.imwrite('result.jpg', large_image)