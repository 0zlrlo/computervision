import cv2 as cv
import numpy as np
import sys
def process_image(edges, image):
    # 허프 변환을 사용하여 선분 검출
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 25, minLineLength=30, maxLineGap=5)

    # 선분 중에서 4개의 꼭짓점을 선택
    vertices = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        vertices.append((x1, y1))
        vertices.append((x2, y2))

    if len(vertices) >= 4:
        # 기존 점에서 더 멀리 이동할 픽셀 크기 설정
        padding = 10

        # 각 꼭짓점을 padding 픽셀만큼 이동
        top_left = (min(vertices, key=lambda vertex: vertex[0] + vertex[1])[0] - padding, min(vertices, key=lambda vertex: vertex[0] + vertex[1])[1] - padding)
        top_right = (max(vertices, key=lambda vertex: vertex[0] - vertex[1])[0] + padding, max(vertices, key=lambda vertex: vertex[0] - vertex[1])[1] - padding)
        bottom_left = (min(vertices, key=lambda vertex: vertex[0] - vertex[1])[0] - padding, min(vertices, key=lambda vertex: vertex[0] - vertex[1])[1] + padding)
        bottom_right = (max(vertices, key=lambda vertex: vertex[0] + vertex[1])[0] + padding, max(vertices, key=lambda vertex: vertex[0] + vertex[1])[1] + padding)
        # 꼭짓점을 이미지에 그리기
        # cv.circle(image, top_left, 10, (0, 0, 255), -1)
        # cv.circle(image, top_right, 10, (0, 255, 0), -1)
        # cv.circle(image, bottom_left, 10, (255, 0, 0), -1)
        # cv.circle(image, bottom_right, 10, (255, 255, 0), -1)

        # 원근 변환에 사용할 4개의 꼭짓점 좌표
        src_points = np.float32([top_left, top_right, bottom_right, bottom_left])

        # 변환 후 이미지의 가로, 세로 크기 정의
        width = 800
        height = 800

        # 변환 후 꼭짓점 좌표 정의
        dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        # 원근 변환 행렬 계산
        perspective_matrix = cv.getPerspectiveTransform(src_points, dst_points)

        # 원근 변환 적용
        warped_image = cv.warpPerspective(image, perspective_matrix, (width, height))

        return warped_image
    else:
        print("꼭짓점을 찾을 수 없습니다.")

def detect_lines_and_angles(image):
    # 그레이스케일 이미지로 변환
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 가장자리 검출
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    # 허프 변환을 이용한 직선 검출
    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)
    
    angles = []
    if lines is not None:
        for rho, theta in lines[:,0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            # 직선의 각도를 계산 (라디안에서 도로 변환)
            angle = np.rad2deg(theta)
            angles.append(angle)
    
    return angles

# Function to count the number of angles that are exactly 90 degrees
def count_exact_90(angles):
    return sum(angle == 90 for angle in angles)

if len(sys.argv) > 1 :
    filename = sys.argv[1]

image = cv.imread(filename)
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
equalized_image = cv.equalizeHist(gray_image)
blurred_image = cv.GaussianBlur(equalized_image, (5, 5), 0)

edges = cv.Canny(blurred_image, 50, 70)
edges2 = cv.Canny(gray_image, 50, 70)
warped_image_with_blur = process_image(edges, image.copy())
warped_image_without_blur = process_image(edges2, image.copy())

angles_with_blur = detect_lines_and_angles(warped_image_with_blur)
angles_without_blur= detect_lines_and_angles(warped_image_without_blur)

# Count the number of lines near 90 degrees for each image
count_with_blur = count_exact_90(angles_with_blur)
count_without_blur = count_exact_90(angles_without_blur)

# Display the image with the most lines near 90 degrees
if count_with_blur > count_without_blur:
    src = warped_image_with_blur
else:
    src = warped_image_without_blur

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)   
edges = cv.Canny(gray, 50, 150, apertureSize=3)  
median = cv.medianBlur(gray, 5)

circles = cv.HoughCircles(median, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=26, minRadius=10, maxRadius=40)
param2 = 25
while circles is None and param2 > 0:
    param2 -= 1
    circles = cv.HoughCircles(median, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=param2, minRadius=10, maxRadius=40)

if circles is not None:
    circles = np.uint16(np.around(circles))
else:
    print("No circles were found")
    
dst = src.copy()

for i in circles[0, :]:
    cv.circle(dst, (i[0], i[1]), i[2], (255, 0, 0), 2)  
    cv.circle(dst, (i[0], i[1]), 1, (0, 0, 255), 2)   
# 원의 중심점의 색상을 분석하여 밝은색과 어두운색 원을 구분합니다.
light_circles = 0
dark_circles = 0

# 각 원에 대해 반복
for i in circles[0, :]:
    center_color = src[i[1], i[0]]  # 원의 중심점 색상(BGR)
    brightness = np.mean(center_color)  # 밝기는 BGR 값의 평균으로 결정
    # 임의로 설정한 임계값(123)을 기준으로 밝기 판단
    if brightness > 123:
        light_circles += 1
    else:
        dark_circles += 1

brightness_values = []

# 각 원에 대해 반복하여 밝기 값을 계산
for i in circles[0, :]:
    center_color = src[i[1], i[0]]  # 원의 중심점 색상(BGR)
    brightness = np.mean(center_color)  # 밝기는 BGR 값의 평균으로 결정
    brightness_values.append(brightness)

# 밝기 값의 중간값을 찾습니다.
median_brightness = np.median(brightness_values)

# 중간값을 기준으로 밝은색과 어두운색 원을 구분합니다.
lighter_circles = 0
darker_circles = 0
for brightness in brightness_values:
    if brightness > median_brightness:
        lighter_circles += 1
    else:
        darker_circles += 1
if (lighter_circles == 1 and darker_circles == 0) or (lighter_circles == 0 and darker_circles == 1) :
    print("말이 없습니다.") 
else:
    if light_circles == 0 or darker_circles == 0:
        print(f"밝은 색 말: {lighter_circles}")
        print(f"어두운 색 말: {darker_circles}")
    else:
        print(f"밝은 색 말: {light_circles}")
        print(f"어두운 색 말: {dark_circles}")
