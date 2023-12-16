import cv2
import numpy as np
import sys

def process_image(edges, image):
    # 허프 변환을 사용하여 선분 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, minLineLength=30, maxLineGap=5)

    # 선분 중에서 4개의 꼭짓점을 선택
    vertices = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        vertices.append((x1, y1))
        vertices.append((x2, y2))

    if len(vertices) >= 4:
        # 상하좌우 꼭짓점 찾기
        top_left = min(vertices, key=lambda vertex: vertex[0] + vertex[1])
        top_right = max(vertices, key=lambda vertex: vertex[0] - vertex[1])
        bottom_left = min(vertices, key=lambda vertex: vertex[0] - vertex[1])
        bottom_right = max(vertices, key=lambda vertex: vertex[0] + vertex[1])

        # 꼭짓점을 이미지에 그리기
        # cv2.circle(image, top_left, 10, (0, 0, 255), -1)
        # cv2.circle(image, top_right, 10, (0, 255, 0), -1)
        # cv2.circle(image, bottom_left, 10, (255, 0, 0), -1)
        # cv2.circle(image, bottom_right, 10, (255, 255, 0), -1)

        # 원근 변환에 사용할 4개의 꼭짓점 좌표
        src_points = np.float32([top_left, top_right, bottom_right, bottom_left])

        # 변환 후 이미지의 가로, 세로 크기 정의
        width = 800
        height = 800

        # 변환 후 꼭짓점 좌표 정의
        dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        # 원근 변환 행렬 계산
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # 원근 변환 적용
        warped_image = cv2.warpPerspective(image, perspective_matrix, (width, height))

        return warped_image
    else:
        print("꼭짓점을 찾을 수 없습니다.")

def detect_lines_and_angles(image):
    # 그레이스케일 이미지로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 가장자리 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 허프 변환을 이용한 직선 검출
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
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

image = cv2.imread(filename)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized_image = cv2.equalizeHist(gray_image)
blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

edges = cv2.Canny(blurred_image, 50, 70)
edges2 = cv2.Canny(gray_image, 50, 70)
warped_image_with_blur = process_image(edges, image.copy())
warped_image_without_blur = process_image(edges2, image.copy())
angles_with_blur = detect_lines_and_angles(warped_image_with_blur)
angles_without_blur= detect_lines_and_angles(warped_image_without_blur)

count_with_blur = count_exact_90(angles_with_blur)
count_without_blur = count_exact_90(angles_without_blur)

if count_with_blur > count_without_blur:
    cv2.imshow('Image with More 90 Degree Angles', warped_image_with_blur)
else:
    cv2.imshow('Image with More 90 Degree Angles', warped_image_without_blur)
# cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
