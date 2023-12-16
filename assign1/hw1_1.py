import cv2 as cv
import numpy as np
import sys
import itertools

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

def detect_angles(image):
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
            
            angle = np.rad2deg(theta)
            angles.append(angle)
    
    return angles

def count_exact_90(angles):
    return sum(angle == 90 for angle in angles)

def find_intersections(lines):
    intersections = []
    for line1, line2 in itertools.combinations(lines, 2):
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        if np.linalg.det([[x1 - x2, y1 - y2], [x3 - x4, y3 - y4]]) != 0:
            a1 = y1 - y2
            b1 = x2 - x1
            c1 = x1 * y2 - y1 * x2

            a2 = y3 - y4
            b2 = x4 - x3
            c2 = x3 * y4 - y3 * x4

            x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
            y = (c1 * a2 - c2 * a1) / (a1 * b2 - a2 * b1)

            intersections.append((x, y))
    return intersections

def cluster_intersections(intersections, threshold):
    clusters = {}
    cluster_idx = 0
    for point in intersections:
        added_to_cluster = False
        for idx, cluster_points in clusters.items():
            for cluster_point in cluster_points:
                if np.linalg.norm(np.array(cluster_point) - np.array(point)) < threshold:
                    clusters[idx].append(point)
                    added_to_cluster = True
                    break
            if added_to_cluster:
                break
        if not added_to_cluster:
            clusters[cluster_idx] = [point]
            cluster_idx += 1
    return clusters

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
angles_with_blur = detect_angles(warped_image_with_blur)
angles_without_blur= detect_angles(warped_image_without_blur)

count_with_blur = count_exact_90(angles_with_blur)
count_without_blur = count_exact_90(angles_without_blur)

if count_with_blur > count_without_blur:
    src = warped_image_with_blur
else:
    src = warped_image_without_blur

gray= cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# Canny 엣지 검출기를 이용해 엣지를 찾습니다.
edges = cv.Canny(src, 50, 150, apertureSize=3)
# Hough 변환을 사용하여 직선을 검출합니다.
lines = cv.HoughLines(edges, 1, np.pi/180, 280)
# 세로선의 개수를 셉니다.
vertical_lines = 0

# 검출된 직선을 분석합니다.
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        # 각도를 각도(도)로 변환합니다.
        degrees = np.degrees(theta)
        # 세로선은 theta가 90도(또는 270도) 근처에 있을 것입니다.
        if abs(degrees - 0) < 10 or abs(degrees - 180) < 10:
            vertical_lines += 1
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)
if vertical_lines <= 9:
    print("영/미식 룰: 8X8")
else:
    print("국제 룰: 10X10")