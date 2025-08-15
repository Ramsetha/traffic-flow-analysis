import cv2
import numpy as np

# Adjust lane polygons according to your video resolution
LANE_1 = np.array([[0, 0], [213, 0], [213, 384], [0, 384]])       # left lane
LANE_2 = np.array([[214, 0], [426, 0], [426, 384], [214, 384]])   # middle lane
LANE_3 = np.array([[427, 0], [639, 0], [639, 384], [427, 384]])   # right lane

def get_lane(cx, cy):
    point = (cx, cy)
    if cv2.pointPolygonTest(LANE_1, point, False) >= 0:
        return 1
    elif cv2.pointPolygonTest(LANE_2, point, False) >= 0:
        return 2
    elif cv2.pointPolygonTest(LANE_3, point, False) >= 0:
        return 3
    return None

def draw_lanes(frame):
    cv2.polylines(frame, [LANE_1, LANE_2, LANE_3], isClosed=True, color=(0, 255, 0), thickness=2)
