from utils.sort import Sort


tracker = Sort()

def update_tracker(detections):
    return tracker.update(detections)
