import pyrealsense2 as rs
import cv2
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file("frame_capture.bag", repeat_playback=False)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())

cv2.imshow("RGB from L515 Photo", color_image)
cv2.imwrite("rgb_extracted.jpg", color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

pipeline.stop()