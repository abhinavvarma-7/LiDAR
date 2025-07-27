import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file("your_file.bag", repeat_playback=False)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
device = profile.get_device()
playback = device.as_playback()
playback.set_real_time(False)

out = cv2.VideoWriter("output_yolo.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
        results = model(color_image)
        annotated_frame = results[0].plot()
        output_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLOv8 on L515 Recorded Video", output_bgr)
        out.write(output_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("Finished processing or error:", e)

finally:
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()