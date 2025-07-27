import pyrealsense2 as rs
import cv2
import numpy as np

# Create RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable depth stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start pipeline
pipeline.start(config)

# Create a colorizer object
colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 2)  # 2 = Jet colormap, can change to others

print("Press 'q' to exit")

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Apply color map to depth frame
        colorized_depth = colorizer.colorize(depth_frame)
        color_image = np.asanyarray(colorized_depth.get_data())

        # Display the depth colormap
        cv2.imshow("L515 Depth Scale (Colorized)", color_image)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()