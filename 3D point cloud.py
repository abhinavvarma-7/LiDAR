import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Configure depth stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Create Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window("L515 Point Cloud Viewer")
pcd = o3d.geometry.PointCloud()
geom_added = False

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert RealSense depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert depth image to point cloud
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy

        height, width = depth_image.shape
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        valid = depth_image > 0

        z = depth_image.astype(np.float32) * 0.001  # from mm to meters
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        points = points[valid.reshape(-1)]

        # Update Open3D PointCloud
        pcd.points = o3d.utility.Vector3dVector(points)
        if not geom_added:
            vis.add_geometry(pcd)
            geom_added = True
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    pipeline.stop()
    vis.destroy_window()