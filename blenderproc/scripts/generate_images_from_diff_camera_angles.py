import blenderproc as bproc
import numpy as np
import os
import json
import cv2

# Output folder
out_dir = "output_dataset/"
os.makedirs(out_dir, exist_ok=True)

# Initialize BlenderProc
bproc.init()

light = bproc.types.Light()
light.set_type("POINT")
light.set_location([2, 2, 4])
light.set_energy(1000)

# Create a simple object:
obj = bproc.object.create_primitive("MONKEY")

# Scale object to 1/3 its original size
obj.set_scale([1/10, 1/10, 1/10])

# Set object pose (identity at origin)
# Set object location
obj.set_location([0, 0, 0])

# Set rotation using Euler angles (radians)
obj.set_rotation_euler([0, 0, 0])

# enable depth distance and normals output
#bproc.renderer.enable_distance_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False) # this is the depth image
#bproc.renderer.enable_normals_output()
bproc.renderer.enable_segmentation_output(map_by=["instance"])

# Enable segmentation output
# bproc.renderer.enable_segmentation_output(
#     map_by=["instance", "class", "name"]
# )


scene_gt = {}
scene_camera = {}

# Create random cameras around object
n_views = 50
for i in range(n_views):
    # Random position on a sphere of radius r
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, np.pi/2)  # only above ground
    r = 1

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    cam_location = np.array([x, y, z])
    target = np.array([0, 0, 0])

    forward_vec = target - cam_location

    rotation_mat = bproc.camera.rotation_from_forward_vec(
        forward_vec,
        inplane_rot=0.0
    )

    cam2world = bproc.math.build_transformation_mat(
        cam_location,
        rotation_mat
    )
    

    # Add camera
    #bproc.camera.set_intrinsics_from_blender()
    bproc.camera.add_camera_pose(cam2world)

    T_obj_world = obj.get_local2world_mat()
    T_cam_world = cam2world
    T_obj_cam_blender = np.linalg.inv(T_cam_world) @ T_obj_world

    T_blender_to_cv = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ])

    T_obj_cam_cv = T_blender_to_cv @ T_obj_cam_blender

    R = T_obj_cam_cv[:3, :3]
    t = T_obj_cam_cv[:3, 3] * 1000.0  # mm

    scene_gt[str(i)] = [{
        "obj_id": 1,  # must match models_info.json
        "cam_R_m2c": R.reshape(-1).tolist(),
        "cam_t_m2c": t.tolist()
    }]

    K = bproc.camera.get_intrinsics_as_K_matrix()

    scene_camera[str(i)] = {
            "cam_K": K.reshape(-1).tolist(),
            "depth_scale": 0.0001
        }

    # before rendering set the noise level
    noise_threshold = 0.001
    # the noise threshold should be above 0 and below 0.1
    # this sets the noise per pixel.
    bproc.renderer.set_noise_threshold(noise_threshold) 

with open(os.path.join(out_dir, "scene_gt.json"), "w") as f:
    json.dump(scene_gt, f, indent=2)

with open(os.path.join(out_dir, "scene_camera.json"), "w") as f:
    json.dump(scene_camera, f, indent=2)


# render the images from all cameras at once and write later
data = bproc.renderer.render()

for i in range(n_views):
    # Extract outputs
    rgb = data["colors"][i]
    depth = data["depth"][i] # in meters 
    seg = data["instance_segmaps"][i]

    depth_scaled = depth * 10000 # 0.1 mm scale
    max_possible_uint16_value = np.iinfo(np.uint16).max

    depth_clipped = np.clip(depth_scaled, 0, max_possible_uint16_value)

    vis_mask = (seg.astype(np.float32) / seg.max()) * 255

    scene_dir = os.path.join(out_dir, str(i) + "/")
    os.makedirs(scene_dir, exist_ok=True)
    
    
    cv2.imwrite(os.path.join(scene_dir, f"rgb_{i:04d}.png"), rgb)
    cv2.imwrite(os.path.join(scene_dir, f"seg_{i:04d}.png"), seg)
    cv2.imwrite(os.path.join(scene_dir, f"depth_{i:04d}.png"), depth_clipped)
    cv2.imwrite(os.path.join(scene_dir, f"vis_mask_{i:04d}.png"), vis_mask)