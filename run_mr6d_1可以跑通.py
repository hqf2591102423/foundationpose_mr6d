import os, sys, json
import numpy as np
import trimesh
import cv2
import imageio.v2 as imageio
import argparse
import glob

from estimater import *

class MR6DReader:
    def __init__(self, scene_dir, obj_id=None):
        self.scene_dir = scene_dir
        self.rgb_dir = os.path.join(scene_dir, 'rgb')
        self.depth_dir = os.path.join(scene_dir, 'depth')
        self.mask_dir = os.path.join(scene_dir, 'mask')
        
        with open(os.path.join(scene_dir, 'scene_camera.json'), 'r') as f:
            self.scene_camera = json.load(f)
        with open(os.path.join(scene_dir, 'scene_gt.json'), 'r') as f:
            self.scene_gt = json.load(f)
            
        self.img_ids = sorted([k for k in self.scene_gt.keys()], key=lambda x: int(x))
        
        # Pick the first object in the first frame if obj_id is None
        if obj_id is None:
            obj_id = self.scene_gt[self.img_ids[0]][0]['obj_id']
            self.inst_idx = 0
        else:
            self.inst_idx = None
            for idx, obj in enumerate(self.scene_gt[self.img_ids[0]]):
                if obj['obj_id'] == obj_id:
                    self.inst_idx = idx
                    break
        
        self.obj_id = obj_id
        
        # Convert scale
        self.depth_scale = self.scene_camera[self.img_ids[0]].get('depth_scale', 1.0)
        
    def get_color(self, i):
        img_id = self.img_ids[i]
        color_path = os.path.join(self.rgb_dir, f"{int(img_id):06d}.png")
        if not os.path.exists(color_path):
            color_path = os.path.join(self.rgb_dir, f"{int(img_id):06d}.jpg")
        
        # 加上 .copy() 解决 PyTorch 负步长报错
        return cv2.imread(color_path)[..., ::-1].copy()
        
    def get_depth(self, i):
        img_id = self.img_ids[i]
        depth_path = os.path.join(self.depth_dir, f"{int(img_id):06d}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is not None:
            depth = depth.astype(np.float32) * self.depth_scale / 1000.0 # Convert to meters usually
        return depth
        
    def get_mask(self, i):
        img_id = self.img_ids[i]
        mask_path = os.path.join(self.mask_dir, f"{int(img_id):06d}_{self.inst_idx:06d}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask > 0

    def get_K(self, i):
        img_id = self.img_ids[i]
        K = np.array(self.scene_camera[img_id]['cam_K']).reshape(3, 3)
        return K


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default='/root/mr6d_full/models/obj_000001.ply')
  parser.add_argument('--test_scene_dir', type=str, default='/root/mr6d_full/val/000000')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=2) 
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)
  mesh.vertices = mesh.vertices.astype(np.float32) * 0.001 # Assume mm to meters
  mesh.vertex_normals = mesh.vertex_normals.astype(np.float32)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = MR6DReader(args.test_scene_dir, obj_id=int(args.mesh_file.split('_')[-1].split('.')[0]))
  
  for i in range(len(reader.img_ids)):
    logging.info(f'Processing frame {i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    K = reader.get_K(i)
    
    if i==0:
      mask = reader.get_mask(0)
      pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)
      
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.img_ids[i]}.txt', pose.reshape(4,4))
    
    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      try:
          vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
          vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
      except Exception as e:
          logging.warning(f"Failed to draw vis: {e}")
          vis = color
          
    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.img_ids[i]}.png', vis)

