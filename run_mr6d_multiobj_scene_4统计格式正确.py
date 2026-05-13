import os, json
import numpy as np
import trimesh
import cv2
import imageio.v2 as imageio
import argparse
import logging

from estimater import *


class MR6DScene:
    def __init__(self, scene_dir):
        self.scene_dir = scene_dir
        self.rgb_dir = os.path.join(scene_dir, 'rgb')
        self.depth_dir = os.path.join(scene_dir, 'depth')
        self.mask_dir = os.path.join(scene_dir, 'mask')

        with open(os.path.join(scene_dir, 'scene_camera.json'), 'r') as f:
            self.scene_camera = json.load(f)
        with open(os.path.join(scene_dir, 'scene_gt.json'), 'r') as f:
            self.scene_gt = json.load(f)

        self.img_ids = sorted([k for k in self.scene_gt.keys()], key=lambda x: int(x))
        self.depth_scale = self.scene_camera[self.img_ids[0]].get('depth_scale', 1.0)

    def get_color(self, img_id):
        color_path = os.path.join(self.rgb_dir, f"{int(img_id):06d}.png")
        if not os.path.exists(color_path):
            color_path = os.path.join(self.rgb_dir, f"{int(img_id):06d}.jpg")
        return cv2.imread(color_path)[..., ::-1].copy()

    def get_depth(self, img_id):
        depth_path = os.path.join(self.depth_dir, f"{int(img_id):06d}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is not None:
            depth = depth.astype(np.float32) * self.depth_scale / 1000.0
        return depth

    def get_K(self, img_id):
        return np.array(self.scene_camera[img_id]['cam_K']).reshape(3, 3)

    def get_mask(self, img_id, inst_idx):
        mask_path = os.path.join(self.mask_dir, f"{int(img_id):06d}_{inst_idx:06d}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask > 0


def load_mesh(models_dir, obj_id):
    mesh_path = os.path.join(models_dir, f"obj_{obj_id:06d}.ply")
    mesh = trimesh.load(mesh_path)
    mesh.vertices = mesh.vertices.astype(np.float32) * 0.001
    mesh.vertex_normals = mesh.vertex_normals.astype(np.float32)
    return mesh


def build_estimator(mesh, debug_dir, debug):
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    return FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--scene_dir', type=str, default='/root/mr6d_full/val/000000')
    parser.add_argument('--models_dir', type=str, default='/root/mr6d_full/models')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=2)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    scene = MR6DScene(args.scene_dir)
    os.makedirs(args.debug_dir, exist_ok=True)

    # 用来存储符合 BOP 标准格式的预测结果
    scene_pred = {str(img_id): [] for img_id in scene.img_ids}

    # Use all instances in the first frame
    first_img_id = scene.img_ids[0]
    targets = []
    for inst_idx, obj in enumerate(scene.scene_gt[first_img_id]):
        targets.append((inst_idx, obj['obj_id']))

    logging.info(f"Found {len(targets)} instances in scene {args.scene_dir}")

    for inst_idx, obj_id in targets:
        logging.info(f"Initializing obj_id={obj_id} inst_idx={inst_idx}")
        mesh = load_mesh(args.models_dir, obj_id)

        obj_debug_dir = os.path.join(args.debug_dir, f"obj_{obj_id:06d}_inst_{inst_idx:06d}")
        os.system(f'rm -rf {obj_debug_dir}/* && mkdir -p {obj_debug_dir}/track_vis {obj_debug_dir}/ob_in_cam')

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

        est = build_estimator(mesh, obj_debug_dir, args.debug)
        logging.info("estimator initialization done")

        for i, img_id in enumerate(scene.img_ids):
            logging.info(f'obj_id={obj_id} inst_idx={inst_idx} frame={i}')
            color = scene.get_color(img_id)
            depth = scene.get_depth(img_id)
            K = scene.get_K(img_id)

            if i == 0:
                mask = scene.get_mask(img_id, inst_idx)
                pose = est.register(
                    K=K,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=args.est_refine_iter,
                )
            else:
                pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

            os.makedirs(f'{obj_debug_dir}/ob_in_cam', exist_ok=True)
            np.savetxt(f'{obj_debug_dir}/ob_in_cam/{img_id}.txt', pose.reshape(4, 4))
            
            # 将 numpy array 转换为嵌套的 list 格式，同时把 translation (深度) 从米乘回到毫米
            scene_pred[str(img_id)].append({
                "cam_R_m2c": pose[:3, :3].tolist(),
                "cam_t_m2c": (pose[:3, 3] * 1000.0).tolist(),
                "obj_id": obj_id
            })

            if args.debug >= 1:
                center_pose = pose @ np.linalg.inv(to_origin)
                try:
                    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                    vis = draw_xyz_axis(
                        color,
                        ob_in_cam=center_pose,
                        scale=0.1,
                        K=K,
                        thickness=3,
                        transparency=0,
                        is_input_rgb=True,
                    )
                except Exception as e:
                    logging.warning(f"Failed to draw vis: {e}")
                    vis = color

            if args.debug >= 2:
                os.makedirs(f'{obj_debug_dir}/track_vis', exist_ok=True)
                imageio.imwrite(f'{obj_debug_dir}/track_vis/{img_id}.png', vis)

    # 循环结束后统一保存预测结果
    pred_path = os.path.join(args.debug_dir, 'scene_gt_pred.json')
    with open(pred_path, 'w') as f:
        json.dump(scene_pred, f, indent=4)
    logging.info(f"Saved BOP-format predictions to {pred_path}")
