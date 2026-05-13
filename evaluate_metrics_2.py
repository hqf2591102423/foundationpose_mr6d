import os
import json
import argparse
import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def compute_rotation_error(R_gt, R_pred):
    """计算旋转误差 e_R，单位为度(°)"""
    R_rel = np.dot(R_gt.T, R_pred)
    trace = np.trace(R_rel)
    trace = np.clip(trace, -1.0, 3.0)  # 防浮点数越界
    angle = np.arccos((trace - 1.0) / 2.0)
    return np.degrees(angle)

def compute_translation_error(t_gt, t_pred):
    """计算平移误差 e_t (L2 距离)"""
    return np.linalg.norm(t_gt - t_pred)

def compute_add(R_gt, t_gt, R_pred, t_pred, pts):
    """计算 ADD (平均模型点距离)"""
    pts_gt = np.dot(pts, R_gt.T) + t_gt
    pts_pred = np.dot(pts, R_pred.T) + t_pred
    return np.linalg.norm(pts_gt - pts_pred, axis=1).mean()

def load_models(models_dir, obj_ids):
    """加载 3D 模型以计算 ADD 和直径"""
    models = {}
    diams = {}
    for obj_id in obj_ids:
        path = os.path.join(models_dir, f"obj_{obj_id:06d}.ply")
        if os.path.exists(path):
            mesh = trimesh.load(path)
            # 保持以毫米为单位
            pts = mesh.vertices.astype(np.float32)
            models[obj_id] = pts
            
            # 使用模型顶点间最大距离作为理论外接球直径
            bmin = pts.min(axis=0)
            bmax = pts.max(axis=0)
            diams[obj_id] = np.linalg.norm(bmax - bmin)
        else:
            print(f"Warning: Model not found at {path}")
    return models, diams

def evaluate(gt_json, pred_json, models_dir):
    with open(gt_json, 'r') as f:
        gt_data = json.load(f)
    with open(pred_json, 'r') as f:
        pred_data = json.load(f)

    # 1. 收集所有出现的 obj_id 并加载对应模型
    all_obj_ids = set()
    for items in gt_data.values():
        for item in items:
            all_obj_ids.add(item['obj_id'])
            
    models, diams = load_models(models_dir, all_obj_ids)

    # 结果统计列表
    metrics = {
        'ADD': [],
        'e_R': [],
        'e_t': [],
        'ADD_lt_0.1d': [] # 鲁棒性分析：ADD 是否小于 0.1*Diameter
    }
    
    # 针对单一 obj_id 的细化统计
    obj_metrics = defaultdict(lambda: {'ADD': [], 'e_R': [], 'e_t': [], 'ADD_lt_0.1d': []})

    # 2. 遍历帧并进行匹配评估
    for img_id_str, gt_items in gt_data.items():
        if img_id_str not in pred_data:
            continue
        
        pred_items = pred_data[img_id_str]
        
        # 按 obj_id 分组进行匹配 (同一帧中可能存在多个同类的物体)
        gt_by_obj = defaultdict(list)
        for item in gt_items:
            gt_by_obj[item['obj_id']].append(item)
            
        pred_by_obj = defaultdict(list)
        for item in pred_items:
            pred_by_obj[item['obj_id']].append(item)

        for obj_id in gt_by_obj.keys():
            if obj_id not in models:
                continue
                
            pts = models[obj_id]
            diam = diams[obj_id]
            gt_list = gt_by_obj[obj_id]
            pred_list = pred_by_obj.get(obj_id, [])
            
            if len(pred_list) == 0:
                continue

            # 构建基于平移误差的代价矩阵，用匈牙利算法做最优匹配
            cost_matrix = np.zeros((len(gt_list), len(pred_list)))
            for i, gt_item in enumerate(gt_list):
                t_gt = np.array(gt_item['cam_t_m2c']).flatten()
                for j, pred_item in enumerate(pred_list):
                    t_pred = np.array(pred_item['cam_t_m2c']).flatten()
                    cost_matrix[i, j] = compute_translation_error(t_gt, t_pred)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 计算各项指标
            for i, j in zip(row_ind, col_ind):
                gt_item = gt_list[i]
                pred_item = pred_list[j]
                
                R_gt = np.array(gt_item['cam_R_m2c']).reshape(3, 3)
                t_gt = np.array(gt_item['cam_t_m2c']).flatten()
                R_pred = np.array(pred_item['cam_R_m2c']).reshape(3, 3)
                t_pred = np.array(pred_item['cam_t_m2c']).flatten()
                
                err_R = compute_rotation_error(R_gt, R_pred)
                err_t = compute_translation_error(t_gt, t_pred)
                err_add = compute_add(R_gt, t_gt, R_pred, t_pred, pts)
                is_robust = 1.0 if err_add < 0.1 * diam else 0.0

                metrics['ADD'].append(err_add)
                metrics['e_R'].append(err_R)
                metrics['e_t'].append(err_t)
                metrics['ADD_lt_0.1d'].append(is_robust)
                
                obj_metrics[obj_id]['ADD'].append(err_add)
                obj_metrics[obj_id]['e_R'].append(err_R)
                obj_metrics[obj_id]['e_t'].append(err_t)
                obj_metrics[obj_id]['ADD_lt_0.1d'].append(is_robust)

    # 3. 打印统计结果
    print("="*60)
    print("【全局评估结果】")
    print(f"评估实例总数: {len(metrics['ADD'])}")
    print(f"平均模型点距离 (ADD): {np.mean(metrics['ADD']):.3f} mm")
    print(f"平均旋转误差 (e_R):   {np.mean(metrics['e_R']):.3f} °")
    print(f"平均平移误差 (e_t):   {np.mean(metrics['e_t']):.3f} mm")
    print(f"鲁棒性准确率 (ADD < 0.1d): {np.mean(metrics['ADD_lt_0.1d']) * 100:.2f}%")
    print("="*60)
    
    print("\n【按物体 ID 详细评估指标】")
    for obj_id in sorted(obj_metrics.keys()):
        m = obj_metrics[obj_id]
        print(f"Object {obj_id:2d} (Instances: {len(m['ADD'])}):")
        print(f"    - ADD (mm): {np.mean(m['ADD']):.3f}")
        print(f"    - e_R (°):  {np.mean(m['e_R']):.3f}")
        print(f"    - e_t (mm): {np.mean(m['e_t']):.3f}")
        print(f"    - Robust % (ADD < 0.1d): {np.mean(m['ADD_lt_0.1d']) * 100:.2f}%")
        print("-" * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json', type=str, default='/root/mr6d_full/val/000000/scene_gt.json', help='Ground truth JSON path')
    parser.add_argument('--pred_json', type=str, default='/root/chentianxing/FoundationPose/debug/scene_gt_pred.json', help='Prediction JSON path')
    parser.add_argument('--models_dir', type=str, default='/root/mr6d_full/models', help='Directory containing 3D models (.ply)')
    args = parser.parse_args()
    
    evaluate(args.gt_json, args.pred_json, args.models_dir)
