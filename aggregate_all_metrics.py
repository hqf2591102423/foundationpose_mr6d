import os
import json
import numpy as np

def aggregate_results():
    base_debug_dir = '/root/chentianxing/FoundationPose'
    
    global_metrics = {
        'ADD': [],
        'e_R': [],
        'e_t': [],
        'ADD_lt_0.1d': []
    }
    
    # 用来按物体汇总
    per_obj_metrics = {}
    
    total_scenes = 0
    total_instances = 0
    
    for i in range(16):
        scene_id = f"{i:06d}"
        metrics_file = os.path.join(base_debug_dir, f'debug{scene_id}', 'metrics_val.json')
        
        if not os.path.exists(metrics_file):
            continue
            
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
        total_scenes += 1
        total_instances += data['global']['instances']
        
        # 汇总全局数据
        global_metrics['ADD'].extend(data['global']['ADD'])
        global_metrics['e_R'].extend(data['global']['e_R'])
        global_metrics['e_t'].extend(data['global']['e_t'])
        global_metrics['ADD_lt_0.1d'].extend(data['global']['ADD_lt_0.1d'])
        
        # 汇总各物体数据
        for obj_id_str, m in data['per_object'].items():
            obj_id = int(obj_id_str)
            if obj_id not in per_obj_metrics:
                per_obj_metrics[obj_id] = {'ADD': [], 'e_R': [], 'e_t': [], 'ADD_lt_0.1d': []}
            per_obj_metrics[obj_id]['ADD'].extend(m['ADD'])
            per_obj_metrics[obj_id]['e_R'].extend(m['e_R'])
            per_obj_metrics[obj_id]['e_t'].extend(m['e_t'])
            per_obj_metrics[obj_id]['ADD_lt_0.1d'].extend(m['ADD_lt_0.1d'])

    # 输出总体摘要
    print("\n" + "#" * 60)
    print(f"【最终汇总结果】 (共评估了 {total_scenes} 个场景, 总计 {total_instances} 个实例)")
    print("#" * 60)
    
    global_summary = {
        "Total Scenes Evaluated": total_scenes,
        "Total Instances Evaluated": total_instances,
        "Global ADD (mm)": float(np.mean(global_metrics['ADD'])) if global_metrics['ADD'] else 0.0,
        "Global e_R (deg)": float(np.mean(global_metrics['e_R'])) if global_metrics['e_R'] else 0.0,
        "Global e_t (mm)": float(np.mean(global_metrics['e_t'])) if global_metrics['e_t'] else 0.0,
        "Global Robustness (%)": float(np.mean(global_metrics['ADD_lt_0.1d'])) * 100 if global_metrics['ADD_lt_0.1d'] else 0.0
    }
    
    print(f"总计平均 ADD: {global_summary['Global ADD (mm)']:.3f} mm")
    print(f"总计平均旋转误差 e_R: {global_summary['Global e_R (deg)']:.3f} °")
    print(f"总计平均平移误差 e_t: {global_summary['Global e_t (mm)']:.3f} mm")
    print(f"总计鲁棒性准确率: {global_summary['Global Robustness (%)']:.2f}%")
    print("-" * 60)
    
    obj_summary = {}
    print("【各物体ID最终汇总结果】")
    for obj_id in sorted(per_obj_metrics.keys()):
        m = per_obj_metrics[obj_id]
        if len(m['ADD']) == 0:
            continue
            
        obj_res = {
            "Instances": len(m['ADD']),
            "ADD": float(np.mean(m['ADD'])),
            "e_R": float(np.mean(m['e_R'])),
            "e_t": float(np.mean(m['e_t'])),
            "Robustness (%)": float(np.mean(m['ADD_lt_0.1d'])) * 100
        }
        obj_summary[str(obj_id)] = obj_res
        
        print(f"Object {obj_id:2d} (共 {obj_res['Instances']} 个实例):")
        print(f"    ADD: {obj_res['ADD']:.3f} mm | e_R: {obj_res['e_R']:.3f}° | e_t: {obj_res['e_t']:.3f} mm | Robust: {obj_res['Robustness (%)']:.2f}%")
        
    # 保存结果到文件
    final_output = {
        "global_summary": global_summary,
        "per_object_summary": obj_summary
    }
    
    out_path = os.path.join(base_debug_dir, 'final_evaluation_summary.json')
    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\n✅ 汇总统计已成功保存至 {out_path}")

if __name__ == "__main__":
    aggregate_results()
