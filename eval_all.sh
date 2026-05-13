#!/bin/bash

# 如果发生错误则中断执行
set -e

# 这里从 0 循环到 15 进行自动化评估
for i in {0..15}
do
    # 格式化成 6 位数字
    SCENE_ID=$(printf "%06d" $i)
    
    echo "========================================"
    echo "开始评估场景: $SCENE_ID"
    echo "========================================"
    
    # 指向对应的 Ground Truth 和 对应的预测结果文件夹
    GT_JSON="/root/mr6d_full/val/$SCENE_ID/scene_gt.json"
    PRED_JSON="/root/chentianxing/FoundationPose/debug${SCENE_ID}/scene_gt_pred.json"
    
    # 为了防止某些场景因为没跑到而跳过，这里判断一下预测文件是否存在
    if [ -f "$PRED_JSON" ]; then
        python /root/chentianxing/FoundationPose/evaluate_metrics.py \
            --gt_json $GT_JSON \
            --pred_json $PRED_JSON \
            --models_dir /root/mr6d_full/models \
            --out_json "/root/chentianxing/FoundationPose/debug${SCENE_ID}/metrics_val.json"
    else
        echo "警告: 找不到预测结果 $PRED_JSON ，跳过场景 $SCENE_ID"
    fi
        
    echo ""
done

echo "评估执行完毕！正在对所有结果进行最终汇总..."

# 调用 Python 脚本汇总所有评估出来的 metrics_val.json
python /root/chentianxing/FoundationPose/aggregate_all_metrics.py
