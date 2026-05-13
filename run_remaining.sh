#!/bin/bash

# 如果发生错误则中断执行
set -e

# 根据之前你提到的总共有 16 个场景 (000000 ~ 000015)
# 这里从 0 循环到 15 进行自动化执行
for i in {0..15}
do
    # 格式化成 6 位数字，如 000009
    SCENE_ID=$(printf "%06d" $i)
    
    echo "========================================"
    echo "开始处理场景: $SCENE_ID"
    echo "========================================"
    
    python /root/chentianxing/FoundationPose/run_mr6d_multiobj_scene.py \
        --scene_dir /root/mr6d_full/val/$SCENE_ID \
        --models_dir /root/mr6d_full/models \
        --debug_dir /root/chentianxing/FoundationPose/debug$SCENE_ID
        
    echo "场景 $SCENE_ID 执行完成！"
    echo ""
done

echo "所有指定的场景已全部执行完毕！"
