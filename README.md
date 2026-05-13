# foundationpose_mr6d
We used FoundationPose to load the models from the MR6D dataset and evaluated them on the validation subset.
# 环境搭建：
foundationpose项目链接https://github.com/NVlabs/FoundationPose
本次使用是在autodl 云服务器4090卡上部署foundationpose ，拉取mr6d的val与models两个文件夹

（可选）foundationpose环境搭建参考官方conda方案，数据集和权重需要自己本地下载然后上传到云服务器
或者直接使用autodl的foundationpose镜像https://www.autodl.art/i/NVlabs/FoundationPose/FoundationPose，无需手动搭建foundtionpose环境

由于云服务器没有可视化界面，运行官方示例只能采用无头模式，将结果保存到本地
python run_demo_headless.py
# 使用
单场景检测(每帧加载mask,run_mr6d_multiobj_scene_5逻辑正确.py为仅物体出现首帧加载mask)
python run_mr6d_multiobj_scene.py \
  --scene_dir /root/mr6d_full/val/000000 \
  --models_dir /root/mr6d_full/models \
  --debug_dir /root/chentianxing/FoundationPose/debug

多场景自动检测
bash /root/chentianxing/FoundationPose/run_remaining.sh

单场景评估
python /root/chentianxing/FoundationPose/evaluate_metrics.py \
    --gt_json /root/mr6d_full/val/000002/scene_gt.json \
    --pred_json /root/chentianxing/FoundationPose/debug2/scene_gt_pred.json

多场景自动评估
bash /root/chentianxing/FoundationPose/eval_all.sh


# 采用的数据集
数据集链接https://huggingface.co/datasets/anas-gouda/mr6d
数据集类型
val/
  ├── 000000/
  │    ├── scene_camera.json    # 相机内外参数据
  │    ├── scene_gt.json        # 3D/6D 真值 (位姿、物体 ID)
  │    ├── scene_gt_info.json   # 2D 真值包围盒及遮挡掩码等可见度信息
  │    ├── rgb/                 # RGB 图像
  │    ├── depth/               # 深度图
  │    ├── mask/                # 物体整体投影分割掩码 (包含被遮挡部分)
  │    └── mask_visib/          # 仅可见部分的分割掩码 (不包含被遮挡部分)
  ├── 000001/
托盘模型
'''
(base) root@autodl-container-247f44a202-495ca768:~/mr6d_full/models# tree .
.
├── models_info.json
├── obj_000001.ply
├── obj_000002.ply
├── obj_000003.ply
├── obj_000004.ply
├── obj_000005.ply
├── obj_000005.png
├── obj_000006.ply
├── obj_000006.png
├── obj_000007.ply
├── obj_000007.png
├── obj_000008.ply
├── obj_000008.png
├── obj_000009.ply
├── obj_000009.png
├── obj_000010.ply
├── obj_000010.png
├── obj_000011.ply
├── obj_000011.png
├── obj_000012.ply
├── obj_000012.png
├── obj_000013.ply
├── obj_000013.png
├── obj_000014.ply
├── obj_000014.png
├── obj_000015.ply
├── obj_000015.png
├── obj_000016.ply
└── obj_000016.png

0 directories, 29 files
'''
