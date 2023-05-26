
FORCE_BOXES=True
FORCE_BOXES_PATH="/storage/group/hexm/weiyn/refreasoning/det_results/faster_rcnn_x152_NMS0p3_th0p0_max100_min10"
python tools/test_sg_net.py \
        --config-file sgg_configs/vgattr/vinvl_x152c4.yaml \
        TEST.IMS_PER_BATCH 1 \
        MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth \
        MODEL.ROI_HEADS.NMS_FILTER 1 \
        MODEL.ROI_HEADS.SCORE_THRESH 0.1 \
        DATA_DIR "../maskrcnn-benchmark-1/datasets1" \
        TEST.IGNORE_BOX_REGRESSION False \
        MODEL.ATTRIBUTE_ON False \
        TEST.OUTPUT_FEATURE True \
        MODEL.ROI_HEADS.DETECTIONS_PER_IMG 100 \
        MODEL.ROI_HEADS.MIN_DETECTIONS_PER_IMG 10 \
        DATALOADER.NUM_WORKERS 4 \
        OUTPUT_DIR "/storage/group/hexm/weiyn/refreasoning/det_results/vinvl_with_fastrcnn_boxes" \
        MODEL.ROI_BOX_HEAD.FORCE_BOXES $FORCE_BOXES \
        MODEL.RPN.FORCE_BOXES $FORCE_BOXES \
        MODEL.RPN.FORCE_BOXES_PATH $FORCE_BOXES_PATH \

# gt box: /public/home/weiyn1/coding/uniter/data_wyn/gt_objects/rois_info.pkl
# d2 box: /storage/group/hexm/weiyn/refreasoning/det_results/faster_rcnn_x152_NMS0p3_th0p0_max100_min10
 