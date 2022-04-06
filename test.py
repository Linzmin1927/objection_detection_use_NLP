import numpy as np
import cv2
from utils import *

# path = "./models/desk.jpg"
path = "../test4.jpg"
img_ori = cv2.imread(path)
height, width = img_ori.shape[:2]
org_size = np.array([[height, width]])

img_cv = img_ori.copy()
# img_cv = ResizeShortestEdge(img_cv)
img_cv = img_cv[:, :, ::-1] #RGB
img_cv = preprocess(img_cv)

ort_outs = mymodel.get_features(img_cv)
boxes,cls_feat_stage_1,cls_feat_stage_2,cls_feat_stage_3,mask_pred,proposal_scores = ort_outs


vocabulary = [
    "白色的机器人"
    ]
emb = text_encoder.get_features(vocabulary)



postprocesser = PostProcess()
scores = postprocesser.decode_scores(emb,[cls_feat_stage_1,cls_feat_stage_2,cls_feat_stage_3],[proposal_scores])
score = scores[0]
box = boxes
mask_feat = mask_pred
score_thresh = 0.2
nms_thresh =0.5
mask_thresh =0.3
ori_image_sizes=[height, width]
class_id, box,score, image_mask = postprocesser.decode_box_mask(score
                                ,box
                                ,mask_feat
                                ,score_thresh
                                ,nms_thresh
                                ,mask_thresh
                                ,ori_image_sizes
                                )
res = visualization(img_ori,vocabulary,class_id, box,score, image_mask)
cv2.imwrite("results.jpg",res)
