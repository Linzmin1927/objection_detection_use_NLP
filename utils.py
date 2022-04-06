import numpy as np
import cv2
from skimage.transform import resize
import onnx
import onnxruntime
from transformers import CLIPProcessor, CLIPModel, pipeline

class TextEncoder:
    def __init__(self):
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_features(self,text):
        vocabulary_en = self.translator(text)
        vocabulary_en = [v['translation_text'] for v in vocabulary_en]
        # emb = get_clip_embeddings(vocabulary_en)
        inputs = self.processor(text=vocabulary_en, return_tensors="pt", padding=True)
        emb = self.model.get_text_features(**inputs)
        emb = emb.detach().numpy().T
        emb = np.hstack([emb,np.zeros([512,1])])
        emb = normalize(emb, p=2, dim=0)
        return emb

class MyModel:
    def __init__(self) -> None:
        file = './myModel.onnx'
        onnx_model = onnx.load(file)
        onnx.checker.check_model(onnx_model)
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # sess_options.optimized_model_filepath = './myModel_opt.onnx'
        self.ort_session = onnxruntime.InferenceSession(file,sess_options)

    def get_features(self,image):        
        ort_inputs = {self.ort_session.get_inputs()[0].name: image}#
        ort_outs = self.ort_session.run(None, ort_inputs)
        return ort_outs

def ResizeShortestEdge(img,max_size=1333):
    h,w,_= img.shape
    scale = h/w
    if h>w:
        h_max = max_size
        w_max = int(h_max/scale+1)
    else:
        w_max = max_size
        h_max = int(scale*w_max+1)
    return cv2.resize(img,[w_max,h_max])

def preprocess(img,size_divisibility=32):
    h,w,c = img.shape
    h_ = (h//size_divisibility +1)*size_divisibility
    w_ = (w//size_divisibility +1)*size_divisibility
    # img = cv2.resize(img,[w_,h_])
    img = cv2.resize(img,(1344,800))
    img = img.transpose(2, 0, 1)
    mean = np.array([[[123.6750]],[[116.2800]],[[103.5300]]])
    std = np.array([[[58.3950]],[[57.1200]],[[57.3750]]])
    new_img = ((img-mean)/std).astype("float32")
    new_img = new_img[None]
    return new_img


class PostProcess:
    def __init__(self) -> None:
        pass
    def get_score_per_stage(self,cls_feat,emb):
        def sigmoid(x):
            s = 1/(1 + np.exp(-x))
            return s
        p = np.matmul(cls_feat, emb)
        scores = sigmoid(p)
        return scores
    def nms(self, boxes, scores,thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2-x1+1)*(y2-y1+1)
        res = []
        index = scores.argsort()[::-1]
        while index.size>0:
            i = index[0]
            res.append(i)
            x11 = np.maximum(x1[i],x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i],x2[index[1:]])
            y22 = np.minimum(y2[i],y2[index[1:]])
            w = np.maximum(0,x22-x11+1)
            h = np.maximum(0,y22-y11+1)
            overlaps = w * h
            iou = overlaps/(areas[i]+areas[index[1:]]-overlaps)
            idx = np.where(iou<=thresh)[0]
            index = index[idx+1]
        # print(res)
        return res
    def decode_scores(self,emb,cls_feat_per_stage,proposal_scores):
        num = emb.shape[1]
        # import pdb;pdb.set_trace()
        # s1 = self.get_score_per_stage(cls_feat_per_stage[0],emb)
        # s2 = self.get_score_per_stage(cls_feat_per_stage[1],emb)
        # s3 = self.get_score_per_stage(cls_feat_per_stage[2],emb)
        score_per_stage = [self.get_score_per_stage(cls_feat,emb) for cls_feat in cls_feat_per_stage] 
        # import pdb;pdb.set_trace()
        score_mean = [sum(score_per_stage)/len(cls_feat_per_stage)]
        # scores = [(s * ps[:, None]) ** 0.5 for s, ps in zip(scores, proposal_scores)]
        # scores = [s * (s == s[:, :-1].max(dim=1)[0][:, None]).float() for s in scores]
        scores_m = [(s * ps[:, None]) ** 0.5 for s, ps in zip(score_mean, proposal_scores)]
        scores = [ss*(ss==np.vstack([ss[:, :-1].max(axis=1)]*num).T) for ss in scores_m]
        # import pdb;pdb.set_trace()
        return scores
    
    def decode_box_mask(self ,score
                        ,box
                        ,mask_feat
                        ,score_thresh = 0.3
                        ,nms_thresh =0.5
                        ,mask_thresh =0.5
                        ,ori_image_sizes=[518, 898]                               
                        ):
        score = score[:, :-1]
        filter_mask = score > score_thresh
        filter_inds = filter_mask.nonzero()
        filter_inds = np.vstack(filter_inds).T

        scale_x = 1344 / ori_image_sizes[1]
        scale_y = 800 / ori_image_sizes[0]
        box[:,0] /= scale_x
        box[:,2] /= scale_x
        box[:,1] /= scale_y
        box[:,3] /= scale_y
        box = box[filter_inds[:,0],:]
        score = score[filter_mask]
        keep = self.nms(box, score,nms_thresh)
        box, score  = box[keep], score[keep]
        filter_inds = filter_inds[keep,:]#dim0 id, dim1 pred_classes
        class_id = filter_inds[:, 1]
        mask_feat = mask_feat[filter_inds[:,0],...]
        mask_feat = mask_feat>=mask_thresh
        image_mask = []
        # import pdb;pdb.set_trace()
        # i = 0
        for i in range(len(mask_feat)):
            box_1 = box[i,:]
            msk = mask_feat[i,0,...]
            x1 = int(box_1[0])
            y1 = int(box_1[1])
            x2 = int(box_1[2])
            y2 = int(box_1[3])
            x1 = np.clip(x1,0,ori_image_sizes[1]-1)
            x2 = np.clip(x2,0,ori_image_sizes[1]-1)
            y1 = np.clip(y1,0,ori_image_sizes[0]-1)
            y2 = np.clip(y2,0,ori_image_sizes[0]-1)
            w = int(x2 - x1 +1)
            h = int(y2 - y1 +1)
            msk = resize(msk, (h,w))
            # import cv2
            # msk = cv2.resize(msk,[w,h])
            msk_rsz = np.zeros(ori_image_sizes)
            # import pdb;pdb.set_trace()
            msk_rsz[y1:y2+1,x1:x2+1] = msk.astype('int')
            image_mask.append(msk_rsz)
        box = box.astype('int32')
        return class_id, box,score, image_mask

def normalize(nparray, p=2, dim=-1):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=p, axis=dim, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def visualization(img_ori,vocabulary,class_id, box, score,image_mask):
    if box==[]:
        print(".....None....",box)
        return img_ori
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('font.ttf', 20, encoding="utf-8")

    all_mask = np.zeros_like(img_ori,dtype='uint8')
    color_list = []
    for i in range(len(vocabulary)):
        color = np.random.randint(125, 255, size=(2),dtype='int').tolist()+[255]
        color_list.append(color)

    for i in range(len(class_id)):
        # import pdb;pdb.set_trace()  
        # color = color_list[class_id[i]]
        color = [0,0,255]
        cv2.rectangle(img_ori, (box[i,0],box[i,1]), (box[i,2],box[i,3]), color, 1)
        pilimg = Image.fromarray(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pilimg)  # 图片上打印
        print('box',box)
        print('vocabulary',vocabulary)
        print('class_id',class_id,len(class_id),i)
        string = f"{vocabulary[class_id[i]]},{score[i]:.3f}"
        draw.text((box[i,0],box[i,1]-10), string, (color[2],color[1],color[0]),font=font)
        img_ori = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(image_mask[i].astype('uint8'), cv2.COLOR_GRAY2BGR)
        mask = mask*np.array(color).reshape([1,1,3])
        # all_mask = cv2.addWeighted(all_mask, 1, mask, 1, 0)
        all_mask = all_mask+ mask

    res = cv2.addWeighted(img_ori, 1, all_mask.astype('uint8'), 0.5, 0)
    return res


mymodel=MyModel()
text_encoder = TextEncoder()
