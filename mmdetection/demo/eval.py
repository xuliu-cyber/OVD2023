import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import glob
from tqdm import tqdm
import json
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
images = glob.glob(f'../data/final/data_final_contest/test/*')
anno_dir = f'../data/final/json_final_contest/test.json'
with open(anno_dir) as f:
    annodata = json.load(f)
labels = ['背景']
for cate in annodata['categories']:
    labels.append(cate['name'])
texts = labels

categ_id = {}
image_id = {}

for cate in annodata['categories']:
    categ_id[cate['name']] = cate['id']
for _image in annodata['images']:
    image_id[_image['file_name']] = _image['id']

def cnclip_classify(CLIP_model, CLIP_preprocess, region, texts):
    # 使用Chinese CLIP模型对region进行分类
    region=CLIP_preprocess(region).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = CLIP_model.get_similarity(region, texts)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return labels[np.argmax(probs)], np.max(probs)

if __name__=='__main__':
    # 加载 chinese clip
    CLIP_model, CLIP_preprocess = load_from_name("../data/experiments/muge_finetune_vit-h-14_roberta-huge/checkpoints/epoch_latest.pt",device=device)
    CLIP_model.eval()
    texts = clip.tokenize(texts).to(device)
    
    # 开始测试
    detection_res = [] 
    for imagepath in tqdm(images):
        image_pil = Image.open(imagepath).convert("RGB")
        W,H = image_pil.size
        
        # class agnostic检测结果加载
        with open(f'final_results/preds/'+os.path.basename(imagepath)[0:-3]+'json','r') as f:
            annos = json.load(f)
        scores = annos['scores']
        for i in range(len(scores)):
            if scores[i]<0.3: 
                continue
            box = annos['bboxes'][i]

            # 对region进行分类
            region = image_pil.crop((max(0,box[0]-5), max(0,box[1]-5),min(W,box[2]+5),min(H,box[3]+5)))
            phrase, score = cnclip_classify(CLIP_model, CLIP_preprocess, region, texts)

            if phrase != '背景' and phrase in categ_id.keys():
                detection_res.append({
                    'score': float(scores[i]*score),
                    'category_id': categ_id[phrase],
                    'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]],
                    'image_id': image_id[os.path.basename(imagepath)]
                })
    
    # 保存结果
    filesave = './faster-rcnn-cnCLIP-OVD+bing-final-huge.json'
    with open(filesave,'a') as f:
        json.dump(detection_res,f)
