import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import time
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, augmentation, load_img_pic
from utils.vis import save_obj, vis_mesh, vis_keypoints, visualize_keypoints
from utils.human_models import mano
from InterHand26M import Jr
import copy
import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=str, default=29, dest='test_epoch')
    parser.add_argument('--gpu', type=str, default="0,1", dest='gpu_ids')
    parser.add_argument('--input', type=str, default='example_image2.png', dest='input')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
model = get_model('test')
model = DataParallel(model).cuda()
model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % int(args.test_epoch))
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
img = load_img(args.input)
height, width = img.shape[:2]
bbox = [0, 0, width, height] 
bbox = process_bbox(bbox, width, height)
img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, 'test')  #bb2img_trans=inv_trans=逆仿射變換矩陣  (2, 3) 
img = transform(img.astype(np.float32))/255.
img = img.cuda()[None,:,:,:]
ouput2input_param = [height  / cfg.output_hm_shape[0] , width  / cfg.output_hm_shape[1]]

inputs = {'img': img}
targets = {}
meta_info = {}
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')
    
img = (img[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)   #PyTorch張量轉換回NumPy陣列
rmano_mesh = out['rmano_mesh_cam'][0].cpu().numpy()  #rmano_mesh:(778, 3)
lmano_mesh = out['lmano_mesh_cam'][0].cpu().numpy()  #lmano_mesh:(778, 3)
rel_trans = out['rel_trans'][0].cpu().numpy()        #rel_trans:(3,)
# print(f"shape of \nrmano_mesh:{rmano_mesh.shape} \nlmano_mesh:{lmano_mesh.shape} \nrel_trans:{rel_trans.shape}")
save_obj(rmano_mesh*np.array([1,-1,-1]), mano.face['right'], 'demo_right.obj')
save_obj((lmano_mesh+rel_trans)*np.array([1,-1,-1]), mano.face['left'], 'demo_left.obj')

# cv2.imshow('Keypoints Visualization', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # 將RGB轉換回BGR，因為OpenCV預設使用BGR
# cv2.waitKey(0)  # 等待按鍵事件
# cv2.destroyAllWindows()  # 關閉所有OpenCV窗口

lmano_joint = out['lmano_joint'].cpu().numpy()  # 假設是在使用PyTorch且需要轉換為NumPy數組  lmano_joint = (1, 21, 2)
rmano_joint = out['rmano_joint'].cpu().numpy() 
vert_pred_r = (out['lmano_mesh_cam']).cuda() *1000
print(vert_pred_r)
regressorR = Jr(copy.deepcopy(mano.layer['right'].J_regressor))
pred_joint_proj_r = regressorR(vert_pred_r)
# print(pred_joint_proj_r)
root_r = pred_joint_proj_r[9:10,:]
mesh_vertex = np.array(vert_pred_r.cpu()+ np.array([0,0,0]))   #要加上手腕座標
# mesh_vertex = np.array(vert_pred_r.cpu())
# print (lmano_joint[0])
img = load_img_pic(args.input)
img = np.ascontiguousarray(img)

#直接從熱圖轉座標作圖
kps = np.array([((x), (y)) for x, y in lmano_joint[0]] ) *  ouput2input_param
# kps = np.array([((x), (y)) for x, y, _ in pred_joint_proj_r[0]] ) *  ouput2input_param
# img_with_kps = visualize_keypoints(img, kps)
# cv2.imshow('Keypoints Visualization', img_with_kps)
# cv2.waitKey(0)  # Wait for a key press to close the window
# cv2.destroyAllWindows()
# 官方vis程式
# img_with_kps = vis_keypoints(img, kps, 0.5)
# cv2.imshow('Keypoints Visualization', cv2.cvtColor(img_with_kps, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.resize(img,(240,240))
ouput2input_param_expanded = np.append(ouput2input_param, 1)
img_with_kps = vis_mesh(img, mesh_vertex[0] , 0.5)
cv2.imshow('Keypoints Visualization', cv2.cvtColor(img_with_kps, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

