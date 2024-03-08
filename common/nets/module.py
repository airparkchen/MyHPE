import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from nets.layer import make_conv_layers, make_conv1d_layers, make_deconv_layers, make_linear_layers
from utils.human_models import mano
from utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d
from config import cfg
from nets.crosstransformer import CrossTransformer
from einops import rearrange
from timm.models.vision_transformer import Block

class Transformer(nn.Module):
    def __init__(self, in_chans=512, joint_num=21, depth=4, num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, joint_num, in_chans))
        self.blocks = nn.ModuleList([
            Block(in_chans, num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            for i in range(depth)])
    def forward(self, x):
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x
    
class HandFormer(nn.Module):
    def __init__(self):
        super(HandFormer, self).__init__()
        self.FC = nn.Linear(512*2, 512)
        self.pos_embed = nn.Parameter(torch.randn(1, 1+(8*8), 512))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.SA_T = nn.ModuleList([
            Block(512, 4, 4.0, qkv_bias=False, norm_layer=nn.LayerNorm)  #自注意力塊
            for i in range(4)])
        self.FC2 = nn.Linear(512, 512)


    def forward(self, feat1):
        B, C, H, W = feat1.shape   
        feat1 = rearrange(feat1, 'B C H W -> B (H W) C')   #[N, 512, 7, 7]
        # allhand Token
        #自注意力處理：
        #新增位置資訊self.pos_embed
        token_h = feat1 + self.pos_embed[:,1:] 
        cls_token = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1) 

        token_h = torch.cat((cls_token, token_h), dim=1) 
        for blk in self.SA_T: #自注意力塊（SA_T）
            token_h = blk(token_h)
        token_h = self.FC2(token_h)
        token_h = token_h[:, 1:, :]  #去除cls_token
        output = rearrange(token_h, 'B (H W) C -> B C H W', H=H, W=W) 
        return output

class FuseFormer(nn.Module):
    def __init__(self):
        super(FuseFormer, self).__init__()
        self.FC = nn.Linear(512*2, 512)
        self.pos_embed = nn.Parameter(torch.randn(1, 1+(2*8*8), 512))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.SA_T = nn.ModuleList([
            Block(512, 4, 4.0, qkv_bias=False, norm_layer=nn.LayerNorm)  #自注意力塊
            for i in range(4)])
        self.FC2 = nn.Linear(512, 512)
        #Decoder
        self.CA_T = CrossTransformer()
        self.FC3 = nn.Linear(512, 512)

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape  #特徵圖的形狀為[B, C, H, W]，其中B是批次大小，C是通道數，H和W是高度和寬度。  
        feat1 = rearrange(feat1, 'B C H W -> B (H W) C')   #[N, 512, 8, 8]
        feat2 = rearrange(feat2, 'B C H W -> B (H W) C')   #通過rearrange操作，將特徵圖從[B, C, H, W]轉換為[B, (H W), C]的形式 配合transforme模型的輸入要求
        #[N, 512, 8, 8] --> [N, (8x8), 512] 7x7的特徵圖重排為64個序列元素，每個元素是512維的特徵向量。
        # joint Token
        token_j = self.FC(torch.cat((feat1, feat2), dim=-1)) #利用全連接層（FC）將feat1和feat2進行初步融合 ，生成聯合特徵（token_j）。
        #通過一個全連接層（self.FC），生成SimToken（相似性Token）和JointToken（連接Token）的混合

        #自注意力處理：
        # similar token  token_s的維度[N, Seq_len, Feature_dim]
        #新增位置資訊self.pos_embed
        token_s = torch.cat((feat1, feat2), dim=1) + self.pos_embed[:,1:] #將feat1和feat2直接拼接，並加上位置嵌入（pos_embed），生成相似特徵（token_s）。
        #torch.cat((feat1, feat2), dim=1)將來自兩個來源（例如，兩個不同的特徵圖）的特徵向量feat1和feat2進行拼接。這一步的目的是將兩組特徵結合起來，形成一個更全面的特徵表示。

        #self.pos_embed[:,1:]是位置嵌入，它被加到上一步拼接後的特徵向量上。位置嵌入是Transformer架構中的一個關鍵概念，用於給模型提供序列中各元素的位置信息，使模型能夠理解序列中元素的順序。在這裡，位置嵌入加到特徵向量上，幫助模型捕捉特徵之間的位置關係。
        cls_token = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1) 
        #cls_token是一個特殊的Token，通常用於Transformer模型中表示整個序列的綜合信息。
        #在這裡，cls_token通過與位置嵌入的第一部分相加並通過expand方法複製到每個樣本上，然後將它與拼接後的特徵向量（已加上位置嵌入的其餘部分）一起拼接。這樣，cls_token作為序列的第一個元素被加入，用於捕捉整體的特徵信息。

        token_s = torch.cat((cls_token, token_s), dim=1) #token_s的維度[N, Seq_len, Feature_dim]保持不變，Feature_dim是特徵維度（512）。
        for blk in self.SA_T: #通過一系列的自注意力塊（SA_T），對相似特徵進行處理，強化特徵中的關鍵信息。
            token_s = blk(token_s)
        token_s = self.FC2(token_s) #token_s格式仍然是[N, Seq_len, 512]   #提供了一個額外的變換機會，有助於增強模型的表達能力

        output = self.CA_T(token_j, token_s)  #CA_T（交叉轉換器）進一步處理聯合特徵和經過自注意力處理的相似特徵，進行深入的特徵融合和調整。
        output = self.FC3(output)  
        output = rearrange(output, 'B (H W) C -> B C H W', H=H, W=W) #通過FC3層將處理後的特徵映射回原始特徵圖的空間形狀，即[B, C, H, W] [N, 512, 7, 7]
        return output



class EABlock(nn.Module):
    def __init__(self):
        super(EABlock, self).__init__()
        self.conv_l = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.conv_r = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)  #從整體特徵圖中提取出針對左右手的特定特徵。
        self.conv_all = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.firstExtrace = HandFormer()
        self.Extract = FuseFormer()
        self.Adapt_r = FuseFormer()
        self.Adapt_l = FuseFormer()
        self.conv_l2 = make_conv_layers([512*2, 512*2], kernel=1, stride=1, padding=0)
        self.conv_r2 = make_conv_layers([512*2, 512*2], kernel=1, stride=1, padding=0)

    def forward(self, hand_feat):
        rhand_feat = self.conv_r(hand_feat)  
        lhand_feat = self.conv_l(hand_feat)  #將特徵圖的通道數從2048降低到512  
        allhand_feat = self.conv_all(hand_feat)
        allhand_feat = self.firstExtrace(allhand_feat)
        rhand_feat = self.Extract(rhand_feat, allhand_feat)
        lhand_feat = self.Extract(lhand_feat, allhand_feat)  #add
        inter_feat = self.Extract(rhand_feat, lhand_feat)
        rinter_feat = self.Adapt_r(rhand_feat, inter_feat)
        linter_feat = self.Adapt_l(lhand_feat, inter_feat)
        rhand_feat = self.conv_r2(torch.cat((rhand_feat,rinter_feat),dim=1))
        lhand_feat = self.conv_l2(torch.cat((lhand_feat,linter_feat),dim=1))
        return rhand_feat, lhand_feat



class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.EABlock = EABlock()
        self.conv_r2 = make_conv_layers([512*2, self.joint_num*cfg.output_hm_shape[2]], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_l2 = make_conv_layers([512*2, self.joint_num*cfg.output_hm_shape[2]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, hand_feat):
        rhand_feat, lhand_feat = self.EABlock(hand_feat) 
        rhand_hm = self.conv_r2(rhand_feat)  #EABlock得到的特徵映射到一個新的特徵空間，這個新的特徵空間是為了更好地預測手部關節的熱圖。
        rhand_hm = rhand_hm.view(-1,self.joint_num, cfg.output_hm_shape[2], cfg.output_hm_shape[0], cfg.output_hm_shape[1]) #將輸出的熱圖數據重新排列（reshape）成特定的維度。
        #-1：這是一個占位符，告訴PyTorch自動計算這一維的大小，保證數據元素的總數不變。batch size
        #self.joint_num：關節的總數，每個關節點都會有一個對應的熱圖。
        #cfg.output_hm_shape[2]：熱圖的深度，如果熱圖是三維的話。對於2D熱圖，這個維度可能是1或被忽略。
        #cfg.output_hm_shape[0]和cfg.output_hm_shape[1]：熱圖的高度和寬度。
        rhand_coord = soft_argmax_3d(rhand_hm)

        lhand_hm = self.conv_l2(lhand_feat)
        lhand_hm = lhand_hm.view(-1,self.joint_num, cfg.output_hm_shape[2], cfg.output_hm_shape[0], cfg.output_hm_shape[1])
        lhand_coord = soft_argmax_3d(lhand_hm)
        return rhand_coord, lhand_coord, rhand_feat, lhand_feat



class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.rconv = make_conv_layers([1024,512], kernel=1, stride=1, padding=0)
        self.lconv = make_conv_layers([1024,512], kernel=1, stride=1, padding=0)
        self.rshape_out = make_linear_layers([1024, mano.shape_param_dim], relu_final=False)
        self.rcam_out = make_linear_layers([1024, 3], relu_final=False)
        self.lshape_out = make_linear_layers([1024, mano.shape_param_dim], relu_final=False)
        self.lcam_out = make_linear_layers([1024, 3], relu_final=False)
        #SJT
        self.Transformer_r = Transformer(in_chans=512, joint_num=21, depth=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm)
        self.Transformer_l = Transformer(in_chans=512, joint_num=21, depth=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm)
        #relative translation
        self.root_relative = make_linear_layers([2*(1024),512,3], relu_final=False)
        ##
        self.rroot_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
        self.rpose_out = make_linear_layers([self.joint_num*(512+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint
        self.lroot_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
        self.lpose_out = make_linear_layers([self.joint_num*(512+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint

    def forward(self, rhand_feat, lhand_feat, rjoint_img, ljoint_img):
        batch_size = rhand_feat.shape[0]

        # shape and camera parameters
        rshape_param = self.rshape_out(rhand_feat.mean((2,3)))
        rcam_param = self.rcam_out(rhand_feat.mean((2,3)))
        lshape_param = self.lshape_out(lhand_feat.mean((2,3)))
        lcam_param = self.lcam_out(lhand_feat.mean((2,3)))
        rel_trans = self.root_relative(torch.cat((rhand_feat, lhand_feat), dim=1).mean((2,3)))

        # xyz corrdinate feature
        rhand_feat = self.rconv(rhand_feat)
        lhand_feat = self.lconv(lhand_feat)
        rhand_feat = sample_joint_features(rhand_feat, rjoint_img[:,:,:2]) # batch_size, joint_num, feat_dim
        lhand_feat = sample_joint_features(lhand_feat, ljoint_img[:,:,:2]) # batch_size, joint_num, feat_dim

        # import pdb; pdb.set_trace()
        rhand_feat = self.Transformer_r(rhand_feat)
        lhand_feat = self.Transformer_l(lhand_feat)

        # Relative Translation
        rhand_feat = torch.cat((rhand_feat, rjoint_img),2).view(batch_size,-1)
        lhand_feat = torch.cat((lhand_feat, ljoint_img),2).view(batch_size,-1)

        rroot_pose = self.rroot_pose_out(rhand_feat)
        rpose_param = self.rpose_out(rhand_feat)
        lroot_pose = self.lroot_pose_out(lhand_feat)
        lpose_param = self.lpose_out(lhand_feat)

        return rroot_pose, rpose_param, rshape_param, rcam_param, lroot_pose, lpose_param, lshape_param, lcam_param, rel_trans

