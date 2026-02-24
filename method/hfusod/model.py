from turtle import shape
import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from base.encoder.vgg import vgg
from base.encoder.resnet import resnet
from .Swin import Swintransformer
from torch_geometric.nn import GCNConv

## Attention modules
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        # max_out = self.fc(self.max_pool(x))
        # out = avg_out + max_out
        return self.sigmoid(avg_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

# adaptive interactive fusion
class AIF(nn.Module):
    def __init__(self, swin_ch, resnet_ch):
        super(AIF,self).__init__()
            
        self.channel=resnet_ch
        self.swin_proj=nn.Conv2d(swin_ch,resnet_ch,kernel_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.sa=SpatialAttention()
        self.fc=nn.Sequential(
            nn.Linear(resnet_ch*2,resnet_ch//16),
            nn.ReLU(inplace=True),
            nn.Linear(resnet_ch//16,resnet_ch*2),
        )
            
    def forward(self,swin_feat,resnet_feat):
        swin_feat=self.swin_proj(swin_feat)
            
        # feat_sum=swin_feat+resnet_feat
        swin_feat=swin_feat*self.sa(swin_feat)
        resnet_feat=resnet_feat*self.sa(resnet_feat)
        feat_sum=torch.cat([swin_feat,resnet_feat],dim=1)
        b,c,_,_=feat_sum.size()
        y=self.avg_pool(feat_sum).view(b,c)
        y=self.fc(y)
        y=y.view(b,c,1,1)
            
        # split api ???
        resnet_att,swin_att=torch.split(y,[self.channel,self.channel],dim=1)
        out=resnet_feat*resnet_att+swin_feat*swin_att
            
        return out   

# cross attention
class CrossAttention(nn.Module):
    def __init__(self, channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        batch_size, C, H, W = x.size()
        
        # 计算查询、键和值
        query = self.query_conv(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        key = self.key_conv(y).view(batch_size, -1, H*W)
        value = self.value_conv(y).view(batch_size, -1, H*W)
        
        # 计算注意力图
        attention = self.softmax(torch.bmm(query, key))
        
        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x  

# cross attention fusion
class CAF(nn.Module):
    def __init__(self, swin_ch, resnet_ch):
        super(CAF, self).__init__()
        
        # Reduce the number of channels for projection
        self.swin_proj = nn.Conv2d(swin_ch, resnet_ch // 2, kernel_size=1)
        self.channel_attention = ChannelAttention(resnet_ch // 2)
        self.spatial_attention = SpatialAttention()
        
        self.cross_attention = CrossAttention(resnet_ch // 2)
        
        # Reduce the number of channels in the refine layer
        self.refine = nn.Sequential(
            nn.Conv2d(resnet_ch, resnet_ch, kernel_size=1),
            nn.BatchNorm2d(resnet_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, swin_feat, resnet_feat):
        # Project Swin features to a smaller number of channels
        swin_feat = self.swin_proj(swin_feat)
        resnet_feat=self.swin_proj(resnet_feat)
        
        # Apply channel attention
        ca_swin = self.channel_attention(swin_feat)
        ca_resnet = self.channel_attention(resnet_feat)
        
        # Apply spatial attention
        sa_swin = self.spatial_attention(swin_feat)
        sa_resnet = self.spatial_attention(resnet_feat)
        
        # Enhanced features
        enhanced_swin = swin_feat * ca_swin * sa_swin
        enhanced_resnet = resnet_feat * ca_resnet * sa_resnet
        
        # Cross attention feature interaction
        cross_features = self.cross_attention(enhanced_swin, enhanced_resnet)
        
        # Feature fusion and refinement (change to enhanced_swin)
        concat_feat = torch.cat([cross_features, enhanced_swin], dim=1)
        output = self.refine(concat_feat)
        
        return output

## advance upsampling module
class CBAM(nn.Module):
    """CBAM注意力模块"""
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class AdvancedUpSampling(nn.Module):
    """创新型上采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 多路径上采样 ???
        self.up_bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_transpose = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.up_pixel_shuffle = nn.PixelShuffle(2)
        self.conv_pixel_shuffle = nn.Conv2d(in_channels, in_channels*4, 1)
        
        # 密集连接块
        # self.dense_block = DenseBlock(in_channels//2, growth_rate=32, num_layers=3)
        self.dense_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.dense_bn = nn.BatchNorm2d(in_channels)
        self.dense_relu = nn.ReLU(inplace=True)
        
        # 特征金字塔
        self.pyramid_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(bin_size) for bin_size in [2, 4, 8, 16]
        ])
        pyramid_channels = in_channels//4
        self.pyramid_convs = nn.ModuleList([
            nn.Conv2d(in_channels, pyramid_channels, 1) 
            for _ in range(4)
        ])
        
        # 注意力机制
        # self.cbam = CBAM(in_channels//2+32*3+pyramid_channels*3+out_channels)
        self.cbam = CBAM(in_channels+out_channels)
        
        # 最终融合
        # total_channels = in_channels//2 + 32*3 + pyramid_channels*3 + out_channels
        total_channels = in_channels + out_channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差连接
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
    def forward(self, x1, x2):
        # 多路径上采样
        path1 = self.up_bilinear(x1)
        # path2 = self.up_transpose(x1)
        path3 = self.conv_pixel_shuffle(x1)
        path3 = self.up_pixel_shuffle(path3)
        
        # 特征金字塔池化
        pyramid_features = []
        for pool, conv in zip(self.pyramid_pools, self.pyramid_convs):
            p = pool(path1)
            p = conv(p)
            p = F.interpolate(p, size=path1.shape[2:], mode='bilinear', align_corners=True)
            pyramid_features.append(p)
        
        pyramid_features=torch.cat(pyramid_features,dim=1) # in_channels nums
        
        # 密集连接
        combined = path3
        dense_out = self.dense_conv(combined) # 输出的结果+128通道
        dense_out = self.dense_bn(dense_out)
        dense_out = self.dense_relu(dense_out)
        
        fused = dense_out + pyramid_features        
        
        # 注意力增强
        fused_cat=torch.cat((fused, x2), dim=1)
        fused = self.cbam(fused_cat)
        
        # 最终处理
        out = self.final_conv(fused)
        
        # 残差连接
        residual = self.residual(x1)
        
        return out + residual

## new post refinement network
class PostProcessingNetwork(nn.Module):
    """创新型后处理优化网络"""
    def __init__(self, in_channels):
        super().__init__()
        
        # 多尺度特征提取
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, 64, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, 64, 3, padding=4, dilation=4)
        
        # 自适应特征重校准
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(192, 48, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 192, 1),
            nn.Sigmoid()
        )
        
        # 边界感知模块
        self.boundary = BoundaryAwareModule(192)
        
        # 上下文聚合模块
        self.context = ContextAggregationModule(192)
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Conv2d(192*2, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, 1)
        )

    def forward(self, x):
        # 多尺度特征提取
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feats = torch.cat([feat1, feat2, feat3], dim=1)
        
        # 自适应特征重校准
        se_weight = self.se(feats)
        feats = feats * se_weight
        
        # 边界感知
        boundary_feats = self.boundary(feats)
        
        # 上下文聚合
        context_feats = self.context(feats)
        
        # 特征融合
        refined_feats = torch.cat([boundary_feats, context_feats], dim=1)
        output = self.fusion(refined_feats)
        
        # 残差连接
        return x + output

class BoundaryAwareModule(nn.Module):
    """边界感知模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels*2, in_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        edge = self.conv1(x)
        edge = torch.abs(x - edge)  # 边缘检测
        combined = torch.cat([x, edge], dim=1)
        output = self.relu(self.conv2(combined))
        return output

class ContextAggregationModule(nn.Module):
    """上下文聚合模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(size) for size in [1, 2, 4, 8]
        ])
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels//4, 1) for _ in range(4)
        ])
        self.fusion = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        contexts = []
        for pool, conv in zip(self.pools, self.convs):
            context = pool(x)
            context = conv(context)
            context = F.interpolate(context, size=x.shape[2:], 
                                  mode='bilinear', align_corners=True)
            contexts.append(context)
        
        contexts = torch.cat(contexts, dim=1)
        # output = self.fusion(torch.cat([x, contexts], dim=1))
        output = self.fusion(contexts+x)
        return output
    
## new r1 refinement
class InnovativeFeatureEnhancement(nn.Module):
    """创新型特征增强模块"""
    def __init__(self, channels):
        super().__init__()
        
        # 多尺度特征提取
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels//4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])
        
        # 通道注意力(SE模块改进版)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 自适应特征重标定
        self.calibration = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 自适应噪声抑制
        self.noise_suppress = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        # 残差边缘增强
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征金字塔池化
        self.pyramid_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(s) for s in [2, 4, 8]
        ])
        self.pyramid_convs = nn.ModuleList([
            nn.Conv2d(channels, channels//4, 1) for _ in range(3)
        ])
        
        self.convert=nn.Conv2d(48,64,1)

    def forward(self, x):
        
        # 通道注意力
        channel_att = self.se(x)
        x_channel = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial(spatial)
        x_spatial = x * spatial_att
        
        # 特征金字塔池化
        pyramid_feats = []
        for pool, conv in zip(self.pyramid_pools, self.pyramid_convs):
            p = pool(x)
            p = conv(p)
            p = F.interpolate(p, size=x.shape[2:], mode='bilinear', align_corners=True)
            pyramid_feats.append(p)
        pyramid_out = torch.cat(pyramid_feats, dim=1)
        
        # 自适应特征重标定
        combined = torch.cat([x_channel, x_spatial], dim=1)
        calibrated = self.calibration(combined)
        
        # 自适应噪声抑制
        noise_mask = self.noise_suppress(calibrated)
        denoised = calibrated * noise_mask
        
        # 残差边缘增强
        edge = self.edge_enhance(denoised)
        edge_residual = edge - denoised
        enhanced = denoised + edge_residual
        
        # 融合所有特征
        pyramid_out=self.convert(pyramid_out)
        final_out = enhanced + pyramid_out
        
        return final_out

# whole framework
class Network(nn.Module):
    def __init__(self, config, encoder, feat, cfg=None):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()

        self.config=config
        self.encoder = encoder
        self.cfg = cfg
        
        self.fusion2=AIF(128,128)
        self.fusion3=AIF(256,256)       
        self.fusion4=CAF(512,512)
        self.fusion5=CAF(1024,1024)
        
        self.decoder1 = AdvancedUpSampling(128,64)
        self.decoder2 = AdvancedUpSampling(256,128)
        self.decoder3 = AdvancedUpSampling(512,256)
        self.decoder4 = AdvancedUpSampling(1024,512)

        ## swin_base_384
        self.swin = Swintransformer(img_size=384, patch_size=4, in_chans=3, num_classes=1000,
                                    embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                    window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                    drop_rate=0., attn_drop_rate=0., drop_path_ra512te=0.2)
        self.swin.load_state_dict(torch.load('/home/hwl/Documents/Segmentation/SALOD-master/pre/swin_base_patch4_window12_384_22k.pth')['model'], strict=False)

        if self.cfg is not None and self.cfg.snapshot:
            print('load checkpoint')
            pretrain = torch.load(self.cfg.snapshot)
            new_state_dict = {}
            for k, v in pretrain.items():
                new_state_dict[k[7:]] = v
            self.load_state_dict(new_state_dict, strict=False)

        self.convertr2=nn.Conv2d(256,128,1)
        self.convertr3=nn.Conv2d(512,256,1)
        self.convertr4=nn.Conv2d(1024,512,1)
        self.convertr5=nn.Conv2d(2048,1024,1)
        
        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        self.linearr4 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        
        self.refnet=PostProcessingNetwork(64)
        self.r1ref=InnovativeFeatureEnhancement(64)
        
        self.convertd2=nn.Conv2d(128,64,3,padding=1)
        self.convertd2_bn=nn.BatchNorm2d(64)
        self.convertd2_relu=nn.ReLU(inplace=True)
        
        # 要计算每层特征图输出 和3,1,1有什么区别？？？？
        self.converterr2=nn.Conv2d(256,1,1)
        self.converterr3=nn.Conv2d(512,1,1)
        self.converterr4=nn.Conv2d(1024,1,1)
        self.converterr5=nn.Conv2d(2048,1,1)
        self.converters1=nn.Conv2d(128,1,1)
        self.converters2=nn.Conv2d(256,1,1)
        self.converters3=nn.Conv2d(512,1,1)
        self.converters4=nn.Conv2d(1024,1,1)
    
    def forward(self, x, phase='test', shape=None, mask=None):

        shape = x.size()[2:] if shape is None else shape
        y = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=True)        
      
        s1, s2, s3, s4 = self.swin(y)
        # s1 8,128,96,96
        # s2 8,256,48,48
        # s3 8,512,24,24
        # s4 8,1024,12,12

        r11, r22, r33, r44, r55 = self.encoder(x)
        r2=self.convertr2(r22)
        r3=self.convertr3(r33)
        r4=self.convertr4(r44)
        r5=self.convertr5(r55)
        r2 = F.interpolate(r2,size=s1.shape[2:],mode='bilinear') # 8, 128, 96, 96
        r3 = F.interpolate(r3,size=s2.shape[2:],mode='bilinear') # 8, 256, 48, 48
        r4 = F.interpolate(r4,size=s3.shape[2:],mode='bilinear') # 8, 512, 24, 24
        r5 = F.interpolate(r5,size=s4.shape[2:],mode='bilinear') # 8, 1024, 12, 12         
                
        fuse_feat11=self.fusion2(s1,r2) # 8, 128, 96, 96
        fuse_feat22=self.fusion3(s2,r3) # 8, 256, 48, 48
        fuse_feat33=self.fusion4(s3,r4) # 8, 512, 24, 24
        fuse_feat44=self.fusion5(s4,r5) # 8, 1024, 12, 12
        
        # 尝试不同的解码器及解码框架
        decoder4=self.decoder4(fuse_feat44,fuse_feat33) # 8, 512, 24, 24
        ufd4=decoder4   # 512
        decoder3=self.decoder3(decoder4,fuse_feat22) # 8, 256, 48, 48   
        ufd3=decoder3   # 256
        decoder2=self.decoder2(decoder3,fuse_feat11) # 8, 128, 96, 96    
        ufd2=decoder2   # 128
            
        # decoder1=self.decoder1(fuse_feat1,64) # 8, 64, 192, 192
        # decoder2=self.convertd2(decoder2)
        
        r1=self.r1ref(r11) 
        r1=F.interpolate(r1,size=(192,192),mode='bilinear')
        # decoder2=self.convertd2_relu(self.convertd2_bn(self.convertd2(decoder2)))
        # r1=decoder2+r1
        r1=self.decoder1(decoder2,r1) 
        ufd1=r1   # 64

        salmap = self.refnet(r1)
        salmap = F.interpolate(self.linearp1(salmap), size=shape, mode='bilinear')
        
        decoder4 = F.interpolate(self.linearr4(decoder4), size=shape, mode='bilinear')
        decoder3 = F.interpolate(self.linearr3(decoder3), size=shape, mode='bilinear')
        decoder2 = F.interpolate(self.linearr2(decoder2), size=shape, mode='bilinear')
        decoder1 = F.interpolate(self.linearp1(r1), size=shape, mode='bilinear')

        r22 = F.interpolate(self.converterr2(r22), size=shape, mode='bilinear')
        r33 = F.interpolate(self.converterr3(r33), size=shape, mode='bilinear')
        r44 = F.interpolate(self.converterr4(r44), size=shape, mode='bilinear')
        r55 = F.interpolate(self.converterr5(r55), size=shape, mode='bilinear')
        
        s1 = F.interpolate(self.converters1(s1), size=shape, mode='bilinear')
        s2 = F.interpolate(self.converters2(s2), size=shape, mode='bilinear')
        s3 = F.interpolate(self.converters3(s3), size=shape, mode='bilinear')
        s4 = F.interpolate(self.converters4(s4), size=shape, mode='bilinear')
        
        fuse_feat11 = F.interpolate(self.converters1(fuse_feat11), size=shape, mode='bilinear')
        fuse_feat22 = F.interpolate(self.converters2(fuse_feat22), size=shape, mode='bilinear')
        fuse_feat33 = F.interpolate(self.converters3(fuse_feat33), size=shape, mode='bilinear')
        fuse_feat44 = F.interpolate(self.converters4(fuse_feat44), size=shape, mode='bilinear')
        
        ufd44 = F.interpolate(self.linearr4(ufd4), size=shape, mode='bilinear')
        ufd33 = F.interpolate(self.linearr3(ufd3), size=shape, mode='bilinear')
        ufd22 = F.interpolate(self.linearr2(ufd2), size=shape, mode='bilinear')
        ufd11 = F.interpolate(self.linearp1(ufd1), size=shape, mode='bilinear')

        # All output are save in a Dict, while the key of final prediction is 'final'
        OutDict = {}
        OutDict['final'] = ufd11
        OutDict['sal'] = [salmap, decoder4, decoder3, decoder2, decoder1]
        return OutDict
