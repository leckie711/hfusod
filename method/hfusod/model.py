import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from base.encoder.vgg import vgg
from base.encoder.resnet import resnet
from .Swin import Swintransformer
from torch_geometric.nn import GCNConv

# adaptive interactive fusion
# modified to the former designed architecture
class AIF(nn.Module):
    def __init__(self, swin_ch, resnet_ch):
        super(AIF,self).__init__()
            
        self.channel=resnet_ch
        self.swin_proj=nn.Conv2d(swin_ch,resnet_ch,kernel_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.sa=SA(resnet_ch)
        self.fc=nn.Sequential(
            nn.Linear(resnet_ch*2,resnet_ch//16),
            nn.ReLU(inplace=True),
            nn.Linear(resnet_ch//16,resnet_ch*2),
        )
            
    def forward(self,swin_feat,resnet_feat):
        swin_feat=self.swin_proj(swin_feat)
            
        # feat_sum=swin_feat+resnet_feat
        swin_feat=self.sa(swin_feat)
        resnet_feat=self.sa(resnet_feat)
        feat_sum=torch.cat([swin_feat,resnet_feat],dim=1)
        b,c,_,_=feat_sum.size()
        y=self.avg_pool(feat_sum).view(b,c)
        y=self.fc(y)
        y=y.view(b,c,1,1)
            
        # split api ???
        resnet_att,swin_att=torch.split(y,[self.channel,self.channel],dim=1)
        out=resnet_feat*resnet_att+swin_feat*swin_att
            
        return out   

# cross fusion
# 提高整体的reduction
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
        
        # Feature fusion and refinement
        concat_feat = torch.cat([cross_features, enhanced_resnet], dim=1)
        output = self.refine(concat_feat)
        
        return output

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

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

class ImprovedDecoder(nn.Module):
    def __init__(self,in_channels,skip_channels,out_channels):
        super(ImprovedDecoder,self).__init__()
        
        self.resblk = ResBlock(out_channels)
        self.ca=ChannelAttention(out_channels)
        self.sa=SA(out_channels)
        
        self.upsample=UpsampleBlock(in_channels=in_channels,skip_channels=skip_channels,out_channels=out_channels)

    def forward(self,x,skip):
        x=self.upsample(x,skip)
        x=self.resblk(x)
        x=x*self.ca(x)
        x=self.sa(x)
        # x=self.refine(x)
        return x
    
class UpsampleBlock(nn.Module):
    def __init__(self,in_channels,skip_channels,out_channels):
        super(UpsampleBlock,self).__init__()
        
        self.conv_before_up=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),    
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.up=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.skip_conv=nn.Sequential(
            nn.Conv2d(skip_channels,out_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels),   
        )
        
        self.fusion=nn.Sequential(
            nn.Conv2d(out_channels*2,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self,x,skip=None):
        x=self.conv_before_up(x)
        x=self.up(x)
        
        if skip is not None:
            skipx=self.skip_conv(x)
            x=self.fusion(torch.cat([x,skipx],dim=1))
            
        return x    
    
class ResBlock(nn.Module):
    def __init__(self,channels):
        super(ResBlock,self).__init__()

        self.conv1=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(channels)
        
        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(channels)
        self.relu=nn.ReLU(inplace=True)
        
    def forward(self,x):
        residual=x
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        
        out=self.conv2(out)
        out=self.bn2(out)
        out+=residual
        out=self.relu(out)
        
        return out  
    
class SA(nn.Module):
    def __init__(self,channels):
        super(SA,self).__init__()
        
        self.conv=nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,padding=1,bias=False),    
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,channels,kernel_size=3,padding=1,bias=False),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        return x*self.conv(x)   
    
# R1 spatial attention
class R1SA(nn.Module):
    def __init__(self, in_channel, kernel_size=7):
        super(R1SA, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.seq=nn.Sequential(
            nn.Conv2d(1,1,kernel_size=3,padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1,1,kernel_size=3,padding=1),
            nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.atrous_block1 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(in_channel * 4, in_channel, 1, 1)

    def forward(self, x):
        identity=x

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        x_out = torch.cat([atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1)
        x_out = self.conv_1x1_output(x_out)

        avgout = torch.mean(x_out, dim=1, keepdim=True)
        maxout, _ = torch.max(x_out, dim=1, keepdim=True)
        avgout = self.seq(avgout)
        maxout = self.seq(maxout)
        x_out = torch.cat([avgout, maxout], dim=1)
        # x_out = avgout + maxout
        x_out = self.conv(x_out)
        x_out = self.sigmoid(x_out)
        x_out = x_out.expand_as(identity)
        x_out = identity * x_out
        return x_out
    
# multi-interactive refinement network
class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4=nn.BatchNorm2d(64)
        self.relu4=nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv5=nn.Conv2d(64,64,3,padding=1)
        self.bn5=nn.BatchNorm2d(64)
        self.relu5=nn.ReLU(inplace=True)

        self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.cs45 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.cs34 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.cs23 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.cs12 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self,x):

        hx = x
        hx = self.conv0(hx)  # 8 64 320

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx1 = self.pool1(hx1) # 8 64 160

        hx2 = self.relu2(self.bn2(self.conv2(hx1)))
        hx2 = self.pool2(hx2) # 8 64 80

        hx3 = self.relu3(self.bn3(self.conv3(hx2)))
        hx3 = self.pool3(hx3) # 8 64 40

        hx4 = self.relu4(self.bn4(self.conv4(hx3)))
        hx4=self.pool4(hx4)

        hx5=self.relu5(self.bn5(self.conv5(hx4)))
        hx5=self.pool5(hx5)

        # 每一层特征的操作
        # 与深层特征相乘 
        # 再与本层特征相加
        hx5up=F.interpolate(hx5,size=hx4.shape[2:],mode='bilinear')
        h4xfuse=hx4*hx5up
        h4xfuse=self.cs45(h4xfuse)

        hx4up=F.interpolate(hx4,size=hx3.shape[2:],mode='bilinear')
        h3xfuse=hx3*hx4up
        h4xfuseup=F.interpolate(h4xfuse,size=h3xfuse.shape[2:],mode='bilinear')
        h3xfuse=self.cs34(torch.cat((h3xfuse,h4xfuseup),dim=1))
        
        
        hx3up=F.interpolate(hx3,size=hx2.shape[2:],mode='bilinear')
        h2xfuse=hx2*hx3up
        h3xfuseup=F.interpolate(h3xfuse,size=h2xfuse.shape[2:],mode='bilinear')        
        h2xfuse=self.cs23(torch.cat((h2xfuse,h3xfuseup),dim=1))  #8 64 80 80 

        hx2up=F.interpolate(hx2,size=hx1.shape[2:],mode='bilinear')
        h1xfuse=hx1*hx2up
        h2xfuseup=F.interpolate(h2xfuse,size=h1xfuse.shape[2:],mode='bilinear')
        h1xfuse=self.cs12(torch.cat((h1xfuse,h2xfuseup),dim=1))  #8 64 160 160
        
        h1xfinal=self.cs45(h1xfuse)

        d1=self.upscore2(h1xfinal)
        residual = self.conv_d0(d1)

        return x + residual

class Network(nn.Module):
    def __init__(self, config, encoder, feat, cfg=None):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()

        self.config=config
        self.encoder = encoder
        self.cfg = cfg
        
        # 改 34的 因为hw的计算特别大
        self.fusion2=AIF(128,128)
        self.fusion3=AIF(256,256)       
        self.fusion4=CAF(512,512)
        self.fusion5=CAF(1024,1024)
        
        self.decoder1 = ImprovedDecoder(128,64,64)
        self.decoder2 = ImprovedDecoder(256,128,128)
        self.decoder3 = ImprovedDecoder(512,256,256)
        self.decoder4 = ImprovedDecoder(1024,512,512)

        # # swin_base_384
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
        
        self.r1sa=R1SA(64)
        self.refunet = RefUnet(1,64)
    
    def forward(self, x, phase='test', shape=None, mask=None):

        shape = x.size()[2:] if shape is None else shape
        y = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=True)
        
        # 是否需要更换更好的transformer及其权重？
        s1, s2, s3, s4 = self.swin(y)
        # s1 8,128,96,96
        # s2 8,256,48,48
        # s3 8,512,24,24
        # s4 8,1024,12,12

        r1, r2, r3, r4, r5 = self.encoder(x)
        r2=self.convertr2(r2)
        r3=self.convertr3(r3)
        r4=self.convertr4(r4)
        r5=self.convertr5(r5)
        r2 = F.interpolate(r2,size=s1.shape[2:],mode='bilinear') # 8, 128, 96, 96
        r3 = F.interpolate(r3,size=s2.shape[2:],mode='bilinear') # 8, 256, 48, 48
        r4 = F.interpolate(r4,size=s3.shape[2:],mode='bilinear') # 8, 512, 24, 24
        r5 = F.interpolate(r5,size=s4.shape[2:],mode='bilinear') # 8, 1024, 12, 12
        
        # 是否需要对提取的特征进行处理？
        
        # 尝试其他的融合方式
        fuse_feat1=self.fusion2(s1,r2) # 8, 128, 96, 96
        fuse_feat2=self.fusion3(s2,r3) # 8, 256, 48, 48
        fuse_feat3=self.fusion4(s3,r4) # 8, 512, 24, 24
        fuse_feat4=self.fusion5(s4,r5) # 8, 1024, 12, 12
        
        # 尝试不同的解码器及解码框架
        # 先是全用+试下
        decoder4=self.decoder4(fuse_feat4,512) # 8, 512, 24, 24
        fuse_feat3=fuse_feat3+decoder4
        decoder3=self.decoder3(fuse_feat3,256) # 8, 256, 48, 48   
        fuse_feat2=fuse_feat2+decoder3    
        decoder2=self.decoder2(fuse_feat2,128) # 8, 128, 96, 96        
        fuse_feat1=fuse_feat1+decoder2
        decoder1=self.decoder1(fuse_feat1,64) # 8, 64, 192, 192
        # 融合resnet 第一层特征？？？
        r1=self.r1sa(r1)
        r1=F.interpolate(r1,size=decoder1.size()[2:],mode='bilinear')
        decoder1=decoder1+r1

        salmap = F.interpolate(self.linearp1(decoder1), size=shape, mode='bilinear')
        salmap = self.refunet(salmap)
        
        decoder4 = F.interpolate(self.linearr4(decoder4), size=shape, mode='bilinear')
        decoder3 = F.interpolate(self.linearr3(decoder3), size=shape, mode='bilinear')
        decoder2 = F.interpolate(self.linearr2(decoder2), size=shape, mode='bilinear')

        # All output are save in a Dict, while the key of final prediction is 'final'
        OutDict = {}
        OutDict['final'] = salmap
        OutDict['sal'] = [salmap, decoder4, decoder3, decoder2]
        return OutDict
