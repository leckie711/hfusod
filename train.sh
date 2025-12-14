#!/bin/bash


# =================================================== state-of-the-art test =============================================
# =================================================== state-of-the-art test =============================================
# =================================================== state-of-the-art test =============================================

# python3 train.py dhsnet --gpus=0 --trset=SALOD

# python3 train.py amulet --gpus=0 --trset=SALOD

# python3 train.py nldf --gpus=0 --trset=SALOD

# python3 train.py srm --gpus=0 --trset=SALOD

# python3 train.py picanet --gpus=0 --trset=SALOD

# python3 train.py basnet --gpus=0 --trset=SALOD

# python3 train.py cpd --gpus=0 --trset=SALOD

# python3 train.py poolnet --gpus=0 --trset=SALOD 

# python3 train.py egnet --gpus=0 --trset=SALOD

# python3 train.py scrn --gpus=0 --trset=SALOD

# python3 train.py f3net --gpus=0 --trset=SALOD

# python3 train.py gcpa --gpus=0 --trset=SALOD

# python3 train.py minet --gpus=0 --trset=SALOD

# python3 train.py ldf --gpus=0 --trset=SALOD

# python3 train.py gatenet --gpus=0 --trset=SALOD

# python3 train.py pfsnet --gpus=0 --trset=SALOD

# python3 train.py ctdnet --gpus=0 --trset=SALOD

# python3 train.py edn --gpus=0 --trset=SALOD

# python3 train.py page --gpus=0 --trset=SALOD

# python3 train.py menet  --gpus=0 --trset=SALOD
# python3 train.py isaanet  --gpus=0 --trset=SALOD
# python3 train.py meanet  --gpus=0 --trset=SALOD
# python3 train.py isnet  --gpus=0 --trset=SALOD



# ========================================================= EDMR =================================================
# ========================================================= EDMR =================================================
# ========================================================= EDMR =================================================

python3 train.py edmr --gpus=0 --trset=SALOD


# test the 3 interactive refinement
# python3 train.py basnet1_0 --gpus=0 --trset=SALOD

# # test the conv6 layer and the conv5 from resnet encoder
# python3 train.py basnet1_1 --gpus=0 --trset=SALOD

# # test the extractor from 123sa and 456ca
# python3 train.py basnet1_2 --gpus=0 --trset=SALOD

# python3 train.py basnet2 --gpus=0 --trset=SALOD

# python3 train.py basnet3 --gpus=0 --trset=SALOD

# python3 train.py basnet4 --gpus=0 --trset=SALOD

# python3 train.py basnet5 --gpus=0 --trset=SALOD

# python3 train.py basnet6 --gpus=0 --trset=SALOD

# tree shape experiments to analyze the module

# python3 train.py basnet7 --gpus=0 --trset=SALOD

# python3 train.py basnet7_2 --gpus=0 --trset=SALOD

# python3 train.py basnet7_3 --gpus=0 --trset=SALOD

# python3 train.py basnet7_4 --gpus=0 --trset=SALOD

# python3 train.py basnet7_5 --gpus=0 --trset=SALOD

# python3 train.py basnet7_6 --gpus=0 --trset=SALOD

# python3 train.py basnet8 --gpus=0 --trset=SALOD

# python3 train.py basnet9 --gpus=0 --trset=SALOD

# python3 train.py basnet9_1 --gpus=0 --trset=SALOD

# python3 train.py basnet9_2 --gpus=0 --trset=SALOD

# python3 train.py basnet10 --gpus=0 --trset=SALOD

# python3 train.py basnet11 --gpus=0 --trset=SALOD

# python3 train.py basnet11_2 --gpus=0 --trset=SALOD

# python3 train.py basnet11_3 --gpus=0 --trset=SALOD

# python3 train.py basnet11_4 --gpus=0 --trset=SALOD

# python3 train.py basnet12 --gpus=0 --trset=SALOD

# python3 train.py basnet13 --gpus=0 --trset=SALOD

# python3 train.py unet --gpus=0 --trset=SALOD

# python3 train.py unet_1 --gpus=0 --trset=SALOD

# python3 train.py unet_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_2_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_4_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_5_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_5_3 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_4 --gpus=0 --trset=SALOD

# python3 train.py basnet9_1_1 --gpus=0 --trset=SALOD

# python3 train.py basnet9_1_2_2 --gpus=0 --trset=SALOD

# python3 train.py ctdnet1_1 --gpus=0 --trset=SALOD

# python3 train.py egnet1_1 --gpus=0 --trset=SALOD



# ====================================================== ablation old ==========================================
# ====================================================== ablation old ==========================================
# ====================================================== ablation old ==========================================


# python3 train.py basnet1_1_3_3_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_ablation1 --gpus=0 --trset=SALOD  

# python3 train.py basnet1_1_3_3_2_deepablation1_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_ablation1_2 --gpus=0 --trset=SALOD  

# python3 train.py basnet1_1_3_3_2_ablation2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_ablation3 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_ablation4 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_ablation5 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_ablation6 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_ablation7 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_ablation7_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_deepablation1 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_deepablation2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_deepablation3 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_deepablation3_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_deepablation4 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_deepablation4_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif1 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif7 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_3 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_4 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_5 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_6 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_7 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_8 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_9 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_10 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_11 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_12 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_13 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_copy_14 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_deep3 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_deep4 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_deep5 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_deep6 --gpus=0 --trset=SALOD


# ======================================================= ablation new ===============================================
# ======================================================= ablation new ===============================================
# ======================================================= ablation new ===============================================

# python3 train.py edmr_ablation1 --gpus=0 --trset=SALOD
# python3 train.py edmr_ablation2 --gpus=0 --trset=SALOD
# python3 train.py edmr_ablation3 --gpus=0 --trset=SALOD
# python3 train.py edmr_ablation4 --gpus=0 --trset=SALOD
# python3 train.py edmr_ablation5 --gpus=0 --trset=SALOD
# python3 train.py edmr_ablation6 --gpus=0 --trset=SALOD
# python3 train.py edmr_ablation7 --gpus=0 --trset=SALOD
# python3 train.py edmr_ablation8 --gpus=0 --trset=SALOD

# python3 train.py edmr_ablation9 --gpus=0 --trset=SALOD
# python3 train.py edmr_ablation10 --gpus=0 --trset=SALOD


# python3 train.py edmr_ablation_miref3 --gpus=0 --trset=SALOD
# python3 train.py edmr_ablation_miref5 --gpus=0 --trset=SALOD
# python3 train.py edmr_ablation_miref7 --gpus=0 --trset=SALOD


# python3 train.py edmr_dem_depth1 --gpus=0 --trset=SALOD
# python3 train.py edmr_dem_depth2 --gpus=0 --trset=SALOD
# python3 train.py edmr_dem_depth3 --gpus=0 --trset=SALOD
# python3 train.py edmr_dem_depth4 --gpus=0 --trset=SALOD
# python3 train.py edmr_dem_depth5 --gpus=0 --trset=SALOD
# python3 train.py edmr_dem_depth6 --gpus=0 --trset=SALOD


# python3 train.py edmr_dem_e1024 --gpus=0 --trset=SALOD
# python3 train.py edmr_dem_e4096 --gpus=0 --trset=SALOD
# python3 train.py edmr_dem_e2048_nodem --gpus=0 --trset=SALOD
# python3 train.py edmr_dem_noe2048_nodem --gpus=0 --trset=SALOD


# python3 train.py edmr_miref1 --gpus=0 --trset=SALOD
# python3 train.py edmr_miref2 --gpus=0 --trset=SALOD
# python3 train.py edmr_miref3 --gpus=0 --trset=SALOD
# python3 train.py edmr_miref4 --gpus=0 --trset=SALOD
# python3 train.py edmr_miref5 --gpus=0 --trset=SALOD


# ======================================================= other trials ===============================================
# ======================================================= other trials ===============================================
# ======================================================= other trials ===============================================


# python3 train.py basnet9_1_1 --gpus=0 --trset=SALOD

# python3 train.py basnet9_1_2 --gpus=0 --trset=SALOD

# python3 train.py basnet9_1_2_2 --gpus=0 --trset=SALOD

# python3 train.py basnet9_1_2_2_2 --gpus=0 --trset=SALOD

# python3 train.py basnet9_1_2_2_3 --gpus=0 --trset=SALOD

# python3 train.py swinsod-sa --gpus=0 --trset=SALOD



# python3 train.py nonlocal1 --gpus=0 --trset=SALOD

# python3 train.py nonlocal2 --gpus=0 --trset=SALOD

# python3 train.py swinsod2 --gpus=0 --trset=SALOD

# python3 train.py swinsod3 --gpus=0 --trset=SALOD

# python3 train.py nonlocal1_2 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_2 --gpus=0 --trset=SALOD



# # nanjing  for new ideas



# python3 train.py swinsod2_3 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_4 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_5 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_6 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_7 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_8 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_9 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_10 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_deep5 --gpus=0 --trset=SALOD   # sigmoid

# # 1024 channel
# python3 train.py basnet1_1_3_3_2_deepablation1_2 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_h6 --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_h1 --gpus=0 --trset=SALOD



# python3 train.py nonlocal1_2 --gpus=0 --trset=SALOD

# # python3 train.py swinsod2_2 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_3_1 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_3_1_2 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_3_1_3 --gpus=0 --trset=SALOD

# python3 train.py swinsod2_3_2 --gpus=0 --trset=SALOD

# python3 train.py swinsod3 --gpus=0 --trset=SALOD

# python3 train.py swinsod3_1 --gpus=0 --trset=SALOD 




# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca2  --gpus=0 --trset=SALOD 

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca3  --gpus=0 --trset=SALOD 

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca4  --gpus=0 --trset=SALOD 

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca5  --gpus=0 --trset=SALOD 

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca6  --gpus=0 --trset=SALOD 

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca7  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca8  --gpus=0 --trset=SALOD 

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca9  --gpus=0 --trset=SALOD 

# python3 train.py swinsod2_3_1  --gpus=0 --trset=SALOD 

# python3 train.py swinsod2_3_1_3  --gpus=0 --trset=SALOD 

# python3 train.py swinsod3  --gpus=0 --trset=SALOD 





# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca10  --gpus=0 --trset=SALOD 

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca11  --gpus=0 --trset=SALOD 

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca12  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca13  --gpus=0 --trset=SALOD





# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca14  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca15  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca16  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca17  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca18  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca19  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca20  --gpus=0 --trset=SALOD



# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca16  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca17  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca18  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca19_2  --gpus=0 --trset=SALOD


# test the whole framework

# original
# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca19_2_test1_good  --gpus=0 --trset=SALOD

# attention in the former (has a good performance)
# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca19_2_test3  --gpus=0 --trset=SALOD

# python3 train.py basnet1_1_3_3_2_mif5_2048_asppca19_2_test3_2  --gpus=0 --trset=SALOD




# new framework from recent years

# python3 train.py menet  --gpus=0 --trset=SALOD
# python3 train.py isaanet  --gpus=0 --trset=SALOD
# python3 train.py meanet  --gpus=0 --trset=SALOD
# python3 train.py isnet  --gpus=0 --trset=SALOD

# python3 train.py admnet  --gpus=0 --trset=SALOD
# python3 train.py minet2024  --gpus=0 --trset=SALOD
#       

# python3 train.py fcn  --gpus=0 --trset=SALOD






# ============================================= TNN ========================================================
# ============================================= TNN ========================================================
# ============================================= TNN ========================================================


# python3 train.py swinsod2_3_1_3  --gpus=0 --trset=SALOD

# BCE loss / 是否需要带权重的bce loss ？
# python3 train.py swinsod3_2  --gpus=0 --trset=SALOD

# bce + iou + struct loss
# python3 train.py swinsod3_3  --gpus=0 --trset=SALOD


# python3 train.py swinsod3_2_2_SECross5  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross6  --gpus=0 --trset=SALOD

# mamba test 0.89 precise without the decoder1
# python3 train.py swinsod3_3_mamba  --gpus=0 --trset=SALOD  

# reduction to embrace decoder1
# python3 train.py swinsod3_3_mamba_reduction  --gpus=0 --trset=SALOD


# sefusion
# python3 train.py swinsod3_2_2  --gpus=0 --trset=SALOD

# multi layer fusion
# python3 train.py swinsod3_2_2_SECross  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross2  --gpus=0 --trset=SALOD

# python3 train.py swinsod3_2_2_SECross4  --gpus=0 --trset=SALOD


## 整体实验

# python3 train.py swinsod3_2_2_SECross4_2  --gpus=0 --trset=SALOD

# python3 train.py swinsod3_2_2_SECross4_2_fusion1  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross4_2_fusion2  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross4_2_fusion3  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross4_2_m1  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross4_2_m2  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross4_2_m3  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross4_2_m4  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross4_2_m5  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross4_2_loss1  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross4_2_loss2  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross4_2_loss3  --gpus=0 --trset=SALOD

# python3 train.py swinsod3_2_2_SECross4_2_loss5  --gpus=0 --trset=SALOD
# python3 train.py swinsod3_2_2_SECross4_2_m6  --gpus=0 --trset=SALOD

