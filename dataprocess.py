import os
import cv2
import numpy as np

def underwater_correction(img):
    """
    水下图像增强算法,包含伽马矫正和白平衡
    Args:
        img: 输入的BGR格式图像
    Returns:
        enhanced_img: 增强后的图像
    """
    # 伽马矫正
    def gamma_correction(img, gamma=1.5):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    # 自动白平衡
    def white_balance(img):
        # 分离BGR通道
        b, g, r = cv2.split(img)
        
        # 计算每个通道的均值
        r_avg = cv2.mean(r)[0]
        g_avg = cv2.mean(g)[0]
        b_avg = cv2.mean(b)[0]
        
        # 计算增益系数
        k = (r_avg + g_avg + b_avg) / 3
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg
        
        # 应用增益调整
        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        
        # 合并通道
        balanced_img = cv2.merge([b, g, r])
        return balanced_img
    
    # 应用伽马矫正
    gamma_img = gamma_correction(img)
    
    # 应用白平衡
    # enhanced_img = white_balance(gamma_img)
    
    return gamma_img

