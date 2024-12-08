import torch
import numpy as np
import matplotlib.pyplot as plt
#####################################################################################################################################################

def colormap(label):
  
    Color_bar=np.array([[000,000,000],[255,255,255],[255, 255, 000],[000, 000, 255]])

    H, W = label.shape
    Y_color = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            item=label[i,j]
            Y_color[i,j,:] = Color_bar[item,:]/255

    return Y_color


def result_pic(label, save_path):

    dpi = 10
    fig = plt.figure(frameon=False)
    fig.set_size_inches(label.shape[1]*2.0/dpi, label.shape[0]*2.0/dpi)

    ax  = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(label) #展示图片          
    fig.savefig(save_path, dpi=dpi)

    return 1


def generate_png(GT, Posotion, Y_pred, DatasetName):

    Y_label = np.zeros_like(GT)

    for i in range(Posotion.shape[0]):
        x, y = Posotion[i, :]
        
        # 确保 x 和 y 在 GT 的有效范围内
        if 0 <= x < GT.shape[0] and 0 <= y < GT.shape[1]:
            if GT[x, y] == Y_pred[i]:
                Y_label[x, y] = Y_pred[i]
            else:
                if GT[x, y] == 1 and Y_pred[i] == 0:  
                    Y_label[x, y] = 2  # FN (False Negatives)
                elif GT[x, y] == 0 and Y_pred[i] == 1: 
                    Y_label[x, y] = 3  # FP (False Positives)

    Y_gt_color   = colormap(GT)
    Y_pred_color = colormap(Y_label)

    path='result/'+ DatasetName
    print("GT")
    result_pic(Y_gt_color,  path+'_GT.png')
    
    print("Predict")
    result_pic(Y_pred_color,path+'_Predict.png')

    return 1