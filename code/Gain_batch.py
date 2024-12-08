import torch
import random
import numpy as np
import torch.utils.data as Data

def seed_worker(seed):

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sample_position(GT, train_rate, seed):

    Position_0=np.argwhere(GT==0)
    Position_0=np.array(Position_0)
    Position_1=np.argwhere(GT==1)
    Position_1=np.array(Position_1)

    seed_worker(seed)
    randomint_0=random.sample(range(0,Position_0.shape[0]),Position_0.shape[0])
    randomint_0=np.array(randomint_0)
    randomint_1=random.sample(range(0,Position_1.shape[0]),Position_1.shape[0])
    randomint_1=np.array(randomint_1)

    num_train_0=round(Position_0.shape[0]*train_rate)
    num_train_1=round(Position_1.shape[0]*train_rate)

    if num_train_0<1:
        num_train_0=1
    if num_train_1<1:
        num_train_1=1 

    train_selected_0=randomint_0[:num_train_0]
    train_position_selected_0=Position_0[train_selected_0,:]
    train_selected_1=randomint_1[:num_train_1]
    train_position_selected_1=Position_1[train_selected_1,:]
    train_position = np.concatenate((train_position_selected_0, train_position_selected_1))

    test_selected_0=randomint_0[num_train_0:]
    test_position_selected_0=Position_0[test_selected_0,:]
    test_selected_1=randomint_1[num_train_1:]
    test_position_selected_1=Position_1[test_selected_1,:]
    test_position = np.concatenate((test_position_selected_0, test_position_selected_1))

    return train_position, test_position


def gain_patch(image, x, y, patch_sizes):

    temp_image = image[x:(x+patch_sizes), y:(y+patch_sizes),:]

    return temp_image



def gain_train_test_batch(Time1, Time2, GT, train_rate, patch_sizes, batch_sizes, seed):

    H, W, B = Time1.shape
    r=patch_sizes//2
    Time1_mirror=np.pad(Time1,((r,r),(r,r),(0,0)),mode='reflect')
    Time2_mirror=np.pad(Time2,((r,r),(r,r),(0,0)),mode='reflect')
    train_position, test_position=sample_position(GT, train_rate, seed)
    print('Train samples: ', train_position.shape[0])
    print('Test  samples: ', test_position.shape[0])

    #***********************Train samples**************************
    train_T1=np.zeros((train_position.shape[0], patch_sizes, patch_sizes, B))
    train_T2=np.zeros((train_position.shape[0], patch_sizes, patch_sizes, B))
    train_gt=np.zeros((train_position.shape[0], ))
    for i in range(train_position.shape[0]):
        temp_x, temp_y = train_position[i,0],train_position[i,1]
        train_T1[i,:,:,:]=gain_patch(Time1_mirror, temp_x, temp_y, patch_sizes=patch_sizes)
        train_T2[i,:,:,:]=gain_patch(Time2_mirror, temp_x, temp_y, patch_sizes=patch_sizes)
        train_gt[i,]=GT[temp_x,temp_y]

    T1_train=torch.from_numpy(train_T1.transpose(0,3,1,2)).type(torch.FloatTensor)
    T2_train=torch.from_numpy(train_T2.transpose(0,3,1,2)).type(torch.FloatTensor)
    Y_trian=torch.from_numpy((train_gt)).type(torch.LongTensor)

    batch_train=Data.TensorDataset(T1_train,T2_train, Y_trian)
    Train_loader=Data.DataLoader(dataset=batch_train,
                                 pin_memory=True,
                                 worker_init_fn=seed_worker(seed),
                                 batch_size=batch_sizes,
                                 shuffle=True)
    
    #***********************Test samples**************************
    test_T1=np.zeros((test_position.shape[0], patch_sizes, patch_sizes, B))
    test_T2=np.zeros((test_position.shape[0], patch_sizes, patch_sizes, B))
    test_gt=np.zeros((test_position.shape[0], ))
    for i in range(test_position.shape[0]):
        temp_x, temp_y = test_position[i,0],test_position[i,1]
        test_T1[i,:,:,:]=gain_patch(Time1_mirror, temp_x, temp_y, patch_sizes=patch_sizes)
        test_T2[i,:,:,:]=gain_patch(Time2_mirror, temp_x, temp_y, patch_sizes=patch_sizes)
        test_gt[i,]=GT[temp_x,temp_y]

    T1_test=torch.from_numpy(test_T1.transpose(0,3,1,2)).type(torch.FloatTensor)
    T2_test=torch.from_numpy(test_T2.transpose(0,3,1,2)).type(torch.FloatTensor)
    Y_test=torch.from_numpy((test_gt)).type(torch.LongTensor)
    
    batch_test=Data.TensorDataset(T1_test,T2_test,Y_test)
    Test_loader=Data.DataLoader(dataset=batch_test,
                                pin_memory=True,
                                worker_init_fn=seed_worker(seed),
                                batch_size=batch_sizes,
                                shuffle=True)

    return Train_loader, Test_loader


def gain_total_batch(Time1, Time2, GT, patch_sizes, batch_sizes):

    _, _, B = Time1.shape
    r=patch_sizes//2
    Time1_mirror=np.pad(Time1,((r,r),(r,r),(0,0)),mode='reflect')
    Time2_mirror=np.pad(Time2,((r,r),(r,r),(0,0)),mode='reflect')

    Position_0=np.array(np.argwhere(GT==0))
    Position_1=np.array(np.argwhere(GT==1))
    Total_position = np.concatenate((Position_0, Position_1))
    print('Total samples: ', Total_position.shape[0])

    #***********************Total samples**************************
    total_T1=np.zeros((Total_position.shape[0], patch_sizes, patch_sizes, B))
    total_T2=np.zeros((Total_position.shape[0], patch_sizes, patch_sizes, B))
    total_gt=np.zeros((Total_position.shape[0], ))
    for i in range(Total_position.shape[0]):
        temp_x, temp_y = Total_position[i,0],Total_position[i,1]
        total_T1[i,:,:,:]=gain_patch(Time1_mirror, temp_x, temp_y, patch_sizes=patch_sizes)
        total_T2[i,:,:,:]=gain_patch(Time2_mirror, temp_x, temp_y, patch_sizes=patch_sizes)
        total_gt[i,]=GT[temp_x, temp_y]

    Z_total =torch.from_numpy((Total_position)).type(torch.LongTensor)
    T1_total=torch.from_numpy(total_T1.transpose(0,3,1,2)).type(torch.FloatTensor)
    T2_total=torch.from_numpy(total_T2.transpose(0,3,1,2)).type(torch.FloatTensor)
    Y_total =torch.from_numpy((total_gt)).type(torch.LongTensor)
    
    batch_total=Data.TensorDataset(Z_total, T1_total, T2_total, Y_total)
    Total_loader=Data.DataLoader(dataset=batch_total,
                                 pin_memory=True,
                                 batch_size=batch_sizes,
                                 shuffle=True)
   
    return Total_loader