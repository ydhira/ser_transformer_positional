
# coding: utf-8

# In[63]:
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data.dataloader as dataloader
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from new_utils import Dataset, cuda, self_attn_model, emotions_used, phoneme_class, Net, IEMOCAP, load_raw, collate_lines, get_metrics
import os, sys
from tqdm import tqdm
from scipy.io import savemat, loadmat
import argparse
import numpy as np
from transformer_class import TransformerClassAudio, TransformerClassAudioText
import string
from sklearn.manifold import MDS
# In[2]:
from scipy import spatial
import plda


# MODEL_NAME = '/home/hyd/multi-hop-attention/code/models/0_1464dim.model' # 64 dim model trained on 4 classes 
# ['ang', 'hap','sad', 'neu']
#MODEL_NAME = '/home/hyd/multi-hop-attention/code/models/0_148classes64dim.model' # 64 dim model trained on 9 classes 
# ['ang', 'hap', 'fru', 'sur', 'fea', 'exc', 'dis', 'sad', 'neu']


def passmodel(model, batch_size, loader):
    ## desceibe the model 
    
    features = []
    labels = []
    c = 0
    for x1, target, x2, vad_target, X_bertpad, X_bert_len, phone_target, input_lens, \
            phone_target_sequence, X_bert_extendpad, X_bert_extend_len, phone_target_len , phone_target_sequence_len in tqdm(loader):
        
        input_lens = torch.Tensor(input_lens).cuda()
        X_bert_len = torch.Tensor(X_bert_len).cuda()
        X_bert_extend_len = torch.Tensor(X_bert_extend_len).cuda()
        phone_target_len = torch.Tensor(phone_target_len).cuda()
        phone_target_sequence_len = torch.Tensor(phone_target_sequence_len).cuda()

        with torch.no_grad():
            if modalities == "a_phfeatspace_aligned":
                feature, logits = model(src=x1, txt=phone_target, trans=X_bert_extendpad , wav_len=input_lens, bert_len=X_bert_extend_len)
                features.append(feature)
            labels.extend(target)

    features = torch.cat(features, dim=0).squeeze().cpu().numpy() # (B, Feat_dim)
    return features, labels


# In[508]:
def mds_transform(features):
    print(features.shape)
    ## do mds on these features 
    mds = MDS(n_components=3)
    X_transformed = mds.fit_transform(features)
    return mds, X_transformed

def mds_transform_eval( train_features, train_features_transformed, test_feature):
    # print(train_features.shape, train_features_transformed.shape, test_feature.shape)
    dist_64 = np.sum(((test_feature - train_features) ** 2), axis=1)
    # print(dist_64)
    dim3_test_feat =  np.random.rand(3) #[0,1,2]# 
    N = train_features.shape[0]
    learning_rate = 0.0001
    prev_loss, loss_diff = 1, 2
    i = 0
    while (loss_diff > 0.001):
        dist_3_vec = (dim3_test_feat - train_features_transformed )
        dist_3 = np.sum((( dim3_test_feat  - train_features_transformed ) ** 2), axis=1)
        loss = np.sum((dist_64 - dist_3) ** 2) / N
        # print(iter , " Loss: ", loss)
        
        dim3_test_feat[0] -= learning_rate * (4/N * np.sum((dim3_test_feat[0]**2 - 2 * dim3_test_feat[0] * train_features_transformed[:,0]) * (dim3_test_feat[0] - train_features_transformed[:,0])))
        dim3_test_feat[1] -= learning_rate * (4/N * np.sum((dim3_test_feat[1]**2 - 2 * dim3_test_feat[1] * train_features_transformed[:,1]) * (dim3_test_feat[1] - train_features_transformed[:,1])))
        dim3_test_feat[2] -= learning_rate * (4/N * np.sum((dim3_test_feat[2]**2 - 2 * dim3_test_feat[2] * train_features_transformed[:,2]) * (dim3_test_feat[2] - train_features_transformed[:,2])))
        
        loss_diff = np.abs(prev_loss - loss)
        prev_loss = loss 
        i+=1
    # print(i, loss_diff)
    return dim3_test_feat



def lower_dim_emo(mds_file):
    mds = loadmat(mds_file)
    X_transformed = mds['X_transformed']
    labels = mds['labels'].reshape(-1)
    features = mds['X']
    avg_dims = {}
    for i in range(labels.shape[0]):
        x = X_transformed[i].reshape(1,-1)
        l = labels[i]
        # print(l)
        if l in avg_dims: 
            avg_dims[l]=np.concatenate((avg_dims[l], x), axis=0)
        else: avg_dims[l] = x

    for l, xvec in avg_dims.items():
        avg_dims[l] = np.mean(np.array(xvec), 0)

    print(avg_dims)

    print("Done")
    
    return avg_dims 

def scoring(mds_file, avg_dims, test_features, test_labels):
        # X_transform_test = mds_transform_eval(features, X_transformed, test_features)
    # print(X_transform_test)
    # print(X_transform_test.shape)
    mds = loadmat(mds_file)
    train_features_transformed = mds['X_transformed']
    labels = mds['labels'].reshape(-1)
    train_features = mds['X']
    num_class = 4
    pred = []
    for test_feature in test_features:
        feat3 = mds_transform_eval(train_features, train_features_transformed, test_feature)
        sims = []
        for i in range(num_class):
            vec = avg_dims[i]
            result = 1 - spatial.distance.cosine(feat3, vec)
            sims.append(result)
        p = np.argmin(sims)
        pred.append(p)
    pred = np.array(pred)
    metrics = get_metrics(pred, test_labels)
    print(metrics)
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser( prog='.py', description=' ')
    parser.add_argument('--model_name', default='models/3_49transformer_bs6_a_phfeatspace_aligned.model')
    parser.add_argument('--batchsize', metavar='N', type=int, default=8)
    parser.add_argument('--modalities', default='a_ph_inputspace', choices=['a_ph_inputspace', 'a_inputspace',\
                                                             'a_phfeatspace_aligned', 'a_phfeatspace_aligned_2', 'a_phfeat_unaligned',\
                                                             'a_phfeat_aligned_unaligned', 'a_phfeat_aligned_unaligned-2' ])
    parser.add_argument('--test_speaker', type=int, default=0)
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--mds_file', default=None)
    parser.add_argument('--transform', default=None)

    args = parser.parse_args()
    # lower_dim_emo()
    model_name = args.model_name
    batch_size = args.batchsize
    modalities = args.modalities
    test_speaker = args.test_speaker
    mode = args.mode
    mds_file =args.mds_file
    transform = args.transform

    idx = np.array([0,1,2,3,4,5,6,7,8,9])
    test_idx = [test_speaker]
    dev_idx = [test_speaker]
    train_idx = np.setdiff1d(idx, test_idx + dev_idx)

    loader = IEMOCAP(train_idx, test_idx, dev_idx)
    trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend = loader.train()
    devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend = loader.dev()

    devTY = list(map(lambda sentence: (' ').join(sentence).replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , devTY))
    trainTY = list(map(lambda sentence: (' ').join(sentence).replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , trainTY))

    train = Dataset(trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend)
    dev = Dataset(devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend)

    dataloader_args = dict(shuffle=False, batch_size=batch_size, collate_fn = collate_lines, num_workers=0)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    dev_loader = dataloader.DataLoader(dev, **dataloader_args)

    model = TransformerClassAudioText(
            #tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
            720, 1024, 64, 64, 8, 1, 1, 1024, 4, modalities=modalities, activation=torch.nn.GELU, encoder_module='conformer', normalize_before=True
        )

    device = torch.device("cuda:0" if cuda else "cpu")
    model = model.to(device).cuda() if cuda else model.to(device)
    model = nn.DataParallel(model, device_ids=[0,1])

    checkpoint = torch.load(model_name)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if transform == "mds":

        if mode == 'train':
            
            features, labels = passmodel(model, batch_size, train_loader)
            labels = np.array(labels).reshape(-1)
            
            mds,X_transformed_mds = mds_transform(features)
            print(X_transformed_mds.shape)
            tosave = 'mds/'+modalities+"-spk-"+str(test_speaker)
            savemat(tosave, \
                 {'mds': mds, 
                 'X_transformed': X_transformed_mds, \
                   'labels':labels, \
                   'X': features })


        if mode == 'eval':
            test_features, test_labels = passmodel(model, batch_size, dev_loader)
            # labels = np.array(labels).reshape(-1)
            avg_dims = lower_dim_emo(mds_file)
            scoring(mds_file, avg_dims, test_features, test_labels)
    

    if transform == 'plda':


        features, labels = passmodel(model, batch_size, train_loader)
        labels = np.array(labels).reshape(-1)
        test_features, test_labels = passmodel(model, batch_size, dev_loader)
        test_labels = np.array(test_labels).reshape(-1)

        for ncomp in [5, 10, 15,20,25]:
            print(ncomp)
            classifier = plda.Classifier()
            classifier.fit_model(features, labels, n_principal_components=ncomp)

            pred, log_p_predictions = classifier.predict(test_features)
            metrics = get_metrics(pred, test_labels, mode='test')
            print(metrics)
