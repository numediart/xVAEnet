# Luca La Fisca
# ------------------------------
# Copyright UMONS (C) 2022

import zarr
from fastai.tabular.all import *
from fastai.data.all import *
from fastai.vision.gan import *
from fastai import *
from tsai.all import *
from torch import nn
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from models.autoencoders import stagerNetVAE, stagerNetCritic
from utils import ChangeTargetData, GetLatentSpace, TrainClassif, CheckNorm


retrain_vae = False
retrain_gan = True
retrain_classif1 = True
retrain_classif2 = True
retrain_classif3 = True
load_dls = True
load_dls_gan = False
load_dls_classif1 = False
load_dls_classif2 = False
load_dls_classif3 = False

print("retrain vae is: "+str(retrain_vae))

dev = torch.device('cuda:0')
torch.cuda.set_device(dev)

path = Path('/home/JennebauffeC/pytorchVAE/fastAI/data')

class TensorUnsqueeze(Transform):
    order=1
    "Unsqueeze tensor on `dim`"
    def __init__(self, dim=1):
        self.dim = dim

    def encodes(self, t:TSTensor): return t.unsqueeze(self.dim)

class norm_batch(Transform):
    def __init__(self, eps=1e-08, glob=False) :
        print('NEW norm_batch initiated!')
        self.eps = eps
    def encodes(self, t:TSTensor):
        try:
            if glob:
                l = t[:,:,-1]
                t = t[:,:,:-1]
            mean = torch.nanmean(t, dim=2)
            std = torch.clamp_min(torch.std(t, dim=2), self.eps)
            out = torch.stack([torch.vstack([(t[j,i,:]-mean[j,i])/torch.clamp_min(std[j,i], self.eps)
                                 for i in range(t.shape[1])]) 
                                 for j in range(t.shape[0])],dim=0)
            out = torch.clamp_min(out,-3)
            out = torch.clamp_max(out,3)
            if glob:
                out = torch.dstack((out,l))
        except:
            out = t
        return out.to(device)

y_large_zarr = zarr.open(path/'y_large.zarr', mode='r')
n_train_samples = round(len(y_large_zarr)*.8)
n_total_samples = len(y_large_zarr)
splits = (L(range(n_train_samples), use_list=True),
          L(np.arange(n_train_samples, n_total_samples), use_list=True))
splitter = IndexSplitter(splits[1])
batch_tfms = TSStandardize(by_sample=False,by_var=True, use_single_batch=True)

if load_dls:
    dls = torch.load('dls.pkl')

else:
    #Normalize input data between -1 and 1
    X_large_zarr = zarr.open(path/'X_large.zarr', mode='r')
    t = torch.Tensor(X_large_zarr)

    eps = 1e-08
    t_max, t_min = t.max(dim=2).values, t.min(dim=2).values
    x = torch.stack([torch.vstack([(t[j,i,:]-t_min[j,i])/torch.clamp_min((t_max[j,i]-t_min[j,i]), eps)
                         for i in range(t.shape[1])]) 
                         for j in range(t.shape[0])],dim=0)*2-1

    print('x shape: ',str(x.shape))

    ### train VAE only ###
    getters = [ItemGetter(0), ItemGetter(0)]
    tfms = [TensorUnsqueeze(0),TensorUnsqueeze(0)]
    dblock = DataBlock(blocks=(TSTensorBlock,TSTensorBlock),
                       getters=getters,
                       splitter=splitter,
                       batch_tfms=batch_tfms)
    src = itemify(x,x)
    dls = dblock.dataloaders(src, bs=16, val_bs=32)

    torch.save(dls, 'dls.pkl')

    del X_large_zarr, y_large_zarr
    time.sleep(.2)
    torch.cuda.empty_cache()
    print('memory flushed')

# vae_filename = 'vae_22_10_26_2'
vae_filename = 'vae_22_12_02_2'
autoencoder = stagerNetVAE(0)
autoencoder = autoencoder.to(device)
learn = Learner(dls, autoencoder, loss_func = autoencoder.loss_func, metrics=rmse, opt_func=ranger)
if retrain_vae:
    learning_rate = learn.lr_find()
    print('learning rate: '+str(learning_rate.valley))
    learn.fit_flat_cos(n_epoch=500, lr=learning_rate.valley,
                        cbs=[CheckNorm(norm_y=True),
                            GradientAccumulation(n_acc=64),
                            TrackerCallback(),
                            SaveModelCallback(fname=vae_filename),
                            EarlyStoppingCallback(min_delta=1e-4,patience=10)])

    learn.save(str(vae_filename)+'_learn')
    np.save('results/'+str(vae_filename)+'_values.npy', learn.recorder.values)
    np.save('results/'+str(vae_filename)+'_losses.npy', learn.recorder.losses)

learn.load(vae_filename)
print('vae loaded')


device = torch.device('cpu') #GAN Learner requires CPU for multitask


getters = [ItemGetter(0), ItemGetter(1)]

# define x for dls_gan
x = torch.stack(list(zip(*dls.train_ds))[0],dim=0)
x = torch.vstack((x,torch.stack(list(zip(*dls.valid_ds))[0],dim=0)))
print("global max/min:")
print(x.max(), x.min())

if load_dls_gan:
    dls_gan = torch.load('dls_gan.pkl')
    print("dls_gan loaded")
else:
    # define y for dls_gan
    learn.get_preds(ds_idx=0, cbs=GetLatentSpace(cycle_len=1))
    y = learn.zs

    learn.get_preds(ds_idx=1, cbs=GetLatentSpace(cycle_len=1))
    y = torch.vstack((y,learn.zs))

    print('x and y shapes after get_preds')
    print(x.shape)
    print(y.shape)


    ### Train GAN ###
    dblock = DataBlock(blocks=(TSTensorBlock,TSTensorBlock),
                       getters=getters,
                       splitter=splitter,
                       batch_tfms=norm_batch)

    src = itemify(x.to(device),y.to(device))
    dls_gan = dblock.dataloaders(src, bs=16, val_bs=32)

    del y
    time.sleep(.2)
    torch.cuda.empty_cache()
    print('memory flushed 2')

    torch.save(dls_gan, 'dls_gan.pkl')


# gan_filename = 'gan_22_10_26_3'
gan_filename = 'gan_22_12_02_2'

generator = stagerNetVAE(1)
generator.conv1.load_state_dict(learn.model.conv1.state_dict())
generator.conv2.load_state_dict(learn.model.conv2.state_dict())
generator.conv3.load_state_dict(learn.model.conv3.state_dict())
generator.fc_z.load_state_dict(learn.model.fc_z.state_dict())
generator.fc_mu.load_state_dict(learn.model.fc_mu.state_dict())
generator.fc_var.load_state_dict(learn.model.fc_var.state_dict())
generator.decoder_input.load_state_dict(learn.model.decoder_input.state_dict())
generator.deconv1.load_state_dict(learn.model.deconv1.state_dict())
generator.deconv2.load_state_dict(learn.model.deconv2.state_dict())
generator.deconv3.load_state_dict(learn.model.deconv3.state_dict())

generator = generator.to('cpu')
critic = stagerNetCritic().to('cpu')

def _tk_diff(real_pred, fake_pred):
    return real_pred.mean() - fake_pred.mean()
clip = 0.01
switch_eval = False
learnGan = GANLearner(dls_gan, generator, critic,  generator.gen_loss_func, _tk_diff, clip=clip,
                      switch_eval=switch_eval, opt_func = RMSProp)


if retrain_gan:
    cycle_len = 15
    print('start learnGan fitting with cycle_len = '+str(cycle_len))
    learnGan.fit(500, 1e-3,
                cbs=[CheckNorm(),
                    GradientAccumulation(n_acc=64),
                    TrackerCallback(),
                    SaveModelCallback(fname=gan_filename),
                    EarlyStoppingCallback(min_delta=1e-4,patience=30),
                    GetLatentSpace(cycle_len=cycle_len),
                    ChangeTargetData(cycle_len=cycle_len,splitter=splitter,getters=getters)])
    # print(learnGan.dls.train_ds[42])

    torch.save(learnGan.model.generator.state_dict(),gan_filename+'_gen.pth')
    torch.save(learnGan.model.critic.state_dict(),gan_filename+'_crit.pth')
    learnGan.save(gan_filename+'_learn')

learnGan.load(gan_filename)
print("learn_gan loaded")

device = dev

# labels = np.load("../area_db.npy")
lab_area = np.load("../area_db.npy")
lab_reveil = np.load("../reveil_db.npy")
lab_duration = np.load("../duration_db.npy")
lab_all = np.array(4*lab_area + 2*lab_reveil + lab_duration)
print('all the labels are:')
print(lab_all.shape)
print(lab_all[:42])

tmp = copy(lab_all)
lab_all[tmp==3] = 4
lab_all[tmp==4] = 3

lab3 = deepcopy(lab_all)
lab3[:] = 0
lab3[lab_all>1] = 1
lab3[lab_all>5] = 2
print('lab3: ')
print(np.unique(lab3))

lab4 = deepcopy(lab_all)
lab4[lab_all>0] = 1
lab4[lab_all>3] = 2
lab4[lab_all==7] = 3
print('lab4: ')
print(np.unique(lab4))


if load_dls_classif1:
    dls_classif = torch.load('dls_classif_1.pkl')
    print("dls_classif1 loaded")
else:
    getters = [ItemGetter(0), ItemGetter(1)]
    dblock = DataBlock(blocks=(TSTensorBlock,CategoryBlock),
                       getters=getters,
                       splitter=splitter,
                       batch_tfms=norm_batch)

    lab = torch.Tensor(lab_area).unsqueeze(-1)
    lab_stack = torch.hstack((lab,torch.zeros(len(lab),x.shape[1]-1)))
    x_classif = torch.dstack((x,lab_stack))

    src = itemify(x_classif.to('cpu'),lab_area)
    print("x in classif: ")
    print(x_classif.max(),x_classif.min(), x_classif.shape)
    dls_classif = dblock.dataloaders(src, bs=16, val_bs=32)

    print("one_batch classif:")
    print(torch.tensor(dls_classif.one_batch()[0][0,:8,:8]))
    torch.save(dls_classif, 'dls_classif_1.pkl')

print(torch.stack(list(zip(*dls_classif.train_ds[:8]))[1],dim=0))

#train classifier
print("start classifier training")
print("check saved code 25")
classif = stagerNetVAE(typ=3, nclass=1, dropout_rate=0)
classif.conv1.load_state_dict(learnGan.generator.conv1.state_dict())
classif.conv2.load_state_dict(learnGan.generator.conv2.state_dict())
classif.conv3.load_state_dict(learnGan.generator.conv3.state_dict())
classif.fc_z.load_state_dict(learnGan.generator.fc_z.state_dict())
classif.fc_mu.load_state_dict(learnGan.generator.fc_mu.state_dict())
classif.fc_var.load_state_dict(learnGan.generator.fc_var.state_dict())
classif.decoder_input.load_state_dict(learnGan.generator.decoder_input.state_dict())
classif.deconv1.load_state_dict(learnGan.generator.deconv1.state_dict())
classif.deconv2.load_state_dict(learnGan.generator.deconv2.state_dict())
classif.deconv3.load_state_dict(learnGan.generator.deconv3.state_dict())
classif.fc_crit.load_state_dict(learnGan.critic.fc_crit.state_dict())
classif.fc_crit2.load_state_dict(learnGan.critic.fc_crit2.state_dict())
classif.fc_crit3.load_state_dict(learnGan.critic.fc_crit3.state_dict())

classif = classif.to(dev)

# dls_classif = dls_classif.dataset.to(dev)
print('check cuda')
print(next(classif.conv1.parameters()).is_cuda)
print(dls_classif.dataset[0][1].is_cuda)
print(dls_classif.dataset[0][0].is_cuda)

learnClassif = Learner(dls_classif, classif, loss_func=classif.classif_loss_func,
                       metrics=accuracy, opt_func=ranger)
print('check cuda 2')
print(next(learnClassif.model.conv1.parameters()).get_device())
print(learnClassif.dls.dataset[0][1].get_device())
print(learnClassif.dls.dataset[0][0].get_device())

classif_filename = 'classif1_22_12_02_2'

if retrain_classif1:
    learning_rate = learnClassif.lr_find()
    print('learning rate: '+str(learning_rate.valley))
    learnClassif.fit_flat_cos(500, lr=learning_rate.valley,
                            cbs=[CheckNorm(),
                                GradientAccumulation(n_acc=64),
                                TrackerCallback(),
                                SaveModelCallback(fname=classif_filename),
                                EarlyStoppingCallback(min_delta=1e-4,patience=30),
                                TrainClassif()])

    learnClassif.save(classif_filename+'_learn')

    np.save('results/'+str(classif_filename)+'_losses.npy', learnClassif.recorder.losses)
    np.save('results/'+str(classif_filename)+'_values.npy', learnClassif.recorder.values)


# classifier with 2 classes
classif2_filename = 'classif2_22_12_02_2'

if load_dls_classif2:
    dls_classif = torch.load('dls_classif_2.pkl')
    print("dls_classif2 loaded")
else:
    getters = [ItemGetter(0), ItemGetter(1)]
    dblock = DataBlock(blocks=(TSTensorBlock,CategoryBlock),
                       getters=getters,
                       splitter=splitter,
                       batch_tfms=norm_batch)

    lab = torch.vstack((torch.Tensor(lab_area), torch.Tensor(lab_reveil))).T
    lab_stack = torch.hstack((lab,torch.zeros(len(lab),x.shape[1]-2)))
    x_classif = torch.dstack((x,lab_stack))

    src = itemify(x_classif.to('cpu'),lab3)
    print("x in classif: ")
    print(x_classif.max(),x_classif.min(), x_classif.shape)
    dls_classif = dblock.dataloaders(src, bs=16, val_bs=32)

    print("one_batch classif:")
    print(torch.tensor(dls_classif.one_batch()[0][0,:8,:8]))
    torch.save(dls_classif, 'dls_classif_2.pkl')

classif = stagerNetVAE(typ=3, nclass=2, dropout_rate=0)
classif.load_state_dict(learnClassif.model.state_dict())
learnClassif = Learner(dls_classif, classif, loss_func=classif.classif_loss_func,
                       metrics=accuracy, opt_func=ranger)

learnClassif.load(classif_filename)
print('learnClassif1 loaded')

if retrain_classif2:
    learning_rate = learnClassif.lr_find()
    print('learning rate: '+str(learning_rate.valley))
    learnClassif.fit_flat_cos(500, lr=learning_rate.valley,
                            cbs=[CheckNorm(),
                                GradientAccumulation(n_acc=64),
                                TrackerCallback(),
                                SaveModelCallback(fname=classif2_filename),
                                EarlyStoppingCallback(min_delta=1e-4,patience=30),
                                TrainClassif()])

    learnClassif.save(classif2_filename+'_learn')

    np.save('results/'+str(classif2_filename)+'_losses.npy', learnClassif.recorder.losses)
    np.save('results/'+str(classif2_filename)+'_values.npy', learnClassif.recorder.values)


# classifier with 3 classes
classif3_filename = 'classif3_22_12_02_2'

if load_dls_classif3:
    dls_classif = torch.load('dls_classif_3.pkl')
    print("dls_classif3 loaded")
else:
    getters = [ItemGetter(0), ItemGetter(1)]
    dblock = DataBlock(blocks=(TSTensorBlock,CategoryBlock),
                       getters=getters,
                       splitter=splitter,
                       batch_tfms=norm_batch)

    lab = torch.vstack((torch.Tensor(lab_area), torch.Tensor(lab_reveil), torch.Tensor(lab_duration))).T
    lab_stack = torch.hstack((lab,torch.zeros(len(lab),x.shape[1]-3)))
    x_classif = torch.dstack((x,lab_stack))

    src = itemify(x_classif.to('cpu'),lab4)
    dls_classif = dblock.dataloaders(src, bs=16, val_bs=32)

    print("one_batch classif:")
    print(torch.tensor(dls_classif.one_batch()[0][0,:8,:8]))
    torch.save(dls_classif, 'dls_classif_3.pkl')

classif = stagerNetVAE(typ=3, nclass=3, dropout_rate=0)
learnClassif = Learner(dls_classif, classif, loss_func=classif.classif_loss_func,
                       metrics=accuracy, opt_func=ranger)

learnClassif.load(classif2_filename)
print('learnClassif2 loaded')

if retrain_classif3:
    learning_rate = learnClassif.lr_find()
    print('learning rate: '+str(learning_rate.valley))
    learnClassif.fit_flat_cos(500, lr=learning_rate.valley,
                            cbs=[CheckNorm(),
                                GradientAccumulation(n_acc=64),
                                TrackerCallback(),
                                SaveModelCallback(fname=classif3_filename),
                                EarlyStoppingCallback(min_delta=1e-4,patience=30),
                                TrainClassif()])

    learnClassif.save(classif3_filename+'_learn')

    np.save('results/'+str(classif3_filename)+'_losses.npy', learnClassif.recorder.losses)
    np.save('results/'+str(classif3_filename)+'_values.npy', learnClassif.recorder.values)


# compute and display the latent space
learnClassif.get_preds(ds_idx=0,cbs=[GetLatentSpace(cycle_len=1)])
new_zi = learnClassif.zi_valid
learnClassif.get_preds(ds_idx=1,cbs=[GetLatentSpace(cycle_len=1)])
new_zi = torch.vstack((new_zi,learnClassif.zi_valid))

print("new_zi shape: "+str(new_zi.shape))


pal = sns.color_palette('YlOrBr_r',n_colors=8)
mypal = np.tile(pal[0],(len(lab_all),1))
for i in range(len(lab_all)):
    mycol = 4*lab_area[i]+2*lab_reveil[i]+lab_duration[i]
    # permute value 3 and 4 to obtain the final order: 0 high level < 1 high level < 2 high level < 3 high level
    if mycol == 3: mycol = 4
    elif mycol == 4: mycol = 3
    mypal[i] = pal[mycol]

tmp = mypal

sns.set(rc={'figure.figsize':(11.7,8.27)})

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
predictions_embedded = lda.fit_transform(new_zi.cpu().detach().numpy(),lab4)

plt.figure()
# sns.scatterplot(x=predictions_embedded[:,0], y=np.random.uniform(-500, 500,len(lab_all)), c=mypal)
sns.scatterplot(x=predictions_embedded[:,0], y=predictions_embedded[:,1], c=mypal)

plt.legend(labels=["zi representations in after classif with LDA"])
plt.savefig("results/zi_"+str(classif_filename)+"_lda")

from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
predictions_embedded = tsne.fit_transform(new_zi.cpu().detach().numpy())
print(np.shape(predictions_embedded))
plt.figure()
sns.scatterplot(x=predictions_embedded[:,0], y=predictions_embedded[:,1], c=mypal)

plt.legend(labels=["zi representations in after classif with TSNE"])
plt.savefig("results/zi_"+str(classif_filename)+"_tsne")

# study_name = 'stagerNetVAE_study'
# study = optuna.create_study(direction='minimize',study_name=study_name, storage='sqlite:///results/stagerNetVAE.db')
# study.optimize(objective, n_trials=100)

# display(optuna.visualization.plot_optimization_history(study))
# display(optuna.visualization.plot_param_importances(study))
# display(optuna.visualization.plot_slice(study))
# display(optuna.visualization.plot_parallel_coordinate(study))