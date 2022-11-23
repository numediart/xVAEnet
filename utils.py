# Luca La Fisca
# ------------------------------
# Copyright UMONS (C) 2022

from fastai.tabular.all import *
from tsai.all import *
from torch import nn
from fastai.vision.gan import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

device = torch.device('cuda:1')

dev = torch.device('cpu')

def norm_batch(self, inp: Tuple) -> Tuple:
    x = inp[0]
    x = (x - x.mean(axis=-1).mean(axis=0)) / x.std(axis=-1).std(axis=0)
    out = (x,inp[1])
    return out

class GlobalLoss(Callback):
    def __init__(self, target=None):
        self.target = torch.tensor(target)
    def after_loss(self):
        VAE_weight = .5
        classif_weight = 1/3

        ce = nn.CrossEntropyLoss()
        self.classif_loss = ce(self.pred_class, self.target)

        self.learn.loss=((1-classif_weight)*(VAE_weight*
                        ((1-self.kld_weight)*self.recons_loss + #Reconstruction loss
                        self.kld_weight*self.kld_loss) + #Kullback-Leibler loss
                        (1-VAE_weight)*self.tk_mean) + #GAN generator loss
                        classif_weight*self.classif_loss) #Classif loss

class CheckNorm(Callback):
    order=1
    def __init__(self, norm_y=False):
        self.norm_y = norm_y
    # def before_train(self):
    #     print("check dataset")
    #     print(self.dls.train_ds[0][0].max(dim=1))
    def before_batch(self):
        clamp_val = 3
        if torch.any(self.xb[0].max() > clamp_val) or torch.any(self.xb[0].min() < -clamp_val):
            # print("!!! Weird value ! !!!")
            # print(self.xb[0].shape, self.yb[0].shape, self.xb[0].max(), self.xb[0].min())
            self.xb[0][:] = torch.clamp_min(self.xb[0][:], -clamp_val)
            self.xb[0][:] = torch.clamp_max(self.xb[0][:], clamp_val)
            # print(self.xb[0].shape, self.yb[0].shape, self.xb[0].max(), self.xb[0].min())
            # time.sleep(60)

        if self.norm_y and (torch.any(self.yb[0].max() > clamp_val) or torch.any(self.yb[0].min() < -clamp_val)):
            self.yb[0][:] = torch.clamp_min(self.yb[0][:], -clamp_val)
            self.yb[0][:] = torch.clamp_max(self.yb[0][:], clamp_val)


class FreezeCritic(Callback):
    order = 1
    def before_train(self):
        try:
            self.learn.generator.fc_crit = self.learn.generator.fc_crit.requires_grad_(False)
            self.learn.generator.fc_crit2 = self.learn.generator.fc_crit2.requires_grad_(False)
            self.learn.generator.fc_crit3 = self.learn.generator.fc_crit3.requires_grad_(False)
        except:
            self.learn.model.fc_crit = self.learn.model.fc_crit.requires_grad_(False)
            self.learn.model.fc_crit2 = self.learn.model.fc_crit2.requires_grad_(False)
            self.learn.model.fc_crit3 = self.learn.model.fc_crit3.requires_grad_(False)


class FreezeLayer(Callback):
    order = 1
    def __init__(self, layers_num=None, epochs='all', cycle_len=None):
        self.layers_num = layers_num
        self.epochs = epochs #all or cycle
        self.cycle_len = cycle_len
    def before_train(self):
        n = self.layers_num
        print(n)
        print(self.opt.param_lists)
        if torch.any(n<0):
            frozen_idx = [n[i] if n[i] >= 0 else len(self.opt.param_lists) + n[i] for i in range(len(n))]
        if (torch.any(frozen_idx > len(self.opt.param_lists))):
            warn(f"Freezing {frozen_idx} groups; model has {len(self.opt.param_lists)}; out of bounds.")
        else:
            frozen_idx = n

        if self.epochs == 'cycle':
            if (self.epoch + 1) % cycle_len != 0:
                for i in range(len(frozen_idx)):
                    for o in self.learn.opt.all_params(frozen_idx[i]): self.learn.opt._set_require_grad(True,  *o)
                return
        elif not self.epochs == 'all':
            warn("epochs should be 'all' or 'cycle'!")
            return
        for i in range(len(frozen_idx)):
            for o in self.learn.opt.all_params(frozen_idx[i]): self.learn.opt._set_require_grad(False,  *o)

        print("grad of layer fc_crit3:")
        print(self.model.fc_crit3.weight.grad)


#Define a callback to get the latent space
class GetLatentSpace(Callback):
    def __init__(self, cycle_len=None):
        self.cycle_len_init = cycle_len
        # print('first cycle len: '+str(self.cycle_len_init))
                
    def before_validate(self):
        self.cycle_len = ifnone(self.cycle_len_init,self.n_epoch)
        if (self.epoch+1)% self.cycle_len == 0:
            # print("cycle len: "+str(self.cycle_len)+", at epoch: "+str(self.epoch))
            print("GetLatentSpace, before_validate, Hook starts, Epoch: "+str(self.epoch))
            if (self.epoch+1) % self.cycle_len == 0:
                # self.cycle_len_init = self.cycle_len_init
                def getActivation(name):
                    # the hook signature
                    def hook(model, input, output):
                        self.activation[name] = output.detach()
                    return hook
                self.activation = {}
                try: #for VAE only
                    self.mu = self.learn.model.fc_mu.register_forward_hook(getActivation('mu'))
                    self.std = self.learn.model.fc_var.register_forward_hook(getActivation('std'))
#                     self.zi_a = self.learn.model.fc_z.register_forward_hook(getActivation('zi'))

                except: #for GAN
                    self.mu = self.learn.generator.fc_mu.register_forward_hook(getActivation('mu'))
                    self.std = self.learn.generator.fc_var.register_forward_hook(getActivation('std'))
#                     self.zi_a = self.learn.generator.fc_z.register_forward_hook(getActivation('zi'))
                    
    def after_batch(self):
        if not self.training:
            if (self.epoch+1) % self.cycle_len == 0:
                if hasattr(self, 'mu_valid'):
                    self.learn.mu_valid = torch.vstack((self.learn.mu_valid,self.activation['mu']))
                    self.learn.std_valid = torch.vstack((self.learn.std_valid,self.activation['std']))
                    if hasattr(self, 'zi'):
                        self.learn.zi_valid = torch.vstack((self.learn.zi_valid,self.zi))
                    else:
                        self.learn.zi_valid = torch.vstack((self.learn.zi_valid,self.generator.zi))

                else:
                    self.learn.mu_valid = self.activation['mu']
                    self.learn.std_valid = self.activation['std']
                    if hasattr(self, 'zi'):
                        self.learn.zi_valid = self.zi
                    else:
                        self.learn.zi_valid = self.generator.zi
            
    def after_validate(self):
        if (self.epoch+1)% self.cycle_len == 0:
            # print('my epoch is: '+str(self.epoch))
            std = torch.exp(0.5 * self.learn.std_valid)
            eps = torch.randn_like(std)
            self.learn.zs = eps * std + self.learn.mu_valid
#             self.learn.zs = std + self.learn.mu_valid
            del self.learn.std_valid, self.learn.mu_valid
            torch.cuda.empty_cache()
            # print('after validate, the zs shape is :'+str(self.learn.zs.shape))
            self.mu.remove()
            self.std.remove()
#             self.zi_a.remove()
            

#Define a callback to modify the target zs
class ChangeTargetData(Callback):
    def __init__(self,cycle_len=None,splitter=None,getters=None): 
        self.cycle_len = cycle_len
        self.splitter = splitter
        self.getters = getters

    def after_epoch(self):
        if (self.epoch+1) % self.cycle_len == 0 and self.train_iter>0:
            print("In epoch "+str(self.epoch)+", let's change the target latent space")
            # print("in ChangeTargetData, the new shape of zs valid is: "+str(self.zs.shape))
            x = torch.stack((list(zip(*self.dls.valid_ds))[0]),dim=0)
            y = self.zs

            # self.dls.train = self.dls.train.new(shuffle=False)
            torch.cuda.set_device(device)
            tmp_cbs = self.cbs[-2:]
            self.learn.cbs = self.cbs[:-2]
            self.learn.add_cb(GetLatentSpace(cycle_len=1))
            self.learn.get_preds(ds_idx=0,inner=True)
            self.learn.cbs = self.cbs[:-1]
            self.learn.add_cbs(tmp_cbs)
            y = torch.vstack((self.zs,y))
        # print("in ChangeTargetData, the new shape of zs train is: "+str(self.zs.shape))
            x = torch.vstack((torch.stack((list(zip(*self.dls.train_ds))[0]),dim=0),x))

            # print("in ChangeTargetData, before new dls, x and y shapes are:"+str((x.shape,y.shape)))
            dblock = DataBlock(blocks=(TSTensorBlock,TSTensorBlock),
                              splitter=self.splitter,
                              getters=self.getters,
                              batch_tfms=norm_batch)
            src = itemify(x.to(dev),y.to(dev))
            self.learn.dls = dblock.dataloaders(src,bs=16,val_bs=32)
            print("before changing zs, the loss is: "+str(self.loss))


class TrainClassif(Callback):
    order = 1
    def before_train(self):
        cycle_len = 5
        if (self.epoch+1) % cycle_len == 0:
            self.learn.model.fc_clf = self.learn.model.fc_clf.requires_grad_(True)
            self.learn.model.fc_crit = self.learn.model.fc_crit.requires_grad_(True)
            self.learn.model.fc_crit2 = self.learn.model.fc_crit2.requires_grad_(True)
            self.learn.model.fc_crit3 = self.learn.model.fc_crit3.requires_grad_(True)
            print("!!!!!!!!!!! GRAD SET TO TRUE !!!!!!!!!!!!!")
        elif (self.epoch+1) % cycle_len == 1:
            self.learn.model.fc_clf = self.learn.model.fc_clf.requires_grad_(False)
            self.learn.model.fc_crit = self.learn.model.fc_crit.requires_grad_(False)
            self.learn.model.fc_crit2 = self.learn.model.fc_crit2.requires_grad_(False)
            self.learn.model.fc_crit3 = self.learn.model.fc_crit3.requires_grad_(False)
            print("!!!!!!!!!!! GRAD SET TO FALSE !!!!!!!!!!!!!")

    def after_loss(self):
        cycle_len = 5
        ce = nn.CrossEntropyLoss()
        if (self.epoch+1) % cycle_len == 1:
            #compute loss with equal weight for each single loss
            VAE_weight = .5
            classif_weight = 1/3
            self.learn.tk_mean = 1 - self.tk_mean
            self.learn.loss = ((1-classif_weight)*(VAE_weight*
                    ((1-self.kld_weight)*self.recons_loss + #Reconstruction loss
                    self.kld_weight*self.kld_loss) + #Kullback-Leibler loss
                    (1-VAE_weight)*self.tk_mean) + #GAN generator loss
                    classif_weight*self.classif_loss) #Classif loss
        elif self.nclass == 2 and (self.epoch+1) % 2 == 1:
            self.learn.loss = 1/2 * ce(self.pred_class,self.targ1) + 1/2 * ce(self.pred_class,self.targ2)
        elif self.nclass == 3 and (self.epoch+1) % 3 == 1:
            self.learn.loss = (1/3 * ce(self.pred_class,self.targ1) + 1/3 * ce(self.pred_class,self.targ2) +
                                1/3 * ce(self.pred_class,self.targ3))

    def after_epoch(self):
        print("final loss is: ", str(self.loss))
        print("single losses are: ", str(self.recons_loss), str(self.kld_loss),
            str(self.tk_mean), str(self.classif_loss))