# Luca La Fisca
# ------------------------------
# Copyright UMONS (C) 2022

from fastai.tabular.all import *
from tsai.all import *
from torch import nn
from fastai.vision.gan import *

device = torch.device('cuda:0')

dev = torch.device('cpu')

class norm_batch(Transform):
    def __init__(self, eps=1e-08) :
        self.eps = eps
    def encodes(self, t:TSTensor):
        try:
            mean = torch.nanmean(t, dim=2)
            std = torch.clamp_min(torch.std(t, dim=2), self.eps)
            out = torch.stack([torch.vstack([(t[j,i,:]-mean[j,i])/torch.clamp_min(std[j,i], self.eps)
                                 for i in range(t.shape[1])]) 
                                 for j in range(t.shape[0])],dim=0)
            out = torch.clamp_min(out,-3)
            out = torch.clamp_max(out,3)
        except:
            out = t
        return out.to(device)

class CheckNorm(Callback):
    order=1
    def __init__(self, norm_y=False):
        self.norm_y = norm_y
    def before_batch(self):
        clamp_val = 3
        if torch.any(self.xb[0].max() > clamp_val) or torch.any(self.xb[0].min() < -clamp_val):
            self.xb[0][:] = torch.clamp_min(self.xb[0][:], -clamp_val)
            self.xb[0][:] = torch.clamp_max(self.xb[0][:], clamp_val)

        if self.norm_y and (torch.any(self.yb[0].max() > clamp_val) or torch.any(self.yb[0].min() < -clamp_val)):
            self.yb[0][:] = torch.clamp_min(self.yb[0][:], -clamp_val)
            self.yb[0][:] = torch.clamp_max(self.yb[0][:], clamp_val)


#Define a callback to get the latent space
class GetLatentSpace(Callback):
    def __init__(self, cycle_len=None):
        self.cycle_len_init = cycle_len
                
    def before_validate(self):
        self.cycle_len = ifnone(self.cycle_len_init,self.n_epoch)
        if (self.epoch+1)% self.cycle_len == 0:
            print("GetLatentSpace, before_validate, Hook starts, Epoch: "+str(self.epoch))
            if (self.epoch+1) % self.cycle_len == 0:
                def getActivation(name):
                    # the hook signature
                    def hook(model, input, output):
                        self.activation[name] = output.detach()
                    return hook
                self.activation = {}
                try:
                    self.mu = self.learn.model.fc_mu.register_forward_hook(getActivation('mu'))
                    self.std = self.learn.model.fc_var.register_forward_hook(getActivation('std'))

                except: #for GAN
                    self.mu = self.learn.generator.fc_mu.register_forward_hook(getActivation('mu'))
                    self.std = self.learn.generator.fc_var.register_forward_hook(getActivation('std'))
                    
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
            std = torch.exp(0.5 * self.learn.std_valid)
            eps = torch.randn_like(std)
            self.learn.zs = eps * std + self.learn.mu_valid
            del self.learn.std_valid, self.learn.mu_valid
            torch.cuda.empty_cache()
            self.mu.remove()
            self.std.remove()
            

#Define a callback to modify the target zs
class ChangeTargetData(Callback):
    def __init__(self,cycle_len=None,splitter=None,getters=None): 
        self.cycle_len = cycle_len
        self.splitter = splitter
        self.getters = getters

    def after_epoch(self):
        if (self.epoch+1) % self.cycle_len == 0 and self.train_iter>0:
            print("In epoch "+str(self.epoch)+", let's change the target latent space")
            x = torch.stack((list(zip(*self.dls.valid_ds))[0]),dim=0)
            y = self.zs

            torch.cuda.set_device(device)
            tmp_cbs = self.cbs[-2:]
            self.learn.cbs = self.cbs[:-2]
            self.learn.add_cb(GetLatentSpace(cycle_len=1))
            self.learn.get_preds(ds_idx=0,inner=True)
            self.learn.cbs = self.cbs[:-1]
            self.learn.add_cbs(tmp_cbs)
            y = torch.vstack((self.zs,y))
            x = torch.vstack((torch.stack((list(zip(*self.dls.train_ds))[0]),dim=0),x))

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
            # uncomment to unfreeze frozen classifier layers 
            # self.learn.model.fc_clf = self.learn.model.fc_clf.requires_grad_(True)
            # self.learn.model.fc_clf2 = self.learn.model.fc_clf2.requires_grad_(True)
            # self.learn.model.fc_clf3 = self.learn.model.fc_clf3.requires_grad_(True)
            self.learn.model.fc_crit = self.learn.model.fc_crit.requires_grad_(True)
            self.learn.model.fc_crit2 = self.learn.model.fc_crit2.requires_grad_(True)
            self.learn.model.fc_crit3 = self.learn.model.fc_crit3.requires_grad_(True)
            print("!!!!!!!!!!! GRAD SET TO TRUE !!!!!!!!!!!!!")
        elif (self.epoch+1) % cycle_len == 1:
            # uncomment to freeze classifier layers
            # self.learn.model.fc_clf = self.learn.model.fc_clf.requires_grad_(False)
            # self.learn.model.fc_clf2 = self.learn.model.fc_clf2.requires_grad_(False)
            # self.learn.model.fc_clf3 = self.learn.model.fc_clf3.requires_grad_(False)
            self.learn.model.fc_crit = self.learn.model.fc_crit.requires_grad_(False)
            self.learn.model.fc_crit2 = self.learn.model.fc_crit2.requires_grad_(False)
            self.learn.model.fc_crit3 = self.learn.model.fc_crit3.requires_grad_(False)
            print("!!!!!!!!!!! GRAD SET TO FALSE !!!!!!!!!!!!!")

    def after_loss(self):
        cycle_len = 5
        ce = nn.CrossEntropyLoss()
        if (self.epoch+1) % cycle_len == 0:
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