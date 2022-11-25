# Luca La Fisca
# ------------------------------
# Copyright UMONS (C) 2022

from torch import nn
from torch.autograd import Variable
from fastai.data.all import *
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.model_selection import StratifiedShuffleSplit
from torchmetrics import F1Score

device = torch.device('cuda:1')

class stagerNetVAE(nn.Module):
# class stagerNetVAE(BaseVAE):
    def __init__(self, typ: int=0, nclass: int=1, dropout_rate: float=0.5):
        super(stagerNetVAE, self).__init__()
        
        self.typ = typ
        self.nclass = nclass
        self.latent_dim = 128 #embed_dim
        self.dropout_rate = dropout_rate
        self.channels = 23 #nombre de channels (en spatial) de l'input
        self.kld_weight = 0.5
        self.gan_depth = 16
        
        #=============Encoder=============#
        self.conv1 = nn.Conv2d(1, self.channels, (1, self.channels), stride=(1, 1))
        self.conv2 = nn.Conv2d(1, 16, (50,1), stride=(1,1))
        self.conv3 = nn.Conv2d(16, 16, (50,1), stride=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(13,1), return_indices=True) 
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.bn_lin = nn.BatchNorm1d(num_features=128)

        self.fc_z = nn.Linear(16*13*self.channels, self.latent_dim)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)

        #=============Decoder=============#
        self.decoder_input = nn.Linear(self.latent_dim, 16*13*self.channels)
        self.unpool = nn.MaxUnpool2d(kernel_size = (13,1))
        self.deconv1 = nn.ConvTranspose2d(self.channels, 1, (1, self.channels), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(16, 1, (50,1), stride=(1,1))
        self.deconv3 = nn.ConvTranspose2d(16, 16, (50,1), stride=(1,1))

        #===============GAN===============#
        if self.typ>1 and self.typ<4:
            self.fc_crit = nn.Linear(self.latent_dim, self.latent_dim//4)
            self.bn_crit = nn.BatchNorm1d(num_features=self.latent_dim//4)
            self.fc_crit2 = nn.Linear(self.latent_dim//4, self.latent_dim//16)
            self.bn_crit2 = nn.BatchNorm1d(num_features=self.latent_dim//16)
            self.fc_crit3 = nn.Linear(self.latent_dim//16, 1)
        
        #============Classifier============#
        if self.typ >= 3:
            self.fc_clf = nn.Linear(self.latent_dim, 2)
            self.fc_clf2 = nn.Linear(self.latent_dim, 3)
            self.fc_clf3 = nn.Linear(self.latent_dim, 4)

        
    def encode(self, input: Tensor) -> List[Tensor]:
        input = self.conv1(input)
        input = input.permute(0, 3, 2, 1)
        input = self.conv2(input)
        input_mp1 = input.detach().clone()
        input, ind_maxpool1 = self.pool(input)
        input = F.relu(input)
        input = self.batchnorm1(input)
        input = self.conv3(input)
        input_mp2 = input.detach().clone()
        input, ind_maxpool2 = self.pool(input)
        input = F.relu(input)
        result = self.batchnorm2(input)
        result = torch.flatten(result, start_dim=1)
        result = F.dropout(result, p=self.dropout_rate)
        zi = self.fc_z(result)
        # if self.typ==3:
        zi = F.relu(self.bn_lin(zi))

        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(zi)
        log_var = self.fc_var(zi)
        # if self.typ==3:
        mu = F.relu(self.bn_lin(mu))
        log_var = F.relu(self.bn_lin(log_var))

        return [zi, mu, log_var, ind_maxpool1, ind_maxpool2, input_mp1, input_mp2]
    
    def decode(self, z: Tensor, ind1, ind2, in1, in2) -> Tensor:
        x = self.decoder_input(z)        
        x = x.view(-1, 16, 13, self.channels) 
        x = self.unpool(x, indices=ind2, output_size = in2.size())
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.deconv3(x)
        x = self.unpool(x, indices=ind1, output_size = in1.size())
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.deconv2(x)
        x = x.permute(0, 3, 2, 1)
        result = self.deconv1(x)        
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def latent_gan(self, zi: Tensor) -> Tensor:
        x = zi.view(-1,self.latent_dim)
        x = self.fc_crit(x)
        x = F.leaky_relu(self.bn_crit(x),negative_slope=0.2)
        x = self.fc_crit2(x)
        x = F.leaky_relu(self.bn_crit2(x),negative_slope=0.2)
        x = self.fc_crit3(x)
        x = F.sigmoid(x)
        return x

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        if self.typ == 3:
            self.lab = input[:,:3,-1].type(torch.LongTensor)#.to(device)
            input = input[:,:,:-1]
            # print(self.lab.shape)
            # print(input.shape)
        if self.typ != 2:
            if input.dim() < 4:
                input = input.unsqueeze(1)
            self.vae_inp = input
            input = input.permute(0, 1, 3, 2)
            if next(self.conv1.parameters()).is_cuda:
                zi, mu, log_var, ind1, ind2, in1, in2 = self.encode(input)

            else:
                zi, mu, log_var, ind1, ind2, in1, in2 = self.encode(input.type(torch.FloatTensor).to('cpu'))#.detach().cpu())
            self.zi = zi
            zs = self.reparameterize(mu, log_var)
            zi_gan = zi.view(-1, self.latent_dim, 1)
            if self.typ>1 and self.typ<4:
                self.gan_class = self.latent_gan(zi_gan)
            decoded = self.decode(zs, ind1, ind2, in1, in2)
            self.decoded = decoded.permute(0, 1, 3, 2)
            self.mu = mu
            self.log_var = log_var
            self.ind1 = ind1
            self.ind2 = ind2
            self.in1 = in1
            self.in2 = in2
            
            if self.typ >= 3:
                self.pred_class = F.softmax(self.fc_clf(zi)).to(device)
                out = self.pred_class
                if self.nclass == 2:
                    self.pred_class2 = F.softmax(self.fc_clf2(zi)).to(device)
                    out = self.pred_class2
                elif self.nclass == 3:
                    self.pred_class3 = F.softmax(self.fc_clf3(zi)).to(device)
                    out = self.pred_class3
        else:
            if next(self.conv1.parameters()).is_cuda:
                self.gan_class = self.latent_gan(input)
            else:
                self.gan_class = self.latent_gan(input.type(torch.FloatTensor).to('cpu'))#.detach().cpu())
            
        if self.typ == 0: #autoencoder
            return self.decoded
        elif self.typ == 1 or self.typ == 4: #generator
            return zi_gan.squeeze(-1)
        elif self.typ == 2: #discriminator
            return self.gan_class
        elif self.typ == 3: #classifier
            return out

    def loss_func(self, output, target):
        recons_loss = F.mse_loss(output, target) #Measures the element-wise mean squared error
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        loss = (1-self.kld_weight)*recons_loss + self.kld_weight*kld_loss        
        return loss
    
    def gen_loss_func(self,fake_pred, output, target):
        tk_mean = (1 - fake_pred).mean()
        VAE_weight = .1
        recons_loss = F.mse_loss(self.decoded, self.vae_inp)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        loss = VAE_weight*((1-self.kld_weight)*recons_loss + self.kld_weight*kld_loss) + (1-VAE_weight)*tk_mean
        
        return loss
    
#     def classif_loss_func(self,fake_pred, output, target):
    def classif_loss_func(self, output, target):
        fake_pred = self.gan_class
        self.tk_mean = 1 - fake_pred.mean()
        VAE_weight = .2
        classif_weight = .95
        self.recons_loss = F.mse_loss(self.decoded, self.vae_inp)
        self.kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)

        self.targ1 = self.lab[:,0].to(device)
        self.targ2 = self.lab[:,1].to(device)
        self.targ3 = self.lab[:,2].to(device)
#         self.targ1 = torch.zeros(target.shape).long().to(device)
#         self.targ1[target > 3] = 1
#         self.targ2 = torch.zeros(target.shape).long().to(device)
#         self.targ2[(target % 4) > 1] = 1
#         self.targ3 = target % 2
        if self.nclass == 1:
            targ = self.targ1
        elif self.nclass == 2:
            targ = self.targ2
        elif self.nclass == 3:
            targ = self.targ3

        ce = nn.CrossEntropyLoss()
        # print("print output/target")
        # print(self.pred_class.shape)
        # print(targ.shape)
        # print(output.shape)
        # print(target.shape)
        self.classif_loss = ce(self.pred_class, targ)

        loss = ((1-classif_weight)*(VAE_weight*
                ((1-self.kld_weight)*self.recons_loss + #Reconstruction loss
                self.kld_weight*self.kld_loss) + #Kullback-Leibler loss
                (1-VAE_weight)*self.tk_mean) + #GAN generator loss
                classif_weight*self.classif_loss) #Classif loss

        return loss
#         return self.classif_loss


class stagerNetCritic(nn.Module):
# class stagerNetVAE(BaseVAE):
    def __init__(self):
        super(stagerNetCritic, self).__init__()
        
        self.latent_dim = 128 #embed_dim
        
        self.fc_crit = nn.Linear(self.latent_dim, self.latent_dim//4)
        self.bn_crit = nn.BatchNorm1d(num_features=self.latent_dim//4)
        self.fc_crit2 = nn.Linear(self.latent_dim//4, self.latent_dim//16)
        self.bn_crit2 = nn.BatchNorm1d(num_features=self.latent_dim//16)
        self.fc_crit3 = nn.Linear(self.latent_dim//16, 1)
        
    def discrim(self, zi: Tensor) -> Tensor:
        x = zi.view(-1,self.latent_dim)
        x = self.fc_crit(x)
        x = F.leaky_relu(self.bn_crit(x),negative_slope=0.2)
        x = self.fc_crit2(x)
        x = F.leaky_relu(self.bn_crit2(x),negative_slope=0.2)
        x = self.fc_crit3(x)
        x = F.sigmoid(x)
        return x

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        if next(self.fc_crit.parameters()).is_cuda:
            self.gan_class = self.discrim(input)
        else:
            self.gan_class = self.discrim(input.type(torch.FloatTensor).detach().cpu())
        return self.gan_class








class last_old_stagerNetVAE(nn.Module):
# class stagerNetVAE(BaseVAE):
    def __init__(self, typ: int=0, nclass: int=1, dropout_rate: float=0.5):
        super(last_old_stagerNetVAE, self).__init__()
        
        self.typ = typ
        self.nclass = nclass
        self.latent_dim = 128 #embed_dim
        self.dropout_rate = dropout_rate
        self.channels = 23 #nombre de channels (en spatial) de l'input
        self.kld_weight = 0.5
        self.gan_depth = 16
        
        #=============Encoder=============#
        self.conv1 = nn.Conv2d(1, self.channels, (1, self.channels), stride=(1, 1))
        self.conv2 = nn.Conv2d(1, 16, (50,1), stride=(1,1))
        self.conv3 = nn.Conv2d(16, 16, (50,1), stride=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(13,1), return_indices=True) 
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.bn_lin = nn.BatchNorm1d(num_features=128)

        self.fc_z = nn.Linear(16*13*self.channels, self.latent_dim)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)

        #=============Decoder=============#
        self.decoder_input = nn.Linear(self.latent_dim, 16*13*self.channels)
        self.unpool = nn.MaxUnpool2d(kernel_size = (13,1))
        self.deconv1 = nn.ConvTranspose2d(self.channels, 1, (1, self.channels), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(16, 1, (50,1), stride=(1,1))
        self.deconv3 = nn.ConvTranspose2d(16, 16, (50,1), stride=(1,1))

        #===============GAN===============#
        if self.typ>1 and self.typ<4:
            self.fc_crit = nn.Linear(self.latent_dim, self.latent_dim//4)
            self.bn_crit = nn.BatchNorm1d(num_features=self.latent_dim//4)
            self.fc_crit2 = nn.Linear(self.latent_dim//4, self.latent_dim//16)
            self.bn_crit2 = nn.BatchNorm1d(num_features=self.latent_dim//16)
            self.fc_crit3 = nn.Linear(self.latent_dim//16, 1)
        
        #============Classifier============#
        # self.clf = LinearDiscriminantAnalysis(n_components=1) 
        if self.typ >= 3:
            self.fc_clf = nn.Linear(self.latent_dim, 2)

        
    def encode(self, input: Tensor) -> List[Tensor]:
        input = self.conv1(input)
        input = input.permute(0, 3, 2, 1)
        input = self.conv2(input)
        input_mp1 = input.detach().clone()
        input, ind_maxpool1 = self.pool(input)
        input = F.relu(input)
        input = self.batchnorm1(input)
        input = self.conv3(input)
        input_mp2 = input.detach().clone()
        input, ind_maxpool2 = self.pool(input)
        input = F.relu(input)
        result = self.batchnorm2(input)
        result = torch.flatten(result, start_dim=1)
        result = F.dropout(result, p=self.dropout_rate)
        zi = self.fc_z(result)
        # if self.typ==3:
        zi = F.relu(self.bn_lin(zi))

        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(zi)
        log_var = self.fc_var(zi)
        # if self.typ==3:
        mu = F.relu(self.bn_lin(mu))
        log_var = F.relu(self.bn_lin(log_var))

        return [zi, mu, log_var, ind_maxpool1, ind_maxpool2, input_mp1, input_mp2]
    
    def decode(self, z: Tensor, ind1, ind2, in1, in2) -> Tensor:
        x = self.decoder_input(z)        
        x = x.view(-1, 16, 13, self.channels) 
        x = self.unpool(x, indices=ind2, output_size = in2.size())
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.deconv3(x)
        x = self.unpool(x, indices=ind1, output_size = in1.size())
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.deconv2(x)
        x = x.permute(0, 3, 2, 1)
        result = self.deconv1(x)        
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def latent_gan(self, zi: Tensor) -> Tensor:
        x = zi.view(-1,self.latent_dim)
        x = self.fc_crit(x)
        x = F.leaky_relu(self.bn_crit(x),negative_slope=0.2)
        x = self.fc_crit2(x)
        x = F.leaky_relu(self.bn_crit2(x),negative_slope=0.2)
        x = self.fc_crit3(x)
        x = F.sigmoid(x)
        return x

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        if self.typ == 4:
            self.lab = input[:,0,-1].type(torch.LongTensor)#.to(device)
            input = input[:,:,:-1]
        if self.typ != 2:
            if input.dim() < 4:
                input = input.unsqueeze(1)
            self.vae_inp = input
            input = input.permute(0, 1, 3, 2)
            if next(self.conv1.parameters()).is_cuda:
                zi, mu, log_var, ind1, ind2, in1, in2 = self.encode(input)

            else:
                zi, mu, log_var, ind1, ind2, in1, in2 = self.encode(input.type(torch.FloatTensor).to('cpu'))
            self.zi = zi
            zs = self.reparameterize(mu, log_var)
            zi_gan = zi.view(-1, self.latent_dim, 1)
            if self.typ>1 and self.typ<4:
                self.gan_class = self.latent_gan(zi_gan)
            decoded = self.decode(zs, ind1, ind2, in1, in2)
            self.decoded = decoded.permute(0, 1, 3, 2)
            self.mu = mu
            self.log_var = log_var
            self.ind1 = ind1
            self.ind2 = ind2
            self.in1 = in1
            self.in2 = in2
            
            # time.sleep(2)

            if self.typ >= 3:
                self.pred_class = F.softmax(self.fc_clf(zi)).to(device)
        else:
            if next(self.conv1.parameters()).is_cuda:
                self.gan_class = self.latent_gan(input)
            else:
                self.gan_class = self.latent_gan(input.type(torch.FloatTensor).to('cpu'))#.detach().cpu())
            
        if self.typ == 0: #autoencoder
            return self.decoded
        elif self.typ == 1 or self.typ == 4: #generator
            return zi_gan.squeeze(-1)
        elif self.typ == 2: #discriminator
            return self.gan_class
        elif self.typ == 3: #classifier
            out = self.pred_class
            return out

    def loss_func(self, output, target):
        #print('In loss function!!!')
#         if self.my_loss == 0: # autoencoder only
        # print("out/targ")
        # print(output[0,:2,:10])
        # print(target[0,:2,:10])
        recons_loss = F.mse_loss(output, target) #Measures the element-wise mean squared error
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        loss = (1-self.kld_weight)*recons_loss + self.kld_weight*kld_loss        
        # print('my losses are: '+str(float(loss))+'; kld: '+ str(float(kld_loss))+
        #       ';reconstr: '+str(float(recons_loss))) #+ ';wasserstein: '+str(float(gan_loss)))
        
        # # print("loss: "+str(loss))
        # print("loss grad: ",str(loss.requires_grad))
        # time.sleep(2)
        return loss
    
    def wasserstein_loss(y_true, y_pred):
        return mean(y_true * y_pred)
    
    def gen_loss_func(self,fake_pred, output, target):
#         print('In gen_loss_func')
# #         print(fake_pred.shape+output.shape+target.shape+test.shape)
#         print(fake_pred.shape)
#         print(output.shape)
#         print(target.shape)
        tk_mean = fake_pred.mean()
#         print('show inp/out shapes: ')
#         print(self.decoded.shape)
#         print(self.vae_inp.shape)
        VAE_weight = .2
        recons_loss = F.mse_loss(self.decoded, self.vae_inp)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        loss = VAE_weight*((1-self.kld_weight)*recons_loss + self.kld_weight*kld_loss) + (1-VAE_weight)*tk_mean
        
#         print('my losses are: '+str(float(loss))+
#               '; kld: '+ str(float(.1*self.kld_weight*kld_loss))+
#               '; reconstr: '+str(float(.1*(1-self.kld_weight)*recons_loss)) +
#               '; wasserstein: '+str(float(tk_mean)))
        
        return loss
    
#     def classif_loss_func(self,fake_pred, output, target):
    def classif_loss_func(self, output, target):
        # def _contiguous(x): return TensorBase(x.transpose(-1,-1))
        # output, target = map(_contiguous, (output,target))
        fake_pred = self.gan_class
        self.tk_mean = 1 - fake_pred.mean()
        VAE_weight = .2
        classif_weight = .95
        self.recons_loss = F.mse_loss(self.decoded, self.vae_inp)
        self.kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)

        self.targ1 = torch.zeros(target.shape).long().to(device)
        self.targ1[target > 3] = 1
        self.targ2 = torch.zeros(target.shape).long().to(device)
        self.targ2[(target % 4) > 1] = 1
        self.targ3 = target % 2
        if self.nclass == 1:
            targ = self.targ1
        elif self.nclass == 2:
            targ = self.targ2
        elif self.nclass == 3:
            targ = self.targ3

        ce = nn.CrossEntropyLoss()
        self.classif_loss = ce(output, targ)

        loss = ((1-classif_weight)*(VAE_weight*
                ((1-self.kld_weight)*self.recons_loss + #Reconstruction loss
                self.kld_weight*self.kld_loss) + #Kullback-Leibler loss
                (1-VAE_weight)*self.tk_mean) + #GAN generator loss
                classif_weight*self.classif_loss) #Classif loss

        return loss

    def global_loss_func(self, fake_pred, output, target):
        # print("before callback, loss is:", str(self.loss))
        # VAE_weight = .5
        # classif_weight = 1/3
        VAE_weight = 1/3
        classif_weight = 1/4
        self.recons_loss = F.mse_loss(self.decoded, self.vae_inp)
        self.kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        self.tk_mean = fake_pred.mean()

        ce = nn.CrossEntropyLoss()
        self.classif_loss = ce(self.pred_class.to('cpu'), self.lab.to('cpu'))

        # loss = ((1-classif_weight)*(VAE_weight*
        #         (self.recons_loss + #Reconstruction loss
        #         self.kld_loss) + #Kullback-Leibler loss
        #         (1-VAE_weight)*self.tk_mean) + #GAN generator loss
        #         classif_weight*self.classif_loss) #Classif loss
        loss = ((1-classif_weight)*VAE_weight*
                (self.recons_loss + #Reconstruction loss
                self.kld_loss + #Kullback-Leibler loss
                self.tk_mean) + #GAN generator loss
                classif_weight*self.classif_loss) #Classif loss

        return loss



class old2_stagerNetVAE(nn.Module):
# class stagerNetVAE(BaseVAE):
    def __init__(self, typ: int=0, my_loss: int=0, dropout_rate: float=.5):
        super(stagerNetVAE, self).__init__()
        
        self.typ = typ
        self.my_loss = my_loss
        self.latent_dim = 128 #embed_dim
        self.dropout_rate = dropout_rate
        self.channels = 23 #nombre de channels (en spatial) de l'input
        self.kld_weight = 0.5
        self.gan_depth = 16
        
        #=============Encoder=============#
        self.conv1 = nn.Conv2d(1, self.channels, (1, self.channels), stride=(1, 1))
        self.conv2 = nn.Conv2d(1, 16, (50,1), stride=(1,1))
        self.conv3 = nn.Conv2d(16, 16, (50,1), stride=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(13,1), return_indices=True) 
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.fc_z = nn.Linear(16*13*self.channels, self.latent_dim)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)

        #=============Decoder=============#
        self.decoder_input = nn.Linear(self.latent_dim, 16*13*self.channels)
        self.unpool = nn.MaxUnpool2d(kernel_size = (13,1))
        self.deconv1 = nn.ConvTranspose2d(self.channels, 1, (1, self.channels), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(16, 1, (50,1), stride=(1,1))
        self.deconv3 = nn.ConvTranspose2d(16, 16, (50,1), stride=(1,1))

        #===============GAN===============#
        self.fc_crit = nn.Linear(self.latent_dim, self.latent_dim//4)
        self.fc_crit2 = nn.Linear(self.latent_dim//4, self.latent_dim//16)
        self.fc_crit3 = nn.Linear(self.latent_dim//16, 1)
        
        #============Classifier============#
        # self.clf = RandomForestClassifier(random_state=0)
        # self.clf = DecisionTreeClassifier(max_depth=3, random_state=0)
        # self.clf = SGDClassifier()
        self.fc_clf = nn.Linear(self.latent_dim, 1)
        self.clf = LinearDiscriminantAnalysis(n_components=1) 

        
    def encode(self, input: Tensor) -> List[Tensor]:
        input = self.conv1(input)
        input = input.permute(0, 3, 2, 1)
        input = self.conv2(input)
        input_mp1 = input.detach().clone()
        input, ind_maxpool1 = self.pool(input)
        input = F.relu(input)
        input = self.batchnorm1(input)
        input = self.conv3(input)
        input_mp2 = input.detach().clone()
        input, ind_maxpool2 = self.pool(input)
        input = F.relu(input)
        result = self.batchnorm2(input)
        result = torch.flatten(result, start_dim=1)
        result = F.dropout(result, p=self.dropout_rate)
        zi = self.fc_z(result)

        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(zi)
        log_var = self.fc_var(zi)
        return [zi, mu, log_var, ind_maxpool1, ind_maxpool2, input_mp1, input_mp2]
    
    def decode(self, z: Tensor, ind1, ind2, in1, in2) -> Tensor:
        x = self.decoder_input(z)        
        x = x.view(-1, 16, 13, self.channels) 
        x = self.unpool(x, indices=ind2, output_size = in2.size())
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.deconv3(x)
        x = self.unpool(x, indices=ind1, output_size = in1.size())
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.deconv2(x)
        x = x.permute(0, 3, 2, 1)
        result = self.deconv1(x)        
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def latent_gan(self, zi: Tensor) -> Tensor:
        x = zi.view(-1,self.latent_dim)
        x = self.fc_crit(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = self.fc_crit2(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = self.fc_crit3(x)
        x = F.sigmoid(x)
        return x

    def forward(self, input: Tensor, **kwargs) -> Tensor:
#         print('I am in forward,\n my input is: '+str(input.shape)+str(type(input))+'\n My type  is: '+str(self.typ))
        if self.typ != 2:
            if input.dim() < 4:
                input = input.unsqueeze(1)
            self.vae_inp = input
            input = input.permute(0, 1, 3, 2)
            if next(self.conv1.parameters()).is_cuda:
                zi, mu, log_var, ind1, ind2, in1, in2 = self.encode(input)
            else:
                zi, mu, log_var, ind1, ind2, in1, in2 = self.encode(input.type(torch.FloatTensor).detach().cpu())
            zs = self.reparameterize(mu, log_var)
            zi_gan = zi.view(-1, self.latent_dim, 1)
            #print('shape of input latent space: '+str(zi_gan.shape))
            self.gan_class = self.latent_gan(zi_gan)
            decoded = self.decode(zs, ind1, ind2, in1, in2)
            self.decoded = decoded.permute(0, 1, 3, 2)
            self.mu = mu
            self.log_var = log_var
            self.zi = zi
            
            if self.typ == 3:
                x = self.fc_clf(zi)
                self.pred_class = F.sigmoid(x)
                # self.pred_class = torch.Tensor(self.clf.predict(zi.to('cpu').detach().numpy())).unsqueeze(1).to(device)
                # self.pred_class = torch.Tensor(self.clf.predict(zi.to('cpu').detach().numpy())).to(device)

        else:
            if next(self.conv1.parameters()).is_cuda:
                self.gan_class = self.latent_gan(input)
            else:
                self.gan_class = self.latent_gan(input.type(torch.FloatTensor).detach().cpu())
            
        if self.typ == 0: #autoencoder
            return self.decoded
        elif self.typ == 1: #generator
            #print('final result gen: '+str(self.gan_class))
            return zi_gan.squeeze(-1)
        elif self.typ == 2: #discriminator
            #print('final result discrim: '+str(self.gan_class))
            return self.gan_class
        elif self.typ == 3: #classifier
            return self.pred_class

    def loss_func(self, output, target):
        recons_loss = F.mse_loss(output, target) #Measures the element-wise mean squared error
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        loss = (1-self.kld_weight)*recons_loss + self.kld_weight*kld_loss        
        
        return loss
    
    def wasserstein_loss(y_true, y_pred):
        return mean(y_true * y_pred)
    
    def gen_loss_func(self,fake_pred, output, target):
        tk_mean = fake_pred.mean()
        VAE_weight = .2
        recons_loss = F.mse_loss(self.decoded, self.vae_inp)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        loss = VAE_weight*((1-self.kld_weight)*recons_loss + self.kld_weight*kld_loss) + (1-VAE_weight)*tk_mean
        
        return loss
    
    def classif_loss_func(self, output, target):
        print('out/targ:',str((output.shape,target.shape)))
        def _contiguous(x): return TensorBase(x.transpose(-1,-1))
        fake_pred = self.gan_class
        self.tk_mean = 1 - fake_pred.mean()
        self.kld_weight = .8
        VAE_weight = .9
        classif_weight = 1
        self.recons_loss = F.mse_loss(self.decoded, self.vae_inp)
        self.kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
#         print('shapes output/target within loss function = '+str((output.shape,target.shape)))
        # fscore = metrics.f1_score(target.to('cpu').detach().numpy(), output.to('cpu').detach().numpy())
        # f1 = F1Score(num_classes=1,multiclass=False).to(device)
        # mae = MeanAbsoluteError().to(device)
        # print((output.shape,target.shape))
        # out = torch.Tensor(self.clf.predict(self.zi.to('cpu').detach().numpy())).unsqueeze(1).to(device)
        # out, target = map(_contiguous, (out,target))
        out = torch.from_numpy(self.clf.predict(self.zi.to('cpu').detach().numpy())).unsqueeze(1).to(device)

        self.classif_loss = torch.mean(abs(out - target))#.type(torch.FloatTensor)
        print(self.classif_loss)
        # self.classif_loss = torch.Tensor(1.0-float(fscore))
        # try:
        #     print('in try \n')
        #     self.classif_loss = torch.vstack((self.classif_loss,closs))
        #     loss = (((1-classif_weight)*VAE_weight*
        #             ((1-self.kld_weight)*self.recons_loss + #Reconstruction loss
        #             self.kld_weight*self.kld_loss) + #Kullback-Leibler loss
        #             (1-VAE_weight)*self.tk_mean) + #GAN generator loss
        #             classif_weight*self.classif_loss[-1]) #Classif loss
        # except:
        #     print('in except \n')
        #     time.sleep(3)
        #     self.classif_loss = closs
        #     print(self.classif_loss)
        #     loss = (((1-classif_weight)*VAE_weight*
        #             ((1-self.kld_weight)*self.recons_loss + #Reconstruction loss
        #             self.kld_weight*self.kld_loss) + #Kullback-Leibler loss
        #             (1-VAE_weight)*self.tk_mean) + #GAN generator loss
        #             classif_weight*self.classif_loss) #Classif loss

        # # if self.learn.iter % 42 == 0:
        # print("my loss is: ", str(self.classif_loss.to('cpu').detach().numpy().T))
        # print("And fscore: ", str(fscore))
    
        return self.classif_loss
       











class old_stagerNetVAE(nn.Module):
    def __init__(self, typ: int = 0):
        super(old_stagerNetVAE, self).__init__()
        
        self.typ = typ
        self.latent_dim = 128 #embed_dim
        self.dropout_rate=0.5
        self.channels = 23 #nombre de channels (en spatial) de l'input
        self.kld_weight = 0.5
        self.gan_depth = 16
        
        #=============Encoder=============#
        self.conv1 = nn.Conv2d(1, self.channels, (1, self.channels), stride=(1, 1))
        self.conv2 = nn.Conv2d(1, 16, (50,1), stride=(1,1))
        self.conv3 = nn.Conv2d(16, 16, (50,1), stride=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(13,1), return_indices=True) 
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.fc_z = nn.Linear(16*13*self.channels, self.latent_dim)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)

        #=============Decoder=============#
        self.decoder_input = nn.Linear(self.latent_dim, 16*13*self.channels)
        self.unpool = nn.MaxUnpool2d(kernel_size = (13,1))
        self.deconv1 = nn.ConvTranspose2d(self.channels, 1, (1, self.channels), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(16, 1, (50,1), stride=(1,1))
        self.deconv3 = nn.ConvTranspose2d(16, 16, (50,1), stride=(1,1))

        #===============GAN===============#
        self.fc_crit = nn.Linear(self.latent_dim, self.latent_dim//4)
        self.fc_crit2 = nn.Linear(self.latent_dim//4, self.latent_dim//16)
        self.fc_crit3 = nn.Linear(self.latent_dim//16, 1)


    def encode(self, input: Tensor) -> List[Tensor]:
        input = self.conv1(input)
        input = input.permute(0, 3, 2, 1)
        input = self.conv2(input)
        input_mp1 = input.detach().clone()
        input, ind_maxpool1 = self.pool(input)
        input = F.relu(input)
        input = self.batchnorm1(input)
        input = self.conv3(input)
        input_mp2 = input.detach().clone()
        input, ind_maxpool2 = self.pool(input)
        input = F.relu(input)
        result = self.batchnorm2(input)
        result = torch.flatten(result, start_dim=1)
        result = F.dropout(result, p=self.dropout_rate)
        zi = self.fc_z(result)
        #print('in encode, shape of zi: '+str(zi.shape))

        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(zi)
        log_var = self.fc_var(zi)
        return [zi, mu, log_var, ind_maxpool1, ind_maxpool2, input_mp1, input_mp2]
    
    def decode(self, z: Tensor, ind1, ind2, in1, in2) -> Tensor:
        x = self.decoder_input(z)        
        x = x.view(-1, 16, 13, self.channels) 
        x = self.unpool(x, indices=ind2, output_size = in2.size())
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.deconv3(x)
        x = self.unpool(x, indices=ind1, output_size = in1.size())
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.deconv2(x)
        x = x.permute(0, 3, 2, 1)
        result = self.deconv1(x)        
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def latent_gan(self, zi: Tensor) -> Tensor:
        x = zi.view(-1,self.latent_dim)
        x = self.fc_crit(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = self.fc_crit2(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = self.fc_crit3(x)
        x = F.sigmoid(x)
        return x


    def forward(self, input: Tensor, **kwargs) -> Tensor:
#         print('I am in forward,\n my input is: '+str(input.shape)+str(type(input))+'\n My type  is: '+str(self.typ))
        if self.typ != 2:
            if input.dim() < 4:
                input = input.unsqueeze(1)
            self.vae_inp = input
            input = input.permute(0, 1, 3, 2)
            zi, mu, log_var, ind1, ind2, in1, in2 = self.encode(input)
            zs = self.reparameterize(mu, log_var)
            zi_gan = zi.view(-1, self.latent_dim, 1)
            #print('shape of input latent space: '+str(zi_gan.shape))
            self.gan_class = self.latent_gan(zi_gan)
            decoded = self.decode(zs, ind1, ind2, in1, in2)
            self.decoded = decoded.permute(0, 1, 3, 2)
            # return  decoded
            self.mu = mu
            self.log_var = log_var
        else:
            #print('shape of input latent space: '+str(input.shape))
            self.gan_class = self.latent_gan(input)
            
        if self.typ == 0: #autoencoder
            return self.decoded
        elif self.typ == 1: #generator
            #print('final result gen: '+str(self.gan_class))
            return zi_gan
        elif self.typ == 2: #discriminator
            #print('final result discrim: '+str(self.gan_class))
            return self.gan_class
        elif self.typ == 3: #classifier
            return my_class

    def loss_func(self, output, target):
        recons_loss =F.mse_loss(output, target) #Measures the element-wise mean squared error
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        loss = (1-self.kld_weight)*recons_loss + self.kld_weight*kld_loss
        #print('my losses are: '+str(float(loss))+'; kld: '+str(float(kld_loss))+'; reconstr: '+str(float(recons_loss)))
        
        return loss

    def gen_loss_func(self,fake_pred, output, target):
        VAE_weights = .2
        tk_mean = fake_pred.mean()
        recons_loss = F.mse_loss(self.decoded, self.vae_inp)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        loss = VAE_weights*((1-self.kld_weight)*recons_loss + self.kld_weight*kld_loss) + (1-VAE_weights)*tk_mean

        return loss


class stagerNetClassif(nn.Module):
# class stagerNetVAE(BaseVAE):
    def __init__(self, dropout_rate: float=0.5):
        super(stagerNetClassif, self).__init__()
        
        self.latent_dim = 128 #embed_dim
        self.dropout_rate = dropout_rate
        self.channels = 23 #nombre de channels (en spatial) de l'input
        
        #=============Encoder=============#
        self.conv1 = nn.Conv2d(1, self.channels, (1, self.channels), stride=(1, 1))
        self.conv2 = nn.Conv2d(1, 16, (50,1), stride=(1,1))
        self.conv3 = nn.Conv2d(16, 16, (50,1), stride=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(13,1), return_indices=True) 
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.fc_z = nn.Linear(16*13*self.channels, self.latent_dim)
        #============Classifier===========#
        self.fc_clf = nn.Linear(self.latent_dim, 2)

    def encode(self, inp: Tensor) -> List[Tensor]:
        inp = self.conv1(inp)
        inp = inp.permute(0, 3, 2, 1)
        inp = self.conv2(inp)
        input_mp1 = inp.detach().clone()
        inp, ind_maxpool1 = self.pool(inp)
        inp = F.relu(inp)
        inp = self.batchnorm1(inp)
        inp = self.conv3(inp)
        input_mp2 = inp.detach().clone()
        inp, ind_maxpool2 = self.pool(inp)
        inp = F.relu(inp)
        result = self.batchnorm2(inp)
        result = torch.flatten(result, start_dim=1)
        result = F.dropout(result, p=self.dropout_rate)
        zi = self.fc_z(result)
        pred_class = self.fc_clf(zi)
        pred_class = F.softmax(pred_class) #get result between 0 and 1

        return pred_class

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        if input.dim() < 4:
            input = input.unsqueeze(1)
        self.vae_inp = input
        input = input.permute(0, 1, 3, 2)
        if next(self.conv1.parameters()).is_cuda:
            pred_class = self.encode(input)
        else:
            pred_class = self.encode(input.to('cpu'))

        print('pred type: ')
        print(pred_class.dtype)

        return pred_class

    def loss_func(self, output, target):
        print("check loss: ")
        target = target.squeeze(1)
        # print(target)
        # print(output)
        # print("then")
        # y = target*1.0
        # print(y.dtype)
        out = torch.randn(1, 8, 10)
        tar = torch.randint(0, 10, (1,8))
        # print("test targte : ")
        # print(tar)
        # print(out)

        tst = nn.CrossEntropyLoss()
        # test = tst(out,tar)
        # print("test loss: ",str(test))
        loss = tst(output, target)
        # print("my loss is: ", str(loss))
        return loss

