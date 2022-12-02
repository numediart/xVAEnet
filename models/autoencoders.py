# Luca La Fisca
# ------------------------------
# Copyright UMONS (C) 2022

from torch import nn
from torch.autograd import Variable
from fastai.data.all import *
import time

from torchmetrics import F1Score

device = torch.device('cuda:0')

class stagerNetVAE(nn.Module):
    def __init__(self, typ: int=0, nclass: int=1, dropout_rate: float=0.5):
        super(stagerNetVAE, self).__init__()
        
        self.typ = typ
        self.nclass = nclass
        self.latent_dim = 128 #embed_dim
        self.dropout_rate = dropout_rate
        self.channels = 23 #number of input channels (spatial)
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
        zi = F.relu(self.bn_lin(zi))

        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(zi)
        log_var = self.fc_var(zi)
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
            # get the labels
            self.lab = input[:,:3,-1].type(torch.LongTensor)
            input = input[:,:,:-1]

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
        if self.typ==3:
            self.gan_class = self.latent_gan(zi_gan)
        decoded = self.decode(zs, ind1, ind2, in1, in2)
        self.decoded = decoded.permute(0, 1, 3, 2)
        self.mu = mu
        self.log_var = log_var
        
        if self.typ == 3:
            self.pred_class = F.softmax(self.fc_clf(zi)).to(device)
            out = self.pred_class
            if self.nclass == 2:
                self.pred_class2 = F.softmax(self.fc_clf2(zi)).to(device)
                out = self.pred_class2
            elif self.nclass == 3:
                self.pred_class3 = F.softmax(self.fc_clf3(zi)).to(device)
                out = self.pred_class3

            
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

        if self.nclass == 1:
            targ = self.targ1
        elif self.nclass == 2:
            targ = self.targ2
        elif self.nclass == 3:
            targ = self.targ3

        ce = nn.CrossEntropyLoss()
        self.classif_loss = ce(self.pred_class, targ)

        loss = ((1-classif_weight)*(VAE_weight*
                ((1-self.kld_weight)*self.recons_loss + #Reconstruction loss
                self.kld_weight*self.kld_loss) + #Kullback-Leibler loss
                (1-VAE_weight)*self.tk_mean) + #GAN generator loss
                classif_weight*self.classif_loss) #Classif loss

        return loss

class stagerNetCritic(nn.Module):
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