import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import uniform, normal
import random
import cython
import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import ot.gpu

import os
import numpy as np
from sklearn.metrics import accuracy_score

from models import Generator, Discriminator, MLPClassifier, Resnet101
from datautils import ZSLDataset

class Trainer:
    def __init__(
        self, device, x_dim, z_dim, attr_dim, **kwargs):
        '''
        Trainer class.
        Args:
            device: CPU/GPU
            x_dim: Dimension of image feature vector
            z_dim: Dimension of noise vector
            attr_dim: Dimension of attribute vector
            kwargs
        '''
        self.device = device

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.attr_dim = attr_dim

        self.n_critic = kwargs.get('n_critic', 5)
        self.lmbda = kwargs.get('lmbda', 10.0)
        self.beta = kwargs.get('beta', 0.01)
        self.bs = kwargs.get('batch_size', 32)#Here, must be 128?
        self.dataset = kwargs.get('dataset')

        self.gzsl = kwargs.get('gzsl', False)
        self.n_train = kwargs.get('n_train')
        self.n_test = kwargs.get('n_test')
        if self.gzsl:
            self.n_test = self.n_train + self.n_test

        self.Z_dist = normal.Normal(0, 1)
        self.z_shape = torch.Size([self.bs, self.z_dim])

        self.net_G = Generator(self.z_dim, self.attr_dim).to(self.device)
        self.optim_G = optim.Adam(self.net_G.parameters(), lr=1e-4)

        #classifier for judging the output of generator, currently not used anywhere. Eventually below classifier will be removed.
        self.classifier = MLPClassifier(
            self.x_dim + self.attr_dim, self.n_train
        ).to(self.device)
        self.optim_cls = optim.Adam(self.classifier.parameters(), lr=1e-4)

        #f, the attribute classifier.
        self.att_classifier = MLPClassifier(
            self.x_dim, self.attr_dim
        ).to(self.device)
        self.optim_f = optim.Adam(self.att_classifier.parameters(), lr=1e-4)

        # Final classifier trained on augmented data for GZSL
        self.final_classifier = MLPClassifier(
            self.x_dim + self.attr_dim, self.n_test
        ).to(self.device)
        self.optim_final_cls = optim.Adam(self.final_classifier.parameters(), lr=1e-4)

        self.criterion_cls = nn.CrossEntropyLoss()

        self.model_save_dir = "saved_models"
        #kaggle edit
        os.chdir("/kaggle/working/")
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

    def get_conditional_input(self, X, C_Y):
        new_X = torch.cat([X, C_Y], dim=1).float()
        return autograd.Variable(new_X).to(self.device)

    def get_cosine_similarity(self, X, X_syn):
        cosine_sim = nn.CosineSimilarity()
        output = cosine_sim(X , X_syn)
        return 1 - output

    def fit_classifier(self, img_features, label_attr, label_idx):
        '''
        Train the classifier in supervised manner on a single
        minibatch of available data
        Args:
            img         : bs X 2048
            label_attr  : bs X 85
            label_idx   : bs
        Returns:
            loss for the minibatch
        '''
        img_features = autograd.Variable(img_features).to(self.device)
        label_attr = autograd.Variable(label_attr).to(self.device)
        label_idx = autograd.Variable(label_idx).to(self.device)

        X_inp = self.get_conditional_input(img_features, label_attr)
        Y_pred = self.classifier(X_inp)

        self.optim_cls.zero_grad()
        loss = self.criterion_cls(Y_pred, label_idx)
        loss.backward()
        self.optim_cls.step()

        return loss.item()

    #Here, change name of function to fit_f_and_g
    def fit_fg(self, img_features, label_attr, label_idx, use_cls_loss=True):
        ''' Bs, Bu, Breal are generated samples from seen classes, generated samples from unseen classes, and
        input sampled directly from provided data, respectively'''
        L_gen = 0
        L_p =0

        img_features_Bs, label_attr_Bs, label_idx_Bs = self.create_Bs(img_features, label_attr, label_idx)
        dset= self.dataset
        train_dataset = ZSLDataset(dset, self.n_train, self.n_test, self.gzsl)
        img_features_Bu, label_attr_Bu, label_idx_Bu = self.create_Bu(train_dataset.test_classmap, train_dataset.attributes)

        img_features = autograd.Variable(img_features.float()).to(self.device)
        label_attr = autograd.Variable(label_attr.float()).to(self.device)
        label_idx = label_idx.to(self.device)

        img_features_Bs = autograd.Variable(img_features_Bs.float()).to(self.device)
        label_attr_Bs = autograd.Variable(label_attr_Bs.float()).to(self.device)
        label_idx_Bs = label_idx_Bs.to(self.device)

        img_features_Bu = autograd.Variable(img_features_Bu.float()).to(self.device)
        label_attr_Bu = autograd.Variable(label_attr_Bu.float()).to(self.device)
        label_idx_Bu = label_idx_Bu.to(self.device)

        z = uniform.Uniform(0,1)
        z = z.sample(torch.Size([1]))
        a= gauss(self.bs, m=0, s= 1)
        b= gauss(self.bs, m=0, s= 1)
        if z <= 0.9:
            #to avoid divide by zero error
            img_features_Bs_corrected = img_features_Bs+1e-8
            img_features_corrected = img_features+1e-8
            #to avoid 'cant call numpy on tensor that requires grad error'
            img_features_Bs_corrected = img_features_Bs_corrected.cpu().detach().numpy()
            img_features_corrected = img_features_corrected.cpu().detach().numpy()

            C = ot.gpu.dist(img_features_Bs_corrected, img_features_corrected)
            T = ot.gpu.sinkhorn(a, b, C, 0.5)
            C= torch.FloatTensor(C).to(self.device).requires_grad_()
            T = torch.FloatTensor(T).to(self.device).requires_grad_()
        else:
            label_idx_list = label_idx.tolist()
            label_idx_Bs_list = label_idx_Bs.tolist()
            T = []
            for i in range(self.bs):
                T_row = []
                for j in range(self.bs):
                    if label_idx[i] == label_idx_Bs[j]:
                        a_n = label_idx_list.count(label_idx[i])
                        a_m = label_idx_Bs_list.count(label_idx_Bs[j])
                        Tij = 1/a_n*a_m
                        T_row.append(Tij)
                    else:
                        Tij = 0
                        T_row.append(Tij)
                T.append(T_row)

            #to avoid divide by zero error
            img_features_Bs_corrected = img_features_Bs+1e-8
            img_features_corrected = img_features+1e-8
            #to avoid 'cant call numpy on tensor that requires grad error'
            img_features_Bs_corrected = img_features_Bs_corrected.cpu().detach().numpy()
            img_features_corrected = img_features_corrected.cpu().detach().numpy()
            #using metric = cosine in ot.gpu.dist() gives NotImplementedError. So sticking with euclidean for now
            C = ot.gpu.dist(img_features_Bs_corrected, img_features_corrected)
            C = torch.FloatTensor(C).to(self.device).requires_grad_()
            T = torch.FloatTensor(T).to(self.device).requires_grad_()

        L_gen = torch.mm(T,C).trace()

        #Here, might have to remove the if statement altogether or replace with say,"if  use_attr_reg:" or something like that
        if use_cls_loss:
            self.classifier.eval()
            A_f = self.att_classifier(img_features)

            Z = self.Z_dist.sample(self.z_shape).to(self.device)
            Z = self.get_conditional_input(Z, label_attr)
            X_gen = self.net_G(Z)
            A_f_hat = self.att_classifier(X_gen)

            label_attr_corrected = label_attr.detach().cpu().numpy()
            A_f_corrected = A_f.detach().cpu().numpy()
            A_f_hat_corrected = A_f_hat.detach().cpu().numpy()

            d = ot.dist(label_attr_corrected ,A_f_corrected, metric = 'cosine')
            d_hat = ot.dist(label_attr_corrected ,A_f_hat_corrected, metric = 'cosine')

            d = torch.FloatTensor(d).to(self.device)
            d_hat = torch.FloatTensor(d_hat).to(self.device)

            #Y_pred = F.softmin(0.5*self.classifier(d), dim=0) #0.5 is gamma^2. Changed this to below:
            Y_pred = F.softmin(0.5*d, dim=0) #0.5 is gamma^2.
            Y_pred_hat = F.softmin(0.5*d_hat, dim=0)

            log_prob = torch.log(torch.gather(Y_pred, 1, label_idx.unsqueeze(1)))
            log_prob_hat = torch.log(torch.gather(Y_pred_hat, 1, label_idx.unsqueeze(1)))

            L_p = -1 * torch.mean(log_prob) -1 * torch.mean(log_prob_hat)
            L_gen += self.beta * L_p

        self.optim_f.zero_grad()
        L_gen.backward()
        self.optim_f.step()

        self.optim_G.zero_grad()
        L_gen.backward()
        self.optim_G.step()

        return L_p.item() , L_gen.item()

    def fit_final_classifier(self, img_features, label_attr, label_idx):
        img_features = autograd.Variable(img_features.float()).to(self.device)
        label_attr = autograd.Variable(label_attr.float()).to(self.device)
        label_idx = label_idx.to(self.device)

        X_inp = self.get_conditional_input(img_features, label_attr)
        Y_pred = self.final_classifier(X_inp)

        self.optim_final_cls.zero_grad()
        loss = self.criterion_cls(Y_pred, label_idx)
        loss.backward()
        self.optim_final_cls.step()

        return loss.item()

    def create_syn_dataset(self, test_labels, attributes, seen_dataset, n_examples=400):# Here, n_examples should be 1 I think.
        '''
        Creates a synthetic dataset based on attribute vectors of unseen class
        Args:
            test_labels: A dict with key as original serial number in provided
                dataset and value as the index which is predicted during
                classification by network
            attributes: A np array containing class attributes for each class
                of dataset
            seen_dataset: A list of 3-tuple (x, orig_label, y) where x belongs to one of the
                seen classes and y is classification label. Used for generating
                latent representations of seen classes in GZSL
            n_samples: Number of samples of each unseen class to be generated(Default: 400)
        Returns:
            A list of 3-tuple (z, _, y) where z is latent representations and y is
        '''
        syn_dataset = []
        for test_cls, idx in test_labels.items():
            attr = attributes[test_cls - 1]
            z = self.Z_dist.sample(torch.Size([n_examples, self.z_dim]))
            c_y = torch.stack([torch.FloatTensor(attr) for _ in range(n_examples)]) #here we have taking n_examples number of attribute vectors from the same class(so all rows are same)

            z_inp = self.get_conditional_input(z, c_y)
            X_gen = self.net_G(z_inp)

            syn_dataset.extend([(X_gen[i], test_cls, idx) for i in range(n_examples)])

        if seen_dataset is not None:
            syn_dataset.extend(seen_dataset)

        return syn_dataset

    #the attributes and generates data for each of them
    def create_Bs(self, img_features, label_attr, label_idx):
        z = self.Z_dist.sample(self.z_shape)
        x_cond = self.get_conditional_input(z, label_attr)
        x_gen = self.net_G(x_cond)
        return x_gen, label_attr, label_idx

    def create_Bu(self, test_labels, attributes):
        attr =[]
        attr_temp = []
        idx_temp = []
        for test_cls, idx in test_labels.items():
            attr_temp.append(attributes[test_cls - 1])
            idx_temp.append(idx)
        attr.append(attr_temp)
        attr.append(idx_temp)
        if len(attr)<self.bs:
            idx_list =[]
            att_list = []
            for _ in range(self.bs):
                rand_int = random.choice(range(len(attr[0])))
                att_list.append(attr[0][rand_int])
                idx_list.append(attr[1][rand_int])
            att_batch = torch.FloatTensor(att_list)
            idx_batch = torch.FloatTensor(idx_list)
        else:
            idx_list =[]
            att_list = []
            for _ in range(self.bs):
                rand_int = random.sample(range(len(attr[0])), self.bs)
                att_list.append(attr[0][rand_int])
                idx_list.append(attr[1][rand_int])
            att_batch = torch.FloatTensor(att_list)
            idx_batch = torch.FloatTensor(idx_list)

        #above lines were to get the attribute matrix and index. Below we'll generate their corresponding features.
        z = self.Z_dist.sample(self.z_shape)
        x_cond = self.get_conditional_input(z,att_batch)
        x_gen = self.net_G(x_cond)
        return x_gen, att_batch, idx_batch


    def test(self, data_generator, pretrained=False):
        if pretrained:
            model = self.classifier
        else:
            model = self.final_classifier

        # eval mode
        model.eval()
        batch_accuracies = []
        for idx, (img_features, label_attr, label_idx) in enumerate(data_generator):
            img_features = img_features.to(self.device)
            label_attr = label_attr.to(self.device)

            X_inp = self.get_conditional_input(img_features, label_attr)
            with torch.no_grad():
                Y_probs = model(X_inp)
            _, Y_pred = torch.max(Y_probs, dim=1)

            Y_pred = Y_pred.cpu().numpy()
            Y_real = label_idx.cpu().numpy()

            acc = accuracy_score(Y_pred, Y_real)
            batch_accuracies.append(acc)
        return np.mean(batch_accuracies)

    def save_model(self, model=None):
        if "disc_classifier" in model:
            ckpt_path = os.path.join(self.model_save_dir, model + ".pth")
            torch.save(self.classifier.state_dict(), ckpt_path)

        elif "f_and_g" in model:
            dset_name = model.split('_')[0]
            g_ckpt_path = os.path.join(self.model_save_dir, "%s_generator.pth" % dset_name)
            torch.save(self.net_G.state_dict(), g_ckpt_path)

            d_ckpt_path = os.path.join(self.model_save_dir, "%s_attribute_classifier.pth" % dset_name)
            torch.save(self.net_D.state_dict(), d_ckpt_path)

        elif "final_classifier" in model:
            ckpt_path = os.path.join(self.model_save_dir, model + ".pth")
            torch.save(self.final_classifier.state_dict(), ckpt_path)

        else:
            raise Exception("Trying to save unknown model: %s" % model)

    def load_model(self, model=None):
        if "disc_classifier" in model:
            ckpt_path = os.path.join(self.model_save_dir, model + ".pth")
            if os.path.exists(ckpt_path):
                self.classifier.load_state_dict(torch.load(ckpt_path))
                return True

        elif "f_and_g" in model:
            f1, f2 = False, False
            dset_name = model.split('_')[0]
            g_ckpt_path = os.path.join(self.model_save_dir, "%s_generator.pth" % dset_name)
            if os.path.exists(g_ckpt_path):
                self.net_G.load_state_dict(torch.load(g_ckpt_path))
                f1 = True

            d_ckpt_path = os.path.join(self.model_save_dir, "%s_attribute_classifier.pth" % dset_name)
            if os.path.exists(d_ckpt_path):
                self.att_classifier.load_state_dict(torch.load(d_ckpt_path))
                f2 = True

            return f1 and f2

        elif "final_classifier" in model:
            ckpt_path = os.path.join(self.model_save_dir, model + ".pth")
            if os.path.exists(ckpt_path):
                self.final_classifier.load_state_dict(torch.load(ckpt_path))
                return True

        else:
            raise Exception("Trying to load unknown model: %s" % model)

        return False
