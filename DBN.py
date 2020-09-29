######################################################################################################
### Edited by Troy Cao
### Time: 2020.08.07
### Title: Restricted Boltzmann Machine
### Description: As a fundanmental realization of DBN; Support the Project of LWB
### Email: troycao777@gmail.com
######################################################################################################

import torch
import torch.nn as nn
from RBM import RBM

class DBN(nn.Module):

    def __init__(self, visible_unit=256, hidden_unit=[64, 100], k=2, learning_rate=1e-5,
                 learning_rate_decay=False, xavier_init=False, use_gpu=True):
        super(DBN, self).__init__()

        self.use_gpu = use_gpu
        self.n_layer = len(hidden_unit)
        self.rbm_layers = []
        self.rbm_nodes = []

        for i in range(self.n_layer):
            input_size = 0
            if i == 0:
                input_size = visible_unit
            else:
                input_size = hidden_unit[i-1]
            rbm = RBM(visible_unit=input_size, hidden_unit=hidden_unit[i], k=k, learning_rate=learning_rate,
                      learning_rate_decay=learning_rate_decay, xavier_init=False, use_gpu=True)
            self.rbm_layers.append(rbm)

        self.W_rec = [nn.Parameter(self.rbm_layers[i].W.data.clone()) for i in range(self.n_layer-1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].W.data) for i in range(self.n_layer-1)]
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].h_bias.data.clone()) for i in range(self.n_layer-1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].v_bias.data) for i in range(self.n_layer-1)]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].W.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].h_bias.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].v_bias.data)

        for i in range(self.n_layer-1):
            self.register_parameter('W_rec{}'.format(i), self.W_rec[i])
            self.register_parameter('W_gen{}'.format(i), self.W_gen[i])
            self.register_parameter('bias_rec{}'.format(i), self.bias_rec[i])
            self.register_parameter('bias_gen{}'.format(i), self.bias_gen[i])

    def forward(self, input_data):
        v = input_data
        for i in range(len(self.rbm_layers)):
            v = v.view((v.shape[0], -1)).type(torch.FloatTensor) # Flatten the input data
            p_v, v = self.rbm_layers[i].to_hidden(v)

        return p_v, v

    def reconstruct(self, input_data):
        h = input_data
        p_h = 0
        for i in range(len(self.rbm_layers)):
            h = h.view((h.shape[0], -1)).type(torch.FloatTensor)
            if self.use_gpu:
                h = h.cuda()
            p_h, h = self.rbm_layers[i].to_hidden(h)

        v = h
        for i in range(len(self.rbm_layers)-1, -1, -1):
            v = v.view((v.shape[0], -1)).type(torch.FloatTensor)
            if self.use_gpu:
                v = v.cuda()
            p_v, v = self.rbm_layers[i].to_visible(v)
        return p_v, v

    def train_stat(self, train_data, train_labels, num_epoch=20, batch_size=100):

        tmp = train_data

        for i in range(len(self.rbm_layers)):
            print("-"*20)
            print('Training the {} rbm layer'.format(i+1))

            tensor_x = tmp.type(torch.FloatTensor)
            tensor_y = train_labels.type(torch.FloatTensor)

            _dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
            _dataloader = torch.utils.data.DataLoader(_dataset, batch_size=batch_size, drop_last=True)

            self.rbm_layers[i].train(_dataloader, num_epoch, batch_size)
            v = tmp.view((tmp.shape[0], -1)).type(torch.FloatTensor)
            if self.use_gpu:
                v = v.cuda()
            p_v, v = self.rbm_layers[i].forward(v)
            tmp = v
        return p_v

    def train_ith(self, train_data, train_labels, num_epoch, batch_size, ith_layer):

        if (ith_layer - 1) > len(self.rbm_layers) or ith_layer <=0:
            print('Layer index out of range!')
            return

        ith_layer = ith_layer - 1
        v = train_data.view((train_data.shape[0], -1)).type(torch.FloatTensor)

        for ith in range(ith_layer):
            p_v, v = self.rbm_layers[ith].foward(v)

        tmp = v
        tensor_x = tmp.type(torch.FloatTensor)
        tensor_y = train_labels.type(torch.FloatTensor)
        _dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        _dataloader = torch.utils.data.DataLoader(_dataset, batch_size=batch_size)
        self.rbm_layers[ith_layer].train(_dataloader, num_epoch, batch_size)
        return





