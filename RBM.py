######################################################################################################
### Edited by Troy Cao
### Time: 2020.08.05
### Title: Restricted Boltzmann Machine
### Description: As a fundanmental realization of DBN; Support the Project of LWB
### Email: troycao777@gmail.com
######################################################################################################

import torch
import torch.nn as nn
from tqdm import tqdm


class RBM(nn.Module):

    def __init__(self, visible_unit, hidden_unit, k, learning_rate, learning_rate_decay, xavier_init=False, use_gpu=True,
                 batch_size=16):
        super(RBM, self).__init__()

        self.visible_unit = visible_unit
        self.hidden_unit = hidden_unit
        self.k = k
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.xavier_init = xavier_init
        self.use_gpu = use_gpu
        self.batch_size = batch_size

        if not self.xavier_init:
            self.W = torch.randn(self.visible_unit, self.hidden_unit) * 0.01
        else:
            self.xavier_value = torch.sqrt(torch.FloatTensor([1.0 / (self.visible_unit + self.hidden_unit)]))
            self.W = -self.xavier_value + torch.randn(self.visible_unit, self.hidden_unit) * (2 * self.xavier_value)

        self.h_bias = torch.zeros(self.hidden_unit)
        self.v_bias = torch.zeros(self.visible_unit)

        if self.use_gpu:
            self.W = self.W.cuda()
            self.h_bias = self.h_bias.cuda()
            self.v_bias = self.v_bias.cuda()

    def to_hidden(self, X): #Convert the data in Visible layer to the Hidden layer also does sampling;
        X_porb = torch.matmul(X, self.W)
        X_prob = torch.add(X_porb, self.h_bias)
        X_prob = torch.sigmoid(X_prob)

        sample_X_prob = self.sampling(X_prob)

        return X_prob, sample_X_prob

    def to_visible(self, X): # reconstruct data from hidden layer also does sampling
        X_prob = torch.matmul(X, self.W.transpose(0, 1))
        X_prob = torch.add(X_prob, self.v_bias)
        X_prob = torch.sigmoid(X_prob)

        sample_X_prob = self.sampling(X_prob)

        return X_prob, sample_X_prob

    def sampling(self, prob): # Bernoulli sampling based on probabilities s
        s = torch.distributions.Bernoulli(prob).sample()
        return s

    def reconstruct_error(self, data):
        return self.contrastive_divergence(data, False)

    def reconstruct(self, X, n_gibbs):
        v = X
        if self.use_gpu:
            v = v.cuda()
        for i in range(n_gibbs):
            prob_h_, h = self.to_hidden(v)
            prob_v_, v = self.to_visible(prob_h_)
        return prob_v_, v

    def contrastive_divergence(self, input_data, training=True, n_gibbs_sampling_steps=1, lr=0.001):
        if self.use_gpu:
            input_data.cuda()
        positive_hidden_probabilities, positive_hidden_act = self.to_hidden(input_data)

        positive_associations = torch.matmul(input_data.t(), positive_hidden_act)

        hidden_activations = positive_hidden_act
        if self.use_gpu:
            hidden_activations.cuda()
        for i in range(n_gibbs_sampling_steps):
            visible_probabilities, _ = self.to_visible(hidden_activations)  # v_dim * v_dim
            hidden_probabilities, hidden_activations = self.to_hidden(visible_probabilities) # v_dim * h_dim

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_association = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        if (training):
            batch_size = self.batch_size

            g = (positive_associations - negative_association)
            grad_update = g / batch_size  # v_dim * h_dim
            v_bias_update = torch.sum(input_data - negative_visible_probabilities, dim=0) / batch_size
            h_bias_update = torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0) / batch_size

            self.W += lr * grad_update
            self.v_bias += lr * v_bias_update
            self.h_bias += lr * h_bias_update

        error = torch.mean(torch.sum((input_data - negative_visible_probabilities)**2, dim=0))
        return error, torch.sum(torch.abs(grad_update))

    def forward(self, input_data):
        return self.to_hidden(input_data)

    def step(self, input_data, epoch):
        n_gibbs_sampling_steps = self.k
        if self.learning_rate_decay:
            lr = self.learning_rate / epoch
        else:
            lr = self.learning_rate

        return self.contrastive_divergence(input_data, True, n_gibbs_sampling_steps, lr)

    def train(self, input_data, num_epochs=20, batch_size=16):
        self.batch_size = batch_size

        if (isinstance(input_data, torch.utils.data.DataLoader)):
            train_loader = input_data
        else:
            train_loader = torch.utils.data.DataLoader(input_data, batch_size=batch_size)

        for epoch in range(1, num_epochs + 1):
            n_batches = int(len(train_loader))
            # print(n_batches)

            cost_ = torch.FloatTensor(n_batches, 1)
            grad_ = torch.FloatTensor(n_batches, 1)

            for i, batch in tqdm(enumerate(train_loader), ascii=True):

                batch = batch.view(len(batch), self.visible_unit)

                if (self.use_gpu):
                    batch = batch.cuda()
                cost_[i - 1], grad_[i - 1] = self.step(batch, epoch)
        print("Epoch:{} finished, avg_cost = {}, std_cost = {}, avg_grad = {}, std_grad = {}".format(num_epochs,
                                                                                                torch.mean(cost_),
                                                                                                torch.std(cost_),
                                                                                                torch.mean(grad_),
                                                                                                torch.std(grad_)))
        return



