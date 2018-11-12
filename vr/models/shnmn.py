import numpy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vr.models.layers import init_modules, ResidualBlock, SimpleVisualBlock, GlobalAveragePool, Flatten
from vr.models.layers import build_classifier, build_stem, ConcatBlock
import vr.programs

from torch.nn.init import kaiming_normal, kaiming_uniform, xavier_uniform, xavier_normal, constant, uniform

from vr.models.filmed_net import FiLM, FiLMedResBlock, ConcatFiLMedResBlock, coord_map
from functools import partial


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _random_tau(num_modules):
    tau_0 = torch.zeros(num_modules, num_modules+1)
    tau_1 = torch.zeros(num_modules, num_modules+1)
    xavier_uniform(tau_0)
    xavier_uniform(tau_1)
    return tau_0, tau_1


def _chain_tau():
    tau_0 = torch.zeros(3, 4)
    tau_1 = torch.zeros(3, 4)
    tau_0[0][1] = tau_1[0][0] = 100 #1st block - lhs inp img, rhs inp sentinel
    tau_0[1][2] = tau_1[1][0] = 100 #2nd block - lhs inp 1st block, rhs inp sentinel
    tau_0[2][3] = tau_1[2][0] = 100 #3rd block - lhs inp 2nd block, rhs inp sentinel
    return tau_0, tau_1

def _chain_with_shortcuts_tau():
    tau_0 = torch.zeros(3, 4)
    tau_1 = torch.zeros(3, 4)
    tau_0[0][1] = tau_1[0][0] = 100 #1st block - lhs inp img, rhs inp sentinel
    tau_0[1][2] = tau_1[1][1] = 100 #2nd block - lhs inp 1st block, rhs img
    tau_0[2][3] = tau_1[2][1] = 100 #3rd block - lhs inp 2nd block, rhs img 
    return tau_0, tau_1


def _tree_tau():
    tau_0 = torch.zeros(3, 4)
    tau_1 = torch.zeros(3, 4)
    tau_0[0][1] = tau_1[0][0] = 100 #1st block - lhs inp img, rhs inp sentinel
    tau_0[1][1] = tau_1[1][0] = 100 #2st block - lhs inp img, rhs inp sentinel
    tau_0[2][2] = tau_1[2][3] = 100 #3rd block - lhs inp 1st block, rhs inp 2nd block
    return tau_0, tau_1


def correct_alpha_init_xyr(alpha):
    alpha.zero_()
    alpha[0][0] = 100
    alpha[1][2] = 100
    alpha[2][1] = 100

    return alpha

def correct_alpha_init_rxy(alpha, use_stopwords=True):
    alpha.zero_()
    alpha[0][1] = 100
    alpha[1][0] = 100
    alpha[2][2] = 100

    return alpha

def correct_alpha_init_xry(alpha, use_stopwords=True):
    alpha.zero_()
    alpha[0][0] = 100
    alpha[1][1] = 100
    alpha[2][2] = 100

    return alpha

def _shnmn_func(question, img, num_modules, alpha, tau_0, tau_1, func):
    sentinel = torch.zeros_like(img) # B x 1 x C x H x W
    h_prev = torch.cat([sentinel, img], dim=1) # B x 2 x C x H x W

    for i in range(num_modules):
        alpha_curr = F.softmax(alpha[i], dim=0)
        tau_0_curr = F.softmax(tau_0[i, :(i+2)], dim=0)
        tau_1_curr = F.softmax(tau_1[i, :(i+2)], dim=0)

        question_rep = torch.sum(alpha_curr.view(1,-1,1)*question, dim=1) #(B,D)
        # B x C x H x W
        lhs_rep = torch.sum(tau_0_curr.view(1, (i+2), 1, 1, 1)*h_prev, dim=1)
        # B x C x H x W
        rhs_rep = torch.sum(tau_1_curr.view(1, (i+2), 1, 1, 1)*h_prev, dim=1)
        h_i = func(question_rep, lhs_rep, rhs_rep) # B x C x H x W

        h_prev = torch.cat([h_prev, h_i.unsqueeze(1)], dim=1)

    return h_prev


class FindModule(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding = kernel_size // 2)

    def forward(self, question_rep, lhs_rep, rhs_rep):
        out = F.relu(self.conv_1(torch.cat([lhs_rep, rhs_rep], 1))) # concat along depth
        question_rep = question_rep.view(-1, self.dim, 1, 1)
        return F.relu(self.conv_2(out*question_rep))

class ResidualFunc:
    def __init__(self, dim, kernel_size):
        self.dim = dim
        self.kernel_size = kernel_size

    def __call__(self, question_rep, lhs_rep, rhs_rep):
        cnn_weight_dim = self.dim * self.dim * self.kernel_size * self.kernel_size
        cnn_bias_dim = self.dim
        proj_cnn_weight_dim = 2 * self.dim * self.dim
        proj_cnn_bias_dim = self.dim
        if (question_rep.size(1) !=
              proj_cnn_weight_dim + proj_cnn_bias_dim
              + 2 * (cnn_weight_dim + cnn_bias_dim)):
            raise ValueError

        # pick out CNN and projection CNN weights/biases
        cnn1_weight = question_rep[:,:cnn_weight_dim]
        cnn2_weight = question_rep[:,cnn_weight_dim:2 * cnn_weight_dim]
        cnn1_bias = question_rep[:,2 * cnn_weight_dim:(2 * cnn_weight_dim) + cnn_bias_dim]
        cnn2_bias = question_rep[:,(2 * cnn_weight_dim) + cnn_bias_dim:2 * (cnn_weight_dim + cnn_bias_dim)]
        proj_weight = question_rep[:, 2 * (cnn_weight_dim + cnn_bias_dim) :
                                  2 * (cnn_weight_dim + cnn_bias_dim) + proj_cnn_weight_dim]
        proj_bias   = question_rep[:, 2*(cnn_weight_dim + cnn_bias_dim) + proj_cnn_weight_dim:]

        cnn_out_total = []
        bs = question_rep.size(0)

        for i in range(bs):
            cnn1_weight_curr = cnn1_weight[i].view(self.dim, self.dim, self.kernel_size, self.kernel_size)
            cnn1_bias_curr   = cnn1_bias[i]
            cnn2_weight_curr = cnn2_weight[i].view(self.dim, self.dim, self.kernel_size, self.kernel_size)
            cnn2_bias_curr   = cnn2_bias[i]

            proj_weight_curr = proj_weight[i].view(self.dim, 2*self.dim, 1, 1)
            proj_bias_curr = proj_bias[i]

            cnn_inp = F.relu(F.conv2d(torch.cat([lhs_rep[[i]], rhs_rep[[i]]], 1),
                               proj_weight_curr,
                               bias=proj_bias_curr, padding=0))

            cnn1_out = F.relu(F.conv2d(cnn_inp, cnn1_weight_curr, bias=cnn1_bias_curr, padding=self.kernel_size // 2))
            cnn2_out = F.conv2d(cnn1_out, cnn2_weight_curr, bias=cnn2_bias_curr,padding=self.kernel_size // 2)

            cnn_out_total.append(F.relu(cnn_inp + cnn2_out) )

        return torch.cat(cnn_out_total)



class ConvFunc:
    def __init__(self, dim, kernel_size):
        self.dim = dim
        self.kernel_size = kernel_size

    def __call__(self, question_rep, lhs_rep, rhs_rep):
        cnn_weight_dim = self.dim*self.dim*self.kernel_size*self.kernel_size
        cnn_bias_dim = self.dim
        proj_cnn_weight_dim = 2*self.dim*self.dim
        proj_cnn_bias_dim = self.dim
        if (question_rep.size(1) !=
              proj_cnn_weight_dim + proj_cnn_bias_dim
              + cnn_weight_dim + cnn_bias_dim):
            raise ValueError

        # pick out CNN and projection CNN weights/biases
        cnn_weight = question_rep[:, : cnn_weight_dim]
        cnn_bias = question_rep[:, cnn_weight_dim : cnn_weight_dim + cnn_bias_dim]
        proj_weight = question_rep[:, cnn_weight_dim+cnn_bias_dim :
                                  cnn_weight_dim+cnn_bias_dim+proj_cnn_weight_dim]
        proj_bias   = question_rep[:, cnn_weight_dim+cnn_bias_dim+proj_cnn_weight_dim:]

        cnn_out_total = []
        bs = question_rep.size(0)

        for i in range(bs):
            cnn_weight_curr = cnn_weight[i].view(self.dim, self.dim, self.kernel_size, self.kernel_size)
            cnn_bias_curr   = cnn_bias[i]
            proj_weight_curr = proj_weight[i].view(self.dim, 2*self.dim, 1, 1)
            proj_bias_curr = proj_bias[i]

            cnn_inp = F.conv2d(torch.cat([lhs_rep[[i]], rhs_rep[[i]]], 1),
                               proj_weight_curr,
                               bias=proj_bias_curr, padding=0)
            cnn_out_total.append(F.relu(F.conv2d(
                cnn_inp, cnn_weight_curr, bias=cnn_bias_curr, padding=self.kernel_size // 2)))

        return torch.cat(cnn_out_total)

INITS = {'xavier_uniform' : xavier_uniform,
         'constant' : constant,
         'uniform' : uniform,
         'correct' : correct_alpha_init_xyr,
         'correct_xry' : correct_alpha_init_xry,
         'correct_rxy' : correct_alpha_init_rxy}

class SHNMN(nn.Module):
    def __init__(self,
        vocab,
        feature_dim,
        module_dim,
        module_kernel_size,
        stem_dim,
        stem_num_layers,
        stem_subsample_layers,
        stem_kernel_size,
        stem_padding,
        stem_batchnorm,
        classifier_fc_layers,
        classifier_proj_dim,
        classifier_downsample,classifier_batchnorm,
        num_modules,
        hard_code_alpha=False,
        hard_code_tau=False,
        tau_init='random',
        alpha_init='xavier_uniform',
        model_type ='soft',
        model_bernoulli=0.5,
        use_module = 'conv',
        use_stopwords = True,
        **kwargs):

        super().__init__()
        self.num_modules = num_modules
        # alphas and taus from Overleaf Doc.
        self.hard_code_alpha = hard_code_alpha
        self.hard_code_tau = hard_code_tau

        num_question_tokens = 3

        if alpha_init.startswith('correct'):
            print('using correct initialization')
            alpha = INITS[alpha_init](torch.Tensor(num_modules, num_question_tokens))
        elif alpha_init == 'constant':
            alpha = INITS[alpha_init](torch.Tensor(num_modules, num_question_tokens), 1)
        else:
            alpha = INITS[alpha_init](torch.Tensor(num_modules, num_question_tokens))
        print('initial alpha ')
        print(alpha)


        if hard_code_alpha:
            assert(alpha_init.startswith('correct'))

            self.alpha = Variable(alpha)
            self.alpha = self.alpha.to(device)
        else:
            self.alpha = nn.Parameter(alpha)


        # create taus
        if tau_init == 'tree':
            tau_0, tau_1 = _tree_tau()
            print("initializing with tree.")
        elif tau_init == 'chain':
            tau_0, tau_1 = _chain_tau()
            print("initializing with chain")
        elif tau_init == 'chain_with_shortcuts':
            tau_0, tau_1 = _chain_with_shortcuts_tau() 
            print("initializing with chain and shortcuts")

        else:
            tau_0, tau_1 = _random_tau(num_modules)

        if hard_code_tau:
            assert(tau_init in ['chain', 'tree', 'chain_with_shortcuts'])
            self.tau_0 = Variable(tau_0)
            self.tau_1 = Variable(tau_1)
            self.tau_0 = self.tau_0.to(device)
            self.tau_1 = self.tau_1.to(device)
        else:
            self.tau_0   = nn.Parameter(tau_0)
            self.tau_1   = nn.Parameter(tau_1)



        if use_module == 'conv':
            embedding_dim_1 = module_dim + (module_dim*module_dim*module_kernel_size*module_kernel_size)
            embedding_dim_2 = module_dim + (2*module_dim*module_dim)

            question_embeddings_1 = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_1)
            question_embeddings_2 = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_2)

            stdv_1 = 1. / math.sqrt(module_dim*module_kernel_size*module_kernel_size)
            stdv_2 = 1. / math.sqrt(2*module_dim)

            question_embeddings_1.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_2.weight.data.uniform_(-stdv_2, stdv_2)
            self.question_embeddings = nn.Embedding(len(vocab['question_idx_to_token']), embedding_dim_1+embedding_dim_2)
            self.question_embeddings.weight.data = torch.cat([question_embeddings_1.weight.data,
                                                              question_embeddings_2.weight.data],dim=-1)

            self.func = ConvFunc(module_dim, module_kernel_size)

        elif use_module == 'residual':
            embedding_dim_1 = module_dim + (module_dim*module_dim*module_kernel_size*module_kernel_size)
            embedding_dim_2 = module_dim + (2*module_dim*module_dim)

            question_embeddings_a = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_1)
            question_embeddings_b = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_1)
            question_embeddings_2 = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_2)

            stdv_1 = 1. / math.sqrt(module_dim*module_kernel_size*module_kernel_size)
            stdv_2 = 1. / math.sqrt(2*module_dim)

            question_embeddings_a.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_b.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_2.weight.data.uniform_(-stdv_2, stdv_2)
            self.question_embeddings = nn.Embedding(len(vocab['question_idx_to_token']), 2*embedding_dim_1+embedding_dim_2)
            self.question_embeddings.weight.data = torch.cat([question_embeddings_a.weight.data, question_embeddings_b.weight.data,
                                                              question_embeddings_2.weight.data],dim=-1)
            self.func = ResidualFunc(module_dim, module_kernel_size)

        else:
            self.question_embeddings = nn.Embedding(len(vocab['question_idx_to_token']), module_dim)
            self.func = FindModule(module_dim, module_kernel_size)


        # stem for processing the image into a 3D tensor
        self.stem = build_stem(feature_dim[0], stem_dim, module_dim,
                   num_layers=stem_num_layers,
                   subsample_layers=stem_subsample_layers,
                   kernel_size=stem_kernel_size,
                   padding=stem_padding,
                   with_batchnorm=stem_batchnorm)

        tmp = self.stem(Variable(torch.zeros([1, feature_dim[0], feature_dim[1], feature_dim[2]])))
        module_H = tmp.size(2)
        module_W = tmp.size(3)
        num_answers = len(vocab['answer_idx_to_token'])
        self.classifier = build_classifier(module_dim, module_H, module_W, num_answers,
                  classifier_fc_layers,
                  classifier_proj_dim,
                  classifier_downsample,
                  with_batchnorm=classifier_batchnorm)

        self.model_type = model_type
        self.use_module = use_module
        p = model_bernoulli
        tree_odds = -numpy.log((1 - p) / p)
        self.tree_odds = nn.Parameter(torch.Tensor([tree_odds]))


    def forward_hard(self, image, question):
        question = self.question_embeddings(question)
        stemmed_img = self.stem(image).unsqueeze(1) # B x 1 x C x H x W

        chain_tau_0, chain_tau_1 = _chain_tau()
        chain_tau_0 = chain_tau_0.to(device)
        chain_tau_1 = chain_tau_1.to(device)
        h_chain = _shnmn_func(question, stemmed_img,
                        self.num_modules, self.alpha,
                        Variable(chain_tau_0), Variable(chain_tau_1), self.func)
        h_final_chain = h_chain[:, -1, :, :, :]
        tree_tau_0, tree_tau_1 = _tree_tau()
        tree_tau_0 = tree_tau_0.to(device)
        tree_tau_1 = tree_tau_1.to(device)
        h_tree  = _shnmn_func(question, stemmed_img,
                        self.num_modules, self.alpha,
                        Variable(tree_tau_0), Variable(tree_tau_1), self.func)
        h_final_tree = h_tree[:, -1, :, :, :]

        p_tree = torch.sigmoid(self.tree_odds[0])
        self.tree_scores = self.classifier(h_final_tree)
        self.chain_scores = self.classifier(h_final_chain)
        output_probs_tree  = F.softmax(self.tree_scores, dim=1)
        output_probs_chain = F.softmax(self.chain_scores, dim=1)
        probs_mixture = p_tree * output_probs_tree + (1.0 - p_tree) * output_probs_chain
        eps = 1e-6
        probs_mixture = (1 - eps) * probs_mixture + eps
        return torch.log(probs_mixture)


    def forward_soft(self, image, question):
        question = self.question_embeddings(question)
        stemmed_img = self.stem(image).unsqueeze(1) # B x 1 x C x H x W

        self.h = _shnmn_func(question, stemmed_img, self.num_modules,
                              self.alpha, self.tau_0, self.tau_1, self.func)
        h_final = self.h[:, -1, :, :, :]
        return self.classifier(h_final)

    def forward(self, image, question):
        if self.model_type == 'hard':
            return self.forward_hard(image, question)
        else:
            return self.forward_soft(image, question)
