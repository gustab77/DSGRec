# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans






class DSGRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DSGRec, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.cf_model = config['cf_model']
        self.n_mm_layer = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.n_hyper_layer = config['n_hyper_layer']
        self.hyper_num = config['hyper_num']
        self.keep_rate = config['keep_rate']
        self.alpha = config['alpha']
        self.cl_weight = config['cl_weight']
        self.reg_weight = config['reg_weight']
        self.beta = config['beta']
        self.tau = 0.2
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.n_nodes = self.n_users + self.n_items

        self.hgnnLayer = HGNNLayer(self.n_hyper_layer)
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))

        # init user and item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.drop = nn.Dropout(p=1 - self.keep_rate)


        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
            self.hyper_image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)

        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
            self.hyper_text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        # load item modal features and define hyperedges embeddings
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.item_image_trs = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))
            self.v_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.hyper_num)))
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.item_text_trs = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))
            self.t_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.hyper_num)))

        # MGCN

        # load dataset info
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        self.num_inters, self.norm_adj = self.get_adj_mat()
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        self.softmax = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        #
        # self.gate_t_hyper_prefer = nn.Sequential(
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.Sigmoid()
        # )
        #
        # self.gate_v_hyper_prefer = nn.Sequential(
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.Sigmoid()
        # )

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(float)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=float)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        data_dict = dict(zip(zip(self.interaction_matrix.row, self.interaction_matrix.col + self.n_users), [1] * self.interaction_matrix.nnz))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1)) + 1e-7
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return sumArr, norm_adj_mat.tocsr()

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)

    # def get_norm_adj_mat(self):
    #     A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
    #     inter_M = self.interaction_matrix
    #     inter_M_t = self.interaction_matrix.transpose()
    #     data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
    #     data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
    #     A._update(data_dict)
    #     # norm adj matrix
    #     sumArr = (A > 0).sum(axis=1)
    #     # add epsilon to avoid Devide by zero Warning
    #     diag = np.array(sumArr.flatten())[0] + 1e-7
    #     diag = np.power(diag, -0.5)
    #     D = sp.diags(diag)
    #     L = D * A * D
    #     # covert norm_adj matrix to tensor
    #     L = sp.coo_matrix(L)
    #     return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size((self.n_nodes, self.n_nodes)))

    # collaborative graph embedding
    def cge(self):
        if self.cf_model == 'mf':
            bd_embs = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            bd_embs = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
                bd_embs += [ego_embeddings]
            bd_embs = torch.stack(bd_embs, dim=1)
            bd_embs = bd_embs.mean(dim=1, keepdim=False)
        return bd_embs

    # nterest-guided Preference Embedding
    def ip(self, str='v'):
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)
        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[:self.n_users]
        # user_feats = self.user_embedding.weight
        ip_feats = torch.concat([user_feats, item_feats], dim=0)
        for _ in range(self.n_mm_layer):
            ip_feats = torch.sparse.mm(self.norm_adj, ip_feats)
        return ip_feats

    def forward(self):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        # Behavior-Guided Purifier
        image_item_embeds = torch.multiply(self.item_id_embedding.weight,
                                           self.gate_v(image_feats))  # image_item_embeds:  torch.Size([7050, 64])
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_feats))

        # hyperedge dependencies constructing
        if self.v_feat is not None:
            iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper)
            uv_hyper = torch.mm(self.adj, iv_hyper)
            iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)
        if self.t_feat is not None:
            it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)
            ut_hyper = torch.mm(self.adj, it_hyper)
            it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)

        # Item-Item View
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # Hypergraph-guided Cooperative Signal Enhancement
        bd_embs = self.cge()

        if self.v_feat is not None and self.t_feat is not None:
            # ip: nterest-guided Preference
            v_feats = self.ip('v')
            t_feats = self.ip('t')
            # Dual-path Embeddings Fusion
            ip_embs = F.normalize(v_feats) + F.normalize(t_feats)
            hnic_embs = bd_embs + ip_embs
            # GHE: global hypergraph embedding
            uv_hyper_embs, iv_hyper_embs = self.hgnnLayer(self.drop(iv_hyper), self.drop(uv_hyper),
                                                          bd_embs[self.n_users:])
            ut_hyper_embs, it_hyper_embs = self.hgnnLayer(self.drop(it_hyper), self.drop(ut_hyper),
                                                          bd_embs[self.n_users:])
            av_hyper_embs = torch.concat([uv_hyper_embs, iv_hyper_embs], dim=0)
            at_hyper_embs = torch.concat([ut_hyper_embs, it_hyper_embs], dim=0)

            # Behavior-Aware Fuser
            # att_common = torch.cat([self.query_common(av_hyper_embs), self.query_common(at_hyper_embs)], dim=-1)
            # weight_common = self.softmax(att_common)
            # common_embeds = weight_common[:, 0].unsqueeze(dim=1) * av_hyper_embs + weight_common[:, 1].unsqueeze(
            #     dim=1) * at_hyper_embs
            # sep_av_hyper_embeds = av_hyper_embs - common_embeds
            # sep_at_hyper_embeds = at_hyper_embs - common_embeds
            #
            # av_hyper_prefer = self.gate_v_hyper_prefer(bd_embs)
            # at_hyper_prefer = self.gate_t_hyper_prefer(bd_embs)
            # sep_av_hyper_embeds = torch.multiply(av_hyper_prefer, sep_av_hyper_embeds)
            # sep_at_hyper_embeds = torch.multiply(at_hyper_prefer, sep_at_hyper_embeds)
            Eh = av_hyper_embs + at_hyper_embs
            # Behavior-aware Modal Signal Augmentation
            att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
            weight_common = self.softmax(att_common)
            common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
                dim=1) * text_embeds
            sep_image_embeds = image_embeds - common_embeds
            sep_text_embeds = text_embeds - common_embeds

            image_prefer = self.image_prefer(bd_embs)
            text_prefer = self.text_prefer(bd_embs)
            sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
            sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
            Ef = (sep_image_embeds + sep_text_embeds + common_embeds) / 3
            # local embeddings + alpha * global embeddings
            all_embs = hnic_embs + self.alpha * F.normalize(Eh) + self.beta * F.normalize(Ef)
        else:
            all_embs = bd_embs

        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)



        return u_embs, i_embs, [uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs], Ef, bd_embs

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss

    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        ua_embeddings, ia_embeddings, hyper_embeddings, Ef, bd_embs = self.forward()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        [uv_embs, iv_embs, ut_embs, it_embs] = hyper_embeddings
        batch_hcl_loss = self.ssl_triple_loss(uv_embs[users], ut_embs[users], ut_embs) + self.ssl_triple_loss(
            iv_embs[pos_items], it_embs[pos_items], it_embs)

        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        Ef_users, Ef_items = torch.split(Ef, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(bd_embs, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(Ef_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            Ef_users[users], content_embeds_user[users], 0.2)

        # loss = batch_bpr_loss + self.reg_weight * batch_reg_loss + self.cl_weight * batch_hcl_loss + self.cl_loss * cl_loss
        loss = batch_bpr_loss + self.reg_weight * batch_reg_loss  + self.cl_weight * batch_hcl_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, _, _, _ = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores


class HGNNLayer(nn.Module):
    def __init__(self, n_hyper_layer):
        super(HGNNLayer, self).__init__()

        self.h_layer = n_hyper_layer

    def forward(self, i_hyper, u_hyper, embeds):
        i_ret = embeds
        for _ in range(self.h_layer):
            lat = torch.mm(i_hyper.T, i_ret)
            i_ret = torch.mm(i_hyper, lat)
            u_ret = torch.mm(u_hyper, lat)
        return u_ret, i_ret
