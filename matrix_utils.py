import torch
import numpy as np
import lpips
from lpips import LPIPS, spatial_average, upsample
import time
from scipy.spatial.distance import squareform
import graspologic
import argparse
from datasets_prep import get_dataset
import networkx as nx 
import matplotlib.pyplot as plt
import ot
import seaborn as sns
import pandas as pd
import torchvision
from tabulate import tabulate



class LPIPS_PDIST(LPIPS):
    def __init__(self, 
                 pretrained=True, 
                 net='alex', 
                 version='0.1', 
                 lpips=True, 
                 spatial=False, 
                 pnet_rand=False, 
                 pnet_tune=False, 
                 use_dropout=True, 
                 model_path=None, 
                 eval_mode=True, 
                 verbose=True):
        super().__init__(pretrained, 
                         net, 
                         version, 
                         lpips, 
                         spatial, 
                         pnet_rand, 
                         pnet_tune, 
                         use_dropout, 
                         model_path, 
                         eval_mode, 
                         verbose)
        
    def calc_pdist(self, xfeats):
        b = xfeats.shape[0]
        dists = []
        for ii in range(b):
            for jj in range(ii+1, b):
                dists.append((xfeats[jj] -xfeats[ii])**2)
        dists = torch.stack(dists)
        return dists
                
    
    def forward(self, images, retPerLayer=False, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            images = 2 * images  - 1

        image_input = self.scaling_layer(images) if self.version=='0.1' else images
        feats = self.net.forward(image_input)
        diffs = {}

        for kk in range(self.L):
            feat = lpips.normalize_tensor(feats[kk].to("cpu"))
            diffs[kk] = self.calc_pdist(feat).to("cuda")
            
        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=images.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=images.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]
            
        if(retPerLayer):
            return (val, res)
        else:
            return val
         
        
def get_image_adjacency_matrix(lpips_model, images):
    start = time.time()
    batch = images.shape[0]
    adj_matrix = torch.zeros((batch, batch))
    for ii in range(batch): 
        for jj in range(ii+1, batch):
            dist_ = lpips_model(images[ii], images[jj])
            adj_matrix[ii][jj] = adj_matrix[jj][ii] = dist_.squeeze()
    print("Slow Duration: {}".format(time.time()-start))
    return adj_matrix


def get_image_adjacency_matrix_fast(lpips_model, images):
    # start = time.time()
    dists = lpips_model(images).squeeze().to("cpu")
    adj_matrix = squareform(dists)
    # print("Fast Duration: {}".format(time.time()-start))
    return adj_matrix



def get_noise_adjacency_matrix(noises: torch.Tensor):
    # start = time.time()
    tmp = torch.nn.functional.pdist(noises.view(noises.size(0), -1)).to("cpu")
    adj_matrix = squareform(tmp)
    # print("Noise Duration: {}".format(time.time()-start))
    return adj_matrix


def graph_matching(model, images, noises):
    # start_time = time.time()
    image_adj = get_image_adjacency_matrix_fast(model, images)
    noise_adj = get_noise_adjacency_matrix(noises)
    result = graspologic.match.graph_match(image_adj, noise_adj)
    # print("Total time: {}".format(time.time()-start_time))
    # print(result)
    noises_idx = result.indices_B
    noises = noises[noises_idx]
    return images, noises


def graph_reorder(matrix_adj, order):
    reorder_mat = np.zeros_like(matrix_adj)
    N = reorder_mat.shape[0]
    for ii in range(N):
        for jj in range(N):
            reorder_mat[order[ii]][order[jj]] = matrix_adj[ii][jj]
    return reorder_mat


def get_adjacency(model, images, noises, knn_num, mode):
    image_adj = get_image_adjacency_matrix_fast(model, images)
    image_knn = knn_graph(image_adj, knn_num=knn_num, mode=mode)
    noise_adj = get_noise_adjacency_matrix(noises)
    noise_knn = knn_graph(noise_adj, knn_num=knn_num, mode=mode)
    plot_matrix(image_adj, noise_adj)
    return image_knn, noise_knn


def graph_matching(image_adj, noise_adj, mode):
    if mode == "simple":
        result = graspologic.match.graph_match(image_adj, noise_adj)
    elif mode == "gm":
        N = image_adj.shape[0]
        p = ot.unif(N) # n_samples is number of images
        q = ot.unif(N)
        result, log0 = ot.gromov.gromov_wasserstein(image_adj, noise_adj, p, q, 'square_loss', verbose=True, log=True)
    return result
    
    
    
def draw_graph(image_adj, noise_adj, name):
    G_image = nx.from_numpy_array(image_adj)
    G_noise = nx.from_numpy_array(noise_adj)
    subax1 = plt.subplot(121)
    nx.draw(G_image, with_labels=True, font_weight='bold')
    subax2 = plt.subplot(122)
    nx.draw(G_noise, with_labels=True, font_weight='bold')
    plt.savefig("graph_{}.png".format(name))



def knn_graph(emb_distance_matrix, knn_num, mode):
    """
    Using Bao code
    """
    N = emb_distance_matrix.shape[0]
    adj_list = np.argpartition(emb_distance_matrix, knn_num + 1)[:, 0:knn_num + 1]
    adj_matrix = np.zeros((N, N))
    for i in range(N):
        adj_matrix[np.ix_([i], adj_list[i])] += 1
        if mode != "non_mutual":
            adj_matrix[np.ix_(adj_list[i], [i])] += 1
        else:
            adj_matrix[np.ix_(adj_list[i], [i])] += 0
    
    adj_matrix = adj_matrix * (1-np.identity(N))
    if mode == "and_mutual":
        adj_matrix = np.where(adj_matrix < 2, 0, 1)
    elif mode == "or_mutual":
        adj_matrix = np.where(adj_matrix > 0, 1, adj_matrix)
    elif mode == "non_mutual":
        adj_matrix = adj_matrix

    return adj_matrix*emb_distance_matrix


def plot_matrix(image_matrix, noise_matrix):
    N = image_matrix.shape[0]
    
    cm = sns.light_palette("blue", as_cmap=True)
    x = pd.DataFrame(image_matrix)
    print(tabulate(x, tablefmt='psql'))

    cm = sns.light_palette("blue", as_cmap=True)
    x = pd.DataFrame(noise_matrix)
    print(tabulate(x, tablefmt='psql'))


def batchify(model, image_matrix, noise_matrix):
    image_adj_, noise_adj_ = get_adjacency(model, image_matrix, noise_matrix, knn_num = 4, mode = "or_mutual")
    image_adj, noise_adj = (image_adj_-np.min(image_adj_))/(np.max(image_adj_)-np.min(image_adj_)), (noise_adj_-np.min(noise_adj_))/(np.max(noise_adj_)-np.min(noise_adj_))
    result = graph_matching(image_adj, noise_adj, mode="simple")
    noise_matrix = noise_matrix[result.indices_B]
    return image_matrix, noise_matrix
 
if __name__ == "__main__":
    print("testing")
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', default=32, help='name of dataset')
    parser.add_argument('--datadir', default = "./dataset/celeba")
    args = parser.parse_args()
    
    model = LPIPS_PDIST(net="vgg").to("cuda")
    model = model.cuda()
    for p in model.parameters():
        p.requires_grad = False 
        
    dataset = get_dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=100,
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=True,
                                            drop_last = True)
    batch, label = next(iter(dataloader))
    
    index_dict = {}
    for ii in range(10):
        index_dict[ii] = []
        for idx in range(len(label)):
            if label[idx] == ii:
                index_dict[ii].append(idx)
    
    class_min = 256
    for ii in range(10):
        if len(index_dict[ii]) < class_min:
            class_min = len(index_dict[ii])
            
    batchs = []
    for ii in range(4):
        batchs.append(batch[index_dict[ii][:class_min],:,:,:])
    batch = torch.cat(batchs).to("cuda")
    
    torchvision.utils.save_image(batch, "image.png", normalize=True, nrow=class_min)
    noise = torch.randn_like(batch).to("cuda")
    start_time = time.time()
    image_adj_, noise_adj_ = get_adjacency(model, batch, noise, knn_num = 2, mode = "or_mutual")
    image_adj, noise_adj = (image_adj_-np.min(image_adj_))/(np.max(image_adj_)-np.min(image_adj_)), (noise_adj_-np.min(noise_adj_))/(np.max(noise_adj_)-np.min(noise_adj_))
    image_adj_m = np.where(image_adj == 0, 2, image_adj)
    noise_adj_m = np.where(noise_adj == 0, 2, noise_adj)
    
    print("Adjacency Matrix Computation: {}".format(time.time()-start_time))
    
    start_time = time.time()
    result = graph_matching(image_adj_m, noise_adj_m, mode="simple")
    print("Graph Matching Computation: {}".format(time.time()-start_time))    
    
    noise_adj_reorder = graph_reorder(noise_adj, order=result.indices_B)
    
    plot_matrix(image_adj_, graph_reorder(noise_adj_, order=result.indices_B))
    
    draw_graph(image_adj, noise_adj_reorder, name="order_")

        
    