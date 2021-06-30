import argparse
import os
import time
import numpy as np
import h5py
import torch
import openslide
import copy
import pickle
import pandas as pd
from collections import OrderedDict
from veb import VEB
from models.vqvae import LargeVectorQuantizedVAE_Encode
from torchvision.models import densenet121
from torchvision import transforms
from tqdm import tqdm


def set_everything(seed):
    """
    Function used to set all random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_latent_semantic(latent, codebook_semantic):
    """
    Convert the original VQ-VAE latent code by using re-ordered codebook
    Input:
        latent (64 x 64 np.array): The original latent code from VQ-VAE encoder
        codebook_semantic (dict): The dictionary that map old codewords to
        the new ones
    Output:
        latent_semantic: The latent code with new codewords
    """
    latent_semantic = np.zeros_like(latent)
    for i in range(latent_semantic.shape[0]):
        for j in range(latent_semantic.shape[1]):
            latent_semantic[i][j] = codebook_semantic[latent[i][j]]
    return latent_semantic


def slide_to_index(latent, codebook_semantic, pool_layers, pool=None):
    """
    Convert VQ-VAE latent code into an integer.
    Input:
        latent (N x 64 x 64 np array): The latent code from VQ-VAE enecoder
        codebook_semantic (128 x 256): The codebook from VQ-VAE encoder
        pool_layers (torch.nn.Sequential): A series of pool layers that convert the latent code into an integer
    Output:
        index (int): An integer index that represents the latent code
    """
    if pool is None:
        result = to_latent_semantic(latent[0], codebook_semantic)
        feat = torch.from_numpy(np.expand_dims(result, 0))
    else:
        iterable = [(lt, codebook_semantic) for lt in latent]
        result = pool.starmap(to_latent_semantic, iterable)
        feat = torch.from_numpy(np.array(result))

    num_level = list(range(len(pool_layers) + 1))
    level_sum_dict = {level: None for level in num_level}

    for level in num_level:
        if level == 0:
            level_sum_dict[level] = torch.sum(feat, (1, 2)).numpy().astype(float)
        else:
            feat = pool_layers[level - 1](feat)
            level_sum_dict[level] = torch.sum(feat, (1, 2)).numpy().astype(float)

    level_power = [0, 0, 1e6, 1e11]
    index = 0
    for level, power in enumerate(level_power):
        if level == 1:
            index = copy.deepcopy(level_sum_dict[level])
        elif level > 1:
            index += level_sum_dict[level] * power

    return index


def min_max_binarized(feat):
    """
    Min-max algorithm proposed in paper: Yottixel-An Image Search Engine for Large Archives of
    Histopathology Whole Slide Images.
    Input:
        feat (1 x 1024 np.arrya): Features from the last layer of DenseNet121.
    Output:
        output_binarized (str): A binary code of length  1024
    """
    prev = float('inf')
    output_binarized = []
    for ele in feat:
        if ele < prev:
            code = 0
            output_binarized.append(code)
        elif ele >= prev:
            code = 1
            output_binarized.append(code)
        prev = ele
    output_binarized = "".join([str(e) for e in output_binarized])
    return output_binarized


def compute_latent_features(patch_rescaled, patch_id, save_path, transform, vqvae):
    """
    Copmute the latent code of input patch by VQ-VAE.
    Input:
        patch_rescaled (PIL.Image): 1024 x 1024 patch
        patch_id (str): An unique patch identifier
        save_path (str): The path to store the latent code of patch
        transform (torch.transforms): The transform applied to image before
        feeding into VQ-VAE
        vqvae (torch.models): VQ-VAE encoder along with codebook with weight
        from the checkpoints
    Output:
        feature (np.array): 1 x 64 x 64 latent feature
    """
    save_vqvae_path = os.path.join(save_path, 'vqvae', patch_id + ".h5")
    with torch.no_grad():
        inp = transform(patch_rescaled)
        inp = torch.unsqueeze(inp, 0)
        inp = inp.to(device, non_blocking=True)
        feature = vqvae(inp)
        feature = feature.cpu().numpy()
        with h5py.File(save_vqvae_path, 'w') as hf:
            hf.create_dataset('features', data=feature)
    return feature


def compute_densenet_features(patch_rescaled, patch_id, save_path, transform, densenet):
    """
    Copmute the texture feature of input patch by DenseNet121.
    Input:
        patch_rescaled (PIL.Image): 1024 x 1024 patch
        patch_id (str): An unique patch identifier
        save_path (str): The path to store the latent code of patch
        transform (torch.transforms): The transform applied to image before
        feeding into VQ-VAE
        densenet (torch.models): A pretrained Densenet121 model loaded from pytorch
    Output:
        feature_binarzied (str): A string of binarzied feature of length 1024
    """
    save_dense_path = os.path.join(save_path, 'densenet', patch_id + ".pkl")
    with torch.no_grad():
        inp = transform(patch_rescaled)
        inp = torch.unsqueeze(inp, 0)
        inp = inp.to(device, non_blocking=True)
        feature = densenet(inp)
        feature = feature.cpu().numpy()
    feature = np.squeeze(feature)
    feature_binarized = min_max_binarized(feature)
    with open(save_dense_path, 'wb') as handle:
        pickle.dump(feature_binarized, handle)
    return feature_binarized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build daatbase for patch data')
    parser.add_argument("--exp_name", type=str, choices=['kather100k'],
                        help="Patch data name for the experiment")
    parser.add_argument("--patch_label_file", type=str, required=True,
                        help="The csv file that contain patch name and its label")
    parser.add_argument("--patch_data_path", type=str, required=True,
                        help="Path to the folder that contains all patches")
    parser.add_argument("--codebook_semantic", type=str, default="./checkpoints/codebook_semantic.pt",
                        help="Path to semantic codebook")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints//model_9.pt",
                        help="Path to VQ-VAE checkpoint")
    args = parser.parse_args()
    device = torch.device("cuda:0,1") if torch.cuda.is_available() else torch.device('cpu')

    # Create the saving paths
    if not os.path.exists("DATABASES_PATCH"):
        os.makedirs("DATABASES_PATCH")
    if not os.path.exists(os.path.join("DATABASES_PATCH", args.exp_name)):
        os.makedirs(os.path.join("DATABASES_PATCH", args.exp_name))
    save_path_latent = os.path.join("./DATA_PATCH/{}_latent/".format(args.exp_name))
    save_path_indextree = os.path.join("DATABASES_PATCH", args.exp_name, "index_tree")
    save_path_indexmeta = os.path.join("DATABASES_PATCH", args.exp_name, "index_meta")
    if not os.path.exists(save_path_indextree):
        os.makedirs(save_path_indextree)
    if not os.path.exists(save_path_indexmeta):
        os.makedirs(save_path_indexmeta)
    if not os.path.exists(save_path_latent):
        os.makedirs(os.path.join(save_path_latent, 'vqvae'))
        os.makedirs(os.path.join(save_path_latent, 'densenet'))

    # Load codebook and label file
    codebook_semantic = torch.load(args.codebook_semantic)
    patch_label_file = pd.read_csv(args.patch_label_file)

    # initialize image transform
    transform_densenet = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    transform_vqvqe = transforms.Compose([transforms.Lambda(lambda x: 2 * transforms.ToTensor()(x) - 1)])

    # Load densenet and vqvae
    densenet = densenet121(pretrained=True)
    densenet = torch.nn.Sequential(*list(densenet.children())[:-1],
                                   torch.nn.AvgPool2d(kernel_size=(32, 32)))
    densenet.to(device)
    densenet.eval()
    vqvae = LargeVectorQuantizedVAE_Encode(256, 128)
    if torch.cuda.device_count() > 1:
        vqvae = torch.nn.DataParallel(vqvae, device_ids=[0, 1])
    vqvae_weight_value = torch.load(args.checkpoint)['model']
    vqvae_weight_enc = OrderedDict({k: v for k, v in vqvae_weight_value.items()
                                    if 'encoder' in k or 'codebook' in k})
    vqvae.load_state_dict(vqvae_weight_enc)
    vqvae = vqvae.to(device)
    vqvae.eval()

    t_enc_start = time.time()
    database = {}
    key_list = []
    pool_layers = [torch.nn.AvgPool2d(kernel_size=(2, 2)),
                   torch.nn.AvgPool2d(kernel_size=(2, 2)),
                   torch.nn.AvgPool2d(kernel_size=(2, 2))]
    for idx in tqdm(range(len(patch_label_file))):
        t_start = time.time()
        patch_name = patch_label_file.loc[idx, 'Patch Names']
        label = patch_label_file.loc[idx, 'label']
        patch = openslide.open_slide(os.path.join(args.patch_data_path, patch_name))
        if args.exp_name == 'kather100k':
            patch_rescaled = patch.read_region((0, 0), 0, (224, 224)).convert('RGB').resize((1024, 1024))
        else:
            # Implementation of customized method that fit your data to
            # scale your data to 1024 x 1024
            pass
        latent = compute_latent_features(patch_rescaled, patch_name.split(".")[0],
                                         save_path_latent,
                                         transform_vqvqe, vqvae)
        dense_feat = compute_densenet_features(patch_rescaled, patch_name.split(".")[0],
                                               save_path_latent,
                                               transform_densenet, densenet)
        slide_index = slide_to_index(latent, codebook_semantic,
                                     pool_layers=pool_layers)
        key = int(slide_index[0])
        tmp = {'patch_name': patch_name.split(".")[0],
               'dense_binarized': dense_feat,
               'diagnosis': label}
        if key not in database:
            database[key] = [tmp]
        else:
            database[key].append(tmp)
        key_list.append(int(key))
    print("")
    print("Encoding takes {}".format(time.time() - t_enc_start))
    t_db_start = time.time()
    database_keys = key_list
    universe = max(database_keys)
    number_of_index = len(database_keys)

    veb = VEB(universe)
    for k in database_keys:
        veb.insert(int(k))
    with open(os.path.join(save_path_indextree, "veb.pkl"), 'wb') as handle:
        pickle.dump(veb, handle)
    with open(os.path.join(save_path_indexmeta, "meta.pkl"), 'wb') as handle:
        pickle.dump(database, handle)
