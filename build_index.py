import argparse
import os
import time
import numpy as np
import h5py
import glob
import torch
import multiprocessing as mp
import openslide
import copy
import pickle
from collections import OrderedDict
from veb import VEB
from models.vqvae import LargeVectorQuantizedVAE_Encode
from dataset import Mosaic_Bag_FP
from torchvision.models import densenet121
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_everything(seed):
    """
    Function used to set all random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_hdf5(output_path, asset_dict, mode='a'):
    """
    Function that used to store hdf5 file chunk by chunk
    """
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape,
                                       maxshape=maxshape, chunks=chunk_shape,
                                       dtype=data_type)
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val

    file.close()
    return output_path


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
    Convert VQ-VAE latent code into an integer
    Input:
        latent (N x 64 x 64 np array): The latent code from VQ-VAE enecoder
        codebook_semantic (128 x 256): The codebook from VQ-VAE encoder
        pool_layers (torch.nn.Sequential): A series of pool layers that convert the latent code into an integer
    Output:
        index (int): An integer index that represents the latent code
    """
    if pool is None:
        result = to_latent_semantic(latent, codebook_semantic)
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


def compute_latent_features(wsi, mosaic_path, save_path, resolution, transform,
                            vqvae, batch_size=8, num_workers=4):
    """
    Copmute the latent code of input by VQ-VAE encoder
    Input:
        wsi (openslide objet): The slide to compute
        mosaic_path (str): The path that store wsi mosaic
        save_path (str): Path to store latent code
        resolution (str): The resolution of wsi (e.g., 20x or 40x)
        transoform (torch.transforms): The transform applied to image before
        feeding into VQ-VAE
        vqvae (torch.models): VQ-VAE encoder along with codebook with weight
        from the checkpoints
        batch_size (int): The number of data processed by VQ-VAE per loop
        num_workers (int): Number of cpu used by Dataloader to load the data
    Output:
        feature list (list): list of vq-vqae latent code of length = #mosaics
        in the wsi
    """
    dataset = Mosaic_Bag_FP(mosaic_path, wsi, int(resolution[:-1]),
                            custom_transforms=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False,
                            pin_memory=True)
    count = 0
    total = len(dataset)
    if total == 0:
        return None
    features_list = []
    mode = 'w'
    save_vqvae_path = os.path.join(save_path, 'vqvae', os.path.basename(mosaic_path))
    with torch.no_grad():
        for mosaic, coord in dataloader:
            mosaic = torch.squeeze(mosaic, 1)
            mosaic = mosaic.to(device, non_blocking=True)
            features = vqvae(mosaic)
            features = features.cpu().numpy()
            features_list.append(features)
            count += features.shape[0]
            print("Numebr of latent code processed {}/{}".format(count, total))
            asset_dict = {'features': features, 'coords': coord.numpy()}
            save_hdf5(save_vqvae_path, asset_dict, mode=mode)
            mode = 'a'
    return np.concatenate(features_list, 0)


def compute_densenet_features(wsi, mosaic_path, save_path, resolution,
                              transform, densenet,
                              batch_size=8, num_workers=4):
    """
    Copmute the texture feature (h_{i}) of input by DenseNet121.
    Input:
        wsi (openslide objet): The slide to compute
        mosaic_path (str): The path that store wsi mosaic
        save_path (str): Path to store latent code
        resolution (str): The resolution of wsi (e.g., 20x or 40x)
        transoform (torch.transforms): The transform applied to image before
        feeding into DenseNet.
        densenet (torch.models): A pretrained Densenet121 model loaded from pytorch
        batch_size (int): The number of data processed by VQ-VAE per loop
        num_workers (int): Number of cpu used by Dataloader to load the data
    Output:
        feature list (list): list of binarized texture feature of length = #mosaics
        in the wsi
    """
    dataset = Mosaic_Bag_FP(mosaic_path, wsi, int(resolution[:-1]),
                            custom_transforms=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False,
                            pin_memory=True)
    features_list = []
    count = 0
    total = len(dataset)
    save_dense_path = os.path.join(save_path, 'densenet',
                                   os.path.basename(mosaic_path).replace(".h5",".pkl"))
    with torch.no_grad():
        for mosaic, coord in dataloader:
            mosaic = torch.squeeze(mosaic, 1)
            mosaic = mosaic.to(device, non_blocking=True)
            features = densenet(mosaic)
            features = features.cpu().numpy()
            features_list.append(features)
            count += features.shape[0]
            print("Number of dense feature rocessed {}/{}".format(count, total))

    features_list = np.squeeze(np.concatenate(features_list, 0))
    features_binarized = [min_max_binarized(feat) for feat in features_list]
    with open(save_dense_path, 'wb') as handle:
        pickle.dump(features_binarized, handle)
    return features_binarized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build database of FISH')
    parser.add_argument("--mosaic_path", type=str, default="./DATA/MOSAICS/",
                        help="Path to mosaics")
    parser.add_argument("--slide_path", type=str, default="./DATA/WSI",
                        help="Path to WSIs")
    parser.add_argument("--site", type=str, required=True,
                        help="The anatomic site where the database is built on")
    parser.add_argument("--slide_ext", default='.svs',
                        help="Slide file format")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/model_9.pt",
                        help="Path to VQ-VAE checkpoint")
    parser.add_argument("--codebook_semantic", type=str, default="./checkpoints/codebook_semantic.pt", 
                        help="Path to semantic codebook")
    args = parser.parse_args()

    # Create the save path of database
    if not os.path.exists("DATABASES"):
        os.makedirs("DATABASES")
    if not os.path.exists(os.path.join("DATABASES")):
        os.makedirs(os.path.join("DATABASES"))
    save_path_indextree = os.path.join("DATABASES", args.site, "index_tree")
    save_path_indexmeta = os.path.join("DATABASES", args.site, "index_meta")
    if not os.path.exists(save_path_indextree):
        os.makedirs(save_path_indextree)
    if not os.path.exists(save_path_indexmeta):
        os.makedirs(save_path_indexmeta)

    # Set up cpu and device
    device = torch.device("cuda:0,1") if torch.cuda.is_available() else torch.device('cpu')
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)

    t_total_start = time.time()

    # initialize the database related object
    database = {}
    key_list = []

    # initialize transforms for densenet and vq-vae
    transform_densenet = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])
    transform_vqvqe = transforms.Compose([transforms.Lambda(lambda x: 2 * transforms.ToTensor()(x) - 1)])

    # Load the Densenet
    densenet = densenet121(pretrained=True)
    densenet = torch.nn.Sequential(*list(densenet.children())[:-1],
                                   torch.nn.AvgPool2d(kernel_size=(32, 32)))
    densenet.to(device)
    densenet.eval()

    # Load the codebook and vq-vae encoder
    codebook_semantic = torch.load(args.codebook_semantic)
    vqvae = LargeVectorQuantizedVAE_Encode(256, 128)
    if torch.cuda.device_count() > 1:
        vqvae = torch.nn.DataParallel(vqvae, device_ids=[0, 1])
    vqvae_weight_value = torch.load(args.checkpoint)['model']
    vqvae_weight_enc = OrderedDict({k: v for k, v in vqvae_weight_value.items()
                                    if 'encoder' in k or 'codebook' in k})
    vqvae.load_state_dict(vqvae_weight_enc)
    vqvae = vqvae.to(device)
    vqvae.eval()

    # Initialize pooling layer
    pool_layers = [torch.nn.AvgPool2d(kernel_size=(2, 2)),
                   torch.nn.AvgPool2d(kernel_size=(2, 2)),
                   torch.nn.AvgPool2d(kernel_size=(2, 2))]
    t_enc_start = time.time()
    mosaic_all = os.path.join(args.mosaic_path, args.site,
                              "*", "*", "coord_clean", "*")
    total = len(glob.glob(mosaic_all))
    count = 0
    number_of_mosaic = 0
    for mosaic_path in tqdm(glob.glob(mosaic_all)):
        print(mosaic_path)
        t_start = time.time()
        resolution = mosaic_path.split("/")[-3]
        diagnosis = mosaic_path.split("/")[-4]
        slide_id = os.path.basename(mosaic_path).replace(".h5", "")
        slide_path = os.path.join(args.slide_path, args.site, diagnosis,
                                  resolution, slide_id + ".svs")
        save_path_latent = os.path.join("./DATA/LATENT/",
                                        args.site, diagnosis, resolution)
        if not os.path.exists(save_path_latent):
            os.makedirs(os.path.join(save_path_latent, 'vqvae'))
            os.makedirs(os.path.join(save_path_latent, 'densenet'))

        with h5py.File(mosaic_path, 'r') as hf:
            mosaic_coord = hf['coords'][:]
        wsi = openslide.open_slide(slide_path)
        latent = compute_latent_features(wsi, mosaic_path, save_path_latent,
                                         resolution, transform_vqvqe, vqvae)
        dense_feat = compute_densenet_features(wsi, mosaic_path,
                                               save_path_latent, resolution,
                                               transform_densenet, densenet)
        slide_index = slide_to_index(latent, codebook_semantic,
                                     pool_layers=pool_layers, pool=pool)

        for idx, key in enumerate(slide_index):
            tmp = {'slide_name': slide_id, 'dense_binarized': dense_feat[idx],
                   'x': mosaic_coord[idx][0], 'y': mosaic_coord[idx][1],
                   'slide_ext': args.slide_ext, 'diagnosis': diagnosis,
                   'site': args.site}
            if key not in database:
                database[key] = [tmp]
            else:
                database[key].append(tmp)
            key_list.append(int(key))
        count += 1
        print("{}/{} Processing slide {} with diagnosis {} takes {}".
              format(count, total, slide_id, diagnosis, time.time() - t_start))
        number_of_mosaic += len(mosaic_coord)

    print("")
    print("Encoding takes {}".format(time.time() - t_enc_start))
    database_keys = key_list
    universe = max(database_keys)
    number_of_index = len(database_keys)
    print("Universe size of veb tree:", universe)
    veb = VEB(universe)
    for k in database_keys:
        veb.insert(int(k))
    with open(os.path.join(save_path_indextree, "veb.pkl"), 'wb') as handle:
        pickle.dump(veb, handle)
    with open(os.path.join(save_path_indexmeta, "meta.pkl"), 'wb') as handle:
        pickle.dump(database, handle)
