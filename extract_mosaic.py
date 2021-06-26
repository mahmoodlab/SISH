import h5py
import os
import glob
import pickle
import openslide
import argparse
import time
import cv2 as cv
import numpy as np
import multiprocessing as mp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
np.random.seed(0)

color = ('r', 'g', 'b')  # Denote the channel used to extract 5x histogram
num_cluster = 9  # Number of cluster used in the first stage K-mean clustering
sample_rate = 0.05  # Number of cluster (sample_rate * cluster_size) used in the second stage K-mean clustering (spatial clustering)


def local_binary_pattern_hist(img_imp):
    """
    Calculate the local binary pattern of the given input.
    Input:
        img_imp (PIL.Image): The input grey scale image represented.
    Output:
        hist (np.array):The histogram of the local binary pattern of input image.
    """
    lbp = local_binary_pattern(img_imp, 8, 1, 'ror')
    hist = np.histogram(lbp, density=True, bins=128, range=(0, 128))[0]
    return hist


def pre_filtering(coord, slide_name, patch_size):
    """
    Filter out the white region and calculate the rgb/lbp histogram for a patch in the given slide.
    Input:
        slide_name (str): The slide to process
        coord (np.array): The coordinate of the patch in the slide
        patch_size (int): The height and width of the patch
    Output:
        hist_feat (np.array): RGB histogram of patch in coord from the slide
        lbp_feat (np.array): LBP histogram of patch in the coord from the slide
    """
    hist_feat = []
    wsi = openslide.open_slide(slide_name)
    patch = wsi.read_region((coord[0], coord[1]), 0, (patch_size, patch_size))

    # Convert to 5x to do filtering
    patch_grey = patch.convert('L').resize((256, 256))
    _, white_region = cv.threshold(np.array(patch_grey), 235, 255, cv.THRESH_BINARY)
    if np.sum(white_region == 255) / (256 * 256) > 0.9:
        return None, None

    # Convert to 5x to extract RGB histogram
    patch_rgb = patch.convert("RGB").resize((256, 256))
    patch_rgb = np.array(patch_rgb).astype('float32')

    for i, col in enumerate(color):
        histr = cv.calcHist([patch_rgb], [i], None, [256], [0, 256])
        hist_feat.append(histr.T)
    hist_feat = np.concatenate(hist_feat, 1)

    lbp_feat = local_binary_pattern_hist(patch_grey)
    return hist_feat, lbp_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_data_path", required=True)
    parser.add_argument("--slide_patch_path", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()
    num_cpu = mp.cpu_count()
    ignore_slide_id = ['TCGA-06-1086-01Z-00-DX2.e1961f1f-a823-4775-acf7-04a46f05e15e',
                       'C3N-02678-21','TCGA-AN-A0XW-01Z-00-DX1.811E11E7-FA67-46BB-9BC6-1FD0106B789D',
                       'TCGA-DQ-5630-01Z-00-DX1.07FE0581-2412-43DA-96A9-0DA192DAED3D']

    # Loading trash region filter classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
    with open("./checkpoints/trash_lgrlbp.pkl", 'rb') as handle:
        clf = pickle.load(handle)

    if not os.path.exists(args.save_path):
        os.makedirs(os.path.join(args.save_path, 'coord'))
    progress = 1
    total = len(glob.glob(os.path.join(args.slide_patch_path, "*")))
    pool = mp.Pool(8)
    for slide_to_process in glob.glob(os.path.join(args.slide_patch_path, "*")):
        slide_key = os.path.basename(slide_to_process).replace(".h5", "")
        if slide_key in ignore_slide_id:
            continue
        print("Proceessing {}/{} {}:".format(progress, total, slide_key))
        if slide_key + ".h5" in os.listdir(os.path.join(args.save_path, 'coord')):
            print("skip", slide_key)
            progress += 1
            continue
        t_start = time.time()
        slide_path = os.path.join(args.slide_data_path, "{}.svs".format(slide_key))
        patch_path = os.path.join(args.slide_patch_path, "{}.h5".format(slide_key))

        print("Loading")
        with h5py.File(patch_path, 'r') as hf:
            coords = hf['coords'][:]
            patch_size = hf['coords'].attrs['patch_size']
        iterable = [(coord, slide_path, patch_size) for coord in coords]

        print("Getting histogram")
        results = pool.starmap(pre_filtering, iterable)
        print("Total patches", len(coords))

        white_index = []
        for r in results:
            if r[0] is not None:
                white_index.append(0)
            else:
                white_index.append(1)
        print("White index: ", sum(white_index))
        white_index = np.array(white_index)

        # If None is contained in r, it means it's the patch filtered out
        # by the trash threshold
        slide_rgbhist_feat = [r[0] for r in results if r[0] is not None]
        slide_lbphist_feat = [np.expand_dims(r[1], 0) for r in results if r[1] is not None]
        coords_white = coords[white_index == 1]
        coords_nonwhite = coords[white_index == 0]
        slide_rgbhist_feat = np.concatenate(slide_rgbhist_feat, 0)
        slide_lbphist_feat = np.concatenate(slide_lbphist_feat, 0)
        print(slide_rgbhist_feat.shape, slide_lbphist_feat.shape)

        # Use pretrained trash/tissue model to clean the results
        trash_pred = clf.predict(slide_lbphist_feat)
        print("Trash rate: ", sum(trash_pred == 1) / len(trash_pred))
        slide_rgbhist_feat_clean = slide_rgbhist_feat[trash_pred == 0]
        coords_clean = coords_nonwhite[trash_pred == 0]
        coords_trash = coords_nonwhite[trash_pred == 1]
        print("Clean patches", len(coords_clean))
        model = KMeans(n_clusters=num_cluster, random_state=0)

        model.fit(slide_rgbhist_feat_clean)
        patch_cluster = model.predict(slide_rgbhist_feat_clean)
        mosaic = []

        # Two stage K-mean to select mosaics from the slide
        for cluster in range(num_cluster):
            coord_select = coords_clean[patch_cluster == cluster]
            num_coord_cluster = int(sample_rate * len(coord_select))
            if num_coord_cluster == 0:
                model_coord = KMeans(n_clusters=len(coord_select),
                                     random_state=0)
            else:
                model_coord = KMeans(n_clusters=num_coord_cluster,
                                     random_state=0)
            model_coord.fit(coord_select)
            mosaic.append(model_coord.cluster_centers_.astype(int))
        mosaic = np.concatenate(mosaic, 0)
        print("Mosaic size: ", len(mosaic))
        save_name = os.path.join(args.save_path, 'coord', slide_key + ".h5")
        with h5py.File(save_name, 'w') as hf:
            hf.create_dataset("coords", data=mosaic)
        print("Proceesing {} takes:{}".format(slide_key, time.time() - t_start))
        print("")
        progress += 1
    pool.close()
