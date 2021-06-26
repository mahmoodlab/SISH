import openslide
import h5py
import numpy as np
import glob
import os
import cv2
import time
import argparse
import multiprocessing as mp


def artifacts_removal(coord, slide_name, patch_size):
    """
    Remove the patch if the white area is larger than 90 percent
    Input:
        coord (np.array): The coordinate of patche in the slide
        slide_name (str): The slide to process
        patch_size (int): The patch size used to patch the slide
    Output:
        (bool): 1: The patch is white, otherwise, 0
    """
    wsi = openslide.open_slide(slide_name)
    region = wsi.read_region(coord, 0, (patch_size, patch_size)).convert("L").resize((256, 256))
    _, white_region = cv2.threshold(np.array(region), 235, 255, cv2.THRESH_BINARY)
    if np.sum(white_region == 255) / (256 * 256) > 0.9:
        return 1
    else:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_slide_path", required=True)
    parser.add_argument("--site_mosaic_path", required=True)
    args = parser.parse_args()
    pool = mp.Pool(4)
    total = len(glob.glob(os.path.join(args.site_mosaic_path, "*", "*",
                "coord", "*")))
    progress = 0

    for mosaic in glob.glob(os.path.join(args.site_mosaic_path,
                            "*", "*", "coord", "*")):
        print("{}/{} Process mosaic: {}".format(progress, total, mosaic))
        diagnosis = mosaic.split("/")[-4]
        resolution = mosaic.split("/")[-3]
        slide_id = mosaic.split("/")[-1].replace(".h5", "")
        with h5py.File(mosaic, 'r') as hf:
            coords = hf['coords'][:]
        if resolution == '20x':
            patch_size = 1024
        elif resolution == '40x':
            patch_size = 2048
        print("Original mosaic size:", len(coords))

        # Remove the white artifacts
        slide_path = os.path.join(args.site_slide_path, diagnosis, resolution, slide_id + ".svs")
        t_start = time.time()
        iterable = [(coord, slide_path, patch_size) for coord in coords]
        artifacts_indicator = pool.starmap(artifacts_removal, iterable)
        coord_clean = coords[np.array(artifacts_indicator) == 0]
        coord_artifacts = coords[np.array(artifacts_indicator) == 1]
        print("Clean mosaic size:", len(coord_clean))
        print("Removal takes: ", time.time() - t_start)

        save_path_clean = os.path.join(args.site_mosaic_path, diagnosis,
                                       resolution, "coord_clean")
        save_path_artifacts = os.path.join(args.site_mosaic_path, diagnosis,
                                           resolution, "coord_artifacts")
        # Save the results
        if not os.path.exists(save_path_clean):
            os.makedirs(save_path_clean)
        if not os.path.exists(save_path_artifacts):
            os.makedirs(save_path_artifacts)

        with h5py.File(os.path.join(save_path_clean, slide_id + ".h5"), 'w') as hf:
            hf.create_dataset("coords", data=coord_clean)
        if len(coord_clean) == len(coords):
            print("")
            progress += 1
            continue
        else:
            with h5py.File(os.path.join(save_path_artifacts, slide_id + ".h5"), 'w') as hf:
                hf.create_dataset("coords", data=coord_artifacts)
            progress += 1
            print("")
