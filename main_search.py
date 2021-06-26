import argparse
import time
import os
import h5py
import pickle
import glob
from fish_database import HistoDatabase
from tqdm import tqdm

# Slides which are in poor quality
IGNORE_SLIDES = ['TCGA-C5-A8YT-01Z-00-DX1.5609D977-4B7E-4B49-A3FB-50434D6E49F9',
                 'TCGA-06-1086-01Z-00-DX2.e1961f1f-a823-4775-acf7-04a46f05e15e',
                 'C3N-02678-21']

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Search for WSI query in the database")
    parser.add_argument("--slide_path", type=str, default="./DATA/WSI",
                        help="The path to all slides")
    parser.add_argument("--latent_path", type=str, default="./DATA/LATENT",
                        help="The path to all mosaic latent code and text features")
    parser.add_argument("--site", type=str, required=True,
                        help="The site where the database is built")
    parser.add_argument("--db_index_path", type=str, required=True,
                        help="Path to the veb tree that stores all indices")
    parser.add_argument("--index_meta_path", type=str, required=True,
                        help="Path to the meta data of each index")
    parser.add_argument("--codebook_semantic", type=str, default="./checkpoints/codebook_semantic.pt", 
                        help="Path to the semantic codebook from vq-vae")
    args = parser.parse_args()

    if args.site == 'organ':
        save_path = os.path.join("QUERY_RESULTS", args.site)
        latent_all = os.path.join(args.latent_path, "*", "*", "*", "vqvae", "*")
        speed_record_path = os.path.join("QUERY_SPEED", args.site)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(speed_record_path):
            os.makedirs(speed_record_path)
    else:
        save_path = os.path.join("QUERY_RESULTS", args.site)
        latent_all = os.path.join(args.latent_path, args.site,
                                  "*", "*", "vqvae", "*")
        speed_record_path = os.path.join("QUERY_SPEED", args.site)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(speed_record_path):
            os.makedirs(speed_record_path)

    db = HistoDatabase(database_index_path=args.db_index_path,
                       index_meta_path=args.index_meta_path,
                       codebook_semantic=args.codebook_semantic)
    t_acc = 0
    results = {}
    for latent_path in tqdm(glob.glob(latent_all)):
        print(latent_path)
        resolution = latent_path.split("/")[-3]
        diagnosis = latent_path.split("/")[-4]
        anatomic_site = latent_path.split("/")[-5]
        slide_id = os.path.basename(latent_path).replace(".h5", "")
        if slide_id in IGNORE_SLIDES:
            continue
        # Remove the current patient from the database for leave-one-patient out evaluation
        # Implement your own to fit your own to fit your data.
        if not slide_id.startswith('TCGA'):
            # Implementation of your own leave-one out strategy to fit your data
            pass
        else:
            # Leave-one-patient out in TCGA cohort
            patient_id = slide_id.split("-")[2]
            db.leave_one_patient(patient_id)

        slide_path = os.path.join(args.slide_path, anatomic_site, diagnosis,
                                  resolution, slide_id + ".svs")
        densefeat_path = latent_path.replace("vqvae", "densenet").replace(".h5", ".pkl")
        with h5py.File(latent_path, 'r') as hf:
            feat = hf['features'][:]
            coords = hf['coords'][:]
        with open(densefeat_path, 'rb') as handle:
            densefeat = pickle.load(handle)

        t_start = time.time()
        tmp_res = []
        for idx, patch_latent in enumerate(feat):
            res = db.query(patch_latent, densefeat[idx])
            tmp_res.append(res)
        t_elapse = time.time() - t_start
        with open(os.path.join(speed_record_path, "speed_log.txt"), 'a') as fw:
            fw.write(slide_id + "," + str(t_elapse) + "\n")
        t_acc += t_elapse
        print("Search takes ", t_elapse)

        key = slide_id
        results[key] = {'results': None, 'label_query': None}
        results[key]['results'] = tmp_res
        if args.site == 'organ':
            results[key]['label_query'] = anatomic_site
        else:
            results[key]['label_query'] = diagnosis
    print("Total search takes: ", t_acc)
    with open(os.path.join(save_path, "results.pkl"), 'wb') as handle:
        pickle.dump(results, handle)
