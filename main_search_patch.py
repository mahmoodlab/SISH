import argparse
import time
import os
import h5py
import pickle
import pandas as pd
from fish_database import HistoDatabase
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Search for patch query in the database")
    parser.add_argument("--patch_label_file", type=str, required=True)
    parser.add_argument("--patch_data_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str, choices=['kather100k'])
    parser.add_argument("--db_index_path", type=str, required=True)
    parser.add_argument("--index_meta_path", type=str, required=True)
    parser.add_argument("--codebook_semantic", type=str, default="./checkpoints/codebook_semantic.pt")
    args = parser.parse_args()

    # Create saving path
    save_path = os.path.join("QUERY_RESULTS", "PATCH", "{}".format(args.exp_name))
    speed_record_path = os.path.join("QUERY_SPEED", "PATCH", args.exp_name)
    topk_MV = 5
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(speed_record_path):
        os.makedirs(speed_record_path)

    # Load label file and database
    patch_label_file = pd.read_csv(args.patch_label_file)
    db = HistoDatabase(database_index_path=args.db_index_path,
                       index_meta_path=args.index_meta_path,
                       codebook_semantic=args.codebook_semantic,
                       is_patch=True)
    t_acc = 0
    query_count = 0
    results = {}
    for idx in tqdm(range(len(patch_label_file))):
        t_start = time.time()
        patch_name = patch_label_file.loc[idx, 'Patch Names']
        patch_id = patch_name.split(".")[0]
        label = patch_label_file.loc[idx, 'label']
        print(patch_id)
        db.leave_one_patient(patch_id)

        latentfeat_path = os.path.join("./DATA_PATCH/", "{}_latent"
                                       .format(args.exp_name), 'vqvae', patch_id + ".h5")
        densefeat_path = os.path.join("./DATA_PATCH/", "{}_latent"
                                      .format(args.exp_name), 'densenet', patch_id + ".pkl")
        with h5py.File(latentfeat_path, 'r') as hf:
            feat = hf['features'][:]
        with open(densefeat_path, 'rb') as handle:
            densefeat = pickle.load(handle)

        tmp_res = []
        t_start = time.time()
        res = db.query(feat[0], densefeat)
        tmp_res.append(res)
        t_elapse = time.time() - t_start
        t_acc += t_elapse
        print("Search takes ", time.time() - t_start)

        with open(os.path.join(speed_record_path, "speed_log.txt"), 'a') as handle:
            handle.write("{},{}\n".format(patch_id, t_elapse))
        key = patch_id
        tmp_res = tmp_res[0]
        tmp_clean = []
        for r in tmp_res:
            if r['patch_name'] == patch_id:
                continue
            else:
                tmp_clean.append((r['hamming_dist'], r['diagnosis'], r['patch_name']))
        top5 = sorted(tmp_clean, key=lambda x: x[0])[0:topk_MV]
        results[key] = {'results': None, 'label_query': None}
        results[key]['results'] = top5
        results[key]['label_query'] = label

    print("Total search takes: ", t_acc)
    with open(os.path.join(save_path, "results.pkl"), 'wb') as handle:
        pickle.dump(results, handle)
