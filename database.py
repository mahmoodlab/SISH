import pickle
import torch
import copy
import numpy as np


class HistoDatabase(object):
    """
    The FISH database that perform O(1) search
    Attributes:
        database_index_path (str): The path to the database index stored in veb tree
        index_meta_path (str): The path to the dictionary that stores the meta data for each index
        coodebook_semantic (str): The path to the semantic codebook from vq-vae encoder
        is_path (bool): Whether to use patch only mode (for patch only database)
    """

    def __init__(self, database_index_path, index_meta_path,
                 codebook_semantic, is_patch=False):
        """
        The intializer for HistoDatabase
        Input:
            database_index_path (str): The path to the database index stored in veb tree
            index_meta_path (str): The path to the dictionary that stores the meta data for each index
            coodebook_semantic (str): The path to the semantic codebook from vq-vae encoder
            is_path (bool): Whether to use patch only mode (for patch only database)
        Output: None
        """
        self.database_index_path = database_index_path
        self.index_meta_path = index_meta_path
        self.is_patch = is_patch

        print("Loading database index...")
        with open(self.database_index_path, 'rb') as handle:
            self.vebtree = pickle.load(handle)

        print("Loading index meta...")
        with open(self.index_meta_path, 'rb') as handle:
            self.meta = pickle.load(handle)

        print("Loading semantic codebook")
        self.codebook_semantic = torch.load(codebook_semantic)
        self.pool_layers = [torch.nn.AvgPool2d(kernel_size=(2, 2)),
                            torch.nn.AvgPool2d(kernel_size=(2, 2)),
                            torch.nn.AvgPool2d(kernel_size=(2, 2))]

    def leave_one_patient(self, patient_id):
        """
        The function used to remove the patient id used in leave-one-patient-out evaluation.
        Input:
            patient_id (str): Unique patient id.
        """
        if self.is_patch:
            self.meta_clean = self.meta
        else:
            meta_tmp = {}
            for key, val in self.meta.items():
                val_tmp = []
                for idx in range(len(val)):
                    if val[idx]['slide_name'].split("-")[2] != patient_id:
                        val_tmp.append(val[idx])
                meta_tmp[key] = val_tmp
            self.meta_clean = meta_tmp

    def query(self, patch, dense_feat,
              pre_step=375, succ_step=375,
              C=50, T=10, thrsh=128):
        """
        Query the database by taking the mosaic index and texture features under default search
        parameters.
        patch (int): Integer index of the mosaic
        densefeat (str): Texture feature of the mosaic
        pre_step (int): The number of step in the backward algorithm
        succ_step (int): The number of step in the forward algorithm
        C (int): The width of interval to expand the given search index
        T (int): The number fo time to expand the index
        """
        index = self.preprocessing(patch)
        indices_nn = self.search(index, dense_feat,
                                 pre_step=pre_step, succ_step=succ_step,
                                 C=C, T=T, thrsh=thrsh)
        results = self.postprocessing(indices_nn)
        return results

    def search(self, query_index, dense_feat, pre_step, succ_step,
               C, T, thrsh):
        """
        Implementation of backward and forward search in the paper
        Input:
            query_index (int): The integer index of the mosaic (m_{i})
            dense_feat (str): Texture feature of the mosaic (h_{i})
            pre_step (int): The number of step in the backward algorithm
            succ_step (int): The number of step in the forward algorithm
            C (int): The width of interval to expand the given search index
            T (int): The number fo time to expand the index
        Output:
            res (list): The list of tuple mainly composed of
            (query mosaic index,
             hamming distance between query and result mosiac,
             the slide name associated with the result mosaic,
             the diagnosis of the slide associated with the result mosaic
             the (x, y) coordinate in the slide where the result mosaic is located)
        """
        res = []
        pre = None
        succ = None
        seed_index = []
        seed_index_pre = [int(query_index - m * C * 1e11) for m in range(T)]
        seed_index_succ = [int(query_index + m * C * 1e11) for m in range(T)]
        seed_index.extend(seed_index_pre)
        seed_index.extend(seed_index_succ)
        visited = {}

        for index in seed_index:
            # Backward search
            pre_prev = index
            p_count = 0
            while p_count < pre_step:
                pre = self.vebtree.predecessor(pre_prev)
                if pre is None or pre in visited:
                    break
                if len(self.meta_clean[pre]) == 0:
                    pre_prev = pre
                    continue
                # If there are multiple mosaic shared with the same index
                # find the one that has minimum hamming distance
                if len(self.meta_clean[pre]) > 1:
                    tmp = []
                    for idx, _ in enumerate(self.meta_clean[pre]):
                        pre_dense = self.meta_clean[pre][idx]['dense_binarized']
                        hamming_dist_tmp = bin(int(pre_dense, 2) ^ int(dense_feat, 2)).count('1')
                        tmp.append(hamming_dist_tmp)
                    min_index = np.argmin(tmp)
                    hamming_dist = tmp[min_index]
                else:
                    min_index = 0
                    pre_dense = self.meta_clean[pre][min_index]['dense_binarized']
                    hamming_dist = bin(int(pre_dense, 2) ^ int(dense_feat, 2)).count('1')

                # Only select high quality mosaic with threshold less than thrsh (128)
                if hamming_dist <= thrsh:
                    index_meta = self.meta_clean[pre][min_index]
                    visited[pre] = 1
                    if not self.is_patch:
                        res.append((query_index, pre, np.abs(pre - query_index),
                                    hamming_dist, index_meta['slide_name'],
                                    index_meta['diagnosis'], index_meta['site'],
                                    index_meta['x'], index_meta['y']))
                    else:
                        res.append((query_index, pre, np.abs(pre - query_index),
                                    hamming_dist, index_meta['patch_name'],
                                    index_meta['diagnosis']))
                p_count += 1
                pre_prev = pre

            # Forward search
            s_count = 0
            succ_prev = index
            while s_count < succ_step:
                succ = self.vebtree.successor(succ_prev)
                if succ is None or succ in visited:
                    break
                if len(self.meta_clean[succ]) == 0:
                    succ_prev = succ
                    continue
                # If there are multiple mosaic shared with the same index
                # find the one that has minimum hamming distance
                if len(self.meta_clean[succ]) > 1:
                    tmp = []
                    for idx, _ in enumerate(self.meta_clean[succ]):
                        succ_dense = self.meta_clean[succ][idx]['dense_binarized']
                        hamming_dist_tmp = bin(int(succ_dense, 2) ^ int(dense_feat, 2)).count('1')
                        tmp.append(hamming_dist_tmp)
                    min_index = np.argmin(tmp)
                    hamming_dist = tmp[min_index]
                else:
                    min_index = 0
                    succ_dense = self.meta_clean[succ][min_index]['dense_binarized']
                    hamming_dist = bin(int(succ_dense, 2) ^ int(dense_feat, 2)).count('1')

                # Only select high quality mosaic with threshold less than thrsh (128)
                if hamming_dist <= thrsh:
                    visited[succ] = 1
                    index_meta = self.meta_clean[succ][min_index]
                    if not self.is_patch:
                        res.append((query_index, succ, np.abs(succ - query_index),
                                    hamming_dist, index_meta['slide_name'],
                                    index_meta['diagnosis'], index_meta['site'],
                                    index_meta['x'], index_meta['y']))
                    else:
                        res.append((query_index, succ, np.abs(succ - query_index),
                                    hamming_dist, index_meta['patch_name'],
                                    index_meta['diagnosis']))
                s_count += 1
                succ_prev = succ
        return res

    def preprocessing(self, latent):
        """
        Implementation of the pipeline that converts the original latent code
        from vq-vae encoder to an integer
        Input:
            latent (np.array): 64 x 64 latent code from vq-vae
        Output:
            mosaic_index: The interger index of the latent code (m_{i})
        """
        if self.is_patch:
            mosaic_index = self._slide_to_index(latent)
            return mosaic_index
        else:
            mosaic_index = self._slide_to_index(latent)

            return mosaic_index

    def postprocessing(self, res_tmp):
        """
        Sorting the result based on the hamming distance and
        converting it into dictionary
        Input:
            res_tmp (list): List of tuples from search
        Output:
            res_srt_dict (dict): Sorted results in dictionary
        """
        if self.is_patch:
            attribute_list = ['query', 'index', 'global_dist',
                              'hamming_dist', 'patch_name', 'diagnosis']
            res_srt = sorted(res_tmp, key=lambda x: x[3])
            res_srt_dict = [dict(zip(attribute_list, res)) for res in res_srt]
            return res_srt_dict
        else:
            attribute_list = ['query', 'index', 'global_dist', 'hamming_dist',
                              'slide_name', 'diagnosis', 'site', 'x', 'y']
            res_srt = sorted(res_tmp, key=lambda x: x[3])
            res_srt_dict = [dict(zip(attribute_list, res)) for res in res_srt]
            return res_srt_dict

    def _to_latent_semantic(self, latent):
        """
        Convert the original latent code from vq-vae by 
        re-ordered semantic codebook
        Input:
            latent (np.array): latent code of size 64 x 64
        Ouput:
            latent_semantic (np.array): The converted latent code of size 64 x 64
        """
        latent_semantic = np.zeros_like(latent)
        for i in range(latent_semantic.shape[0]):
            for j in range(latent_semantic.shape[1]):
                latent_semantic[i][j] = self.codebook_semantic[latent[i][j]]
        return latent_semantic

    def _slide_to_index(self, latent):
        """
        The pipeline that convert the latent code into an integer
        Input:
            latent (np.array): 64 x 64 latent code from the vq-vae encoder
        Output:
            mosaic_index: The index that represents the given mosaic (m_{i})
        """
        result = self._to_latent_semantic(latent)
        feat = torch.unsqueeze(torch.from_numpy(np.array(result)), 0)

        num_level = list(range(len(self.pool_layers) + 1))
        level_sum_dict = {level: None for level in num_level}

        for level in num_level:
            if level == 0:
                level_sum_dict[level] = torch.sum(feat, (1, 2)).numpy().astype(float)
            else:
                feat = self.pool_layers[level - 1](feat)
                level_sum_dict[level] = torch.sum(feat, (1, 2)).numpy().astype(float)

        level_power = [0, 0, 1e6, 1e11]
        mosaic_index = 0
        for level, power in enumerate(level_power):
            if level == 1:
                mosaic_index = copy.deepcopy(level_sum_dict[level])
            elif level > 1:
                mosaic_index += level_sum_dict[level] * power
        return int(mosaic_index[0])

    def __str__(self):
        """
        Tell how the database is built
        """
        return "Database built from {} with meta data from {}"\
               .format(self.database_index_path, self.index_meta_path)
