import pickle
import time
import numpy as np
import operator
import argparse
import copy
import os
import math
import glob
from collections import Counter, defaultdict


def Uncertainty_Cal(bag, is_organ=False):
    """
    Implementation of Weighted-Uncertainty-Cal in the paper.
    Input:
        bag (list): A list of dictionary which contain the searhc results for each mosaic
    Output:
        ent (float): The entropy of the mosaic retrieval results
        label_count (dict): The diagnois and the corresponding weight for each mosaic
        hamming_dist (list): A list of hamming distance between the input mosaic and the result
    """
    if len(bag) >= 1:
        label = []
        hamming_dist = []
        label_count = defaultdict(float)
        for bres in bag:
            if is_organ:
                label.append(bres['site'])
            else:
                label.append(bres['diagnosis'])
            hamming_dist.append(bres['hamming_dist'])

        # Counting the diagnoiss by weigted count
        # If the count is less than 1, round to 1
        for lb_idx, lb in enumerate(label):
            label_count[lb] += (1. / (lb_idx + 1)) * weight[lb]
        for k, v in label_count.items():
            if v < 1.0:
                v = 1.0
            else:
                label_count[k] = v

        # Normalizing the count to [0,1] for entropy calculation
        total = 0
        ent = 0
        for v in label_count.values():
            total += v
        for k in label_count.keys():
            label_count[k] = label_count[k] / total
        for v in label_count.values():
            ent += (-v * np.log2(v))
        return ent, label_count, hamming_dist
    else:
        return None, None, None


def Clean(len_info, bag_summary):
    """
    Implementation of Clean in the paper
    Input:
        len_info (list): The length of retrieval results for each mosaic
        bag_summary (list): A list that contains the positional index of mosaic,
        entropy, the hamming distance list, and the length of retrieval results
    Output:
        bag_summary (list): The same format as input one but without low quality result
        (i.e, result with large hamming distance)
        top5_hamming_distance (float): The mean of average hamming distance in top 5
        retrival results of all mosaics
    """
    LOW_FREQ_THRSH = 3
    LOW_PRECENT_THRSH = 5
    HIGH_PERCENT_THRSH = 95
    len_info = [b[-1] for b in bag_summary]
    if len(set(len_info)) <= LOW_FREQ_THRSH:
        pass
    else:
        bag_summary = [b for b in bag_summary if b[-1]
                       > np.percentile(len_info, LOW_PRECENT_THRSH)
                       and b[-1] < np.percentile(len_info, HIGH_PERCENT_THRSH)]

    # Remove the mosaic if its top5 mean hammign distance is bigger than average
    top5_hamming_dist = np.mean([np.mean(b[2][0:5]) for b in bag_summary])

    bag_summary = sorted(bag_summary, key=lambda x: (x[1]))  # sort by certainty
    bag_summary = [b for b in bag_summary if np.mean(b[2][0:5]) <= top5_hamming_dist]
    return bag_summary, top5_hamming_dist


def Filtered_BY_Prediction(bag_summary, label_count_summary):
    """
    Implementation of Filtered_By_Prediction in the paper
    Input:
        bag_summary (list): The same as the output from Clean
        label_count_summary (dict): The dictionary storing the diagnosis occurrence 
        of the retrieval result in each mosaic
    Output:
        bag_removed: The index (positional) of moaic that should not be considered 
        among the top5
    """
    voting_board = defaultdict(float)
    for b in bag_summary[0:5]:
        bag_index = b[0]
        for k, v in label_count_summary[bag_index].items():
            voting_board[k] += v
    final_vote_candidates = sorted(voting_board.items(), key=lambda x: -x[1])
    fv_pointer = 0
    while True:
        final_vote = final_vote_candidates[fv_pointer][0]
        bag_removed = {}
        for b in bag_summary[0:5]:
            bag_index = b[0]
            max_vote = max(label_count_summary[bag_index].items(), key=operator.itemgetter(1))[0]
            if max_vote != final_vote:
                bag_removed[bag_index] = 1
        if len(bag_removed) != len(bag_summary[0:5]):
            break
        else:
            fv_pointer += 1
    return bag_removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate the result of slide level search")
    parser.add_argument("--result_path", required=True, help="The path to the query result")
    parser.add_argument("--site", required=True, help="The anatomic site where the database is built upon")
    args = parser.parse_args()

    # Load the result file and corresponding slide's diagnosis label
    with open(args.result_path, 'rb') as handle:
        results = pickle.load(handle)

    # Count the number of slide in each diagnosis (organ)
    if args.site == 'organ':
        topK_mMV = 10
        type_of_diagnosis = [os.path.basename(e) for e in glob.glob("./DATA/LATENT/*")]
    else:
        topK_mMV = 5
        type_of_diagnosis = [os.path.basename(e) for e in glob.glob("./DATA/LATENT/{}/*"
                             .format(args.site))]
    total_slide = {k: 0 for k in type_of_diagnosis}
    for k, v in results.items():
        total_slide[v['label_query']] += 1

    # Using the inverse count as a weight for each diagnosis
    sum_inv = 0
    for v in total_slide.values():
        sum_inv += (1./v)

    # Set a parameter k  to make the weight sum to k (k = 10, here)
    if args.site == 'organ':
        norm_fact = 30 / sum_inv
    else:
        norm_fact = 10 / sum_inv
    weight = {k: norm_fact * 1./v for k, v in total_slide.items()}

    metric_dict = {k: {'Acc': 0, 'Percision': 0, 'total_slide': 0}
                   for k in weight.keys()}
    bag_for_ret = {k: {} for k in weight.keys()}
    t_start = time.time()

    # Evaluating the result diagnosis by diagnosis
    for evlb in weight.keys():
        eval_label = evlb
        corr = 0
        percision = 0
        avg_percision = 0
        count = 0
        for test_slide in results.keys():
            test_slide_result = results[test_slide]['results']
            label_query = results[test_slide]['label_query']
            if label_query != eval_label:
                continue
            else:
                # Filter out complete failure case (i.e.,
                # All mosaics fail to retrieve a patch that meet the criteria)
                ttlen = 0
                for tt in test_slide_result:
                    ttlen += len(tt)
                if ttlen == 0:
                    count += 1
                    continue
                bag_result = []
                bag_summary = []
                len_info = []
                label_count_summary = {}
                for idx, bag in enumerate(test_slide_result):
                    if args.site == 'organ':
                        ent, label_cnt, dist = Uncertainty_Cal(bag, is_organ=True)
                    else:
                        ent, label_cnt, dist = Uncertainty_Cal(bag)
                    if ent is not None:
                        label_count_summary[idx] = label_cnt
                        bag_summary.append((idx, ent, dist, len(bag)))
                        len_info.append(len(bag))

                bag_summary_dirty = copy.deepcopy(bag_summary)
                bag_summary, hamming_thrsh = Clean(len_info, bag_summary)
                bag_removed = Filtered_BY_Prediction(bag_summary, label_count_summary)

                # Process to calculate the final ret slide
                ret_final = []
                visited = {}
                for b in bag_summary:
                    bag_index = b[0]
                    uncertainty = b[1]
                    res = results[test_slide]['results'][bag_index]
                    for r in res:
                        if uncertainty == 0:
                            if r['slide_name'] not in visited:
                                if args.site == 'organ':
                                    ret_final.append((r['slide_name'], r['hamming_dist'],
                                                      r['site'], uncertainty,
                                                      bag_index))
                                else:
                                    ret_final.append((r['slide_name'], r['hamming_dist'],
                                                      r['diagnosis'], uncertainty,
                                                      bag_index))
                                visited[r['slide_name']] = 1
                        else:
                            if (r['hamming_dist'] <= hamming_thrsh) and\
                               (r['slide_name'] not in visited):
                                if args.site == 'organ':
                                    ret_final.append((r['slide_name'], r['hamming_dist'],
                                                      r['site'], uncertainty,
                                                      bag_index))
                                else:
                                    ret_final.append((r['slide_name'], r['hamming_dist'],
                                                      r['diagnosis'], uncertainty,
                                                      bag_index))
                                visited[r['slide_name']] = 1

                ret_final_tmp = [(e[1], e[2], e[3], e[-1]) for e in
                                 sorted(ret_final, key=lambda x: (x[3], x[1]))
                                 if e[-1] not in bag_removed]
                ret_final = [e[2] for e in
                             sorted(ret_final, key=lambda x: (x[3], x[1]))
                             if e[-1] not in bag_removed][0:topK_mMV]

                # MAP calculation
                ap_at_k = 0
                corr_index = []
                for l in range(len(ret_final)):
                    if ret_final[l] == eval_label:
                        corr_index.append(l)
                if len(corr_index) == 0:
                    avg_percision += ap_at_k
                else:
                    for i_corr in corr_index:
                        ap_at_idx_tmp = 0
                        for j in range(i_corr + 1):
                            if ret_final[j] == eval_label:
                                ap_at_idx_tmp += 1
                        ap_at_idx_tmp /= (i_corr + 1)
                        ap_at_k += ap_at_idx_tmp
                    ap_at_k /= 5
                    avg_percision += ap_at_k

                hit_label = Counter(ret_final).most_common(1)[0][0]
                if hit_label == label_query:
                    if len(ret_final) == topK_mMV:
                        corr += 1
                    elif len(ret_final) < topK_mMV and\
                            Counter(ret_final).most_common(1)[0][1] >= math.ceil(topK_mMV / 2):
                        corr += 1
                count += 1
        metric_dict[evlb]['Acc'] = corr / count
        metric_dict[evlb]['Percision'] = avg_percision / count
        metric_dict[evlb]['total_slide'] = count
    print(time.time() - t_start)
    print(metric_dict)
