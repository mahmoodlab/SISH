import pickle
import time
import argparse
import math
from collections import Counter, defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate the result of patch level search")
    parser.add_argument("--result_path", required=True)
    args = parser.parse_args()

    # Load the result file and corresponding slide's diagnosis label
    with open(args.result_path, 'rb') as handle:
        results = pickle.load(handle)
    total_slide = defaultdict(int)
    for v in results.values():
        total_slide[v['label_query']] += 1

    metric_dict = {k: {'Acc': 0, 'Percision': 0, 'total_slide': 0}
                   for k in total_slide.keys()}
    topk_MV = 5
    ret_dict = defaultdict(list)
    t_start = time.time()
    for evlb in total_slide.keys():
        # Evaluating the result diagnosis by diagnoiss
        corr = 0
        percision = 0
        avg_percision = 0
        count = 0
        for patch in results.keys():
            test_patch_result = results[patch]['results']
            label_query = results[patch]['label_query']
            if label_query != evlb:
                continue
            else:
                # Process to calculate the final ret slide
                ret_final = [r[1] for r in test_patch_result[0:topk_MV]]
                ap_at_k = 0
                corr_index = []
                for lb in range(len(ret_final)):
                    if ret_final[lb] == evlb:
                        corr_index.append(lb)
                if len(corr_index) == 0:
                    avg_percision += ap_at_k
                else:
                    for i_corr in corr_index:
                        ap_at_idx_tmp = 0
                        for j in range(i_corr + 1):
                            if ret_final[j] == evlb:
                                ap_at_idx_tmp += 1
                        ap_at_idx_tmp /= (i_corr + 1)
                        ap_at_k += ap_at_idx_tmp
                    ap_at_k /= 5
                    avg_percision += ap_at_k
                if len(ret_final) != 0:
                    hit_label = Counter(ret_final).most_common(1)[0][0]
                else:
                    hit_label = 'NA'

                if hit_label == label_query:
                    if len(ret_final) == topk_MV:
                        corr += 1
                    elif len(ret_final) < topk_MV and\
                            Counter(ret_final).most_common(1)[0][1] >= math.ceil(topk_MV):
                        corr += 1
                else:
                    pass
                count += 1
        metric_dict[evlb]['Acc'] = corr / count
        metric_dict[evlb]['Percision'] = avg_percision / count
        metric_dict[evlb]['total_slide'] = count
    print(time.time() - t_start)
    print(metric_dict)
