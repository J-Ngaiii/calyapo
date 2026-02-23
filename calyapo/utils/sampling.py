from typing import Iterable, List, Dict    
import random
import numpy as np

from calyapo.configurations.config import UNIVERSAL_RANDOM_SEED

random.seed(UNIVERSAL_RANDOM_SEED)
np.random.seed(UNIVERSAL_RANDOM_SEED)

class ReservoirSample:
    def __init__(self, values: Iterable, k: int, n: int):
        self.reservoir = []
        self.out_sample = []

        self.k = k
        self.n = n

        # begin sampling initially inputed list of values
        for i in range(k):
            self.reservoir.append(values[k])

        counter = k
        while counter < n:
            j = random.randrange(counter+1)

            if j < k:
                self.out_sample.append(self.reservoir[j])
                self.reservoir[j] = values[counter]
            else:
                # datapoint didn't roll into sample
                self.out_sample.append(values[counter])
            
            counter +=1 
    
    def get(self, sample: str):
        if sample == 'reservoir':
            return self.reservoir
        elif sample == 'out_sample':
            return self.out_sample
        
def _num_unique_individuals(value_buckets: Iterable[Iterable]):
    unique_pool = set()
    for b in value_buckets:
        for item in b:
            uid = item.get('id') if isinstance(item, dict) else item
            unique_pool.add(uid)
    return len(unique_pool), unique_pool
        
def exhaustive_hierarchal_sample(value_buckets: Iterable[Iterable[Dict]], targ_bucket_idx: int, bucket_distrib: Iterable, silent_handle: bool = False, debug: bool = False):
    assert targ_bucket_idx < len(bucket_distrib), f"Target bucket idx '{targ_bucket_idx}' exceeds number of buckets given distribution '{len(bucket_distrib)}'"
    assert len(value_buckets) == len(bucket_distrib), f"Number of values buckets '{len(value_buckets)}' does not match distributions specified '{len(bucket_distrib)}'"
    
    if debug:
        ids_in_targ = [indiv.get('id') for indiv in value_buckets[targ_bucket_idx]]
        print(f"(exhaus_heir_samp | Debug) bucket cleanliness check: total entries '{len(ids_in_targ)}' vs num unique entries: '{len(set(ids_in_targ))}'")

    num_elem_lst = [len(sublist) for sublist in value_buckets]
    total_unique_count, global_id_pool = _num_unique_individuals(value_buckets)
    targ_num = bucket_distrib[targ_bucket_idx] * total_unique_count

    if debug:
        print(f"(exhaus_heir_samp | Debug) smallest bucket length <idx = '{targ_bucket_idx}'>: '{len(value_buckets[targ_bucket_idx])}'")
        print(f"(exhaus_heir_samp | Debug) distrib: '{bucket_distrib}'")
        print(f"(exhaus_heir_samp | Debug) num unique indivs: '{total_unique_count}'")
        print(f"(exhaus_heir_samp | Debug) targ num: '{targ_num}'")

    if targ_num > num_elem_lst[targ_bucket_idx]:
        err_msg = f"(exhaustive sampler) target number '{targ_num}' higher than total number of elements in given bucket '{num_elem_lst[targ_bucket_idx]}' exiting"
        if silent_handle: 
            print(err_msg)
            return 
        else:
            raise ValueError(err_msg)
    
    # for the bucket with the least datapoints (eg. val w/ only 1169) --> we reservoir sample until we meet 10% (eg. 900 individuals) then redistribute
    sampler = ReservoirSample(values=value_buckets[targ_bucket_idx], k=int(targ_num), n=num_elem_lst[targ_bucket_idx])

    output_buckets = [[] for _ in range(len(value_buckets))]
    output_buckets[targ_bucket_idx] = sampler.get(sample='reservoir')
    consumed_ids = set([indiv_map.get('id') for indiv_map in output_buckets[targ_bucket_idx]])

    remaining_distribution = [prob / (1 - bucket_distrib[targ_bucket_idx]) for i, prob in enumerate(bucket_distrib) if i != targ_bucket_idx]
    remaining_buckets = [i for i in range(len(value_buckets)) if i != targ_bucket_idx]

    if debug:
        print(f"(exhaus_heir_samp | Debug) remaining distrib: '{remaining_distribution}'")
        print(f"(exhaus_heir_samp | Debug) remaining buckets: '{remaining_buckets}'")
        print(f"(exhaus_heir_samp | Debug) num in out_sample: '{len(sampler.get(sample='out_sample'))}'")
        
    # sweep through out sample 
    for val in sampler.get(sample='out_sample'):
        val_id = val.get('id')
        if val_id in consumed_ids:
            continue

        bucket_idx = np.random.choice(remaining_buckets, p=remaining_distribution)
        output_buckets[bucket_idx].append(val)

        consumed_ids.add(val_id)

    if debug:
        print(f"(exhaus_heir_samp | Debug) initial counts per bucket after processing out sample:\n")
        for i in range(len(output_buckets)):
            print(f"Index '{i}'; length '{len(output_buckets[i])}'")

    # sweep thru everyone else
    for indiv_map in global_id_pool:
        val_id = indiv_map.get('id')
        if val_id not in consumed_ids:
            # need to assign individual to a split they're actually eligible for not just random choice it
            bucket_idx = np.random.choice(remaining_buckets, p=remaining_distribution)
            output_buckets[bucket_idx].append(indiv_map)
            consumed_ids.add(val_id)

    return output_buckets
