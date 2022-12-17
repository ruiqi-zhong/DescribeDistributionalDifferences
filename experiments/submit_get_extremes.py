import pickle as pkl
import os
import random

applications = pkl.load(open('data/benchmark_applications_1stdraft.pkl', 'rb'))
prioritized_pair_ids = [3, 5, 6, 12, 24, 25, 30, 32, 34, 36, 37, 41, 47, 53, 55, 57, 60, 61, 64, 66, 69, 70, 73, 75, 76, 77, 81, 83, 86, 88, 94, 98, 101, 102, 103, 107, 112, 115, 122, 122, 123, 130, 132, 133, 138, 140, 142, 143, 143, 145, 148, 149, 150, 153, 159, 164, 165, 167, 167, 168, 170, 171, 173, 177, 178, 181, 183, 186, 188, 192, 193, 196, 197, 198, 202, 202, 204, 206, 207, 208, 212, 213, 215, 219, 221, 222, 225, 226, 227, 227, 230, 233, 237, 243, 254, 261, 264, 267, 269, 270, 298, 309, 322, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 362, 469, 492, 506, 534, 589, 612, 635, 679, 789, 799, 828, 866, 966, 1004, 1005, 1028, 1057, 1322, 1455, 1859]
others = list(set(application['pair_id'] for application in applications) - set(prioritized_pair_ids))
pair_ids = prioritized_pair_ids + others

for i, pair_id in enumerate(pair_ids):
    all_choices = ['sunstone', 'rainbowquartz', 'balrog', 'saruman']
    restricted_choices = ['sunstone', 'rainbowquartz']
    if i < 134:
        partition = random.choice(all_choices)
    else:
        partition = random.choice(restricted_choices)
    cmd = 'sbatch -p jsteinhardt -w %s --gres=gpu:1 --export=PAIR_ID="%d" get_extreme.sh' % (partition, pair_id)
    print(cmd)
    os.system(cmd)
