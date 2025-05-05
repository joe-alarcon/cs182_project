import os
import numpy as np

np.random.seed(0)

OUT_DIR = "./data/"
try:
    os.open(OUT_DIR, os.O_RDONLY)
except FileNotFoundError:
    os.mkdir(OUT_DIR)

in_distribution_min = 0
in_distribution_max = 201 # exclusive

out_distribution_1_min = 201
out_distribution_1_max = 401 # exclusive

params_dict = {
    1: {
        "name": "data_no_curriculum.txt", # txt file name
        "p_length": [50], # length of prompts - ie. number of examples
        "num_p": lambda x: 2000, # total number of prompts, for each length of prompt
        "base": 10,
    },
    # 2: {
    #     "name": "data_curriculum.txt",
    #     "p_length": [5,10,15], #[50,45,40,35,30,25,20,15,10]
    #     "num_p": lambda x: 3000 - 200 * int(x/5),
    #     "base": 10,
    # },
    # 3: {
    #     "name": "data_no_curriculum_bases.txt",
    #     "p_length": [50],
    #     "num_p": lambda x: 2000,
    #     "base": 2,
    # },
    # 4: {
    #     "name": "data_curriculum_bases.txt",
    #     "p_length": [5,10,15,20,25,30,35,40,45,50], #[50,45,40,35,30,25,20,15,10]
    #     "num_p": lambda x: 3000 - 200 * (x/5),
    #     "base": 2,
    # },
    5: {
        "name": "data_train_5.txt",
        "p_length": [5], #[50,45,40,35,30,25,20,15,10]
        "num_p": lambda x: 10000,
        "base": 10,
    },
    6: {
        "name": "test_length_5.txt",
        "p_length": [5], #[50,45,40,35,30,25,20,15,10]
        "num_p": lambda x: 1000,
        "base": 10,
    },
    7: {
        "name": "data_train_10.txt",
        "p_length": [10],
        "num_p": lambda x: 10000,
        "base": 10,
    },
    8: { 
        "name": "data_train_length_mixture.txt",
        "p_length": [-1],
        "num_p": lambda x: 10000,
        "base": 10
    },
    9: {
        "name": "test_int_noise.txt",
        "p_length": list(range(1,16)),
        "num_p": lambda x: 50,
        "base": 10,
    },
    10: {
        "name": "test_ood_length.txt",
        "p_length": list(range(1,16)),
        "num_p": lambda x: 50,
        "base": 10,
    },
    11: {
        "name": "test_ood_nums.txt",
        "p_length": list(range(1,16)),
        "num_p": lambda x: 50,
        "base": 10,
    },
}

run = 5

is_test_set = run in [6, 9, 10, 11]
if is_test_set:
    np.random.seed(1)

is_ood_nums = run in [11]
if is_ood_nums:
    in_distribution_min = out_distribution_1_min
    in_distribution_max = out_distribution_1_max

is_noisy_dataset = run in [9]
if is_noisy_dataset:
    def noise(label):
        p = 0.15
        noise_num = 0
        positive_noise = (np.random.rand(1) < p).astype(int)[0]
        negatvie_noise = (np.random.rand(1) < p).astype(int)[0]
        noise_num += positive_noise
        noise_num -= negatvie_noise
        return label + noise_num
else:
    noise = lambda x: x

def out_of_distribution_pair(i,j):
    return (i == 3 and j == 8) or (i == 7 and j == 2) or (i == 5 or j == 6) or (i == j)

name = params_dict[run]["name"]
prompt_length_list = params_dict[run]["p_length"]
number_of_prompts_func = params_dict[run]["num_p"]
base = params_dict[run]["base"]

with open(OUT_DIR + name, "w") as f:
    prompt_idx = 0
    for prompt_length in prompt_length_list:
        number_of_prompts = number_of_prompts_func(prompt_length)
        for i in range(number_of_prompts):
            random_base_1 = np.random.randint(2,10)
            random_base_2 = np.random.randint(base,11)
            if base == 2:
                while out_of_distribution_pair(random_base_1, random_base_2):
                    random_base_1 = np.random.randint(2,10)
                    random_base_2 = np.random.randint(base,11)
            if prompt_length == -1:
                curr_prompt_length = np.random.randint(1,11)
            else:
                curr_prompt_length = prompt_length
            for j in range(curr_prompt_length):
                integer = np.random.randint(in_distribution_min,in_distribution_max)
                f.write(f"{np.base_repr(integer,random_base_1)}->{np.base_repr(noise(integer),random_base_2)}")
                if j < (curr_prompt_length - 1):
                    f.write(",")
                else:
                    f.write(f";{random_base_1}->{random_base_2}")
            if i < number_of_prompts-1:
                f.write("\n")
        # Separate different sections of the training data
        prompt_idx += 1
        if prompt_idx < len(prompt_length_list):
            f.write("*")

