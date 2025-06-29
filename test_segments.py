import os
import pickle as pkl
import cv2
from tqdm import tqdm
from time import perf_counter

from image_captioner import IC
from Clip import Clip
from XIC import XIC
from utils import log_print

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def get_proposed_mask(file_path, stage='final'):
    segments_dict = pkl.load(open(file_path,'rb'))
    return segments_dict[f'{stage} proposed_mask']


captioner = IC(dataset='flickr8k', weights_dir='./weights')
clip_model = Clip(selected_model='ViT-B/32')

modes = ['XIC', 'iXIC', 'iXIC_clip']
DSs = ['coco', 'lvis']
exps = ['120_0_True_True_False', '120_0_False_True_False', '120_0_False_True_True',
        '120_1_False_True_False', '120_1_True_True_False', '120_0_False_True_True',
        'auto_1_True_True_False', 'auto_1_True_True_True', '20_0_False_False_False']


for ds in DSs:
    for mode in modes:
        for exp in exps:

            stage='final' if exp[:2]!='20' else 'stage 2'

            images_dir = f'.\\data\\test\\{ds}\\images_dir'
            answers_dir = f'E:\PhD\Publications\\2023_CogInfoCom\\test_output\\{ds}\\save_dir\\{exp}'
            # answers_dir = f'E:\\PhD\\Publications\\2023_SACI\\test_data\\data\\test\\flickr8k\\instance_seg_{ds}\save_dir' # for 20 exps
            save_dir = f'.\\data\\test\\{ds}\\save_dir - Drive\\{exp}\\test_results'

            if not os.path.isdir(save_dir): os.mkdir(save_dir)

            image_query_file = f'E:\PhD\Publications\\2023_CogInfoCom\\test_output\\{ds}\\save_dir - Drive\\{exp}\\image_query.txt'
            # image_query_file = f'.\\data\\test\\{ds}\\save_dir - Drive\\{exp}\\image_query.txt' # for 20 exps

            image_query_dict = dict()
            with open(image_query_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split()
                    key = f'{items[0]}_{items[1]}'
                    value = ' '.join(items[2:])
                    image_query_dict[key] = value

            results = list()
            log_print(f"calculating {ds} {exp} {mode} results")
            start = perf_counter()
            for filename in tqdm(os.listdir(answers_dir)):
                if stage in filename:
                    idx_1 = filename.index('_')
                    idx_2 = filename.rindex('_')
                    image_id = filename[:idx_1]
                    query_id = filename[idx_1+1:idx_2]

                    file = os.path.join(answers_dir, filename)
                    proposed_mask = get_proposed_mask(file, stage)
                    image = os.path.join(images_dir, f'{image_id}.jpg')
                    image = cv2.imread(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    query = image_query_dict[f'{image_id}_{query_id}']
                    score, test_images_captions = XIC(image_id=image_id, query_id=query_id, image=image, query=query, answer=proposed_mask, captioner=captioner, clip_model=clip_model, mode=mode, b_ksize=120, save_dir=save_dir)
                    results.append([image_id, query_id, query, score] + test_images_captions)


            pref_time = perf_counter() - start
            log_print(f"Testing is done in less than {int(pref_time/60)+1} min ({pref_time} sec)!")

            with open(os.path.join(save_dir, f'{mode}_results.txt'), 'w') as f:
                f.write(f'image_id\tquery_id\tquery\t{mode}\tblack\twhite\tgrey\n')
                for result in results:
                    f.write(f'{result[0]}\t{result[1]}\t{result[2]}\t{result[3]}\t{result[4]}\t{result[5]}\t{result[6]}\n')