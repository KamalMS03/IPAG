import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.append(".\CLIP_Explainability\code")
import re
import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import numpy as np
from tqdm import tqdm
import pickle as pkl
from time import perf_counter, gmtime, strftime
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import spacy
from nltk.corpus import stopwords
from PIL import Image
from Clip import Clip
from utils import Chunker, log_print
from clip_ import load, tokenize
from vit_cam import interpret_vit #saliency Map for CLIP


class Explainer():
    def __init__(self, captioner, instance_seg, random_seg, text_encoder, \
                 blur_ksize=100, clip_mode='0', pobj_mode=False, improved_XIC=False, \
                 stage_hi_sim=False, clip_model="ViT-B/32", spacy_model="en_core_web_sm") -> None:
        self.CaptioningModel = captioner
        self.InstanceSegModel = instance_seg
        self.RandomSegModel = random_seg
        self.text_encoder = text_encoder

        self.ClipModel = Clip(selected_model=clip_model)
        self.SpacyModel = spacy.load(spacy_model)

        self.image_path = None
        self.main_caption = None
        self.main_caption_dependency = None
        self.query = None
        self.query_dependency = None
        self.image = None
        self.blurred = None
        self.font = {
    'size': 15,
}
        self.current_stage = None
        self.current_stage_proposals = None
        self.device = "cpu"
        self.stages_results = self.initialize_stages_results()
        self.model_vit, self.preprocess = load("ViT-B/32", device=self.device, jit=False)
        self.chunker = Chunker()
        self.stops = stopwords.words('english')
        self.ori_preprocess = Compose([Resize((224), interpolation=Image.BICUBIC),CenterCrop(size=(224, 224)),ToTensor()])
        self.test_time = None
        self.blur_ksize = blur_ksize
        self.image_blur_ksize = None
        self.clip_mode = clip_mode
        self.pobj_mode = pobj_mode
        self.improved_XIC = improved_XIC
        self.stage_hi_sim = stage_hi_sim

    def initialize_stages_results(self):
        stage_dict = {'selected_ids ': [], 'max_score': [], 'proposed_mask': [], 'proposal': [], 'segments':[], 'captions':[]}
        results_dict = {'stage 1': stage_dict.copy(), 'stage 2': stage_dict.copy(), 'final': stage_dict.copy()}
        return results_dict
    
    def explain(self, image_path, mode='loop', test_query='', save_dir='', image_id='', query_id=''):
        loop_cond = True
        self.validate_image_path(image_path)
        self.im = self.ori_preprocess(Image.open(image_path))
        if self.image_path:
            if not self.main_caption: self.caption_image(self.image, main=True)
            while(loop_cond):
                if mode=='loop':
                    loop_cond = self.get_query()
                    if not loop_cond: break
                elif mode=='test':
                    self.query = test_query.lower()
                    loop_cond = False
                self.set_query_dependency()
                log_print(f"Query is {self.query}")
                log_print(f'Query dependency is: {self.query_dependency}')
                log_print("Query accepted!")
                log_print("Generating segments!")
                start = perf_counter()
                self.instance_segmentation()
                self.random_segmentation()
                pref_time = perf_counter() - start
                log_print(f"Segments generation is done in less than {int(pref_time/60)+1} min ({pref_time} sec)!")
                start = perf_counter()
                log_print("Stage 1 ...")
                self.stageI()
                pref_time = perf_counter() - start
                log_print(f"Stage 1 is done in less than {int(pref_time/60)+1} min ({pref_time} sec)!")
                start = perf_counter()
                self.stageII()
                pref_time = perf_counter() - start
                log_print(f"Stage 2 is done in less than {int(pref_time/60)+1} min ({pref_time} sec)!")
                self.answer(save_dir, image_id, query_id)

    def validate_image_path(self, image_path):
        if os.path.isfile(image_path):
            self.image_path = image_path
            image = cv2.imread(self.image_path)
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #self.image = cv2.resize(self.image,(224,224))
            log_print("Setting blurring kernel size!")
            self.set_blur_ksize()
            log_print(f"Blurring kernel size is: {self.image_blur_ksize}")
            self.blurred = cv2.blur(self.image, (self.image_blur_ksize, self.image_blur_ksize))
        else:
            log_print('Image was not found!')

    def set_blur_ksize(self):
        if self.blur_ksize == 'auto':
            log_print("Automatic setting!")
            image_caption = self.remove_stops(self.caption_image(self.image))
            image_caption = image_caption.split()
            for blur_ksize in range(20, 140, 20):
                blur_image = cv2.blur(self.image, (blur_ksize, blur_ksize))
                blur_caption = self.remove_stops(self.caption_image(blur_image))
                blur_caption = blur_caption.split()
                if not(set(image_caption) & set(blur_caption)): break
            self.image_blur_ksize = blur_ksize
        else:
            log_print("Manual setting!")
            self.image_blur_ksize = int(self.blur_ksize)

    def caption_image(self, image, main=False):
        caption = self.CaptioningModel.caption_image(image).lower()
        if main:
            self.main_caption = caption
            self.extract_caption_dependency()
            log_print(f'\nOriginal caption:{self.main_caption}\n')
            plt.imshow(self.image)
            plt.title('Original image')
            plt.show()
        return caption

    def extract_caption_dependency(self):
        main_caption_dependency = dict()
        for token in self.SpacyModel(self.main_caption):
            main_caption_dependency[str(token)] = token.dep_
        self.main_caption_dependency = main_caption_dependency
    
    def set_query_dependency(self):
        query_dependency = list()
        for token in self.query.split():
            query_dependency.append(self.main_caption_dependency[token])
        self.query_dependency = ' '.join(query_dependency)

    def get_query(self):
        good_query = False
        while(not good_query):
            query = input('Enter query (from the original caption) or "NAN" to exit:\n').lower().strip()
            if (query != '') and (re.search(r"\b{}\b".format(query), self.main_caption.strip())):
                good_query = True
            elif query=='nan':
                log_print("Exit!")
                return False
            else:
                log_print(f'"{query}" is not a valid query!')
        self.query = query
        return True

    def instance_segmentation(self):
        if len(self.stages_results['stage 1']['segments'])==0:
            log_print("Instance segmentation segments are NOT available!")
            self.InstanceSegModel.predict(self.image)
            self.InstanceSegModel.create_proposals(self.image_blur_ksize)
            self.stages_results['stage 1']['segments'] = self.InstanceSegModel.proposals
        else:
            log_print("Instance segmentation segments are READY!")
            self.InstanceSegModel.proposals = self.stages_results['stage 1']['segments']

    def random_segmentation(self):   
        if len(self.stages_results['stage 2']['segments'])==0:
            log_print("Low-level segmentation segments are NOT available!")
            self.RandomSegModel.create_segments(self.image) # it is only done if the image does not have segments
        self.stages_results['stage 2']['segments'] = self.RandomSegModel.segments
        log_print("Low-level segmentation segments are READY!")

    def stageI(self):
        self.current_stage = 'stage 1'
        self.current_stage_proposals = None
        self.current_stage_proposals = self.InstanceSegModel.proposals
        self.nominate_proposal()

    def nominate_proposal(self):
        ids, scores = self.calculate_similarity_scores()
        self.select_highest(ids, scores)
        self.merge_proposals() # useful when there is more than region with the max similarity score
        if self.current_stage == 'stage 2':
            selected_stage = 'stage 2'
            if self.stage_hi_sim:
                selected_stage = 'stage 1' if self.stages_results['stage 1']['max_score'] > self.stages_results['stage 2']['max_score'] else 'stage 2'
            
            self.stages_results['final']['max_score'] = self.stages_results[selected_stage]['max_score']
            self.stages_results['final']['proposal'] = self.stages_results[selected_stage]['proposal']
            self.stages_results['final']['proposed_mask'] = self.stages_results[selected_stage]['proposed_mask']

    def calculate_similarity_scores(self):
        scores = list()
        ids = list()
        log_print(f'{self.current_stage} calculate_similarity_scores starts')
        start = perf_counter()

        for idx, image_size, mask_size, proposed_segment, mask in tqdm(self.current_stage_proposals):
            ids.append(idx)
            sim_score = self.calculate_similarity_score(proposed_segment)

            weight = 1
            if (self.current_stage=='stage 1'):
                if not self.pobj_mode:
                    weight = (mask_size/image_size)
                elif not ('pobj' in self.query_dependency):
                    weight = (mask_size/image_size)

            log_print(f'{str(sim_score)} / {str(weight)}')
            sim_score_w = sim_score/weight
            if sim_score_w >= 1: sim_score_w = 1
            scores.append(sim_score_w)

        pref_time = perf_counter() - start
        log_print(f'\nmeasuring similarity score for {len(ids)} segments is done in {pref_time} seconds...\n')
        log_print(f'{self.current_stage} calculate_similarity_scores is done')
        ids = [x for _, x in sorted(zip(scores, ids), reverse=True)]
        scores.sort(reverse=True)
        log_print(f'{self.current_stage} score: {max(scores)}')
        return ids, scores

    def calculate_similarity_score(self, proposed_segment):
        if (self.clip_mode=='1' and self.current_stage=='stage 1') or \
           (self.clip_mode=='2' and self.current_stage=='stage 2') or \
           (self.clip_mode=='both'):
            return self.calculate_clip_similarity_score(proposed_segment, self.query)
        else:
            return self.calculate_cosine_similarity_score(proposed_segment, self.query)

    def remove_stops(self, sent):
        words = [word for word in sent.split() if word.lower() not in self.stops]
        text = ' '.join(words)
        return text

    def calculate_clip_similarity_score(self, image, query):
        self.stages_results[self.current_stage]['captions'].append("CLIP") # no caption is used in calculating CLIP score
        return self.ClipModel.measure_similarity(image, query)[0][0]

    def calculate_cosine_similarity_score(self, proposed_segment, query):
        caption = self.caption_image(proposed_segment)
        self.stages_results[self.current_stage]['captions'].append(caption)
        return self.calculate_text_cosine_similarity_score(caption, query, remove_stops=False)

    def calculate_text_cosine_similarity_score(self, text_1, text_2, remove_stops=True):
        if remove_stops:
            text_1 = self.remove_stops(text_1)
            text_2 = self.remove_stops(text_2)
        corpus = [text_1, text_2]

        if self.text_encoder=='count':
            text_encoder_model = CountVectorizer().fit_transform(corpus)
            corpus = text_encoder_model.toarray()
        elif self.text_encoder=='bert':
            text_encoder_model = SentenceTransformer('bert-base-nli-mean-tokens')
            corpus = text_encoder_model.encode(corpus)
        elif self.text_encoder=='roberta':
            text_encoder_model = SentenceTransformer('stsb-roberta-large')
            corpus = text_encoder_model.encode(corpus)
        
        vect_1 = corpus[0].reshape(1, -1)
        vect_2 = corpus[1].reshape(1, -1)

        return cosine_similarity(vect_1, vect_2)[0][0]
    
    def select_highest(self, sorted_ids, scores):
        max_score = max(scores)
        selected_ids = list()
        for idx in range(len(sorted_ids)):
            if scores[idx] != max_score: break # the highest score only
            selected_ids.append(sorted_ids[idx])
        # self.stageI_selected_ids = selected_ids
        self.stages_results[self.current_stage]['selected_ids'] = selected_ids
        # self.stageI_max_score = max_score
        self.stages_results[self.current_stage]['max_score'] = max_score

    def merge_proposals(self):
        selected_ids = self.stages_results[self.current_stage]['selected_ids']
        selected_proposals = [self.current_stage_proposals[i] for i in selected_ids]
        #print(selected_propo
        #print("\n\n\n",selected_proposals[0][4])
        proposed_mask = selected_proposals[0][4].copy() # [0] the first item in the list, [4] the binary mask
        #print(proposed_mask.shape)
        for proposal in selected_proposals:
            proposed_mask = np.logical_or(proposed_mask, proposal[4])
        self.stages_results[self.current_stage]['proposed_mask'] = proposed_mask
        proposal = self.blurred.copy()
        #print(self.blurred.shape)
        #print(proposed_mask.shape)
        proposal[proposed_mask] = self.image[proposed_mask]
        self.stages_results[self.current_stage]['proposal'] = proposal
        
    def stageII(self):
        self.current_stage = 'stage 2'
        self.current_stage_proposals = None
        proposed_mask = self.stages_results['stage 1']['proposed_mask']
        self.RandomSegModel.create_proposals(self.image, proposed_mask, self.image_blur_ksize)
        self.current_stage_proposals = self.RandomSegModel.proposals
        self.nominate_proposal()
    
    def answer(self, save_dir='', image_id='', query_id=''):
        log_print(f"Caption: {self.main_caption}")
        log_print(f"Query: {self.query}")
        selected_stage = 'final'
        log_print(f"Answer similarity score: {self.stages_results[selected_stage]['max_score']}")

        masked_image = np.zeros_like(self.image)
        masked_image[:,:] = (255, 0, 0)
        masked_image[self.stages_results[selected_stage]['proposed_mask']!=True] = self.image[self.stages_results[selected_stage]['proposed_mask']!=True]
        p_image = self.im.unsqueeze(0).to(self.device)
        t_text = tokenize(self.query).to(self.device)
        text_features = self.model_vit.encode_text(t_text)
        if save_dir and image_id:
            save_path = str(os.path.join(save_dir, f'{image_id}_{query_id}_blurred.jpg'))
            plt.imsave(save_path, self.stages_results[selected_stage]['proposal'])
            save_path = str(os.path.join(save_dir, f'{image_id}_{query_id}_masked.jpg'))
            plt.imsave(save_path, masked_image)

            results_file = save_dir + f'results_{self.test_time}.txt'
            with open(results_file, 'a') as file:
                result_line = f"\n{image_id}, {self.main_caption}, {query_id}, {self.query}, "
                result_line += f"{'1' if self.is_good_answer() else '0'}, "
                result_line += f"{self.stages_results[selected_stage]['max_score']}, "
                result_line += f"{self.image.shape[0]*self.image.shape[1]}, "
                result_line += f"{int(np.count_nonzero(self.stages_results[selected_stage]['proposed_mask'])/3)}, "
                result_line += f"{self.image_blur_ksize}, "
                result_line += f"{self.query_dependency}"

                file.write(result_line)

            stage = 'stage 1'
            results_file = save_dir + f'{image_id}_{stage}.bin'
            self.dump_results(results_file, stage)

            stage = 'stage 2'
            results_file = save_dir + f'{image_id}_{query_id}_{stage}.bin'
            self.dump_results(results_file, stage)

            stage = 'final'
            results_file = save_dir + f'{image_id}_{query_id}_{stage}.bin'
            self.dump_results(results_file, stage)

            plt.figure(figsize=(20,20))
            plt.tight_layout()
            plt.subplot(191)
            plt.imshow(self.image)
            plt.axis('off')
            plt.title("original", **self.font, y=-0.20)
            
            plt.subplot(192)
            plt.imshow(masked_image)
            plt.axis('off')
            plt.title(f"{self.query}", **self.font, y=-0.20)
            
            plt.subplot(193)
            interpret_vit(p_image.type((self.model_vit).dtype), text_features, self.model_vit.visual ,self.device)
            plt.axis('off')
            plt.title(f"{self.query}", **self.font, y=-0.20)
            
            plt.show()
            

    def dump_results(self, file_path, stage):
        segments_dict = dict()
        segments_dict[f'{stage} segments'] = self.stages_results[stage]['segments']
        segments_dict[f'{stage} proposal'] = self.stages_results[stage]['proposal']
        segments_dict[f'{stage} proposed_mask'] = self.stages_results[stage]['proposed_mask']
        segments_dict[f'{stage} captions'] = self.stages_results[stage]['captions']
        pkl.dump(segments_dict, open(file_path,'wb'))

        
    def read_results(self, file_path, stage):
        segments_dict = pkl.load(open(file_path,'rb'))
        self.stages_results[stage]['segments'] = segments_dict[f'{stage} segments']
        self.stages_results[stage]['proposal'] = segments_dict[f'{stage} proposal']
        self.stages_results[stage]['proposed_mask'] = segments_dict[f'{stage} proposed_mask']
        self.stages_results[stage]['captions'] = segments_dict[f'{stage} captions']

        
    def test_images(self, save_dir, images_dir):
        if not os.path.isdir(images_dir):
            log_print('Invalid test directory!')
            return False

        self.test_time = strftime("%m_%d_%H_%M_%S", gmtime())
        results_file = save_dir + f'results_{self.test_time}.txt'
        if not os.path.isfile(results_file):
            with open(results_file, 'w') as file:
                file.write(f'instance_seg_ds\t{self.InstanceSegModel.dataset}, \
                           random_seg_algo\t{self.RandomSegModel.algo_type}, blur_mode\t{self.blur_ksize}, clip_mode\t{self.clip_mode}, \
                           pobj_mode\t{self.pobj_mode}, improved_XIC\t{self.improved_XIC}\n')
                file.write('image_id, caption, query_id, query, good_query, sim_score, image_size, answer_size, blur_kszie, query_dependency')

        images_names = next(os.walk(images_dir), (None, None, []))[2]
        # partial test
        images_names = [int(name[:name.index('.')]) for name in images_names]
        images_names.sort()
        images_names = [str(name)+'.jpg' for name in images_names]
        
        for name in tqdm(images_names):
            segments_dict = dict()
            self.reset_parameters()
            ext_idx = name.index('.')
            image_id = name[:ext_idx]
            image_results = self.test_image(image_path=images_dir+name, save_dir=save_dir, image_id=image_id)   

    def reset_parameters(self):
        self.image_path = None
        self.main_caption = None
        self.main_caption_dependency = None
        self.query = None
        self.query_dependency = None
        self.image = None
        self.blurred = None
        self.current_stage = None
        self.current_stage_proposals = None
        self.stages_results = self.initialize_stages_results()

    def test_image(self, image_path, save_dir, image_id):
        log_print(f'Testing image {image_id}')
        results_dict = dict()
        self.validate_image_path(image_path)
        if self.image_path: self.caption_image(self.image, main=True)
        queries = self.generate_test_quries()
        log_print(f"Queries: {queries}")

        stage = 'stage 1'
        results_file = save_dir + f'{image_id}_{stage}.bin'
        if os.path.isfile(results_file): self.read_results(results_file, stage)

        self.random_segmentation()  # Moda: improvement , random segmentation for the whole image is done only one time per image, in Stage 2 they are masked based on Stage 1 proposal mask

        for query_id, query in enumerate(queries):
            self.explain(image_path=self.image_path, mode='test', test_query=query, save_dir=save_dir, image_id=image_id, query_id=query_id)
            results_dict[query] = 1 if self.is_good_answer() else 0
        return results_dict

    def generate_test_quries(self):
        queries = self.chunker.get_chunks(self.main_caption)
        queries = list(set(queries))
        return queries

    def is_good_answer(self):
        test_images = self.generate_test_images()
        for image in test_images:
            caption = self.caption_image(image)
            if self.query in caption.lower(): return True
        return False

    def generate_test_images(self):
        test_images = list()
        mask = self.stages_results[self.current_stage]['proposed_mask']

        if self.improved_XIC:
            blurred = self.blurred.copy()
            blurred[mask] = self.image[mask]
            test_images = [blurred]
        else:
            shape = self.image.shape
            black = np.full(shape, 0)
            white = np.full(shape, 1)
            grey = np.random.normal(loc=0.5, scale=0.1, size=shape)
            grey = np.clip(grey, 0, 1)

            black[mask] = self.image[mask]
            grey[mask] = self.image[mask]
            white[mask] = self.image[mask]

            test_images = [black, grey, white]

        return test_images


if __name__ == "__main__":
    pass