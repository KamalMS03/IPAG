from nltk.chunk import ChunkParserI
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.tag import UnigramTagger, BigramTagger
from nltk import NgramTagger, word_tokenize, pos_tag
from nltk.corpus import treebank_chunk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import filters
from PIL import Image

from time import perf_counter, gmtime, strftime
import os


IMAGE_SHAPE=(224, 224, 3)
def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn(img, attn_map, blur=True):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    plt.show()

def load_image1(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SHAPE[:-1])
    return img

def masked_loss(labels, preds):  
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)

  mask = (labels != 0) & (loss < 1e8) 
  mask = tf.cast(mask, loss.dtype)

  loss = loss*mask
  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss

def masked_acc(labels, preds):
  mask = tf.cast(labels!=0, tf.float32)
  preds = tf.argmax(preds, axis=-1)
  labels = tf.cast(labels, tf.int64)
  match = tf.cast(preds == labels, mask.dtype)
  acc = tf.reduce_sum(match*mask)/tf.reduce_sum(mask)
  return acc

def conll_tag_chunks(chunk_data):
	
	tagged_data = [tree2conlltags(tree) for
					tree in chunk_data]
	
	return [[(t, c) for (w, t, c) in sent]
			for sent in tagged_data]

def log_print(message_to_print, file_path='./test_log'):
  
  if not os.path.exists(file_path):
    os.makedirs(file_path)

  log_file = f'{file_path}/output.txt'
  print(message_to_print)
  with open(log_file, 'a') as f:
    f.write(message_to_print + '\n')

class TagChunker(ChunkParserI):
  def __init__(self, train_chunks,
				tagger_classes =[UnigramTagger, BigramTagger]):
            
         train_data = conll_tag_chunks(train_chunks)
         # self.tagger = backoff_tagger(train_data, tagger_classes)
         tagger = None
         for n in range(1,4):  # start at unigrams (1) up to and including trigrams (3)
            tagger = NgramTagger(n, train_data, backoff=tagger)
         self.tagger = tagger

  def parse(self, tagged_sent):
   if not tagged_sent:
      return None
   (words, tags) = zip(*tagged_sent)
   chunks = self.tagger.tag(tags)
   wtc = zip(words, chunks)
   
   return conlltags2tree([(w, t, c) for (w, (t, c)) in wtc])

class Chunker():
  def __init__(self) -> None:
    train_data = treebank_chunk.chunked_sents()[:3000]
    #  self.test_data = treebank_chunk.chunked_sents()[3000:]
    self.chunker = TagChunker(train_data)
      
  def get_chunks(self, sentence):
    tokenized_text = word_tokenize(sentence)
    tagged = pos_tag(tokenized_text)
    chunked = self.chunker.parse(tagged)
    chunks = list()
    for s in chunked.subtrees(lambda t: t.height() == 2):
      chunk = ' '.join([i[0] for i in s.leaves()])
      # print("subtree: ", chunk)
      chunks.append(chunk)
    return chunks

