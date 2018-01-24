
# coding: utf-8

# In[1]:

import tensorflow as tf
from model import BiLSTM_CRF
import numpy as np
import os, argparse, time, random
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding,vocab_build


# In[2]:

#hypterparameters
tf.flags.DEFINE_string('train_data','data_path','train data source')
tf.flags.DEFINE_string('test_data', 'data_path', 'test data source')
tf.flags.DEFINE_integer('batch_size', 64, 'sample of each minibatch')
tf.flags.DEFINE_integer('epoch', 15, 'epoch of traing')
tf.flags.DEFINE_integer('hidden_dim', 300, 'dim of hidden state')
tf.flags.DEFINE_string('optimizer', 'Adam', 'Adam/Adadelta/Adagrad/RMSProp/Momentum/SG')
tf.flags.DEFINE_boolean('CRF',True, 'use CRF at the top layer. if False, use Softmax')
tf.flags.DEFINE_float('lr', 0.001, 'learing rate')
tf.flags.DEFINE_float('clip', 5.0, 'gradient clipping')
tf.flags.DEFINE_float('dropout', 0.5, 'dropout keep_prob')
tf.flags.DEFINE_boolean('update_embeddings', True, 'update embeddings during traings')
tf.flags.DEFINE_string('pretrain_embedding', 'random', 'use pretrained char embedding or init it randomly')
tf.flags.DEFINE_integer('embedding_dim', 300, 'random init char embedding_dim')
tf.flags.DEFINE_boolean('shuffle', True, 'shuffle training data before each epoch')
tf.flags.DEFINE_string('mode', 'train', 'train/test/demo')
tf.flags.DEFINE_string('demo_model', '1499785643', 'model for test and demo')
tf.flags.DEFINE_string('wordPath', 'data_path/word', 'train/test/demo')
args = tf.flags.FLAGS


# In[3]:

import pickle
def create_voabulary(wordPath=args.wordPath):
    cache_path = "data_path/word_voabulary.pkl"
    #print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))
    # load the cache file if exists
    if os.path.exists(cache_path):
        #print(cache_path)
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word = pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        vocabulary_word2index = {}
        vocabulary_index2word = {}
        words = open(wordPath).readlines()
        print("vocabulary:", len(words))
        for i, vocab in enumerate(words):
            vocabulary_word2index[vocab] = i + 1
            vocabulary_index2word[i + 1] = vocab

        # save to file system if vocabulary of words is not exists.
        print(len(vocabulary_word2index))
        if not os.path.exists(cache_path):
            with open(cache_path, 'wb') as data_f:
                pickle.dump((vocabulary_word2index, vocabulary_index2word), data_f)
    return vocabulary_word2index, vocabulary_index2word
#word2id, vocabulary_index2word = create_voabulary(wordPath=args.wordPath)


# In[4]:

#vocab_build("data_path/word_voabulary.pkl", "data_path/all_data", 1)


# In[5]:

## get char embeddings
word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))


# In[6]:

if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


# In[7]:

if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data,'train_data1')
    test_path = os.path.join('.', args.test_data, 'test_data1')
    print(train_path, test_path)
    '''
    with open(train_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in lines:
        line = line.replace("\n","")
        if line != '\n':
            #print(line)
            if len(line.strip().split())==2:
                [char, label] = line.strip().split()
    '''
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    
    test_size = len(train_data)
    print(test_size)


# In[ ]:

## paths setting
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
result_path = os.path.join(output_path, "results")
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
get_logger(log_path).info(str(args))


# In[ ]:

## training model
if args.mode == 'train':
    print("==========lr====", args.lr)
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                       model_path=ckpt_prefix, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=args.CRF, update_embedding=args.update_embeddings)
    
    model.build_graph()
    
    print("train data len=", len(train_data))
    model.train(train_data, test_data)
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                       model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=args.CRF, update_embedding=args.update_embedding)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim,
                       embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                       model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=args.CRF, update_embedding=args.update_embeddings)
    
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session as sess:
        print("========demo===========")
        saver.restore(sess, ckpt_file)
        
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG = get_entity(tag, demo_sent)
                print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
                                                                           


# In[ ]:




# In[ ]:



