#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import numpy as np
import re
from collections import Counter
import tensorflow as tf
import keras
from keras.models import Model
from keras import optimizers
from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Input, concatenate, Dropout, Reshape
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support as fscore
from sklearn.metrics.pairwise import cosine_similarity as CS
from sklearn.metrics import matthews_corrcoef, roc_auc_score, jaccard_score, brier_score_loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import pickle
import spacy

TEST = False
TEST_METRICS = False
GENERATE_FEATS = False
if len(sys.argv) > 1 and sys.argv[1] == 'TEST':
    TEST = True

if len(sys.argv) > 1 and sys.argv[1] == 'METRICS':
    TEST_METRICS = True

if len(sys.argv) > 1 and sys.argv[1] == 'FEATS':
    TEST_METRICS = True
    GENERATE_FEATS = True

# In[2]:


my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
my_devices
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''
    def __init__(self):
        self.epoch = 0
        self.cur_time = datetime.datetime.now()

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch),flush=True)
        self.cur_time = datetime.datetime.now()

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch),flush=True)
        delta = datetime.datetime.now()-self.cur_time
        print("Time taken : ",delta,flush=True)
        #output_path = get_tmpfile('file_epoch{}.model'.format(self.epoch))
        #model.save(output_path)
        #print("Temp Saved at ",output_path,flush=True)
        self.epoch += 1


# In[3]:


CLEANING_PATTERSN = re.compile("[\s\n\r\t.,:;\-_\'\"?!#&()*]")
HANDCODED_FEATS = 20
LSTM_HIDDEN_SIZE = 100
MAX_TIME = 30 #MAXIMUM SIZE OF A COMMENT TO BE PASSED TO LSTM
VOCAB_SIZE = 10000 #MAX VOCAB SIZE
DROPOUT = 0.5
ANNOTATIONS = {'N':0,'P':1,'U':2,'n':0}
LEARNING_RATE = 0.00005
NUM_EPOCHS = 10
if TEST:
    NUM_EPOCHS = 5
BATCH_SIZE = 100
FILE_TYPE = 'all' #should be one of 'all', 'ProgramDomain', 'ProblemDomain', 'ProjectManagement'
MIDDLE_LAYER_ACTIVATION = keras.layers.LeakyReLU #Activation function in middle layers.
FINAL_LAYER_ACTIVATION = 'softmax' #Activation function of final layer.
K = 5 #Parameter for K-fold Cross Validation
TRAIN_LOGS_DIR = 'MODELS_NEW'
EMBEDDINGS_MODEL = Word2Vec.load('../../../corpus/corpus_book.bin')


# In[4]:


Z = pd.read_csv('ML_DATASHEETS/Z_LATEST_FEATURES_cal.csv',delimiter='\t') #Z contains the comment text
if TEST:
    Z = pd.read_csv('ML_DATASHEETS/TEST_Z_LATEST_FEATURES_cal.csv',delimiter='\t') #Z contains the comment text
FEATS = pd.read_csv('ML_DATASHEETS/LATEST_FEATURES_cal.csv',delimiter='\t') #Features for training
if TEST:
    FEATS = pd.read_csv('ML_DATASHEETS/TEST_LATEST_FEATURES_cal.csv',delimiter='\t') #Features for training
print(FEATS.head())
FEATS = FEATS.drop(columns=['Descriptional'])



# In[5]:


comments = np.array(Z['F2'])
comments_files = np.array(Z['FILES'])
X = np.array(FEATS)[:,:-1]
# if FILE_TYPE == 'all':
#     Y = np.array(FEATS[['ProgramDomain','ProjectManagement','ProblemDomain']])
# else:
#     Y = np.array(FEATS['Class'])
Y = np.array(FEATS)[:,-1]
Y = np.array(Y,dtype=np.int32)-1

print(len(X), len(Y), len(comments))

# In[6]:


# Comments Cleaning
ctr = Counter()
mp = {}
sentences = []
for comment in comments:
    sent = [x.strip() for x in CLEANING_PATTERSN.split(comment) if x!='']
    ctr[len(sent)] += 1
    sentences.append(sent)
    if len(sent) not in mp:
        mp[len(sent)] = []
    mp[len(sent)].append(sent)


# In[7]:


ctr = Counter()
for sent in sentences:
    for word in sent:
        ctr[word] += 1


sentences_for_parsing = [' '.join(x) for x in sentences]
print("LEN SENTENCES ",len(sentences_for_parsing))
#parse_results = dependency_parser.raw_parse_sents(sentences_for_parsing)

nlp = spacy.load("en_core_web_sm")
pos_tags_score = []

def get_pos_score(taglist):
    score = 0
    for tag in taglist:
        if tag == 'NNP' or tag == 'NNPS' or tag == 'SYM':
            score += 5
        elif tag[:2] == 'NN' or tag[:2] == 'VB':
            score += 3
        elif tag[:2] == 'JJ' or tag[:2] == 'RB':
            score += 1
    return score

# In[8]:

if not (TEST_METRICS or GENERATE_FEATS):
	for sent in sentences_for_parsing:
	    doc = nlp(sent)
	    taglist = []
	    for token in doc:
	        if token.tag_ != None:
	            taglist.append(token.tag_)
	    pos_tags_score.append(get_pos_score(taglist))

	pos_tags_score = np.array(pos_tags_score)
	pmean = np.mean(pos_tags_score)
	pstd = np.std(pos_tags_score)
	pos_tags_score = (pos_tags_score - pmean)/pstd
	pos_tags_score = np.tanh(pos_tags_score)
	pos_tags_score = pos_tags_score.reshape(-1,1)
#print(pos_tags_score)


tag_pefix = ""
if TEST:
    tag_pefix = "test"
if not (TEST_METRICS or GENERATE_FEATS):
    print("Dumping POS TAG LIST")
    with open(tag_pefix+'POS_TAGS_LIST.list','wb') as f:
        pickle.dump(pos_tags_score,f)

print("Loading POS TAG LIST")
with open(tag_pefix+"POS_TAGS_LIST.list",'rb') as f:
    pos_tags_score = pickle.load(f)


X = np.concatenate((X,pos_tags_score), axis=1)
print("Features shape - ",X.shape)




# For creating a vocabulary and convert a sentence (vector of words) to vector of indices
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

tok_prefix = ""
if TEST:
	tok_prefix = "test"

if (not TEST_METRICS) and (not GENERATE_FEATS):
	print("Saving Tokenizer")
	with open(os.path.join(TRAIN_LOGS_DIR,tok_prefix+'TOKENIZER.pkl'),'wb') as f:
		pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(TRAIN_LOGS_DIR,tok_prefix+'TOKENIZER.pkl'),'rb') as f:
	tokenizer = pickle.load(f)


# In[9]:


len(tokenizer.word_index)

wi = tokenizer.word_index
embeddingMatrix = np.zeros((len(wi)+1,100))
for word, i in wi.items():
    if word not in EMBEDDINGS_MODEL.wv.vocab:
        continue
    embeddingMatrix[i] = EMBEDDINGS_MODEL.wv[word]

# In[10]:


# train_sent are Comment texts to be passed for training. (Input to model)
train_sent = tokenizer.texts_to_sequences(sentences)
train_sent = pad_sequences(train_sent, maxlen=MAX_TIME,padding='post')


# In[11]:


# if FILE_TYPE == 'all':
#     train_y = Y
# else:
#     train_y = to_categorical(Y)
# def preprocess_y(y):
#     temp = []
#     for k in y:
#         temp.append(ANNOTATIONS[k])
#     return np.array(temp)
train_y = to_categorical(Y)
print(train_y.shape)


# In[12]:


# Train/Test Split
# perm = np.random.permutation(len(X))

with open('split_perm','rb') as f:
    perm = pickle.load(f)

X = X[perm]
train_y = train_y[perm]
train_sent = train_sent[perm]
comments = comments[perm]
comments_files = comments_files[perm]
NUM_TRAIN = int(0.9*len(X))
print(NUM_TRAIN)
train_x = X[:NUM_TRAIN]
test_x = X[NUM_TRAIN:]
train_y, test_y = train_y[:NUM_TRAIN], train_y[NUM_TRAIN:]
train_sent, test_sent = train_sent[:NUM_TRAIN], train_sent[NUM_TRAIN:]
print(train_x.shape, train_y.shape, train_sent.shape, test_x.shape, test_y.shape, test_sent.shape)


# In[13]:


def divide_into_k_folds(train_x, train_y, train_sent,k):
    xs = []
    ys = []
    sents = []
    each = int(len(train_x)/k)
    for i in range (k-1):
        xs.append(train_x[i*each:(i+1)*each])
        ys.append(train_y[i*each:(i+1)*each])
        sents.append(train_sent[i*each:(i+1)*each])
    xs.append(train_x[(k-1)*each:])
    ys.append(train_y[(k-1)*each:])    
    sents.append(train_sent[(k-1)*each:])    
    return np.array(xs), np.array(ys), np.array(sents)

def get_fold(train_x, train_y, train_sent,i,k):
    ids = [x for x in range(k) if x != i]
    print(i,k,ids)
    return np.concatenate(train_x[ids],axis=0), np.concatenate(train_y[ids],axis=0),         np.concatenate(train_sent[ids],axis=0)

def get_all_data_from_folds(train_x, train_y, train_sent):
    return np.concatenate(train_x,axis=0), np.concatenate(train_y,axis = 0),            np.concatenate(train_sent,axis=0)


# In[14]:


train_x, train_y, train_sent = divide_into_k_folds(train_x, train_y, train_sent, K)
print(train_x.shape)

def build_model(optimizer='rmsprop',lr=LEARNING_RATE,middle_act=MIDDLE_LAYER_ACTIVATION,
               final_act=FINAL_LAYER_ACTIVATION,dropout=DROPOUT,lstm_hidden=LSTM_HIDDEN_SIZE): 
    
    sent_input = Input(shape=(MAX_TIME,)) #Input 1 - Comment text
    extracted_feats = Input(shape=(HANDCODED_FEATS,)) #Input 2 - 12 Features
    print(sent_input.shape, extracted_feats.shape)
    
    embeddingLayer = Embedding(embeddingMatrix.shape[0], embeddingMatrix.shape[1], input_length=MAX_TIME,  trainable=True, weights=[embeddingMatrix])
    sent = embeddingLayer(sent_input)
    _, h1, c1 = LSTM(lstm_hidden,dropout=dropout,return_state=True)(sent) #Feed the comments to LSTM
    print(h1.shape)
    # Concat h1 and 12 features
    feature_vector = concatenate([h1,extracted_feats],axis=1) #Concat output of LSTM with the 12 features
    print(feature_vector.shape)
    probs = Dense(64,activation=None)(feature_vector) #Dense layer over LSTM_HIDEEN_SIZE + 12 features
    probs = middle_act()(probs)
    probs = Dropout(dropout)(probs)
    print(probs.shape)
    probs = Dense(3,activation=final_act)(probs) #Final Activation. Use sigmoid and NOT Softmax here.
    print(probs.shape)
    model = Model(inputs=[sent_input,extracted_feats],outputs=[probs,feature_vector])
    if optimizer == 'rmsprop':
        optimizer = optimizers.rmsprop(lr=lr)
    elif optimizer == 'adam':
        optimizer = optimizers.adam(lr=lr)
    else:
        print("Optimizer not supported!")
        return
    model.compile(loss='categorical_crossentropy',
                 optimizer=optimizer,
                 metrics=['acc'],
                 loss_weights=[1,0])
    return model


# In[16]:


# Find fscore for a model
def find_fs(model):
    predictions = model.predict([test_sent,test_x],batch_size=BATCH_SIZE)[0]
    predictions = predictions.argmax(axis=1)
    fs = fscore(test_y.argmax(axis=1),predictions)
    return fs


# In[17]:


# Run, takes parameters for model. Returns K-models from K-cross validation (We use only final one) 
# and Fscore Statistics from all of them.

def run(optimizer='rmsprop',lr=LEARNING_RATE,middle_act=MIDDLE_LAYER_ACTIVATION,
               final_act=FINAL_LAYER_ACTIVATION,dropout=DROPOUT,lstm_hidden=LSTM_HIDDEN_SIZE):
    MODELS = []
    FSS = []
    model = build_model(optimizer,lr,middle_act,final_act,dropout,lstm_hidden)
    for k in range(K):
        print("-----------------Running Fold - ",k+1," of ",K,"-------------------")
        MODELS.append(model)
        curr_train_x, curr_train_y, curr_train_sent = get_fold(train_x, train_y, train_sent,k,K)
        print(curr_train_x.shape)
        dummy_y = np.zeros((len(curr_train_y),lstm_hidden+HANDCODED_FEATS))
        dummy_y2 = np.zeros((len(train_sent[k]),lstm_hidden+HANDCODED_FEATS))
        model.fit([curr_train_sent,curr_train_x],[curr_train_y,dummy_y],epochs=NUM_EPOCHS,batch_size=BATCH_SIZE,verbose=1,
              validation_data=([train_sent[k], train_x[k]],[train_y[k],dummy_y2]))
        FSS.append(find_fs(model))
        model_prefix = ""
        if TEST:
            model_prefix = "test"
        model.save(os.path.join(TRAIN_LOGS_DIR,model_prefix+'model_'+FILE_TYPE+'_fold_'+str(k)+'.h5'))
    return MODELS, FSS


# In[18]:


# TO CONTINUE TRAINING FOR MORE EPOCHS
# for k in range(K):
#     print("-----------------Running Fold - ",k+1," of ",K,"-------------------")
#     model = MODELS[k]
#     model.fit([train_sent[k],train_x[k]],train_y[k],epochs=NUM_EPOCHS,batch_size=BATCH_SIZE,verbose=1,
#           validation_data=([test_sent, test_x],test_y))
#     model.save('model_'+FILE_TYPE+'_fold_'+str(k)+'.h5')


# In[19]:


# Get predictions for an ensemble for models. 
def get_predictions(test_x, test_sent,models_arr=None):
    prediction_scores = np.zeros((len(test_x),3))
    k = len(models_arr)
    for mod in models_arr:
        predictions = mod.predict([test_sent, test_x],batch_size=BATCH_SIZE)
        if FILE_TYPE == 'all':
            predictions = np.where(predictions > 0.5,1,0)
        else:
            predictions = predictions.argmax(axis=1)
        prediction_scores += predictions
    print(prediction_scores)
    return np.where(prediction_scores > k/2, 1, 0)


# In[20]:


# predictions = get_predictions(test_x, test_sent)


# In[21]:


# if FILE_TYPE == 'all':
#     fs = fscore(test_y,predictions)
# else:
#     fs = fscore(test_y.argmax(axis=1),predictions)
# fs


# In[22]:


# model.save('model_'+FILE_TYPE+".h5")


# # Ensemble

# In[23]:





# In[24]:


ENSEMBLE_FSS = {} #Key - experiment name. Value - FScore Statistics of the experiment.
if not os.path.exists(TRAIN_LOGS_DIR):
    os.mkdir(TRAIN_LOGS_DIR)
if os.path.exists(os.path.join(TRAIN_LOGS_DIR,'LSTM_ENSEMBLE_MODELS_SUMMARY.map')):
    with open(os.path.join(TRAIN_LOGS_DIR,'LSTM_ENSEMBLE_MODELS_SUMMARY.map'),'rb') as f:
        ENSEMBLE_FSS = pickle.load(f)
# Saves all the information for an experiment. Saves the FScore Stats in ENSEMBLE_FSS, 
# saves the models in folder ensemble_models, and dumps the ENSEMBLE_FSS to be read later.\
# Input parameters - MODELS as returned by run(), FSS as returned by run(), name of the experiment.
def _put(m,f,name):
    for j,model in enumerate(m):
        model.save(os.path.join(TRAIN_LOGS_DIR,'model_'+name+str(j)+'.h5'))
    ENSEMBLE_FSS[name] = f
    with open(os.path.join(TRAIN_LOGS_DIR,'LSTM_ENSEMBLE_MODELS_SUMMARY.map'),'wb') as f:
        pickle.dump(ENSEMBLE_FSS,f)
# Running different experiments.

print("HERE")
# Default model

if not TEST_METRICS:
    m, f = run(optimizer='adam')
    _put(m,f,'default')
# 2*LSTM_HIDDEN
# m, f = run(lstm_hidden=2*LSTM_HIDDEN_SIZE)
# _put(m,f,'2LSTM_HIDDEN')
# # 4*LSTM_HIDDEN
# m,f = run(lstm_hidden=4*LSTM_HIDDEN_SIZE)
# _put(m,f,'4LSTM_HIDDEN')


# In[ ]:


# Ensemble Prediction
# with open('LSTM_ENSEMBLE_MODELS_SUMMARY.map','rb') as f:
#     ENSEMBLE_FSS = pickle.load(f)
# ENSEMBLE_MODELS = []
# for k,v in ENSEMBLE_FSS.items():
#     # Taking only last fold model
#     m = keras.models.load_model('ensemble_models_new/model_'+k+str(len(v)-1)+'.h5')
#     ENSEMBLE_MODELS.append(m)
# predictions = get_predictions(test_x, test_sent, ENSEMBLE_MODELS)
# if FILE_TYPE == 'all':
#     fs = fscore(test_y,predictions)
# else:
#     fs = fscore(test_y.argmax(axis=1),predictions)
# fs

model_prefix = ""
if TEST:
    model_prefix = "test"
model = keras.models.load_model(os.path.join(TRAIN_LOGS_DIR,model_prefix+'model_'+FILE_TYPE+'_fold_'+str(K-1)+'.h5'))

def write_in_latex(fs,mic,mac, text):
    print("{ \\bf %s Precision } & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline"%(text,fs[0][0]*100,fs[0][1]*100,fs[0][2]*100,mic[0]*100,mac[0]*100))
    print("{ \\bf %s Recall } & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline"%(text,fs[1][0]*100,fs[1][1]*100,fs[1][2]*100,mic[1]*100,mac[1]*100))
    print("{ \\bf %s F1-score } & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline"%(text,fs[2][0]*100,fs[2][1]*100,fs[2][2]*100,mic[2]*100,mac[2]*100))



def get_metrics(y_true, y_prob, text):
    y_pred = np.argmax(y_prob,axis=1)
    # print("FSCORE: ")
    fs = fscore(y_true, y_pred)
    mic = fscore(y_true, y_pred, average='micro')
    mac = fscore(y_true, y_pred, average='macro')
    write_in_latex(fs, mic, mac, text)


def get_validation_metrics():
    for k in range(1):
        # print("Fold ",k)
        curr_train_x, curr_train_y, curr_train_sent = get_fold(train_x, train_y, train_sent,k,K)
        with open('split_details/train_split.list','wb') as f:
            temp = [curr_train_sent, curr_train_x, curr_train_y]
            pickle.dump(temp, f)

        predictions = model.predict([curr_train_sent, curr_train_x], batch_size=BATCH_SIZE)[0]
        get_metrics(curr_train_y.argmax(axis=1), predictions, 'Train')
        # print("Val: ")
        with open('split_details/val_split.list','wb') as f:
            temp = [train_sent[k], train_x[k], train_y[k]]
            pickle.dump(temp, f)
        predictions = model.predict([train_sent[k], train_x[k]], batch_size=BATCH_SIZE)[0]
        get_metrics(train_y[k].argmax(axis=1), predictions, 'Validation')

        with open('split_details/test_split.list','wb') as f:
            temp = [test_sent, test_x, test_y]
            pickle.dump(temp, f)
        predictions = model.predict([test_sent,test_x],batch_size=BATCH_SIZE)[0]
        get_metrics(test_y.argmax(axis=1), predictions, 'Test')


if TEST_METRICS:
    print("LATEX")
    get_validation_metrics()


# In[25]:

def get_metrics(y_true, y_prob):
	y_pred = np.argmax(y_prob,axis=1)
	print("FSCORE: ")
	print(fscore(y_true, y_pred))
	print("---------------------------------------------------------------------------------")

	print("Mathhews Corr Coeff:")
	print(matthews_corrcoef(y_true, y_pred))
	print("---------------------------------------------------------------------------------")

	print("Jaccard Score:")
	print(jaccard_score(y_true, y_pred,average = None))
	print("---------------------------------------------------------------------------------")

	print("ROC AUC Score:")
	print(roc_auc_score(y_true, y_prob,average='macro', multi_class='ovr'))
	print("=================================================================================")








# In[46]:


all_x, all_y, all_sent = get_all_data_from_folds(train_x, train_y, train_sent)


# In[47]:


probabilities = model.predict([all_sent,all_x],batch_size=BATCH_SIZE)[0]
print("---------------------------------METRICS: ALL TRAIN DATA-------------------------------------")
get_metrics(all_y.argmax(axis=1), probabilities)
# predictions


# # In[48]:


# pred = np.argmax(predictions,axis=1)
# pred


# # In[49]:

# print("---------------------------------FSCORE: ALL TRAIN DATA-------------------------------------")
# print(fscore(all_y.argmax(axis=1),pred))


# In[50]:


extracted_feats = model.predict([all_sent,all_x],batch_size=BATCH_SIZE)[1]
extracted_feats_test = model.predict([test_sent,test_x],batch_size=BATCH_SIZE)[1]
extracted_feats.shape

probabilities = model.predict([test_sent,test_x],batch_size=BATCH_SIZE)[0]
print("---------------------------------METRICS: TEST DATA-------------------------------------")
get_metrics(test_y.argmax(axis=1), probabilities)


# In[51]:

if (not TEST_METRICS) or GENERATE_FEATS:

    with open('MODELS_NEW/EXTRACTED_FEATS.pkl','wb') as f:
        pickle.dump(extracted_feats,f)


    # In[52]:


    with open('ML_DATASHEETS/EXTRACTED/20kx220.csv','w') as f:
        f.write(','.join(['F'+str(i+1) for i in range(LSTM_HIDDEN_SIZE+HANDCODED_FEATS)]+['Class']))
        f.write('\n')
        for j,el in enumerate(extracted_feats):
            f.write(','.join([str(e) for e in el]+[str(np.argmax(all_y[j]))]))
            f.write('\n')
        for j,el in enumerate(extracted_feats_test):
            f.write(','.join([str(e) for e in el]+[str(np.argmax(test_y[j]))]))
            f.write('\n')

        

all_x = np.concatenate((all_x,test_x),axis=0)
all_y = np.concatenate((all_y,test_y),axis=0)
all_sent = np.concatenate((all_sent,test_sent),axis=0)
all_extracted_feats = np.concatenate((extracted_feats, extracted_feats_test),axis=0)

probabilities = model.predict([all_sent,all_x],batch_size=BATCH_SIZE)[0]
print("---------------------------------METRICS: ALL DATA-------------------------------------")
get_metrics(all_y.argmax(axis=1), probabilities)
preds = np.argmax(probabilities,axis=1)

import csv
if TEST_METRICS or GENERATE_FEATS:

	with open('ML_DATASHEETS/EXTRACTED/all_results.csv','w') as f:
		writer = csv.writer(f,delimiter='\t')
		writer.writerow(["Filename","Comment"]+['F'+str(i+1) for i in range(HANDCODED_FEATS)]+["Actual","Predicted"])
		for j,el in enumerate(preds):
			writer.writerow([comments_files[j],comments[j]]+[str(e) for e in all_extracted_feats[j,LSTM_HIDDEN_SIZE:]]+[str(np.argmax(all_y[j])),str(el)])


print("-----------COMPLETED SUCCESSFULLY-----------------")
# In[45]:


# np.argmax(all_y[0])


# # In[ ]:





# # # Embeddings Visualisation

# # In[ ]:


# # Visulaising Embeddings
# embeddings = model.layers[1].get_weights()[0]


# # In[ ]:


# def embed(word):
#     return embeddings[tokenizer.word_index[word]].reshape(1,-1)


# # In[ ]:


# NUM_WORDS_FOR_ANALYSIS = 50
# SIM = []
# ALL_WORDS = []
# all_words = tokenizer.word_index.keys()
# for word in all_words:
#     ALL_WORDS.append(word)
# all_words = ALL_WORDS
# for i in range(NUM_WORDS_FOR_ANALYSIS):
#     for j in range(i+1,NUM_WORDS_FOR_ANALYSIS):
#         SIM.append((all_words[i],all_words[j],CS(embed(all_words[i]),embed(all_words[j]))[0][0]))


# # In[ ]:


# SS = sorted(SIM,reverse=True,key=(lambda x:abs(x[2])))


# # In[ ]:


# def tsne_plot():
#     "Creates and TSNE model and plots it"
#     labels = []
#     tokens = []

#     for word in all_words[:50]:
#         tokens.append(embed(word)[0])
#         labels.append(word)
    
#     tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
#     new_values = tsne_model.fit_transform(tokens)

#     x = []
#     y = []
#     for value in new_values:
#         x.append(value[0])
#         y.append(value[1])
        
#     plt.figure(figsize=(16, 16)) 
#     for i in range(len(x)):
#         plt.scatter(x[i],y[i])
#         plt.annotate(labels[i],
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#     #plt.show()
#     plt.savefig('SP.svg',format='svg')


# # In[ ]:


# tsne_plot()


# # In[ ]:


# train_x.shape


# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[11]:


# len(train_sent)


# # In[28]:


# model = keras.models.load_model('ensemble_models_new/model_default0.h5')


# # In[30]:


# print(model.summary)


# # In[31]:


# model.layers


# # In[113]:


# x,y,s = get_all_data_from_folds(train_x, train_y, train_sent)


# # In[114]:


# np.sum(np.argmax(y,axis=1)==0),np.sum(np.argmax(y,axis=1)==1),np.sum(np.argmax(y,axis=1)==2),


# # In[68]:


# y.shape


# # In[112]:


# train_y


# # In[94]:





# # In[ ]:





# # In[ ]:




