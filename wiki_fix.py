#!/usr/bin/env python
# coding: utf-8

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
import multiprocessing
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile
import datetime
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

start = datetime.datetime.now()
print("Starting")
wiki = WikiCorpus("./data/enwiki-latest-pages-articles1.xml-p10p30302.bz2")
print("Wiki loaded")
print(datetime.datetime.now())
print(datetime.datetime.now() - start)
start = datetime.datetime.now()
class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument(content, [title])

class EpochLoggerDM(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0
        self.start = datetime.datetime.now()

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
        self.start = datetime.datetime.now()
        print(self.start)

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
        model.save(output_path)
        self.epoch += 1
        print(datetime.datetime.now() - self.start)
    
    def on_train_begin(self, model):
        print("Training for DM")
    
    def on_train_end(self, model):
        print("Training end for DM")

class EpochLoggerDBOW(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, path_prefix):
        self.epoch = 0
        self.path_prefix = path_prefix
        self.start = datetime.datetime.now()

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
        self.start = datetime.datetime.now()
        print(self.start)

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
        model.save(output_path)
        self.epoch += 1
        print(datetime.datetime.now() - self.start)
    
    def on_train_begin(self, model):
        print("Training for DBOW")
    
    def on_train_end(self, model):
        print("Training end for DBOW")


documents = TaggedWikiDocument(wiki)
print("tagged docs prepared")
print(datetime.datetime.now())
print(datetime.datetime.now() - start)
start = datetime.datetime.now()

cores = multiprocessing.cpu_count() - 1

models = [
    # PV-DBOW 
    Doc2Vec(dm=0, dbow_words=1, size=300, window=10, min_count=20, iter=10, workers=cores, callbacks=[EpochLoggerDBOW("DBOW")]),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=300, window=10, min_count=20, iter=10, workers=cores, callbacks=[EpochLoggerDM("DM")]),
]
print("build vocab start")

models[0].build_vocab(documents)
print(str(models[0]))
models[1].reset_from(models[0])
print(str(models[1]))

print("build vocab end, training start")
print(datetime.datetime.now())
print(datetime.datetime.now() - start)
start = datetime.datetime.now()

for model in models:
    model.train(documents, total_examples=model.corpus_count, epochs=model.iter)

print(datetime.datetime.now())
print(datetime.datetime.now() - start)
print ('training end')

# similarity
for model in models:
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    print(model.docvecs.most_similar(positive=["New Year"], topn=20))

models[1].docvecs.most_similar([models[1].infer_vector("New Year".split(' '))], topn=10)


# #### save model

temp_path_dbow = get_tmpfile('wiki_d2v_dbow')
temp_path_db = get_tmpfile('wiki_d2v_db')


# In[88]:


models[0].save(temp_path_dbow)
models[1].save(temp_path_db)


# #### load saved model

# In[89]:


# model_dbow = Doc2Vec.load(temp_path_dbow)
# model_db = Doc2Vec.load(temp_path_db)


# In[90]:


# model_dbow.docvecs.most_similar(positive=["New Year"], topn=10)


# In[91]:


# type(model_dbow.infer_vector("New Year".split()))


# In[92]:


# model_dbow.docvecs.most_similar([model_dbow.infer_vector("New Year is typing".split())], topn=10)


# In[93]:


# model_dbow.wv.most_similar("test")


# In[ ]:





# In[ ]:




