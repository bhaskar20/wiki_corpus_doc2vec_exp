#!/bin/sh

git clone https://github.com/bhaskar20/wiki_corpus_doc2vec_exp.git
cd wiki_corpus_doc2vec_exp
git config --global user.name "bhaskar20"
git config --global user.email "sharmabhaskar13@gmail.com"
git pull origin master
mkdir modelsData
mv ../gensim/modelsData .

tmux new-session -d -s "juyp"
tmux send -t juyp "jupyter notebook" ENTER