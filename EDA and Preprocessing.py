#-*- codeing=utf-8 -*-
#@time: 2020/7/19 10:39
#@Author: Shang-gang Lee

import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords # Import the stop word list
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from transformers import BertTokenizer
# read data
train_data=pd.read_csv(r'./data/train.csv')
test_data=pd.read_csv(r'./data/test.csv')

# print('train_data_shape:',train_data.shape)
# print('test_data_shape:',test_data.shape)
# print('train_data.isnull:\n',train_data.isnull().sum())
# print('test_data.isnull:\n',test_data.isnull().sum())

train_data.dropna(axis=0,how='any',inplace=True) # drop missing values

# print('train_data.isnull:\n',train_data.isnull().sum())

# # analysis of the sentiment colunm
# print(train_data['sentiment'].value_counts())
# print(test_data['sentiment'].value_counts())

# # make a picture in train data
# train_data['sentiment'].value_counts(normalize=True).plot(kind='bar')
# plt.title('sentiment')
# plt.show()
#
# test_data['sentiment'].value_counts(normalize=True).plot(kind='bar')
# plt.title('sentiment')
# plt.show()

# Text Data Preprocessing
def clean_text(text):
    text=BeautifulSoup(text,'lxml').get_text()            #remove html
    latter=re.sub('https?://\S+|www\.\S+', '', text)
    latter=re.sub("[^a-zA-Z]",' ',latter)                 #getting clear sentence
    words=' '.join(latter.lower().split())                #split sentence to indiviual word

    # stop=set(stopwords.words('english'))                #Remove stop words
    # words=[w for w in words if not w in stop]
    return words

#
train_data['text_clean'] = train_data['text'].apply(str).apply(lambda x: clean_text(x))
test_data['text_clean'] = test_data['text'].apply(str).apply(lambda x: clean_text(x))

train_data['text_len'] = train_data['text_clean'].astype(str).apply(len)
train_data['text_word_count'] = train_data['text_clean'].apply(lambda x: len(str(x).split()))

test_data['text_len'] = test_data['text_clean'].astype(str).apply(len)
test_data['text_word_count'] = test_data['text_clean'].apply(lambda x: len(str(x).split()))

#train data Sentiment Text Length Distribution
positive=train_data[train_data['sentiment']=='positive']
negative=train_data[train_data['sentiment']=='negative']
neutral=train_data[train_data['sentiment']=='neutral']
# positive['text_len'].plot(kind='hist',bins=100,title='Train Positive Text Length Distribution',stacked=True)
# plt.show()
# negative['text_len'].plot(kind='hist',bins=100,title='Train Negative Text Length Distribution',stacked=True)
# plt.show()
# neutral['text_len'].plot(kind='hist',bins=100,title='Train Neutral Text Length Distribution',stacked=True)
# plt.show()

#test data Sentiment Text Length Distribution
test_positive=test_data[test_data['sentiment']=='positive']
test_negative=test_data[test_data['sentiment']=='negative']
test_neutral=test_data[test_data['sentiment']=='neutral']

# test_positive['text_len'].plot(kind='hist',bins=100,title='Test Positive Text Length Distribution',stacked=True)
# plt.show()
# test_negative['text_len'].plot(kind='hist',bins=100,title='Test Negative Text Length Distribution',stacked=True)
# plt.show()
# test_neutral['text_len'].plot(kind='hist',bins=100,title='Test Neutral Text Length Distribution',stacked=True)
# plt.show()

# #Train_data Box visual
# plt.figure(figsize=(8,12))
# plt.subplot(131)
# positive['text_len'].plot(kind='box',title='Train Positive Text ')
# plt.xlabel('postitve')
#
# plt.subplot(132)
# negative['text_len'].plot(kind='box',title='Train Negative Text ')
# plt.xlabel('negative')
#
# plt.subplot(133)
# neutral['text_len'].plot(kind='box',title='Train Neutral Text ')
# plt.xlabel('neutral')
# plt.show()
#
# #Test data Box visual
# plt.figure(figsize=(8,12))
# plt.subplot(131)
# test_positive['text_len'].plot(kind='box',title='Test Positive Text ')
# plt.xlabel('postitve')
#
# plt.subplot(132)
# test_negative['text_len'].plot(kind='box',title='Test Negative Text ')
# plt.xlabel('negative')
#
# plt.subplot(133)
# test_neutral['text_len'].plot(kind='box',title='Test Neutral Text ')
# plt.xlabel('neutral')
# plt.show()

# # Text word count analysis
# #Train
# positive['text_word_count'].plot(kind='hist',title='Train Positive Text Length Distribution',stacked=True)
# plt.show()
# negative['text_word_count'].plot(kind='hist',title='Train Negative Text Length Distribution',stacked=True)
# plt.show()
# neutral['text_word_count'].plot(kind='hist',title='Train Neutral Text Length Distribution',stacked=True)
# plt.show()
#
# # test
# test_positive['text_word_count'].plot(kind='hist',title='Test Positive Text Length Distribution',stacked=True)
# plt.show()
# test_negative['text_word_count'].plot(kind='hist',title='Test Negative Text Length Distribution',stacked=True)
# plt.show()
# test_neutral['text_word_count'].plot(kind='hist',title='Test Neutral Text Length Distribution',stacked=True)
# plt.show()
# #Train_data Box visual
# plt.figure(figsize=(8,12))
# plt.subplot(131)
# positive['text_word_count'].plot(kind='box',title='Train Positive Text ')
# plt.xlabel('postitve')
#
# plt.subplot(132)
# negative['text_word_count'].plot(kind='box',title='Train Negative Text ')
# plt.xlabel('negative')
#
# plt.subplot(133)
# neutral['text_word_count'].plot(kind='box',title='Train Neutral Text ')
# plt.xlabel('neutral')
# plt.show()
#
# #Test data Box visual
# plt.figure(figsize=(8,12))
# plt.subplot(131)
# test_positive['text_word_count'].plot(kind='box',title='Test Positive Text ')
# plt.xlabel('postitve')
#
# plt.subplot(132)
# test_negative['text_word_count'].plot(kind='box',title='Test Negative Text ')
# plt.xlabel('negative')
#
# plt.subplot(133)
# test_neutral['text_word_count'].plot(kind='box',title='Test Neutral Text ')
# plt.xlabel('neutral')
# plt.show()

# def get_top_n_words(corpus,n=None):
#     vectorizer=CountVectorizer(stop_words='english')
#     vectorizer.fit(corpus)
#     bag_of_words=vectorizer.transform(corpus)
#     sum_words=bag_of_words.sum(axis=0)
#     words_freq=[(word,sum_words[0, idx]) for word,idx in vectorizer.vocabulary_.items()]
#     words_freq=sorted(words_freq,key= lambda x:x[1],reverse=True)
#     return words_freq[:n]
#
# # train top20 the frequency of words
# positive_freq_words=get_top_n_words(positive['text_clean'],20)
# negative_freq_words=get_top_n_words(negative['text_clean'],20)
# neutral_freq_words=get_top_n_words(neutral['text_clean'],20)
#
# df1=pd.DataFrame(positive_freq_words,columns=['text','count'])
# df1.groupby('text').sum()['count'].sort_values(ascending=True).plot(kind='barh',title='Train Top 20 freq_words in positve text')
# plt.show()
#
# df2=pd.DataFrame(negative_freq_words,columns=['text','count'])
# df2.groupby('text').sum()['count'].sort_values(ascending=True).plot(kind='barh',title='Train Top 20 freq_words in negative text')
# plt.show()
#
# df3=pd.DataFrame(neutral_freq_words,columns=['text','count'])
# df3.groupby('text').sum()['count'].sort_values(ascending=True).plot(kind='barh',title='Train Top 20 freq_words in neutral text')
# plt.show()
#
# #test top20 the frequency of words
# test_positive_freq_words=get_top_n_words(test_positive['text_clean'],20)
# test_negative_freq_words=get_top_n_words(test_negative['text_clean'],20)
# test_neutral_freq_words=get_top_n_words(test_neutral['text_clean'],20)
#
# test_df1=pd.DataFrame(test_positive_freq_words,columns=['text','count'])
# test_df1.groupby('text').sum()['count'].sort_values(ascending=True).plot(kind='barh',title='Test Top 20 freq_words in positve text')
# plt.show()
#
# test_df2=pd.DataFrame(test_negative_freq_words,columns=['text','count'])
# test_df2.groupby('text').sum()['count'].sort_values(ascending=True).plot(kind='barh',title='Test Top 20 freq_words in negative text')
# plt.show()
#
# test_df3=pd.DataFrame(test_neutral_freq_words,columns=['text','count'])
# test_df3.groupby('text').sum()['count'].sort_values(ascending=True).plot(kind='barh',title='Test Top 20 freq_words in neutral text')
# plt.show()

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# check out max length text
# max_len=0
# for text in train_data['text']:
#     input_ids=tokenizer.encode(text,add_special_tokens=True)
#     max_len=max(max_len,len(input_ids))
# print("Max length:",max_len)  #<=150

train_text=train_data['text'].values
train_sentiment=train_data['sentiment'].values

input_ids=[]
attention_masks=[]
token_type_ids=[]
for i in range(len(train_text)):
    encoded=tokenizer.encode_plus(

    # input_ids: list[int],
    # token_type_ids: list[int] if return_token_type_ids is True (default)
    # attention_mask: list[int] if return_attention_mask is True (default)
    # overflowing_tokens: list[int] if the tokenizer is a slow tokenize, else a List[List[int]] if a ``max_length`` is specified and ``return_overflowing_tokens=True``
    # special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True``
    # and return_special_tokens_mask is True

        train_sentiment[i],
        train_text[i],
        add_special_tokens=True,    # add [CLS] and [SEP]
        max_length=150,
        pad_to_max_length=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'         # return tensor in pytorch
    )
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])
    token_type_ids.append(encoded['token_type_ids'])

print('Original text: ',train_text[10])
print(len(input_ids[10]))
print(input_ids[10])
print(attention_masks[10])
print(token_type_ids[10])

