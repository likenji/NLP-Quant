# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 18:43:13 2017

@author: User
"""

from bs4 import BeautifulSoup
import datetime as dt
import math
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool
import numpy as np
import pandas as pd
import pickle
import re
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sqlalchemy import create_engine
import pymysql
import os
import time
import tushare as ts
import jieba
wd=os.getcwd()
jieba.load_userdict(wd+"\\test1.txt")
#?ts.is_holiday
#ts.get_hist_data('600435').p_change
#datetime.timedelta.days 返回时间差的天数
#datetime.timedelta.seconds 返回时间差的秒数
#datetime.timedelta.microseconds 返回时间差的微秒数
user_agent = 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'
headers={'User-Agent':user_agent 
         #,'Referer':'http://www.zhihu.com/articles' 
         }#伪造个headers
engine = create_engine('mysql+pymysql://$USER_NAME:$PASSWORD@127.0.0.1/eastmoney?charset=utf8')
#===========返回主吧帖子信息===============================================================================         
base_url="http://guba.eastmoney.com" #用来拼接外链，贼好用
end_time=100 #435000到5.10号
#raw_pool=[base_url+"/default,3,f_"+str(i)+".html" for i in range(1,end_time+1)]
raw_pool=[base_url+"/default_"+str(i)+".html" for i in range(1,end_time+1)]
def get_urls_info(url):
    resp = requests.get(url ,headers = headers)
    bsObj=BeautifulSoup(resp.text,"lxml")
    urls_colnames=["views","response","init_time","latest_resp_time","bbs_inner_url",\
       "bbs_name","issue_inner_url","topic","author","author_inner_url"]
            #返回 [浏览数、回复数、发帖时间、回帖时间、贴吧内链、贴吧名字、帖子内链、帖子标题、作者、作者内链]
            #遇上非注册用户会挂，没有a[2]标签,另外遇上资讯号也会挂，这两个问题都已经解决
    temp_array=pd.DataFrame([[int(bsobj.findAll('cite')[0].get_text()),
                          int(bsobj.findAll('cite')[1].get_text()),
                        bsobj.findAll('cite')[3].get_text(),
                        bsobj.findAll('cite')[4].get_text(),
                        bsobj.findAll('a')[0].attrs['href'],
                        bsobj.findAll('a')[0].get_text(),
                        bsobj.findAll('a')[1].attrs['href'],
                        bsobj.findAll('a')[1].attrs['title'],
                        bsobj.findAll('cite')[2].get_text(),
                        bsobj.findAll('a')[2].attrs['href'] 
                if bool(bsobj.findAll('cite')[2].find('a')) else None
            ] for bsobj in bsObj.find("ul",{"class":"newlist"}).findAll("li")],columns=urls_colnames)
    return temp_array
    

pool=Pool()    
total_list=pool.map(get_urls_info,raw_pool)
temp=pd.concatenate(np.array(total_list),axis=0)
temp.回复数=temp.回复数.apply(lambda x:int(x))
temp.to_csv("C:/Users/User/Desktop/华南BOSS/NLP+策略/store.csv")
temp1=temp[temp.回复数>4]
url_pool=[i for i in temp1["帖子内链"]]
sub_pool=[url_pool[4]]
sub_pool.append(url_pool[22])

#===================返回小吧帖子信息=====================
def get_suburls_info(url):
    resp = requests.get(url ,headers = headers)
    bsObj=BeautifulSoup(resp.text,"lxml")
    temp_array1=np.array([[
                           int(bsobj.findAll('span')[0].get_text()),
                          int(bsobj.findAll('span')[1].get_text()),
                        bsobj.find('span',{'class':'l6'}).get_text(),#发帖时间
                        bsobj.find('span',{'class':'l5'}).get_text(),#回帖时间
                        bsobj.find('span',{'class':'l3'}).findAll('a')[0].attrs['href'],                         
                        bsobj.find('span',{'class':'l3'}).find('a').attrs['title']
                        if 'title' in bsobj.findAll('span')[2].findAll('a')[0].attrs 
                        else bsobj.findAll('span')[2].findAll('a')[0].get_text(), #帖子标题
                        bsobj.findAll('span')[3].findAll('a')[0].attrs['href'] if bool(bsobj.findAll('span')[3].find('a')) else None,#作者内链
                        bsobj.findAll('span')[3].get_text()] for bsobj in bsObj.findAll("div",{"class":"articleh"})])
    return temp_array1
    
def get_suburls_info_by_code(stock_code, page=100):
    suburl_pool=['http://guba.eastmoney.com/list,'+stock_code+'_'+str(i)+'.html' for i in range(1,page)] 
    names1=["浏览数","回复数","发帖时间","回帖时间","帖子内链","帖子标题","作者内链","作者"]
    pool=Pool() 
    total_sublist=pool.map(get_suburls_info,suburl_pool)
    subtemp=pd.DataFrame(np.concatenate(np.array(total_sublist),axis=0),columns=names1)
    subtemp.回复数=subtemp.回复数.apply(lambda x:int(x))
    subtemp.浏览数=subtemp.浏览数.apply(lambda x:int(x))
    subtemp['阅评比']=subtemp.apply(lambda x:x.回复数/x.浏览数,axis=1)
    return subtemp
temp=get_suburls_info_by_code('600435', page=100)
#subtemp.to_csv("C:/Users/User/Desktop/华南BOSS/NLP+策略/store"+str(stock_code)+".csv")

#筛选帖子
def select_url(temp):
    url_num=list(temp[(temp.回复数>10) & (temp.阅评比>0.0015)].index)
    return url_num

#now=dt.datetime.now()
#def time_delta(timedelta):
#    return timedelta.days+timedelta.seconds/86400
#temp['exist_time']=temp.发帖时间.apply(lambda x:time_delta(now-dt.datetime.strptime('2017-'+x,'%Y-%m-%d %H:%M')))
#newtemp=temp[temp.exist_time<1]
#temp['回复数'].apply(lambda x:int(x)).quantile(0.9)  

#===================爬取评论板块
def get_emotion(bs):
    emo_list=[bsobj.attrs['title'] for bsobj in bs.findAll('img')]
    emo='#'.join(emo_list)
    return emo
    
def get_comments(url):
    resp = requests.get(url ,headers = headers)
    resp.encoding='uft-8'
    bsObj=BeautifulSoup(resp.text,"lxml")
    return np.array([[bsobj.find('div',{"class":"zwlianame"}).get_text().strip(),
    bsobj.find('a').attrs['href'] if bool(bsobj.find('a')) else None,
    bsobj.find('div',{'class':"zwlitime"}).get_text()[4:],
    bsobj.find('div',{"class":"zwlitext stockcodec"}).get_text() 
    if bsobj.find('div',{"class":"zwlitext stockcodec"}) else None,
    get_emotion(bsobj.find('div',{"class":"zwlitext stockcodec"})) 
    if bsobj.find('div',{"class":"zwlitext stockcodec"}) else ''] for bsobj in bsObj.findAll("div",{"class":"zwlitxt"})])
    #评论人，评论人链接,评论时间,评论内容
    
    #columns=['评论者','评论者链接','评论时间','评论内容','评论表情']
def get_comments_by_inner_url(inner_url):
    if(inner_url[0]!='/'):
        i_url="http://guba.eastmoney.com/"+inner_url
    else:
        i_url="http://guba.eastmoney.com"+inner_url
    resp = requests.get(i_url, headers = headers)
    info=[]
    if resp.status_code==200:#链接可能会失效
        maxpage=math.ceil(eval(re.findall("pinglun_num=(.*?);",resp.text,re.S)[0])/30)
        info=np.concatenate([get_comments(i_url[:-5]+"_"+str(i)+".html") for i in range(1,maxpage+1)],axis=0)
    return pd.DataFrame(info,columns=['评论者','评论者链接','评论时间','评论内容','评论表情'])
    #,columns=['评论者','评论者链接','评论时间','评论内容','评论表情'])
comments_list=[get_comments_by_inner_url(i) for i in url_pool]
for i in range(60):
    (comments_list[i]).to_csv("C:/Users/User/Desktop/华南BOSS/NLP+策略/label/comment"+str(i+1)+".csv")
comments=comments_list[0]
split=[sent2word(i) for i in comments.评论内容 if i]              
influence_sample=pool.map(influence,comments_sample.评论者链接[:10])

    
         
#==============================================================================
# def get_urls(base_url="http://guba.eastmoney.com", 
#              begin=1 , end = 1, create_time="00-00 00:00",resp_time="00-00 00:00",
#              headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'}):
#     raw_pool=[base_url+"/default_"+str(i)+".html" for i in range(begin,end+1)]
#     url_pool=set()
#     for i in range(begin,end+1):
#         print(i)
#         resp = requests.get(base_url+"/default_"+str(i)+".html" ,headers = headers) #这行是申请数据
#         resp.encoding = "utf-8"
#         bsObj = BeautifulSoup(resp.text,"lxml")    
#         for info in bsObj.find("ul",{"class":"newlist"}).findAll("li"):
#             if (eval(info.findAll("cite")[0].get_text()) > 5000 and eval(info.findAll("cite")[1].get_text()) > 10 
#                 and info.findAll("cite")[2].get_text()>create_time and info.findAll("cite")[3].get_text()>resp_time):
#                 url_pool.append(base_url+info.span.findAll('a')[1].attrs["href"])
#     return url_pool    
#==============================================================================

#=======================查询用户影响力模块=============================

def influence(id_url,headers=headers):
    if(id_url!=None):#如果不为空
        resp1=requests.get(id_url,headers=headers)
        bsObj1=BeautifulSoup(resp1.text,"lxml")
        flag=bsObj1.find("div",{"id":"influence"}).span.get_text()
        issue=(bsObj1.find("div",{"class":"grtab5"}).findAll("a")[0].get_text()[3:-1])
        #主帖子数
        remark=(bsObj1.find("div",{"class":"grtab5"}).findAll("a")[1].get_text()[3:-1])
        if(flag=="临时禁言"):
            return np.array([id_url,-1,None,None,None,None,issue,remark])
        stars=eval(bsObj1.find("div",{"id":"influence"}).span.attrs["data-influence"])
        #星星
        age=bsObj1.find("div",{"id":"influence"}).findAll('span')[1].get_text()
        #吧龄
        birthday=bsObj1.find("div",{"id":"influence"}).findAll("span")[2].get_text()[1:-1]
        #注册日
        total_guests_number=bsObj1.find("div",{"class":"sumfw"}).findAll('span')[0].get_text()[:-1]
        #总访客
        today_guests_number=bsObj1.find("div",{"class":"sumfw"}).findAll('span')[1].get_text()[:-1]
        #今日访客
        return np.array([id_url,stars,age,birthday,total_guests_number,today_guests_number,issue,remark])
        #return stars
    else:
        #return 0
        return np.array([None for i in range(8)])      
        
def all_influence(idurl_pool):
    pool=Pool()
    return pd.DataFrame(np.array(pool.map(influence,idurl_pool)),
            columns=['作者链接','影响力',"吧龄","注册日","总访客","今日访客","发帖数","评论数"]) 
df=all_influence(comments_sample.评论者链接[:10])
#=============对评论情感判分模块===============
def sent2word(sentence):
    """
    Segment a sentence to words
    Delete stopwords
    """
    segList = jieba.cut(sentence)
    segResult = []    
    for w in segList:
        segResult.append(w)
    return segResult
    
stopTable=pd.read_table("C:/Users/User/Desktop/华南BOSS/NLP+策略/stop_words.txt",encoding="GBK",header=None)
stopTable.columns=["word"]
stop_list=[i.strip() for i in list(stopTable.word)]
del stopTable
senTable=pd.read_table("C:/Users/User/Desktop/华南BOSS/NLP+策略/store/BosonNLP_sentiment_score.txt")
senTable.columns=["word"]
senDict = {} 
for d in senTable.word:
    senDict[d.split(' ')[0]] = d.split(' ')[1]
    
def score(words):
    sum=0
    if words==None:
        return sum
    for word in words:
        if word in stop_list:
            continue
        if word in senDict.keys():
            sum=sum+eval(senDict[word])
    return sum
    
def comment_emotion_score(comment):
    emo_score=score(sent2word(comment))
    return emo_score
#==============================================================================
# def date_score(url_pool,time):
#     date_dict={}
#     for rawurl in url_pool:#对每个
#         resp = requests.get(rawurl ,headers = headers) #这行是申请数据
#         if resp.status_code==404:#链接可能会失效
#             continue
#         resp.encoding = "utf-8"    #编码
#         maxpage=math.ceil(eval(re.findall("pinglun_num=(.*?);",resp.text,re.S)[0])/30)
#         for i in range(1,maxpage+1):
#             url = rawurl[:-5]+"_"+str(maxpage+1-i)+".html"#从后往前
#             resp = requests.get(url ,headers = headers) #这行是申请数据    
#             bsObj = BeautifulSoup(resp.text,"lxml") #解析申请到的数据，.text才是html文本
#             individual = bsObj.findAll('div',{"class":"zwli clearfix"})
#             if len(individual)==0:#空页码break,时间太久break
#                 break
#             for indiv in individual:
#                 comment=indiv.find("div",{"class":"zwlitext stockcodec"})
#                 if comment:
#                     senti_score=score(sent2word(comment.get_text()))
#                     date=indiv.find("div",{"class":"zwlitime"}).get_text()[4:14]
#                     if date not in date_dict:
#                         date_dict[date]=[0,0,0]
#                     elif senti_score>3:
#                         date_dict[date][0]=date_dict[date][0]+1
#                     elif senti_score<-3:
#                         date_dict[date][1]=date_dict[date][1]+1
#                     else:
#                         date_dict[date][2]=date_dict[date][2]+1
#             
#     return date_dict
#==============================================================================
emo_all="[微笑][大笑][鼓掌][不说了][为什么][哭][不屑][怒][滴汗][拜神][胜利][亏大了][赚大了][牛][俏皮][傲][好困惑][兴奋][赞][不赞][摊手][好逊][失望][加油][困顿][想一下][围观][献花][大便][爱心][心碎][毛估估][成交][财力][护城河][复盘][买入][卖出][满仓][空仓][抄底][看多][看空][加仓][减仓]"[1:-1]
emotion_list=['微笑', '大笑', '鼓掌', '不说了', '为什么', '哭', '不屑', '怒', '滴汗', '拜神',
 '胜利', '亏大了', '赚大了', '牛', '俏皮', '傲', '好困惑', '兴奋', '赞', '不赞', '摊手',
 '好逊', '失望', '加油', '困顿', '想一下', '围观', '献花', '大便', '爱心', '心碎', '毛估估',
 '成交', '财力', '护城河', '复盘', '买入', '卖出', '满仓', '空仓', '抄底', '看多', '看空',
 '加仓', '减仓']
emotion_dict={emotion_list[i]:i for i in range(45)}

comment_train=[pd.read_csv("C:/Users/User/Desktop/华南BOSS/NLP+策略/label/comment"+str(i+1)+".csv",encoding='GBK') for i in range(60)]
for df in comment_train:
    df.columns=['label', '评论者', '评论者链接', '评论时间', '评论内容', '评论表情']
comment_train1=pd.concat(comment_train,axis=0,ignore_index=True)

def emo_vectorize(emotion):
    arr=np.array([0 for i in range(45)])
    if(type(emotion)==float):
         return arr
    else:
        emotion_l=emotion.split("#")
        for i in emotion_l:
            arr[emotion_dict[i]]+=1;
        return arr
#特征词        
f = open("C:/Users/User/Desktop/华南BOSS/NLP+策略/store/feature.txt","r",encoding="utf-8")  
issueword = f.readlines()#读取全部内容
issueword[0]=issueword[0][1:]
issueword=[w[:-1] for w in issueword]
issue_word_dict={issueword[i]:i for i in range(len(issueword))}
                 
def text_vectorize(text):    
    arr=np.array([0 for i in range(len(issueword))])
    if(type(text)!=float):        
        word_segment=sent2word(text)
        for word in word_segment:   
            if word in issue_word_dict:
                arr[issue_word_dict[word]]+=1;
    return arr

def comment_vectorize(x):
    return np.concatenate([emo_vectorize(x["评论表情"]),text_vectorize(x["评论内容"])],axis=0)

def comment_df_process(df):
    v=[comment_vectorize(i[1]) for i in df.iterrows()]
    return np.array(v)

def comment_df_label(df):
    return df.label.values
#=====================================================训练模块
rf1=RandomForestClassifier(n_estimators=100) #构建一个由100棵决策树组成的随机森林
Y=comment_df_label(comment_train1)
X=comment_df_process(comment_train1)
rf1.fit(X,Y) #训练模型
predict=rf1.predict_proba(X)
predict_result=[row.argmax()-1 for row in predict]
rf1.score(X,Y)

X1=X#[Y!=0]
Y1=Y#[Y!=0]
rf2=RandomForestClassifier(n_estimators=100)
rf2.fit(X1[:560],Y1[:560]) #训练模型
predict=rf1.predict_proba(X1)
rf2.score(X1[560:],Y1[560:])
np.array(issueword)[zero[45:]]