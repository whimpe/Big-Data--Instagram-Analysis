# Databricks notebook source
spark

# COMMAND ----------

# DBTITLE 1,IMPORTS


# COMMAND ----------

# DBTITLE 1,Import libraries
#Imports
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from collections import Counter
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import udf,count,when,isnan,size,col,to_timestamp
from pyspark.sql.functions import * 
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.sql.functions import count, when, isnan
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml import Pipeline, Transformer
from pyspark.sql.functions import regexp_replace, col
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import VectorAssembler, StandardScaler, VectorIndexer,CountVectorizer,StringIndexer, Word2Vec
from textblob import TextBlob
from pyspark.sql.types import DoubleType, ArrayType, TimestampType,IntegerType,LongType
from pyspark.sql.functions import udf, col
from pyspark.sql import functions as f
import re
import string
from pyspark.ml.feature import OneHotEncoderEstimator,HashingTF
import emoji
from luminoth import Detector, read_image, vis_objects
from PIL import Image
import numpy as np
import os
import subprocess
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, MapType
import datetime
import pandas as pd
from pyspark.sql.functions import udf,desc
import seaborn as sns
from datetime import datetime
 





# COMMAND ----------

# Topic modeling imports

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
!pip install gensim
!pip install spacy
!pip install pyLDAvis


import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.tfidfmodel import TfidfModel
# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
  
#20 sec

# COMMAND ----------

# DBTITLE 1,FUNCTIONS 


# COMMAND ----------

#check which checkpoints we need to download
a = !lumi checkpoint list
print(a)

# COMMAND ----------

# DBTITLE 1,download de checkpoint (1x runnen)
#eenmaal runnen
!lumi checkpoint refresh
!lumi checkpoint download e1c2565b51e9   
!lumi checkpoint download aad6912e94d9   

# COMMAND ----------

# DBTITLE 1,Luminoth detector opzetten (1x runnen anders foutmelding -> clear state) 
#eenmaal runnen anders clear state
detector = Detector(checkpoint='fast', config = None,prob=0.1, classes=None)

# COMMAND ----------

# DBTITLE 1,Luminoth functions 1
# hadoop = sc._jvm.org.apache.hadoop
# fs = hadoop.fs.FileSystem
# conf = hadoop.conf.Configuration() 

def luminoth_predict(directory):
  directory1 = "/dbfs/FileStore/tables/" + str(directory) + "/"
   
  height = list()
  width = list()
  timestamp =list()
  predictions = list()
  #returns all the files that are in this folder (directory)
  for picture in os.listdir(directory1):
     #directory vanaf filestore      
    directory_pic = directory1[5:] + str(picture)
    print(directory_pic)
    pic =spark.read.format("image").load(directory_pic)
    height_local =pic.select('image.height').collect()[0][0]
    width_local = pic.select('image.width').collect()[0][0]
    height.append(height_local)
    width.append(width_local)
    timestamp.append(str(picture)[:-4])  # plaatst de titel in een lijst en converts later naar timestamp        
    pic = pic.select('image.data').collect()  #binarytype
    pic = np.asarray(pic)
    pic = pic.reshape(height_local,width_local,-1)
    pic = pic[:,:,::-1] # RGB to BGR  #volgorde omdraaien van de drie dimensies door slicing met -1 als stap
    pred = detector.predict(pic)
    predictions.append(pred)    
    print(timestamp)
    print(height)
    print(width)
    print(predictions)
    
#   Schemad = StructType([StructField("Timestamp", StringType()),
#                         StructField("height", IntegerType()),
#                         StructField("width", IntegerType()),
#                         StructField("Predictions", StringType()),
#                        ])
  
  final_list = [timestamp,height,width,predictions]
  final_list = np.array(final_list).T.tolist()
  df = pd.DataFrame(data =final_list, columns = ['Timestamp','Height','Width','lumi_output'])
  #returns a pandas data frame with the 4 columns and the last one the predictions  
  return df
 

# COMMAND ----------

# DBTITLE 1,Luminoth functions 2
def timestampconverter(row):
  row = row[0:19]
  a= datetime.datetime.strptime(row, '%Y_%m_%d_%H_%M_%S')
  return a
convert_to_timestamp = udf(timestampconverter, TimestampType())


def extract_labels_per_threshold(df, threshold):
  print("threshold is equal to :" + str(threshold))
  column = []
  for dicts in df:
    lijst = []
    for obj in dicts:
      if obj['prob'] > threshold:
        label = str(obj['label']) + "_in_img"
        lijst.append(label)
    lijst = Counter(lijst)
    column.append(lijst)
  #print(column)
  return pd.Series(column)


def create_base_table_picture_data(df,threshold):
  df['new_Predictions'] = extract_labels_per_threshold(df.lumi_output,threshold)
  df = pd.concat([df, pd.DataFrame(list(df.new_Predictions))], axis=1)
  df = df.drop(['new_Predictions','lumi_output'], axis =1)
  df = df.fillna(0)
  
  #spark again
  df = spark.createDataFrame(df)
  df = df.withColumn("Timestamp_id",convert_to_timestamp(df['Timestamp']))
  df = df.drop('Timestamp')
  return df


# COMMAND ----------

# DBTITLE 1,Loading posts functions
# Define a new schema using the StructType method
posts_schema = StructType([
  # Define a StructField for each field
  StructField('co', StringType(), True),
  StructField('Post_ID', StringType(), True),
  StructField('Profile_name', StringType(), True),
  StructField('Post_Date', TimestampType(), True),  
  StructField('Caption',StringType(), True),
  StructField('Nbr_of_Pics',IntegerType(), True),
  StructField('Multiple_Posts',BooleanType(), True),
  StructField('Google_Maps_URL',StringType(), True),
  StructField('Location',StringType(), True),
  StructField('Nbr_of_Likes',IntegerType(), True),
  StructField('Nbr_of_Comments',IntegerType(), True),
  StructField('Is_video',StringType(), True)  
])


def initial_posts_processing(df_name, drop_columns):
  #probleempje oplossen met 2014
  path = "FileStore/tables/" + str(df_name)
  if df_name == "tomorrowland_Posts_2014.csv":
    df = spark.read.csv(path,schema = posts_schema, header = True, multiLine = True, escape = '"', sep = ';').drop('co')
  else:
    df = spark.read.csv(path,schema = posts_schema , header = True, multiLine = True, sep = ';').drop('co')
  
  
  df = df.withColumn("Multiple_Posts_int",df.Multiple_Posts.cast("Int"))
  df = df.withColumn("Is_video_int",F.when(df["Is_video"] == "False\r", 0).otherwise(1))
  #die \r sukt echt en maakt me zorgen!!
  df =df.fillna('Not Available', subset = ['Google_Maps_URL','Location'])
    
  #misschien maar een van de twee nodig -> nog eens checken
  df = df.withColumn("Google_Maps_URL_binary_available",F.when(df["Google_Maps_URL"] == "Not Available", 0).otherwise(1))
  df = df.withColumn("Location_added_to_post",F.when(df["Location"] == "Not Available", 0).otherwise(1))

  if drop_columns:
    #Hoeft niet per se maar in dit geval (2013) beter wel
    df = df.drop('Multiple_Posts','Is_video')
    df = df.drop('Google_Maps_URL','Location')
  return df





# COMMAND ----------

# DBTITLE 1,Loading comments functions
# Define a new schema using the StructType method
comments_schema = StructType([
  # Define a StructField for each field
  StructField('co', StringType(), True),
  StructField('Comment_ID', LongType(), True),
  StructField('Commenter_ID', StringType(), True),
  StructField('Commenter_verified', StringType(), True),
  StructField('Post_ID', StringType(), True),  
  StructField('Text',StringType(), True),
  StructField('created_at\r', TimestampType(), True)
  ])

def initial_comments_processing(df_name):
  path = "FileStore/tables/" + str(df_name)+".csv"
  df = spark.read.csv(path,schema = comments_schema , header = True, multiLine = True, sep = ';').drop('co')
  df = df.withColumnRenamed('created_at\r','created_at')
  df = df.withColumn("Commenter_verified", F.when(df["Commenter_verified"] == "True", 1).otherwise(0))       
  return df




# COMMAND ----------

# DBTITLE 1,Functions needed to model the topics
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, data_words):  
    bigram = gensim.models.Phrases(data_words, min_count=4, threshold=50)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

  
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), np.round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)  
  

def make_corpus(df_with_all_captions):  
  list_with_cleaned_captions = [str(row.only_str_cap) for row in df_with_all_captions.select('only_str_cap').collect()] 
  list_with_cleaned_captions = [x.lower() for x in list_with_cleaned_captions] 
  
  stop_words = stopwords.words('english')
  #stop_words.extend(['from', 'subject', 're', 'edu', 'use' , 'day', 'end' , 'last' , 'end' , 'first' , 'yesterday' ,'tomorrow', 'people', 'tomorrowland', 'today'])
  stop_words.extend(['from', 'subject', 're', 'edu', 'use' , 'end' , 'first' , 'yesterday' ,'tomorrow', 'people', 'tomorrowland','arch', 'tomorrowland' , 'bag' , 'c' , 'tomorrowlandcom' , 'thenorthfaceuk' , 'cet' , 'arminvanbuuren' , 'budweiser' , 'lostfrequencie' , 'afrojack' , 'tiesto' , 'axwell' , 'netskyofficial','s', 'almost', 'around,resident_yvesv'])
  #hier wat extra stopwoorden in zetten om zo de lda beter te maken
  
  data_words = list(sent_to_words(list_with_cleaned_captions))
  
  # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
  !python3 -m spacy download en  
  
  # Remove Stop Words
  data_words_nostops = remove_stopwords(data_words, stop_words)

  # Form Bigrams
  data_words_bigrams = make_bigrams(data_words_nostops, data_words)

  # Do lemmatization keeping only noun, adj, vb, adv
  data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

  # Create Dictionary
  id2word = corpora.Dictionary(data_lemmatized)

  # Create Corpus
  texts = data_lemmatized

  # Term Document Frequency
  corpus = [id2word.doc2bow(text) for text in texts]

  #now adjust using tfidf
  tfidf = TfidfModel(corpus)
  
  #update the corpus with the adjusted weights
  corpus_adjusted = [tfidf[corpus[a]] for a in range(0,len(tfidf[corpus[:]]))]
  
    
  return  corpus_adjusted, id2word, list_with_cleaned_captions, data_lemmatized #zet hierbij wat nodig,
  

def get_dominant_topic_list(df_with_all_captions,  topic_nbr): 
  
  corpus_adjusted , id2word , list_with_cleaned_captions, dont_use_var = make_corpus(df_with_all_captions)
  
  lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_adjusted,  id2word=id2word,
                                           num_topics=topic_nbr, random_state=100,
                                           update_every=1,  chunksize=100,
                                           passes=10,   alpha='auto',
                                           per_word_topics=True)
  
  df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus_adjusted, texts=list_with_cleaned_captions)

  # Format and make common key
  df_dominant_topic = df_topic_sents_keywords.reset_index()
  df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
  df_dominant_topic['New_ID'] = range(10, 10 + len(df_dominant_topic))
  df_dominant_topic = spark.createDataFrame(df_dominant_topic)  
  
  #Add take ID of captions and ad common key
  IDss = df_with_all_captions.select('Post_ID')  
  IDss_pd = IDss.toPandas()
  IDss_pd['New_ID'] = range(10, 10 + len(IDss_pd))
  IDss = spark.createDataFrame(IDss_pd)
    
  #join  
  dominant_topicsss = df_dominant_topic.join(IDss, on = 'New_ID', how = 'inner')
   
  return dominant_topicsss


#use for tuning the best number of topics
def compute_coherence_values(df_with_all_captions, limit, start, step, show_graph):
    corpus_adjusted , id2word , list_with_cleaned_captions , data_lemmatized= make_corpus(df_with_all_captions)
    
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus_adjusted,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)      
      
      
        model_list.append(model)
        coherence_model_lda = CoherenceModel(model=model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')      
        coherence_values.append(coherence_model_lda.get_coherence())  
        
        #make another graph using perplexity
        #lda_model.log_perplexity(corpus_adjusted)
        
        
    if show_graph:
      # Show graph
      x = range(start, limit, step)
      plt.plot(x, coherence_values)
      
      plt.xlabel("Num Topics")
      plt.ylabel("Coherence score")
      plt.legend(("coherence_values"), loc='best')
      plt.show()
      display(plt.show())
      #5 min runnen  

    return model_list, coherence_values

def Compute_Perplexity_and_Coherence(df_with_all_captions, num_topics): 
    corpus_adjusted , id2word , list_with_cleaned_captions , data_lemmatized= make_corpus(df_with_all_captions)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_adjusted,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

    print('\nPerplexity: ', lda_model.log_perplexity(corpus_adjusted))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    return coherence_lda

  

# COMMAND ----------

# DBTITLE 1,CLASSES for to clean caption and extract features
class MakeResponse(Transformer):
    def __init__(self, cols_to_scale, alpha):
        super(MakeResponse, self).__init__()
        self.cols_to_scale = cols_to_scale
        self.alpha = alpha

    def _transform(self, df: DataFrame) -> DataFrame:
          for col in self.cols_to_scale:
            # Define min and max values and collect them
            max_days = df.agg({col: 'max'}).collect()[0][0]
            min_days = df.agg({col: 'min'}).collect()[0][0]
            # Create a new column based off the scaled data
            df = df.withColumn("normalized_{}".format(col), (df[col] - min_days) / (max_days - min_days))
          df = df.withColumn("y",(1-self.alpha) * df.normalized_Nbr_of_Likes + self.alpha * df.normalized_Nbr_of_Comments) #opletten hier is het hard_coded
          return df

#maakt cleanCaption kolom en kolom met aantal hashtags en aantal mentions in caption        
class CleanCaption(Transformer): 
    def __init__(self):
        super(CleanCaption, self).__init__()        

    def _transform(self, df: DataFrame) -> DataFrame:
        df = df.withColumn('no_nbrs', regexp_replace(col('Caption'), r'[0-9]{1,}', '')) #numbers remover
        df = df.withColumn('only_str_cap', regexp_replace(col('no_nbrs'),  "[{0}]".format(re.escape(string.punctuation)), ''))#         
        RT1 = RegexTokenizer(inputCol = "Caption", outputCol = '#words', pattern =  r"\#")
        RT2 = RegexTokenizer(inputCol = "Caption", outputCol = '@words', pattern = r"\@")
         # deze was eerst niet gemaakt RT3
        RT3 = RegexTokenizer(inputCol = "only_str_cap", outputCol = "words", pattern = "\\W+")
        
        pipelineModel = Pipeline(stages =[RT1,RT2,RT3]).fit(df) 
        
       
        
        df = pipelineModel.transform(df)
        df = df.withColumn('Nbr_of_hashtags',size(df["#words"])-1) 
        df = df.withColumn('Nbr_of_mentions',size(df["@words"])-1)
        return df


#aantal woorden in een caption     
class lengte_caption(Transformer):
    def __init__(self):
        super(lengte_caption, self).__init__()        

    def _transform(self, df: DataFrame) -> DataFrame:
        df = df.withColumn('aantal_woorden',size(df["words"]))
        return df
#maakt een dag column wanneer gepost
class DayCreator(Transformer):
    def __init__(self):
        super(DayCreator, self).__init__()        
    def _transform(self, df: DataFrame) -> DataFrame:
        df = df.withColumn('Day',F.date_format('Post_Date','yyyy-MM-dd'))     
        return df      
#returnes int welke dag het gepost is      
class Weekday(Transformer):
  def __init__(self):
      super(Weekday, self).__init__()       
  def _transform(self, df: DataFrame) -> DataFrame:
      weekDay_udf =  udf(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%w'))
      df = df.withColumn('weekDay', weekDay_udf(df["Day"]))
      return df    



# COMMAND ----------

# DBTITLE 1,UDF om periode aan te duiden met een int
import datetime

start13 = datetime.date(2013,7,26)
end13 = datetime.date(2013,7,28)

start14_1 = datetime.date(2014,7,18)
end14_1 = datetime.date(2014,7,20)
start14_2 = datetime.date(2014,7,25)
end14_2 = datetime.date(2014,7,27)

start15 = datetime.date(2015,7,24)
end15 = datetime.date(2015,7,26)

start16 = datetime.date(2016,7,22)
end16 = datetime.date(2016,7,24)

start17_1 = datetime.date(2017,7,21)
end17_1 = datetime.date(2017,7,23)
start17_2 = datetime.date(2017,7,28)
end17_2 = datetime.date(2017,7,30)

start18_1 = datetime.date(2018,7,20)
end18_1 = datetime.date(2018,7,22)
start18_2 = datetime.date(2018,7,27)
end18_2 = datetime.date(2018,7,29)

start19_1 = datetime.date(2019,7,19)
end19_1 = datetime.date(2019,7,21)
start19_2 = datetime.date(2019,7,26)
end19_2 =datetime.date(2019,7,28)




from pyspark.sql.types import DateType


from pyspark.sql.functions import udf
from pyspark.sql.types import *
from datetime import timedelta



def During(date):
  if (((date >= start13) & (date <= end13))|((date >= start14_1) & (date <= end14_1))|((date >= start14_2) & (date <= end14_2))\
     |((date >= start15) & (date <= end15))|((date >= start16) & (date <= end16))|((date >= start17_1) & (date <= end17_1)) \
     |((date >= start17_2) & (date <= end17_2))|((date >= start18_1) & (date <= end18_2))|((date >= start19_1) & (date <= end19_1))
     |((date >= start19_2)&(date<=end19_2))
     ):
    return 1
  else:
    return 0
during_udf = udf(During,IntegerType())

def period_encoder(date):
  
  if date == None:
    return 0 
  #during festival
  elif (((date >= start13) & (date <= end13))|((date >= start14_1) & (date <= end14_2))|((date >= start15) & (date <= end15))\
      |((date >= start16) & (date <= end16))|((date >= start17_1) & (date <= end17_2))|((date >= start18_1) & (date <= end18_2))\
      |((date >= start19_1) & (date <= end19_2))     
     ):
    return 1
  #30 days before start 
  elif (((date >= start13-timedelta(30))&(date < start13))|((date >= start14_1-timedelta(30))&(date < start14_1))\
        |((date >= start15-timedelta(30))&(date < start15))|((date >= start16-timedelta(30))&(date < start16)) \
        |((date >= start17_1-timedelta(30))&(date < start17_1))|((date >= start18_1-timedelta(30))&(date < start18_1))\
        |((date >= start19_1-timedelta(30))&(date < start19_1))
        ):
    return 2
  #30 days after
  elif (((date > end13)&(date <= end13+timedelta(30)))|((date > end14_2)&(date <= end14_2+timedelta(30)))\
        |((date > end15)&(date <= end15+timedelta(30)))|((date > end16)&(date <= end16+timedelta(30)))|((date > end17_2)&(date <= end17_2+timedelta(30)))\
        |((date > end18_2)&(date <= end18_2+timedelta(30)))|((date > end19_2)&(date <= end19_2+timedelta(30)))
       ):
    return 3
  #period around ticket sale
  elif ((date.month == 1 & date.day>=15)|(date.month == 2 & date.day <=15) 
       ):
    return 4
  else:
    return 0
  
period_udf = udf(period_encoder, IntegerType())

class During_festival(Transformer):
  def __init__(self):
      super(During_festival, self).__init__()       
  def _transform(self, df: DataFrame) -> DataFrame:
      df = df.withColumn('boolean_during', during_udf(df.Day_datum))      
      return df 
    
class Period_Year(Transformer):
  def __init__(self):
      super(Period_Year, self).__init__()       
  def _transform(self, df: DataFrame) -> DataFrame:
      df = df.withColumn('period', period_udf(df.Day_datum))
      return df    
    
    
class days_since_last_post(Transformer):
  def __init__(self):
    super(days_since_last_post, self).__init__()
  def _transform(self, df: DataFrame) -> DataFrame:
    my_window = Window.partitionBy().orderBy("Post_ID")
    df = df.withColumn("prev_value", F.lag(df.Post_Date).over(my_window))
    df = df.withColumn("days_since_last_post", F.when(F.isnull(F.datediff(df.Post_Date,df.prev_value)), 0).otherwise(F.datediff(df.Post_Date,df.prev_value )))
    df = df.drop("prev_value")
    return df
  

  
# DD = During_festival()
# PP = Period_Year()
# pipelineModel = Pipeline(stages =[DD,PP,PC]).fit(posts13) 
# posts13_df = pipelineModel.transform(posts13) 
# #posts14_df = pipelineModel.transform(posts14) 
# posts15_df = pipelineModel.transform(posts15) 
# posts16_df = pipelineModel.transform(posts16) 
# posts17_df = pipelineModel.transform(posts17) 
# posts18_df = pipelineModel.transform(posts18) 
# posts19_df = pipelineModel.transform(posts19) 



# COMMAND ----------

# DBTITLE 1,Comment functions
from textblob import TextBlob
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col

def getPolarity(row):
  textBlob_review = TextBlob(row)
  return textBlob_review.sentiment[0]
getPolarityUDF = udf(getPolarity, DoubleType())

def getSubjectivity(row):
  textBlob_review = TextBlob(row)
  return textBlob_review.sentiment[1]
getSubjectivityUDF = udf(getSubjectivity, DoubleType())

#returns het gemiddelde van subjectivity en polarity per post
class Pol_and_Subj(Transformer):
    def __init__(self):
        super(Pol_and_Subj, self).__init__()        

    def _transform(self, df: DataFrame) -> DataFrame:            
      df = df.withColumn('polarity', getPolarityUDF(col('Text'))).withColumn('subjectivity', getSubjectivityUDF(col('Text')))
      df1 = df.select('Post_ID','polarity').groupBy('Post_ID').agg({'polarity': 'avg'}).withColumnRenamed('avg(polarity)', 'totalPolarity')
      df2 = df.select('Post_ID','subjectivity').groupBy('Post_ID').agg({'subjectivity': 'avg'}).withColumnRenamed('avg(subjectivity)', 'totalSubjectivity')
      df = df1.join(df2, ['Post_ID'])
      return df
    

class TotaalAantalMentionsInComments(Transformer):
    def __init__(self):
        super(TotaalAantalMentionsInComments, self).__init__()        

    def _transform(self, df: DataFrame) -> DataFrame:  
      MM = RegexTokenizer(inputCol = "Text", outputCol = '@words', pattern = r"\@")
      pipelineModel = Pipeline(stages =[MM]).fit(df) 
      df = pipelineModel.transform(df)
      df = df.withColumn('Nbr_of_mentions',size(df["@words"])-1)
      df1 = df.groupBy('Post_ID').agg({'Nbr_of_mentions': 'sum'}).withColumnRenamed('sum(Nbr_of_mentions)', 'TotalMentionsInComments')
      return df1

    


# COMMAND ----------

# DBTITLE 1,functies  woorden_columns
import pandas as pd
class word_columns(Transformer):
  def __init__(self):
    super(word_columns, self).__init__()        

  def _transform(self, df: DataFrame, stopwords = []) -> DataFrame: 
    x=1
    lijst =[]
    dfp = df.toPandas()
    dfp = dfp[['Post_ID','filtered']]  
    for i,j in dfp[['filtered']].iteritems():
      for arra in j:
          for w2 in arra:
            if len(w2)==1:
              continue
            elif w2 not in lijst:
              lijst.append(w2)
    #print(lijst)
    #haalt stopwords eruit
    #stopwords = ['tomorrowland','tomorrowworld' ]
    lijst = [word for word in lijst if word not in stopwords]
    
    matrix_ant =[]
    for woord in lijst:
      tss=[]
      for index,pandas in dfp[['filtered']].iteritems():
        for arr in pandas:
          if woord in arr:
            tss.append(1)
          else:
            tss.append(0)
        dfp[woord]=pd.Series(tss)
        
    dfp = dfp.drop(['filtered'],axis=1)    
    
    lijst_struct=[StructField('Post_ID2',StringType(),True)]
    for woord in lijst:
      lijst_struct.append(StructField(woord,IntegerType(),True))    
    p_schema = StructType(lijst_struct)
    df_word_colo = sqlContext.createDataFrame(dfp, p_schema)
    dfn = df.join(df_word_colo,df.Post_ID==df_word_colo.Post_ID2)
    dfn = dfn.drop('Post_ID2','filtered')
    return dfn
    
    
#     lijst =[]
#     dfp = df.toPandas()
#     dfp = dfp[['Post_ID','filtered']]  
#     for i,j in dfp[['filtered']].iteritems():
#       for lijst_met_woorden in j:
#         for woord in lijst_met_woorden:
#           if len(woord)==1:
#             continue
#           if woord in lijst:
#             continue
#           else:
#             lijst.append(woord)
#     print(len(lijst))
#     #haalt stopwords eruit
    
#     #stopwords = ['tomorrowland','tomorrowworld' ]
#     lijst = [word for word in lijst if word not in stopwords]

#     for woord in lijst:
#       tss=[]
#       for i,j in dfp[['filtered']].iteritems():  
#         for arr in j: 
#           #print(arr)
#           if type(arr)==int:
#             tss.append(0)
#           else: 
#             lis = list(arr)
#             if (woord in lis):
#               tss.append(lis.count(woord))
#             else:
#               tss.append(0)
#       dfp[woord]=pd.Series(tss)
#     dfp =dfp.drop(['filtered'], axis=1)  

#     #STRUCTFIELDS
#     lijst_struct=[StructField('Post_ID2',StringType(),True)]
#     dfnew = dfp[['Post_ID']]

# #     #the best words
    
#     l = dfp.drop(['Post_ID'],axis=1).sum().sort_values(ascending=False)
#     words = []
#     for i in range(len(lijst)):          
#       dfnew.insert(i+1, l.index[i], dfp[[l.index[i]]] , True)
#       words.append(l.index[i])    
#       lijst_struct.append(StructField(l.index[i],IntegerType(),True))

    #the worst words

#       l = dfp.drop(['Post_ID'],axis=1).sum().sort_values(ascending=True)
#       words = []
#       for i in range(20):
#         dfnew.insert(i+21, l.index[i], dfp[[l.index[i]]] , True)
#         words.append(l.index[i])    
#         lijst_struct.append(StructField(l.index[i],IntegerType(),True))


#     #convert dfp to pyspark dataframe
#     print(dfnew.head())
#     p_schema = StructType(lijst_struct)
#     df_word_colo = sqlContext.createDataFrame(dfnew, p_schema)
#     dfn = df.join(df_word_colo,df.Post_ID==df_word_colo.Post_ID2)
#     dfn = dfn.drop('Post_ID2')

#     return dfn
  
  
  


# COMMAND ----------

# DBTITLE 1,FUNCTIE period_columns aanmaken
class period_columns(Transformer):
  def __init__(self):
    super(period_columns, self).__init__()        

  def _transform(self, df: DataFrame) -> DataFrame: 
    lijst =[]
    dfp_old = df.toPandas()[['period','Post_ID']]


    periods = ['no_period','during','30 days before','30 days after','ticket_sale']
    new_colo = []
    aa= dfp_old[['period']]
    for i,j in aa.iteritems(): 
      for val in j.iteritems() :    
        tss = [0,0,0,0,0]
        tss[val[1]]=1
        new_colo.append(tss)

    for index,naam in enumerate(periods):
      tss = []
      for arr in new_colo:
        tss.append(arr[index])
      dfp_old[naam] = tss


    lijst_struct=[]  
    lijst_struct.append(StructField('period',IntegerType(),True))
    lijst_struct.append(StructField('Post_ID2',StringType(),True))
    for woord in periods:
      lijst_struct.append(StructField(woord,IntegerType(),True))


    p_schema = StructType(lijst_struct)

    df_word_colo = sqlContext.createDataFrame(dfp_old, p_schema)
    df = df.join(df_word_colo,df.Post_ID==df_word_colo.Post_ID2)
    df = df.drop('period','Post_ID2')
    return df
  
  
class Sunday(Transformer):
  def __init__(self):
    super(Sunday, self).__init__()
  def _transform(self , df: DataFrame) -> DataFrame:
    weekDay_udf =  udf(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d').strftime('%w'))
    df = df.withColumn('weekDay', weekDay_udf(df["Day_datum"]))
    df = df.withColumn('sunday_binary' , F.when(df['weekDay']== '0',1).otherwise(0))
    df = df.drop('weekDay')
    return df
 

# COMMAND ----------

# DBTITLE 1,No extra features basetable fct
def make_simple_table(df):
  cols_to_drop_no_extra_features = [ 'Profile_name', 'Post_Date', 'Caption',  'Nbr_of_Likes','Nbr_of_Comments' ,'normalized_Nbr_of_Likes' , 'normalized_Nbr_of_Comments'      ,'Day_datum',  'no_nbrs', 'only_str_cap', '#words', '@words', 'words', 'filtered']
  
  df = df.drop(*cols_to_drop_no_extra_features)

  cols_to_scale_no_extra_features = ['Nbr_of_Pics', 'Nbr_of_hashtags', 'Nbr_of_mentions']
  df = min_max_scaler(df, cols_to_scale_no_extra_features).drop(*cols_to_scale_no_extra_features).fillna(0)

  return df


# COMMAND ----------

# DBTITLE 1,Basic basetable fct
#basic pipeline
def make_basic_table(df):
  #weekday nog insteken
  pipelineModel_basic=Pipeline(stages =[LC,PY,PC,DUR,DIFF,SUN]).fit(df)
  df_basic = pipelineModel_basic.transform(df)  
  
  print(df_basic.columns)  
  
  #hier iets insteken die de Y waarde interactief maakt voor verdere analyse
  cols_to_drop_basic = [ 'Profile_name', 'Post_Date', 'Caption',  'Nbr_of_Likes', 'Nbr_of_Comments',  'Day_datum', 'normalized_Nbr_of_Likes',                                            'normalized_Nbr_of_Comments', 'no_nbrs', 'only_str_cap', '#words', '@words', 'words', 'filtered']
  
  df_basic = df_basic.drop(*cols_to_drop_basic)
  
  cols_to_scale_basic = ['Nbr_of_Pics', 'Nbr_of_hashtags', 'Nbr_of_mentions' , 'aantal_woorden','days_since_last_post']
  df_basic = min_max_scaler(df_basic, cols_to_scale_basic).drop(*cols_to_scale_basic).fillna(0).drop('boolean_during')

  return df_basic




# COMMAND ----------

# DBTITLE 1,Topic basetable fct
def make_topic_table(df):
  #weekday nog insteken
  pipelineModel_topic=Pipeline(stages =[LC,PY,PC,DUR, DIFF, SUN]).fit(df)
  df_topic = pipelineModel_topic.transform(df)  
  
  print(df_topic.columns)  
  df_topic = add_topics_to_table(df_topic ,dominant_topic_df)
  
  #hard coded werkt dus alleen als we 30 topics hebben
  renaming = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0','9.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '16.0', '17.0',
 '18.0', '19.0', '20.0', '21.0', '22.0', '23.0', '24.0', '25.0', '26.0', '27.0', '28.0', '29.0']
  for a in renaming:   
    df_topic = df_topic.withColumnRenamed(a ,'topic_{}'.format(str(a).split('.')[0] ))
  
  
  
  #hier iets insteken die de Y waarde interactief maakt voor verdere analyse
  cols_to_drop_topic = [ 'Profile_name', 'Post_Date', 'Caption',  'Nbr_of_Likes', 'Nbr_of_Comments',  'Day_datum', 'normalized_Nbr_of_Likes',                                            'normalized_Nbr_of_Comments', 'no_nbrs', 'only_str_cap', '#words', '@words', 'words', 'filtered']
  df_topic = df_topic.drop(*cols_to_drop_topic)
  
  cols_to_scale_topic = ['Nbr_of_Pics', 'Nbr_of_hashtags', 'Nbr_of_mentions' , 'aantal_woorden','days_since_last_post']  
  df_topic = min_max_scaler(df_topic, cols_to_scale_topic).drop(*cols_to_scale_topic).fillna(0).drop('boolean_during')
  
  
  
  return df_topic


# COMMAND ----------

# DBTITLE 1,Individual words basetable fct
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))


def make_subset_words_using_RF_VARIABLEIMPORTANCE(df,nbr_of_best_predictors_per_iteration,size_of_subset):  
  
  
#   print("wat hij binnen krijgt in de tweede functie")
#   print(df.columns)
  
  
  cols_to_drop = ['Post_ID', 'Profile_name', 'Post_Date', 'Caption', 'Nbr_of_Pics', 'Nbr_of_Likes', 'Nbr_of_Comments', 'Multiple_Posts_int',
 'Is_video_int', 'Google_Maps_URL_binary_available', 'Location_added_to_post', 'Day_datum', 'normalized_Nbr_of_Likes', 'normalized_Nbr_of_Comments','no_nbrs', 'only_str_cap',
 '#words', '@words', 'words', 'Nbr_of_hashtags', 'Nbr_of_mentions', 'filtered', 'aantal_woorden', 'no_period', 'during', '30 days before', '30 days after', 'ticket_sale',
 'boolean_during' , 'sunday_binary', 'days_since_last_post']

  
  df = df.drop(*cols_to_drop)
  X = df.drop('y')
  
  kolommen = X.columns
  
  parts = [kolommen[i:i+size_of_subset] for i in range(0, len(kolommen), size_of_subset)]
  
  return_data = pd.DataFrame()
  for deeltje in parts:
    df_binnen_loop =df
    #RANDOM FORREST REGRESSIE
    #print(deeltje)
    VA = VectorAssembler(inputCols=[*deeltje],outputCol='CatFeatures')
    RF = RandomForestRegressor(labelCol = 'y', featuresCol = 'CatFeatures', numTrees = 50)

    pipelinefulldata = Pipeline().setStages([VA,RF]).fit(df_binnen_loop)
    df_binnen_loop = pipelinefulldata.transform(df_binnen_loop)

    
    feature_imp_dataframe = ExtractFeatureImp(pipelinefulldata.stages[-1].featureImportances,df_binnen_loop,'CatFeatures' ).head(nbr_of_best_predictors_per_iteration)
    return_data = return_data.append(feature_imp_dataframe,ignore_index = True)
        
  return return_data
  

def make_words_table(df,nbr_of_best_predictors_per_iteration,size_of_subset):
  
  pipelineModel_words=Pipeline(stages =[LC,PY,PC,DUR, DIFF, SUN]).fit(df)
  df_all = pipelineModel_words.transform(df) 
  
  df_words = df_all
    
  
  cols_to_drop_words = [ 'Profile_name', 'Post_Date', 'Caption',  'Nbr_of_Likes', 'Nbr_of_Comments',  'Day_datum', 'normalized_Nbr_of_Likes',                                            'normalized_Nbr_of_Comments', 'no_nbrs', 'only_str_cap', '#words', '@words', 'words', 'filtered']
  df_words = df_words.drop(*cols_to_drop_words)
  
  
  
  columns_to_keep1 = df_words.columns
  
  pipelineModel_words2=Pipeline(stages =[WC]).fit(df_all)
  df_all = pipelineModel_words2.transform(df_all)  
  

  kolommen = make_subset_words_using_RF_VARIABLEIMPORTANCE(df = df_all ,nbr_of_best_predictors_per_iteration =nbr_of_best_predictors_per_iteration,size_of_subset =size_of_subset)
  
  kolommen_te_behouden = kolommen['name'].tolist()
  
  #print(kolommen_te_behouden)
  
  #hier iets insteken die de Y waarde interactief maakt voor verdere analyse
  cols_all_to_keep = columns_to_keep1 + kolommen_te_behouden
  df_final = df_all.select(*cols_all_to_keep)
  
  cols_to_scale_words = ['Nbr_of_Pics', 'Nbr_of_hashtags', 'Nbr_of_mentions' , 'aantal_woorden', 'days_since_last_post']  
  df_final = min_max_scaler(df_final, cols_to_scale_words).drop(*cols_to_scale_words).fillna(0).drop('boolean_during')
  
  return df_final


#28 minuten om een subset te maken? is best lang

# COMMAND ----------

# DBTITLE 1,Img data fct
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))


def make_subset_img_using_RF_VARIABLEIMPORTANCE(df,nbr_of_best_predictors_per_iteration,size_of_subset):  
  cols_to_drop = ['Post_ID', 'Profile_name', 'Post_Date', 'Caption', 'Nbr_of_Pics', 'Nbr_of_Likes', 'Nbr_of_Comments', 'Multiple_Posts_int',
 'Is_video_int', 'Google_Maps_URL_binary_available', 'Location_added_to_post', 'Day_datum', 'normalized_Nbr_of_Likes', 'normalized_Nbr_of_Comments','no_nbrs', 'only_str_cap',
 '#words', '@words', 'words', 'Nbr_of_hashtags', 'Nbr_of_mentions', 'filtered', 'aantal_woorden', 'no_period', 'during', '30 days before', '30 days after', 'ticket_sale',
 'boolean_during', 'sunday_binary', 'days_since_last_post'  ]

  df = df.drop(*cols_to_drop)
  X = df.drop('y')

  kolommen = X.columns
  parts = [kolommen[i:i+size_of_subset] for i in range(0, len(kolommen), size_of_subset)]
  
  return_data = pd.DataFrame()
  for deeltje in parts:
    df_binnen_loop =df
    #RANDOM FORREST REGRESSIE
    #print(deeltje)
    VA = VectorAssembler(inputCols=[*deeltje],outputCol='CatFeatures')
    RF = RandomForestRegressor(labelCol = 'y', featuresCol = 'CatFeatures', numTrees = 50)

    pipelinefulldata = Pipeline().setStages([VA,RF]).fit(df_binnen_loop)
    df_binnen_loop = pipelinefulldata.transform(df_binnen_loop)
    print("checkpoint")

    feature_imp_dataframe = ExtractFeatureImp(pipelinefulldata.stages[-1].featureImportances,df_binnen_loop,'CatFeatures' ).head(nbr_of_best_predictors_per_iteration)
    return_data = return_data.append(feature_imp_dataframe,ignore_index = True)
    
  return return_data
  

def make_img_table(df,nbr_of_best_predictors_per_iteration,size_of_subset,year,only_person):
  #weekday nog insteken
  pipelineModel_img=Pipeline(stages =[LC,PY,PC,DUR, DIFF, SUN]).fit(df)
  df_all = pipelineModel_img.transform(df)  
  
  df_img = df_all
  
  cols_to_drop_img = ['Profile_name', 'Post_Date', 'Caption',  'Nbr_of_Likes', 'Nbr_of_Comments',  'Day_datum', 'normalized_Nbr_of_Likes',                                            'normalized_Nbr_of_Comments', 'no_nbrs', 'only_str_cap', '#words', '@words', 'words', 'filtered']
  df_img = df_img.drop(*cols_to_drop_img)
  
  columns_to_keep1 = df_img.columns
  
  df_all = img_data_to_table(year, df_all, only_person)
  
    
  #make subset using the variable importance and then drop the remaining columns  
  kolommen = make_subset_img_using_RF_VARIABLEIMPORTANCE(df = df_all ,nbr_of_best_predictors_per_iteration =nbr_of_best_predictors_per_iteration,size_of_subset =size_of_subset)
  #print(kolommen)
  kolommen_te_behouden = kolommen['name'].tolist()
  cols_all_to_keep = columns_to_keep1 + kolommen_te_behouden
  df_all = df_all.select(*cols_all_to_keep)
  
  
  cols_to_scale_img = ['Nbr_of_Pics', 'Nbr_of_hashtags', 'Nbr_of_mentions' , 'aantal_woorden', 'days_since_last_post'] + kolommen_te_behouden 
  df_all = min_max_scaler(df_all, cols_to_scale_img).drop(*cols_to_scale_img).fillna(0).drop('boolean_during')
  
  return df_all





# COMMAND ----------

# DBTITLE 1,Merge Dataframes Functions 
#beetje preprocessen nodig

def column_dropper_for_imgdata(df, threshold):
  # Takes a dataframe and threshold for missing values. Returns a dataframe.
  total_records = df.count()
  for col in df.columns[:-1]:
    # Calculate the percentage of missing values
    missing = df.where(df[col]==0.0).count()
    missing_percent = missing / total_records
    # Drop column if percent of missing is more than threshold
    if missing_percent > threshold:
      df = df.drop(col)
  return df


def img_data_to_table(img_year, postsyear, add_only_persons):
  sql_command = "select * from Image_data_" + str(img_year)  
  df_imagedata = spark.sql(sql_command)

  # Drop columns that are more than 50% missing
  
  #df2018_imagedata_cl = df2018_imagedata_cl.withColumn('resolution' ,df2018_imagedata_cl.Height * df2018_imagedata_cl.Width )
  if add_only_persons:
    df_imagedata = column_dropper_for_imgdata(df_imagedata, 0.5).drop('Height')
    df_imagedata = df_imagedata.groupby('Timestamp_id').agg({'person_in_img':'sum', 'Width':'avg'})
  else:
    df_imagedata = df_imagedata.drop('Height')

  df = postsyear.join(df_imagedata,df_imagedata.Timestamp_id==postsyear.Post_Date).drop('Timestamp_id')
  return df

def add_topics_to_table(df_year, df_dominant_topics):
  pivoted = df_dominant_topics.groupBy("Post_ID").pivot("Dominant_Topic").agg(F.lit(1))
  pivoted = pivoted.na.fill(0)

  basetable = df_year.join(pivoted , 'Post_ID', 'inner')
  
  return basetable


def min_max_scaler(df, cols_to_scale):
  # Takes a dataframe and list of columns to minmax scale. Returns a dataframe.
  for col in cols_to_scale:
    # Define min and max values and collect them
    max_days = df.agg({col: 'max'}).collect()[0][0]
    min_days = df.agg({col: 'min'}).collect()[0][0]
    new_column_name = 'scaled_' + col
    # Create a new column based off the scaled data
    df = df.withColumn("scaled_{}".format(col), 
                      (df[col] - min_days) / (max_days - min_days))
  return df
  
 

# COMMAND ----------

# DBTITLE 1,Evaluate performance regression fct
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd 
 

def run_all_regression(allbasetables=[],names=[],label='y'):
  
  for index,basetable in enumerate(allbasetables):
    
    lrsq,lrmae,lrrmse,lrmse,lrModel = run_lineaire_regression(basetable,label)
    rfsq,rfmae,rfrmse,rfmse,rfModel = run_randomforest_regression(basetable,label)
    dtsq,dtmae,dtrmse,dtmse,dtModel = run_decisiontree_regression(basetable,label)
    
    if (index==0):  
      #4pandas dataframen aanmaken
      data_r2 = {names[index]:[lrsq,rfsq,dtsq]}
      data_mae = {names[index]:[lrmae,rfmae,dtmae]}
      data_rmse = {names[index]:[lrrmse,rfrmse,dtrmse]}
      data_mse = {names[index]:[lrmse,rfmse,dtmse]}
      df_pan_r2 = pd.DataFrame(data_r2, index =['LIN REG', 'RANDOM FOREST', 'DECISION TREE'])
      df_pan_mae = pd.DataFrame(data_mae, index =['LIN REG', 'RANDOM FOREST', 'DECISION TREE'])
      df_pan_rmse = pd.DataFrame(data_rmse, index =['LIN REG', 'RANDOM FOREST', 'DECISION TREE'])
      df_pan_mse = pd.DataFrame(data_mse, index =['LIN REG', 'RANDOM FOREST', 'DECISION TREE'])
    else:      
      df_pan_r2.insert(index, names[index], [lrsq,rfsq,dtsq]) 
      df_pan_mae.insert(index, names[index], [lrmae,rfmae,dtmae]) 
      df_pan_rmse.insert(index, names[index], [lrrmse,rfrmse,dtrmse]) 
      df_pan_mse.insert(index, names[index], [lrmse,rfmse,dtmse]) 
  
  return df_pan_r2,df_pan_mae,df_pan_rmse,df_pan_mse
  
      
  
def run_lineaire_regression(basetable,label):
  
    VA = VectorAssembler(inputCols=[*basetable.drop(label).columns],outputCol='CatFeatures')
    pipelinefulldata = Pipeline().setStages([VA]).fit(basetable)
    basetable = pipelinefulldata.transform(basetable)
    (training, test) = basetable.randomSplit([0.70, 0.30], seed=45)

    #LINEAIRE REGRESSIE
    LR = LinearRegression(labelCol = label, featuresCol = 'CatFeatures', maxIter = 100,elasticNetParam=1.0)  #LASSO BIJ ELASTICNETPARAM =1 , RIDGE REGRESSION BIJ ELASTICNETPARAM=0
    lrModel = LR.fit(training)
    lrPredictions = lrModel.transform(test)

    # LINEAR REGRESSION EVALUATOR
    lrEvaluator = RegressionEvaluator(labelCol = label, predictionCol = 'prediction')
    lrsq = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: 'r2'})
    lrmae = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: 'mae'})
    lrrmse = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: 'rmse'})
    lrmse = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: 'mse'})    
    print('lineaire regressie')
    print('R^2  : %g' % lrsq)
    print('MAE  : %g' % lrmae)
    print('RMSE : %g' % lrrmse)
    print('MSE  : %g' % lrmse)      
    return lrsq,lrmae,lrrmse,lrmse,lrModel
  
def run_randomforest_regression(basetable,label):
  
  VA = VectorAssembler(inputCols=[*basetable.drop(label).columns],outputCol='CatFeatures')
  pipelinefulldata = Pipeline().setStages([VA]).fit(basetable)
  basetable = pipelinefulldata.transform(basetable)
  (training, test) = basetable.randomSplit([0.70, 0.30], seed=45)
  
  #RANDOM FORREST REGRESSIE
  RF = RandomForestRegressor(labelCol = label, featuresCol = 'CatFeatures', numTrees = 500)
  rfModel = RF.fit(training)
  rfPredictions = rfModel.transform(test)
  
  #RANDOM FORREST REGRESSOR EVALUATOR
  rfEvaluator = RegressionEvaluator(labelCol = label, predictionCol = 'prediction')
  rfsq = rfEvaluator.evaluate(rfPredictions, {rfEvaluator.metricName: 'r2'})
  rfmae = rfEvaluator.evaluate(rfPredictions, {rfEvaluator.metricName: 'mae'})
  rfrmse = rfEvaluator.evaluate(rfPredictions, {rfEvaluator.metricName: 'rmse'})
  rfmse = rfEvaluator.evaluate(rfPredictions, {rfEvaluator.metricName: 'mse'})
  print('random forest')
  print('R^2  : %g' % rfsq)
  print('MAE  : %g' % rfmae)
  print('RMSE : %g' % rfrmse)
  print('MSE  : %g' % rfmse)
  return rfsq,rfmae,rfrmse,rfmse,rfModel
  
def run_decisiontree_regression(basetable,label):
  
  VA = VectorAssembler(inputCols=[*basetable.drop(label).columns],outputCol='CatFeatures')
  pipelinefulldata = Pipeline().setStages([VA]).fit(basetable)
  basetable = pipelinefulldata.transform(basetable)
  (training, test) = basetable.randomSplit([0.70, 0.30], seed=45)
  
  #DECISION TREE
  dt = DecisionTreeRegressor(labelCol=label,featuresCol="CatFeatures")
  dtModel = dt.fit(training)
  dtPredictions = dtModel.transform(test)
  
  #DECISION TREE REGRESSOR EVALUATOR
  dtEvaluator = RegressionEvaluator(labelCol = label, predictionCol = 'prediction')
  dtsq = dtEvaluator.evaluate(dtPredictions, {dtEvaluator.metricName: 'r2'})
  dtmae = dtEvaluator.evaluate(dtPredictions, {dtEvaluator.metricName: 'mae'})
  dtrmse = dtEvaluator.evaluate(dtPredictions, {dtEvaluator.metricName: 'rmse'})
  dtmse = dtEvaluator.evaluate(dtPredictions, {dtEvaluator.metricName: 'mse'})
  print('decision tree')
  print('R^2  : %g' % dtsq)
  print('MAE  : %g' % dtmae)
  print('RMSE : %g' % dtrmse)
  print('MSE  : %g' % dtmse)
  return dtsq,dtmae,dtrmse,dtmse,dtModel


# COMMAND ----------

# DBTITLE 1,Evaluate performance classification fct
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator

def groep_UDF(q):
    def get_class_likes(row):
      if row==None:
        return 0
      elif row<q[0]:
        return 0 
      elif (row>=q[0] and row<q[1]):
        return 1
      elif(row>=q[1] and row<q[2]):
        return 2
      else:
        return 3    
    return f.udf(get_class_likes, IntegerType())
  
def groep_half_UDF(q):
    def get_class_likes(row):
      if row==None:
        return 0
      elif row<q[0]:
        return 0       
      else:
        return 1    
    return f.udf(get_class_likes, IntegerType())  


def run_all_classification(allbasetables=[],names=[],label='y',half=False):
  
  for index,basetable in enumerate(allbasetables):
    rf_AUC,rf_ACC,rf_class_Model = run_rf_classification(basetable,label,half)
    lr_AUC,lr_ACC,lr_class_Model = run_logreg_classification(basetable,label,half)
    
    if (index==0):  
      #4pandas dataframen aanmaken
      data_auc = {names[index]:[lr_AUC,rf_AUC]}
      data_acc = {names[index]:[lr_ACC,rf_ACC]}
      
      df_pan_auc = pd.DataFrame(data_auc, index =['LIN REG', 'RANDOM FOREST'])
      df_pan_acc= pd.DataFrame(data_acc, index =['LIN REG', 'RANDOM FOREST'])
      
    else:      
      df_pan_auc.insert(index, names[index],[lr_AUC,rf_AUC]) 
      df_pan_acc.insert(index, names[index], [lr_ACC,rf_ACC]) 
      
  return df_pan_auc,df_pan_acc
  
#   #MULTII-LAYER-CALSSIFIER
#   # 4 input layer, hiddenlayers, output layers
#   layers = [4, 5, 5, 4]
#   # create the trainer and set its parameters
#   trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
#   # train the model
#   MLCmodel = trainer.fit(training).transform(test)
#   predictionAndLabels = MLCmodel.select("prediction", "label_groepen")
#   evaluatorMLC = MulticlassClassificationEvaluator(metricName="accuracy")
#   print("MULTII-LAYER-CALSSIFIER : "+year+" ACC    " + str(evaluatorMLC.evaluate(predictionAndLabels)))
  


def run_rf_classification(basetable,label,half):
  if half==False:      
    quantil = basetable.approxQuantile(label, [0.25,0.5,0.75], 0) 
    basetable = basetable.withColumn('label', groep_UDF(quantil)(f.col(label)))
    basetable = basetable.drop(label)

    VA =VectorAssembler(inputCols=[*basetable.drop('label').columns],outputCol='all_features')
    (training, test) = basetable.randomSplit([0.70, 0.30], seed=45)
    rfClassifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'all_features', numTrees = 500)
  else:
    quantil = basetable.approxQuantile(label, [0.5], 0) 
    basetable = basetable.withColumn('label', groep_half_UDF(quantil)(f.col(label)))
    basetable = basetable.drop(label)

    VA =VectorAssembler(inputCols=[*basetable.drop('label').columns],outputCol='all_features')
    (training, test) = basetable.randomSplit([0.70, 0.30], seed=45)
    rfClassifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'all_features', numTrees = 500)
    
  rf_class_Model= Pipeline().setStages([VA,rfClassifier]).fit(training)
  rf_class_pred = rf_class_Model.transform(test)

  evaluator = BinaryClassificationEvaluator()
  evaluatorM = MulticlassClassificationEvaluator()
  rf_AUC = evaluator.evaluate(rf_class_pred, {evaluator.metricName: 'areaUnderROC'})
  rf_ACC = evaluatorM.evaluate(rf_class_pred, {evaluatorM.metricName: 'accuracy'})
  print('RANDOM FOREST CLASSIFIER : AUC : %s' %(rf_AUC))
  print('RANDOM FOREST CLASSIFIER : ACC : %s' %(rf_ACC))
  return rf_AUC,rf_ACC,rf_class_Model
  
  
def run_logreg_classification(basetable,label,half):
  if half==False:
    quantil = basetable.approxQuantile(label, [0.25,0.5,0.75], 0)
    basetable = basetable.withColumn('label', groep_UDF(quantil)(f.col(label)))
    basetable = basetable.drop(label)
    VA =VectorAssembler(inputCols=[*basetable.drop('label').columns],outputCol='all_features')

    lrClassifier = LogisticRegression(labelCol = 'label', featuresCol = 'all_features', maxIter = 100,elasticNetParam=1, family = 'multinomial')
    (training, test) = basetable.randomSplit([0.70, 0.30], seed=45)                                     

    lr_class_Model= Pipeline().setStages([VA,lrClassifier]).fit(training)
    lr_class_pred = lr_class_Model.transform(test)

    evaluator = BinaryClassificationEvaluator()
    evaluatorM = MulticlassClassificationEvaluator()
    lr_AUC = evaluator.evaluate(lr_class_pred, {evaluator.metricName: 'areaUnderROC'})    
    lr_ACC = evaluatorM.evaluate(lr_class_pred, {evaluatorM.metricName: 'accuracy'})
    print('LOGISTIC REGRESSION : AUC  : %s' %(lr_AUC))
    print('LOGISTIC REGRESSION : ACC : %s' %(lr_ACC))
    return lr_AUC,lr_ACC,lr_class_Model
                                     
  else:
    quantil = basetable.approxQuantile(label, [0.5], 0) 
    basetable = basetable.withColumn('label', groep_half_UDF(quantil)(f.col(label)))
    basetable = basetable.drop(label)
    VA =VectorAssembler(inputCols=[*basetable.drop('label').columns],outputCol='all_features')

    lrClassifier = LogisticRegression(labelCol = 'label', featuresCol = 'all_features',maxIter=100, regParam=0.3, elasticNetParam=1)
    (training, test) = basetable.randomSplit([0.70, 0.30], seed=45)
                                     

    lr_class_Model= Pipeline().setStages([VA,lrClassifier]).fit(training)
    lr_class_pred = lr_class_Model.transform(test)

    evaluator = BinaryClassificationEvaluator()
    evaluatorM = MulticlassClassificationEvaluator()
    lr_AUC = evaluator.evaluate(lr_class_pred, {evaluator.metricName: 'areaUnderROC'})    
    lr_ACC = evaluatorM.evaluate(lr_class_pred, {evaluatorM.metricName: 'accuracy'})
    print('LOGISTIC REGRESSION : AUC  : %s' %(lr_AUC))
    print('LOGISTIC REGRESSION : ACC : %s' %(lr_ACC))
    return lr_AUC,lr_ACC,lr_class_Model

# COMMAND ----------

# DBTITLE 1,CODE 


# COMMAND ----------

# DBTITLE 1,STEP 0: Load all posts and comment data
#True in onderstaande functie
#nu laat ik die kolommen vallen omdat ze geen meerwaarde bieden, in 2019 is er wel een relevante locatie dan beter laten staan

posts13 = initial_posts_processing("tomorrowland_Posts_2013.csv", True)  
posts14 = initial_posts_processing("tomorrowland_Posts_2014.csv", True)  
posts15 = initial_posts_processing("tomorrowland_Posts_2015.csv", True)  
posts16 = initial_posts_processing("tomorrowland_Posts_2016.csv", True)  
posts17 = initial_posts_processing("tomorrowland_Posts_2017.csv", True)  
posts18 = initial_posts_processing("tomorrowland_Posts_2018.csv", True)  
posts19 = initial_posts_processing("tomorrowland_Posts_2019.csv", True)  

posts13_df = posts13.withColumn('Day_datum',F.date_format('Post_Date','yyyy-MM-dd'))
posts13_df = posts13_df.withColumn('Day_datum', posts13_df.Day_datum.cast(DateType()))
posts14_df = posts14.withColumn('Day_datum',F.date_format('Post_Date','yyyy-MM-dd'))
posts14_df = posts14_df.withColumn('Day_datum', posts14_df.Day_datum.cast(DateType()))
posts15_df = posts15.withColumn('Day_datum',F.date_format('Post_Date','yyyy-MM-dd'))
posts15_df = posts15_df.withColumn('Day_datum', posts15_df.Day_datum.cast(DateType()))
posts16_df = posts16.withColumn('Day_datum',F.date_format('Post_Date','yyyy-MM-dd'))
posts16_df = posts16_df.withColumn('Day_datum', posts16_df.Day_datum.cast(DateType()))
posts17_df = posts17.withColumn('Day_datum',F.date_format('Post_Date','yyyy-MM-dd'))
posts17_df = posts17_df.withColumn('Day_datum', posts17_df.Day_datum.cast(DateType()))
posts18_df = posts18.withColumn('Day_datum',F.date_format('Post_Date','yyyy-MM-dd'))
posts18_df = posts18_df.withColumn('Day_datum', posts18_df.Day_datum.cast(DateType()))
posts19_df = posts19.withColumn('Day_datum',F.date_format('Post_Date','yyyy-MM-dd'))
posts19_df = posts19_df.withColumn('Day_datum', posts19_df.Day_datum.cast(DateType()))





comments13 = initial_comments_processing("tomorrowland_Comments_13_rand_sample")  
comments14 = initial_comments_processing("tomorrowland_Comments_14_rand_sample")  
comments15 = initial_comments_processing("tomorrowland_Comments_15_rand_sample")  
comments16 = initial_comments_processing("tomorrowland_Comments_16_rand_sample")  
comments17 = initial_comments_processing("tomorrowland_Comments_17_rand_sample")  
comments18 = initial_comments_processing("tomorrowland_Comments_18_rand_sample")  
comments19 = initial_comments_processing("tomorrowland_Comments_19_rand_sample")  




# COMMAND ----------

# DBTITLE 1,STEP 1: Luminoth code
#Because it takes a long time all this code already has been run and can be found in teh zipfile

#make the pandas dataframe with luminoth predictions (takes a long time)

#df2013_pd= luminoth_predict("2013")
#df2014_pd= luminoth_predict("2014")
#df2015_pd= luminoth_predict("2015")
#df2016_pd= luminoth_predict("2016")
#df2017_pd= luminoth_predict("2017")
#df2018_pd= luminoth_predict("2018")
#df2019_pd= luminoth_predict("2019")




# COMMAND ----------



# COMMAND ----------

#load image data

#df2013_imagedata = spark.sql('select * from Image_data_2013')
#df2014_imagedata = spark.sql('select * from Image_data_2014')
#df2015_imagedata = spark.sql('select * from Image_data_2015')
#df2016_imagedata = spark.sql('select * from Image_data_2016')
#df2017_imagedata = spark.sql('select * from Image_data_2017')
#df2018_imagedata = spark.sql('select * from Image_data_2018')
#df2019_imagedata = spark.sql('select * from Image_data_2019')


# COMMAND ----------

# DBTITLE 1,STEP 2: TOPIC MODELING
#preprocessing
RR0 = CleanCaption() 
Respons = MakeResponse(cols_to_scale = ['Nbr_of_Likes', 'Nbr_of_Comments'], alpha = 0.0)
SWR = StopWordsRemover(inputCol = 'words', outputCol = 'filtered')
pipelineModel = Pipeline(stages =[Respons,RR0,SWR]).fit(posts13_df) 
posts13_df = pipelineModel.transform(posts13_df)
posts14_df = pipelineModel.transform(posts14_df)
posts15_df = pipelineModel.transform(posts15_df)
posts16_df = pipelineModel.transform(posts16_df)
posts17_df = pipelineModel.transform(posts17_df)
posts18_df = pipelineModel.transform(posts18_df)
posts19_df = pipelineModel.transform(posts19_df)


#put all posts together to get a global topic model for all the years together
All_captions = posts13_df.union(posts14_df).union(posts15_df).union(posts16_df).union(posts17_df).union(posts18_df).union(posts19_df)

# COMMAND ----------

#Visually check which number of topics seems good

model_list, coherence_values = compute_coherence_values(df_with_all_captions = All_captions,  start=2, limit=40, step=2, show_graph = True)



# COMMAND ----------

# Print the coherence scores
x = range(2, 40, 2)

for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

# taking 30 seems like a good bet

# COMMAND ----------

Compute_Perplexity_and_Coherence(All_captions, 10)

# when taking into account perplexity as well, 10 seems a way better option      num topics =10 ->>   perp = -8,53 , cohe = 0.56
# as the coherence is similar but the perplexity is 3 times lower


# COMMAND ----------

#check for a chosen number of topics the content of the topics
# here between the brackets 

#--------------
# aantal topics = 2 + 2*x 
# x = (aantal topics -2) /2
# x= 14 -> 30 TOPICS

x =4

optimal_model = model_list[x]
model_topics = optimal_model.show_topics(formatted=False)

pprint(optimal_model.print_topics(num_words=10))


# COMMAND ----------

#Werkt soms niet dus daarom in comment


# #Inspect visually the preferred number of topics and their content

# corpus_adjusted , id2word , dont_use_var1, dont_use_var2 = make_corpus(All_captions)

# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_adjusted,
#                                            id2word=id2word,
#                                            num_topics=30,    #change here number of topics
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=30,
#                                            alpha='auto',
#                                            per_word_topics=True)

# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus_adjusted, id2word)
# vis

# COMMAND ----------

#Make a dataset where every caption gets the dominant topic

dominant_topic_df = get_dominant_topic_list(All_captions, 30)

#inspect the frequencies of the topics
#dominant_topic_df.groupby('Dominant_Topic').count().show(50)

#dominant_topic_df.show()  

# COMMAND ----------

dominant_topic_df.groupby('Dominant_Topic').count().show(50)


# COMMAND ----------

# DBTITLE 1,STEP 3:Extract and enhance features from posts dataframe
#maak instanties aan van de pipeline stappen

#----------------------------------------------------------------------------------------
# Determine here which y is most important, likes or comments
# alpha = 1 -> only comments
# alpha = 0 -> only likes
# alpha = 1 -> equal importance for likes and comments

alpha_value = 0.0
#-----------------------------------------------------------------------------------------

Respons = MakeResponse(cols_to_scale = ['Nbr_of_Likes', 'Nbr_of_Comments'], alpha = alpha_value)   


RR0 = CleanCaption() #Creates onlystr, nr of # , nr of @
DC = DayCreator() #niet gebruiken
WD = Weekday() #niet gebruiken
DUR = During_festival()
PY = Period_Year()
PC = period_columns()
LC = lengte_caption()
SWR = StopWordsRemover(inputCol = 'words', outputCol = 'filtered') #behoud de belangrijkste, en verwijdert onnodige basiswoorden
BOW = CountVectorizer(inputCol = 'filtered', outputCol = 'features') 
WC = word_columns()
#PS = Pol_and_Subj()  
#TT = TotaalAantalMentionsInComments()    
BOW = CountVectorizer(inputCol = 'filtered', outputCol = 'features') 
#W2V = Word2Vec(inputCol = 'filtered', outputCol = 'features')
WC = word_columns()
DIFF = days_since_last_post()
SUN = Sunday()


# COMMAND ----------

#Make simplest dataset of 2016
posts15_simple = make_simple_table(posts15_df).drop('Post_ID')

# COMMAND ----------

posts15_simple.show()

# COMMAND ----------

#make basic dataset of 2016
posts15_basic = make_basic_table(posts15_df).drop('Post_ID')

# COMMAND ----------

posts15_basic.toPandas().head()

# COMMAND ----------

#make dataset with the dominant topic added
posts15_topic_met_post_id= make_topic_table(posts15_df)
posts15_topic = posts15_topic_met_post_id.drop('Post_ID')

# COMMAND ----------

posts19_topic_met_post_id.toPandas().head()

# COMMAND ----------

#make dataset with individual words added 
posts15_words_met_post_id= make_words_table(posts15_df,2,35)
posts15_words =posts15_words_met_post_id.drop('Post_ID')

# COMMAND ----------

posts19_words.toPandas().head()

# COMMAND ----------

def run_lineaire_regression_coeff_features(basetable,year,label):  
  VA = VectorAssembler(inputCols=[*basetable.drop(label).columns],outputCol='CatFeatures')
  pipelinefulldata = Pipeline().setStages([VA]).fit(basetable)
  basetable = pipelinefulldata.transform(basetable)
  (training, test) = basetable.randomSplit([0.70, 0.30], seed=45)
  
  #LINEAIRE REGRESSIE
  LR = LinearRegression(labelCol = label, featuresCol = 'CatFeatures', maxIter = 100,elasticNetParam=1.0)  #LASSO BIJ ELASTICNETPARAM =1 , RIDGE REGRESSION BIJ ELASTICNETPARAM=0
  lrModel = LR.fit(training)
  lrPredictions = lrModel.transform(test)
  
  # LINEAR REGRESSION EVALUATOR
  lrEvaluator = RegressionEvaluator(labelCol = label, predictionCol = 'prediction')
  lrsq = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: 'r2'})
  lrmae = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: 'mae'})
  lrrmse = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: 'rmse'})
  lrmse = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: 'mse'})
  print('lineaire regressie'+year)
  print('R^2  : %g' % lrsq)
  print('MAE  : %g' % lrmae)
  print('RMSE : %g' % lrrmse)
  print('MSE  : %g' % lrmse)
  return_data = pd.DataFrame()
  feature_imp_dataframe = ExtractFeatureImp(lrModel.coefficients,basetable,'CatFeatures')
  return_data = return_data.append(feature_imp_dataframe,ignore_index = True)
  
  return return_data

def run_randomforest_regression_feaures(basetable,label):
  
  VA = VectorAssembler(inputCols=[*basetable.drop(label).columns],outputCol='CatFeatures')
  pipelinefulldata = Pipeline().setStages([VA]).fit(basetable)
  basetable = pipelinefulldata.transform(basetable)
  (training, test) = basetable.randomSplit([0.70, 0.30], seed=45)
  
  #RANDOM FORREST REGRESSIE
  RF = RandomForestRegressor(labelCol = label, featuresCol = 'CatFeatures', numTrees = 500)
  rfModel = RF.fit(training)
  rfPredictions = rfModel.transform(test)
  
  #RANDOM FORREST REGRESSOR EVALUATOR
  rfEvaluator = RegressionEvaluator(labelCol = label, predictionCol = 'prediction')
  rfsq = rfEvaluator.evaluate(rfPredictions, {rfEvaluator.metricName: 'r2'})
  rfmae = rfEvaluator.evaluate(rfPredictions, {rfEvaluator.metricName: 'mae'})
  rfrmse = rfEvaluator.evaluate(rfPredictions, {rfEvaluator.metricName: 'rmse'})
  rfmse = rfEvaluator.evaluate(rfPredictions, {rfEvaluator.metricName: 'mse'})
  print('random forest')
  print('R^2  : %g' % rfsq)
  print('MAE  : %g' % rfmae)
  print('RMSE : %g' % rfrmse)
  print('MSE  : %g' % rfmse)
  return_data = pd.DataFrame()
  feature_imp_dataframe = ExtractFeatureImp(rfModel.featureImportances,basetable,'CatFeatures')
  return_data = return_data.append(feature_imp_dataframe,ignore_index = True)
  
  return return_data

# COMMAND ----------

# DBTITLE 1,All data
# posts16_simple
# posts16_basic
# posts16_topic
# posts16_words
# posts16_img


double_cols = ['Multiple_Posts_int','Is_video_int','Google_Maps_URL_binary_available','Location_added_to_post','no_period','during','30 days before'
               ,'30 days after','ticket_sale','scaled_Nbr_of_Pics','scaled_Nbr_of_hashtags','scaled_Nbr_of_mentions','scaled_aantal_woorden', 'y', 'sunday_binary','scaled_days_since_last_post']

posts15_words_drop = posts15_words_met_post_id.drop(*double_cols)
#posts16_img_drop = posts16_img_met_post_id.drop(*double_cols)

posts15_all = posts15_topic_met_post_id.join(posts15_words_drop, on = 'Post_ID')
#.join(posts16_img_drop, on = 'Post_ID')
posts15_all = posts15_all.drop('Post_ID')

# COMMAND ----------

# DBTITLE 1,STEP 5: Analysis

lijst_loopke = [posts15_simple,posts15_basic,posts15_topic,posts15_words,posts15_all]
lijst_head= ['simple_data','basic_data','topic_data','individual_words_data','all_data']

df_pan_r2,df_pan_mae,df_pan_rmse,df_pan_mse =run_all_regression(lijst_loopke,lijst_head,'y')
  

# COMMAND ----------

df_pan_r2.head()

# COMMAND ----------

df_pan_rmse.head()

# COMMAND ----------

df_pan_mse.head()

# COMMAND ----------

df_pan_mae.head()

# COMMAND ----------

#Classify all datasets 

df_cl_auc,df_cl_acc =run_all_classification(allbasetables=lijst_loopke,names=lijst_head,label = 'y')
  
# duurt 1.62 uren

# COMMAND ----------

df_cl_auc.head()

# COMMAND ----------

df_cl_acc.head()

# COMMAND ----------

coeff = run_lineaire_regression_coeff_features(posts19_words,"2019","y")

# COMMAND ----------

coeff.tail(20)
coeff.head(20)

# COMMAND ----------

output_feature_importance = run_lineaire_regression_coeff_features(posts19_words,y")
import seaborn as sns
#sns.barplot(output_feature_importance.name ,output_feature_importance.score)

figure, ax = plt.subplots(figsize = (8,10))
#sns.scatterplot(index,'Nbr_of_Likes', hue = 'During_festival', data = posts171819.toPandas(),ax = ax)
sns.barplot(output_feature_importance.name[:15] ,output_feature_importance.score[:15],ax = ax)
ax.set_xlabel('features')
#ax.set_ylabel('Likes per post')
# ax.set_xlim(-10,1000)
# ax.set_ylim(-1000,850000)
#ax.set_xticks(xticks)
ax.set_xticklabels(output_feature_importance.name[:15],rotation = 90)
ax.set_title("Feature importances for year 2019")
ax.legend(loc = "upper left")
plt.tight_layout()
display(figure)


#display(sns.barplot(output_feature_importance.name[:10] ,output_feature_importance.score[:10], ))

# COMMAND ----------

# MAGIC %md
# MAGIC ##SNIPPETS

# COMMAND ----------

# cols_to_remove = list()

# for col in preparedclass_words_2019.columns:
#   # Count the number of 1 values in the binary column
#   obs_count = preparedclass_words_2019.agg({col: 'sum'}).collect()[0][0]
#   print(obs_count)
#   # If less than our observation threshold, remove
#   if obs_count <1:
#     cols_to_remove.append(col)
    
    


# COMMAND ----------

from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


cols_to_drop = ['Post_ID', 'Profile_name', 'Post_Date', 'Caption', 'Nbr_of_Pics', 'Nbr_of_Likes', 'Nbr_of_Comments', 'Multiple_Posts_int',
 'Is_video_int', 'Google_Maps_URL_binary_available', 'Location_added_to_post', 'Day_datum', 'normalized_Nbr_of_Likes', 'normalized_Nbr_of_Comments','no_nbrs', 'only_str_cap',
 '#words', '@words', 'words', 'Nbr_of_hashtags', 'Nbr_of_mentions', 'filtered', 'aantal_woorden', 'no_period', 'during', '30 days before', '30 days after', 'ticket_sale',
 'boolean_during'  ]

preparedclass_words_2017 = preparedclass_words_2017.drop(*cols_to_drop)
X = preparedclass_words_2017.drop('y')

kolommen = X.columns
aantal_delen = 40
parts = [kolommen[i:i+aantal_delen] for i in range(0, len(kolommen), aantal_delen)]
parts



# COMMAND ----------

# DBTITLE 1,DESCRIPTIVES: Visual


# COMMAND ----------

# DBTITLE 1,Evolution of likes per post over time
all_posts = posts13.union(posts14).union(posts15).union(posts16).union(posts17).union(posts18).union(posts19)
all_posts = all_posts.withColumn('Year',year('Post_Date'))

# Get likes per post without outliers

likes_13 = posts13.select('Nbr_of_Likes').collect()
mean_13 = np.mean(likes_13)
std_13 = np.std(likes_13)
posts13_wo = posts13.filter(posts13.Nbr_of_Likes<=mean_13+2*std_13)
posts13_outliers_above = posts13.filter(posts13.Nbr_of_Likes>=mean_13+2*std_13)
posts13_outliers_beneath = posts13.filter(posts13.Nbr_of_Likes<=mean_13-1.5*std_13)

likes_14 = posts14.select('Nbr_of_Likes').collect()
mean_14 = np.mean(likes_14)
std_14 = np.std(likes_14)
posts14_wo = posts14.filter(posts14.Nbr_of_Likes<=mean_14+2*std_14)
posts14_outliers_above = posts14.filter(posts14.Nbr_of_Likes>=mean_14+2*std_14)
posts14_outliers_beneath = posts14.filter(posts14.Nbr_of_Likes<=mean_14-1.5*std_14)

likes_15 = posts15.select('Nbr_of_Likes').collect()
mean_15 = np.mean(likes_15)
std_15 = np.std(likes_15)
posts15_wo = posts15.filter(posts15.Nbr_of_Likes<=mean_15+2*std_15)
posts15_outliers_above = posts15.filter(posts15.Nbr_of_Likes>=mean_15+2*std_15)
posts15_outliers_beneath = posts15.filter(posts15.Nbr_of_Likes<=mean_15-1.5*std_15)

likes_16 = posts16.select('Nbr_of_Likes').collect()
mean_16 = np.mean(likes_16)
std_16 = np.std(likes_16)
posts16_wo = posts16.filter(posts16.Nbr_of_Likes<=mean_16+2*std_16)
posts16_outliers_above = posts16.filter(posts16.Nbr_of_Likes>=mean_16+2*std_16)
posts16_outliers_beneath = posts16.filter(posts16.Nbr_of_Likes<=mean_16-1.5*std_16)

likes_17 = posts17.select('Nbr_of_Likes').collect()
mean_17 = np.mean(likes_17)
std_17 = np.std(likes_17)
posts17_wo = posts17.filter(posts17.Nbr_of_Likes<=mean_17+2*std_17)
posts17_outliers_above = posts17.filter(posts17.Nbr_of_Likes>=mean_17+2*std_17)
posts17_outliers_beneath = posts17.filter(posts17.Nbr_of_Likes<=mean_17-1.5*std_17)

likes_18 = posts18.select('Nbr_of_Likes').collect()
mean_18 = np.mean(likes_18)
std_18 = np.std(likes_18)
posts18_wo = posts18.filter(posts18.Nbr_of_Likes<=mean_18+2*std_18)
posts18_outliers_above = posts18.filter(posts18.Nbr_of_Likes>=mean_18+2*std_18)
posts18_outliers_beneath = posts18.filter(posts18.Nbr_of_Likes<=mean_18-1.5*std_18)

likes_19 = posts19.select('Nbr_of_Likes').collect()
mean_19 = np.mean(likes_19)
std_19 = np.std(likes_19)
posts19_wo = posts19.filter(posts19.Nbr_of_Likes<=mean_19+2*std_19)
posts19_outliers_above = posts19.filter(posts19.Nbr_of_Likes>=mean_19+2*std_19)
posts19_outliers_beneath = posts19.filter(posts19.Nbr_of_Likes<=mean_19-1.5*std_19)

posts_wo = posts13_wo.union(posts14_wo).union(posts15_wo).union(posts16_wo).union(posts17_wo).union(posts18_wo).union(posts19_wo)

posts_wo = posts_wo.orderBy('Post_Date')
posts_wo = posts_wo.withColumn('Year',year('Post_Date'))

posts_outliers_above = posts13_outliers_above.union(posts14_outliers_above).union(posts15_outliers_above).union(posts16_outliers_above).union(posts17_outliers_above).union(posts18_outliers_above).union(posts19_outliers_above)
posts_outliers_above = posts_outliers_above.withColumn('Year',year('Post_Date'))

posts_outliers_beneath = posts13_outliers_beneath.union(posts14_outliers_beneath).union(posts15_outliers_beneath).union(posts16_outliers_beneath).union(posts17_outliers_beneath).union(posts18_outliers_beneath).union(posts19_outliers_beneath)
posts_outliers_beneath = posts_outliers_beneath.withColumn('Year',year('Post_Date'))

# COMMAND ----------

display(posts_outliers_above.orderBy(['Year','Nbr_of_Likes'],ascending=[True,False]))

# COMMAND ----------

display(posts_outliers_beneath.orderBy(['Year','Nbr_of_Likes'],ascending=[True,True]))

# COMMAND ----------

years = np.array(posts_wo.select('Year').collect())
xticks = list()
years_ticks = list()
for i in range(len(posts_wo.toPandas())):
  if years[i][0] not in years_ticks:
    years_ticks.append(years[i][0])
    xticks.append(i)
xticks_ax2 = list()
for i in range(len(xticks)):
  if i==len(xticks)-1:
    xtick = xticks[i]+(len(posts_wo.toPandas())-xticks[i])/2
  else:
    xtick = xticks[i]+(xticks[i+1]-xticks[i])/2
  xticks_ax2.append(xtick)

# COMMAND ----------

index1 = np.arange(len(posts_wo.toPandas()))
index2 = xticks_ax2
likes = posts_wo.select('Nbr_of_Likes').collect()
likes_per_year = all_posts.groupBy('Year').mean('Nbr_of_Likes').orderBy('Year').select('avg(Nbr_of_Likes)').collect()
figure, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(index1,y = 'Nbr_of_Likes',data = posts_wo.toPandas(), marker = ".",label = "Likes per post", alpha = 0.8,ax = ax)
ax.plot(index2,likes_per_year,color = 'darkorange', label = 'Average likes per year')
ax.set_xlabel('Year')
ax.set_ylabel('Likes per post')
ax.set_xlim(-10,2150)
ax.set_ylim(-1000,np.max(likes)+10000)
ax.set_xticks(xticks)
ax.set_xticklabels(years_ticks,rotation = 50)
ax.set_title("Evolution of likes per post")
ax.legend(loc = "upper left")
display(figure)

# COMMAND ----------

# DBTITLE 1,Boxplot of likes per year
likes_2013 = np.array(posts13.select('Nbr_of_Likes').collect())
likes_2014 = np.array(posts14.select('Nbr_of_Likes').collect())
likes_2015 = np.array(posts15.select('Nbr_of_Likes').collect())
likes_2016 = np.array(posts16.select('Nbr_of_Likes').collect())
likes_2017 = np.array(posts17.select('Nbr_of_Likes').collect())
likes_2018 = np.array(posts18.select('Nbr_of_Likes').collect())
likes_2019 = np.array(posts19.select('Nbr_of_Likes').collect())

fig, axes= plt.subplots(3,3,figsize = (14,9))
fig.suptitle("Likes per year")
sns.boxplot(likes_2013, ax = axes[0,0],orient = 'v')
axes[0,0].set_xticklabels(["2013"])
sns.boxplot(likes_2014, ax = axes[0,1],orient = 'v')
axes[0,1].set_xticklabels(["2014"])
sns.boxplot(likes_2015, ax = axes[0,2],orient = 'v')
axes[0,2].set_xticklabels(["2015"])
sns.boxplot(likes_2016,ax = axes[1,0],orient = 'v')
axes[1,0].set_xticklabels(["2016"])
sns.boxplot(likes_2017,ax = axes[1,1],orient = 'v')
axes[1,1].set_xticklabels(["2017"])
sns.boxplot(likes_2018,ax = axes[1,2],orient = 'v')
axes[1,2].set_xticklabels(["2018"])
sns.boxplot(likes_2019,ax = axes[2,0],orient = 'v')
axes[2,0].set_xticklabels(["2019"])
axes[2,1].set_axis_off()
axes[2,2].set_axis_off()
display(fig)

# COMMAND ----------

figure, ax = plt.subplots(figsize = (8,6))
sns.boxplot(x = 'Year',y = 'Nbr_of_Likes',data = all_posts.toPandas(), ax = ax)
ax.set_title('Distribution of likes per year')
ax.set_ylabel('Likes')
display(figure)

# COMMAND ----------

# DBTITLE 1,Likes per post evolution with posts during festival in orange
posts171819 = posts17.union(posts18).union(posts19)
posts171819 = posts171819.withColumn('Year',year('Post_Date'))
posts171819 = posts171819.withColumn('Day_Date',to_date('Post_Date','yyyy-MM-dd'))
posts171819 = posts171819.withColumn('During_bool',during_udf('Day_Date'))
posts171819 = posts171819.withColumn('During_festival',when(col('During_bool')==0,False).otherwise(True))
posts171819 = posts171819.orderBy('Post_Date')

years = np.array(posts171819.select('Year').collect())
xticks = list()
years_ticks = list()
for i in range(len(posts171819.toPandas())):
  if years[i][0] not in years_ticks:
    years_ticks.append(years[i][0])
    xticks.append(i)

index = np.arange(len(posts171819.toPandas()))
figure, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(index,'Nbr_of_Likes', hue = 'During_festival', data = posts171819.toPandas(),ax = ax)
ax.set_xlabel('Year')
ax.set_ylabel('Likes per post')
ax.set_xlim(-10,1000)
ax.set_ylim(-1000,850000)
ax.set_xticks(xticks)
ax.set_xticklabels(years_ticks,rotation = 50)
ax.set_title("Likes per post 2017, 2018, 2019")
ax.legend(loc = "upper left")
display(figure)

# COMMAND ----------

# DBTITLE 1,Boxplot likes per year with post about ticket sale
fig, axes= plt.subplots(2,3,figsize = (12,8))
axes[0,0].boxplot(likes_2014)
axes[0,0].scatter([1 for i in range(len(posts14.filter(lower(col('Caption')).contains('sale')).toPandas()))],posts14.filter(lower(col('Caption')).contains('sale')).select('Nbr_of_Likes').collect(), marker = '.')
axes[0,0].scatter(1,np.array(posts14.filter(lower(col('Caption')).contains('sale')).agg({'Nbr_of_Likes':'avg'}).select('avg(Nbr_of_Likes)').collect()),c = 'r', marker = 'x')
axes[0,0].set_xticklabels(["2014"])
axes[0,1].boxplot(likes_2015)
axes[0,1].scatter([1 for i in range(len(posts15.filter(lower(col('Caption')).contains('sale')).toPandas()))],posts15.filter(lower(col('Caption')).contains('sale')).select('Nbr_of_Likes').collect(), marker = '.')
axes[0,1].scatter(1,np.array(posts15.filter(lower(col('Caption')).contains('sale')).agg({'Nbr_of_Likes':'avg'}).select('avg(Nbr_of_Likes)').collect()),c = 'r', marker = 'x')
axes[0,1].set_xticklabels(["2015"])
axes[0,2].boxplot(likes_2016)
axes[0,2].scatter([1 for i in range(len(posts16.filter(lower(col('Caption')).contains('sale')).toPandas()))],posts16.filter(lower(col('Caption')).contains('sale')).select('Nbr_of_Likes').collect(), marker = '.')
axes[0,2].scatter(1,np.array(posts16.filter(lower(col('Caption')).contains('sale')).agg({'Nbr_of_Likes':'avg'}).select('avg(Nbr_of_Likes)').collect()),c = 'r', marker = 'x')
axes[0,2].set_xticklabels(["2016"])
axes[1,0].boxplot(likes_2017)
axes[1,0].scatter([1 for i in range(len(posts17.filter(lower(col('Caption')).contains('sale')).toPandas()))],posts17.filter(lower(col('Caption')).contains('sale')).select('Nbr_of_Likes').collect(), marker = '.')
axes[1,0].scatter(1,np.array(posts17.filter(lower(col('Caption')).contains('sale')).agg({'Nbr_of_Likes':'avg'}).select('avg(Nbr_of_Likes)').collect()),c = 'r', marker = 'x')
axes[1,0].set_xticklabels(["2017"])
axes[1,1].boxplot(likes_2018)
axes[1,1].scatter([1 for i in range(len(posts18.filter(lower(col('Caption')).contains('sale')).toPandas()))],posts18.filter(lower(col('Caption')).contains('sale')).select('Nbr_of_Likes').collect(), marker = '.')
axes[1,1].scatter(1,np.array(posts18.filter(lower(col('Caption')).contains('sale')).agg({'Nbr_of_Likes':'avg'}).select('avg(Nbr_of_Likes)').collect()),c = 'r', marker = 'x')
axes[1,1].set_xticklabels(["2018"])
axes[1,2].boxplot(likes_2019)
axes[1,2].scatter([1 for i in range(len(posts19.filter(lower(col('Caption')).contains('sale')).toPandas()))],posts19.filter(lower(col('Caption')).contains('sale')).select('Nbr_of_Likes').collect(), marker = '.')
axes[1,2].scatter(1,np.array(posts19.filter(lower(col('Caption')).contains('sale')).agg({'Nbr_of_Likes':'avg'}).select('avg(Nbr_of_Likes)').collect()),c = 'r', marker = 'x')
axes[1,2].set_xticklabels(["2019"])
display(fig)

# COMMAND ----------

# DBTITLE 1,Likes per weekday per year
weekDay = udf(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%w'))

posts13_wo = posts13_wo.withColumn('Year',year('Post_Date'))
posts13_wo = posts13_wo.withColumn('Day_str', F.date_format('Post_Date','yyyy-MM-dd'))
posts13_wo = posts13_wo.orderBy('Post_Date')
posts13_wo = posts13_wo.withColumn('weekday',weekDay('Day_str'))

posts14_wo = posts14_wo.withColumn('Year',year('Post_Date'))
posts14_wo = posts14_wo.withColumn('Day_str', F.date_format('Post_Date','yyyy-MM-dd'))
posts14_wo = posts14_wo.orderBy('Post_Date')
posts14_wo = posts14_wo.withColumn('weekday',weekDay('Day_str'))

posts15_wo = posts15_wo.withColumn('Year',year('Post_Date'))
posts15_wo = posts15_wo.withColumn('Day_str', F.date_format('Post_Date','yyyy-MM-dd'))
posts15_wo = posts15_wo.orderBy('Post_Date')
posts15_wo = posts15_wo.withColumn('weekday',weekDay('Day_str'))

posts16_wo = posts16_wo.withColumn('Year',year('Post_Date'))
posts16_wo = posts16_wo.withColumn('Day_str', F.date_format('Post_Date','yyyy-MM-dd'))
posts16_wo = posts16_wo.orderBy('Post_Date')
posts16_wo = posts16_wo.withColumn('weekday',weekDay('Day_str'))

posts17_wo = posts17_wo.withColumn('Year',year('Post_Date'))
posts17_wo = posts17_wo.withColumn('Day_str', F.date_format('Post_Date','yyyy-MM-dd'))
posts17_wo = posts17_wo.orderBy('Post_Date')
posts17_wo = posts17_wo.withColumn('weekday',weekDay('Day_str'))

posts18_wo = posts18_wo.withColumn('Year',year('Post_Date'))
posts18_wo = posts18_wo.withColumn('Day_str', F.date_format('Post_Date','yyyy-MM-dd'))
posts18_wo = posts18_wo.orderBy('Post_Date')
posts18_wo = posts18_wo.withColumn('weekday',weekDay('Day_str'))

posts19_wo = posts19_wo.withColumn('Year',year('Post_Date'))
posts19_wo = posts19_wo.withColumn('Day_str', F.date_format('Post_Date','yyyy-MM-dd'))
posts19_wo = posts19_wo.orderBy('Post_Date')
posts19_wo = posts19_wo.withColumn('weekday',weekDay('Day_str'))

# COMMAND ----------

fig, axes = plt.subplots(3,3,figsize = (14,10))
fig.suptitle('Distribution of likes per weekday per year')
weekdays = ['SUN','MON','TUE','WED','THU','FRI','SAT']
sns.boxplot(x = 'weekday',y = 'Nbr_of_Likes',data = posts13_wo.toPandas(), ax = axes[0,0])
axes[0,0].set_xticklabels(weekdays)
axes[0,0].set_ylabel('')
axes[0,0].set_xlabel('')
axes[0,0].set_title('')
sns.boxplot(x = 'weekday',y = 'Nbr_of_Likes',data = posts14_wo.toPandas(), ax = axes[0,1])
axes[0,1].set_xticklabels(weekdays)
axes[0,1].set_ylabel('')
axes[0,1].set_xlabel('')
axes[0,1].set_title('')
sns.boxplot(x = 'weekday',y = 'Nbr_of_Likes',data = posts15_wo.toPandas(), ax = axes[0,2])
axes[0,2].set_xticklabels(weekdays)
axes[0,2].set_ylabel('')
axes[0,2].set_xlabel('')
axes[0,2].set_title('')
sns.boxplot(x = 'weekday',y = 'Nbr_of_Likes',data = posts16_wo.toPandas(), ax = axes[1,0])
axes[1,0].set_xticklabels(weekdays)
axes[1,0].set_ylabel('')
axes[1,0].set_xlabel('')
axes[1,0].set_title('2016')
sns.boxplot(x = 'weekday',y = 'Nbr_of_Likes',data = posts17_wo.toPandas(), ax = axes[1,1])
axes[1,1].set_xticklabels(weekdays)
axes[1,1].set_ylabel('')
axes[1,1].set_xlabel('')
axes[1,1].set_title('2017')
sns.boxplot(x = 'weekday',y = 'Nbr_of_Likes',data = posts18_wo.toPandas(), ax = axes[1,2])
axes[1,2].set_xticklabels(weekdays)
axes[1,2].set_ylabel('')
axes[1,2].set_xlabel('')
axes[1,2].set_title('2018')
sns.boxplot(x = 'weekday',y = 'Nbr_of_Likes',data = posts14_wo.toPandas(), ax = axes[2,0])
axes[2,0].set_xticklabels(weekdays)
axes[2,0].set_ylabel('')
axes[2,0].set_xlabel('')
axes[2,0].set_title('2019')
axes[2,1].set_axis_off()
axes[2,2].set_axis_off()
display(fig)

# COMMAND ----------

posts_wo = posts13_wo.union(posts14_wo).union(posts15_wo).union(posts16_wo).union(posts17_wo).union(posts18_wo).union(posts19_wo)

posts_wo = posts_wo.orderBy('Post_Date')

# COMMAND ----------

fig, ax = plt.subplots()
ax = sns.boxplot(x = 'weekday',y = 'Nbr_of_Likes',data = posts_wo.toPandas())
ax.set_xticklabels(weekdays)
ax.set_ylabel('Likes')
ax.set_xlabel('')
ax.set_title('Distribution of likes per weekday')
display(fig)

# COMMAND ----------

# DBTITLE 1,Mentions tov likes en comments
PS = Pol_and_Subj()
TAM = TotaalAantalMentionsInComments()

pipelineModel = Pipeline(stages =[TAM]).fit(comments13) 
comments13_mentions = pipelineModel.transform(comments13)
pipelineModel = Pipeline(stages =[TAM]).fit(comments14) 
comments14_mentions = pipelineModel.transform(comments14)
pipelineModel = Pipeline(stages =[TAM]).fit(comments15) 
comments15_mentions = pipelineModel.transform(comments15)
pipelineModel = Pipeline(stages =[TAM]).fit(comments16) 
comments16_mentions = pipelineModel.transform(comments16)
pipelineModel = Pipeline(stages =[TAM]).fit(comments17) 
comments17_mentions = pipelineModel.transform(comments17)
pipelineModel = Pipeline(stages =[TAM]).fit(comments18.filter(comments18.Commenter_ID != 5951195697)) 
comments18_mentions = pipelineModel.transform(comments18.filter(comments18.Commenter_ID != 5951195697))
pipelineModel = Pipeline(stages =[TAM]).fit(comments19) 
comments19_mentions = pipelineModel.transform(comments19)

# COMMAND ----------

comments_mentions = comments13_mentions.union(comments14_mentions).union(comments15_mentions).union(comments16_mentions).union(comments17_mentions).union(comments18_mentions).union(comments19_mentions)

posts_with_mentions = all_posts.join(comments_mentions, on = 'Post_ID', how = 'left').fillna(0, subset=['TotalMentionsInComments'])

# COMMAND ----------

# slight negative relation between likes and mentions
posts_pd = posts_with_mentions.toPandas()
corr_lm = posts_pd['Nbr_of_Likes'].corr(posts_pd['TotalMentionsInComments'])
corr_cm = posts_pd['Nbr_of_Comments'].corr(posts_pd['TotalMentionsInComments'])

fig, axes = plt.subplots(1,2, figsize = (14,5))
sns.scatterplot(x = 'Nbr_of_Likes', y = 'TotalMentionsInComments',data = posts_with_mentions.toPandas(), ax = axes[0])
axes[0].set_title('Relation between likes and mentions, correlation: ' + str(np.round(corr_lm,4)))
axes[0].set_xlabel('Likes')
axes[0].set_ylabel('Mentions')
sns.scatterplot(x = 'Nbr_of_Comments', y = 'TotalMentionsInComments',data = posts_with_mentions.toPandas(), ax = axes[1])
axes[1].set_title('Relation between comments and mentions, correlation: ' + str(np.round(corr_cm,4)))
axes[1].set_xlabel('Comments')
axes[1].set_ylabel('Mentions')
display(fig)

# COMMAND ----------

# DBTITLE 1,Likes tov mentions in caption
CC = CleanCaption()

pipelineModel = Pipeline(stages =[CC]).fit(posts13_wo) 
posts13_mentions = pipelineModel.transform(posts13_wo)
pipelineModel = Pipeline(stages =[CC]).fit(posts14_wo) 
posts14_mentions = pipelineModel.transform(posts14_wo)
pipelineModel = Pipeline(stages =[CC]).fit(posts15_wo) 
posts15_mentions = pipelineModel.transform(posts15_wo)
pipelineModel = Pipeline(stages =[CC]).fit(posts16_wo) 
posts16_mentions = pipelineModel.transform(posts16_wo)
pipelineModel = Pipeline(stages =[CC]).fit(posts17_wo) 
posts17_mentions = pipelineModel.transform(posts17_wo)
pipelineModel = Pipeline(stages =[CC]).fit(posts18_wo) 
posts18_mentions = pipelineModel.transform(posts18_wo)
pipelineModel = Pipeline(stages =[CC]).fit(posts19_wo) 
posts19_mentions = pipelineModel.transform(posts19_wo)

posts_mentions = posts13_mentions.union(posts14_mentions).union(posts15_mentions).union(posts16_mentions).union(posts17_mentions).union(posts18_mentions).union(posts19_mentions)

# COMMAND ----------

posts_pd = posts_mentions.toPandas()
corr_lm = posts_pd['Nbr_of_Likes'].corr(posts_pd['Nbr_of_mentions'])
corr_lh = posts_pd['Nbr_of_Likes'].corr(posts_pd['Nbr_of_hashtags'])

fig, axes = plt.subplots(1,2, figsize = (14,5))
sns.scatterplot(x = 'Nbr_of_mentions', y = 'Nbr_of_Likes',data = posts_mentions.toPandas(), ax = axes[0])
axes[0].set_title('mentions in caption - likes, correlation: ' + str(np.round(corr_lm,4)))
axes[0].set_xlabel('Mentions')
axes[0].set_ylabel('Likes')
sns.scatterplot(x = 'Nbr_of_hashtags', y = 'Nbr_of_Likes',data = posts_mentions.toPandas(), ax = axes[1])
axes[1].set_title('hashtags in caption - likes, correlation: ' + str(np.round(corr_lh,4)))
axes[1].set_xlabel('Hashtags')
axes[1].set_ylabel('Likes')
display(fig)

# COMMAND ----------

# DBTITLE 1,Relation between length of the caption and likes
LC = lengte_caption()
CC = CleanCaption()

pipelineModel = Pipeline(stages =[CC,LC]).fit(posts13_wo) 
posts13_lengths = pipelineModel.transform(posts13_wo)
pipelineModel = Pipeline(stages =[CC,LC]).fit(posts14_wo) 
posts14_lengths = pipelineModel.transform(posts14_wo)
pipelineModel = Pipeline(stages =[CC,LC]).fit(posts15_wo) 
posts15_lengths = pipelineModel.transform(posts15_wo)
pipelineModel = Pipeline(stages =[CC,LC]).fit(posts16_wo) 
posts16_lengths = pipelineModel.transform(posts16_wo)
pipelineModel = Pipeline(stages =[CC,LC]).fit(posts17_wo) 
posts17_lengths = pipelineModel.transform(posts17_wo)
pipelineModel = Pipeline(stages =[CC,LC]).fit(posts18_wo) 
posts18_lengths = pipelineModel.transform(posts18_wo)
pipelineModel = Pipeline(stages =[CC,LC]).fit(posts19_wo) 
posts19_lengths = pipelineModel.transform(posts19_wo)

posts_lengths = posts13_lengths.union(posts14_lengths).union(posts15_lengths).union(posts16_lengths).union(posts17_lengths).union(posts18_lengths).union(posts19_lengths)

# COMMAND ----------

posts_pd = posts_lengths.toPandas()
corr_lw = posts_pd['Nbr_of_Likes'].corr(posts_pd['aantal_woorden'])

fig, ax = plt.subplots()
ax = sns.scatterplot(x = 'aantal_woorden', y = 'Nbr_of_Likes',data = posts_lengths.toPandas())
ax.set_title('length of caption - likes, correlation: ' + str(np.round(corr_lw,4)))
ax.set_xlabel('Number of words')
ax.set_ylabel('Likes')
display(fig)

# COMMAND ----------

# DBTITLE 1,Relation between number of pictures and likes
posts_pd = all_posts.toPandas()
corr_npl = posts_pd['Nbr_of_Likes'].corr(posts_pd['Nbr_of_Pics'])

fig, ax = plt.subplots()
sns.scatterplot('Nbr_of_Pics','Nbr_of_Likes',data = all_posts.toPandas())
ax.set_title('Number of pictures - likes, correlation: '+ str(np.round(corr_npl,4)))
ax.set_xlabel('Number of pictures')
ax.set_ylabel('Likes')
display(fig)

# COMMAND ----------

posts_pd = posts_wo.toPandas()
corr_npl = posts_pd['Nbr_of_Likes'].corr(posts_pd['Nbr_of_Pics'])

fig, ax = plt.subplots()
sns.scatterplot('Nbr_of_Pics','Nbr_of_Likes',data = posts_wo.toPandas())
ax.set_title('Number of pictures - likes, correlation: '+ str(np.round(corr_npl,4)))
ax.set_xlabel('Number of pictures')
ax.set_ylabel('Likes')
display(fig)

# COMMAND ----------

# DBTITLE 1,Difference between posts with multiple pictures and one picture
posts_pd_wo = posts_wo.toPandas()
corr_mpl_wo = posts_pd_wo['Nbr_of_Likes'].corr(posts_pd_wo['Multiple_Posts_int'])
posts_pd = all_posts.toPandas()
corr_mpl = posts_pd['Nbr_of_Likes'].corr(posts_pd['Multiple_Posts_int'])

fig, axes = plt.subplots(1,2,figsize = (14,6))
sns.boxplot('Multiple_Posts_int','Nbr_of_Likes',data = posts_wo.toPandas(),ax = axes[0])
axes[0].set_title('Multiple posts - likes, correlation: '+ str(np.round(corr_mpl_wo,4)))
axes[0].set_xlabel('Multiple post')
axes[0].set_ylabel('Likes')
axes[0].set_xticklabels(['False','True'])
sns.boxplot('Multiple_Posts_int','Nbr_of_Likes',data = all_posts.toPandas(), ax = axes[1])
axes[1].set_title('Multiple posts - likes, correlation: '+ str(np.round(corr_mpl,4)))
axes[1].set_xlabel('Multiple post')
axes[1].set_ylabel('Likes')
axes[1].set_xticklabels(['False','True'])
display(fig)

# COMMAND ----------

# DBTITLE 1,Difference between video post and picture post
posts_pd_wo = posts_wo.toPandas()
corr_vl_wo = posts_pd_wo['Nbr_of_Likes'].corr(posts_pd_wo['Is_video_int'])
posts_pd = all_posts.toPandas()
corr_vl = posts_pd['Nbr_of_Likes'].corr(posts_pd['Is_video_int'])

fig, axes = plt.subplots(1,2, figsize = (14,6))
sns.boxplot('Is_video_int','Nbr_of_Likes',data = posts_wo.toPandas(), ax = axes[0])
axes[0].set_title('Video - likes, correlation: '+ str(np.round(corr_vl_wo,4)))
axes[0].set_xlabel('Video')
axes[0].set_ylabel('Likes')
axes[0].set_xticklabels(['False','True'])
sns.boxplot('Is_video_int','Nbr_of_Likes',data = all_posts.toPandas(), ax = axes[1])
axes[1].set_title('Video - likes, correlation: '+ str(np.round(corr_vl,4)))
axes[1].set_xlabel('Video')
axes[1].set_ylabel('Likes')
axes[1].set_xticklabels(['False','True'])
display(fig)

# COMMAND ----------


