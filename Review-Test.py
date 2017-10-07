from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("BUzDataFram").getOrCreate()

file = spark.read.json(r"C:\Users\kusha\Downloads\BigData\Python\Demo\Reviews_test.json")

regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
tokenized = regexTokenizer.transform(file)
tokenized = tokenized.drop('text')
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
stop_free = remover.transform(tokenized).drop('words')

doc_list = stop_free.rdd.map(lambda x: x.filtered).collect()

import gensim
from gensim import corpora

dictionary = corpora.Dictionary.load('restaurent.dict')

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_list]

lda = gensim.models.ldamodel.LdaModel.load('restaurent.lda')

topics = sorted(lda[doc_term_matrix],key=lambda x:x[1],reverse=True)

print(topics)
print(topics[0][0])