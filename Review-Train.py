from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
# Importing Gensim
import gensim
from gensim import corpora


#Create SparkSession
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("ReviewLDA").getOrCreate()
#Read json file
file = spark.read.json(r"C:\Users\kusha\Downloads\BigData\Python\Demo\Reviews.json")

#Tokenize text by provided regex pattern. Given pattern remove punctuations and convert text into lower case.
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
tokenized = regexTokenizer.transform(file).drop('text')
#tokenized.show(2,False)

#Remove stop words from given text column
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
stop_free = remover.transform(tokenized).drop('words')
#stop_free.show(2,False)

#Create python list from dataframe column
doc_list = stop_free.rdd.map(lambda x: x.filtered).collect()
#print(doc_list)


# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_list)
print(dictionary)
#dictionary.save('restaurent.dict')

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_list]
print(doc_term_matrix)

#Create corpus iterator
#corpora.BleiCorpus.serialize('restaurent.lda-c', doc_term_matrix)
#Load the corpus iterator
corpus = corpora.BleiCorpus('restaurent.lda-c')

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(corpus, num_topics=20, id2word = dictionary, passes=50)
ldamodel.save('restaurent.lda')

print(ldamodel.print_topics(num_topics=20, num_words=10))



spark.stop()

