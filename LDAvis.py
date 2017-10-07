import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
from gensim import corpora


import pyLDAvis.gensim as gensimvis
import pyLDAvis


dictionary = corpora.Dictionary.load('restaurent.dict')

corpus = corpora.BleiCorpus('restaurent.lda-c')

lda = gensim.models.ldamodel.LdaModel.load('restaurent.lda')

if __name__ == "__main__":
    vis_data = gensimvis.prepare(lda, corpus, dictionary)
    pyLDAvis.show(vis_data)

    pyLDAvis.save_html(vis_data, 'vis.html')