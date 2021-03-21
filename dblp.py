#!/usr/bin/env python
# coding: utf-8

# # UK Doctoral Thesis Metadata from EThOS
# 
# The data in this collection comprises the bibliographic metadata for all UK doctoral theses listed in EThOS, the UK's national thesis service.
# 
# 
# https://dblp.uni-trier.de/xml/

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle
import re
import os
from pathlib import Path
import requests
from collections import Counter
import matplotlib.pyplot as plt
from numpy import mean, ones
from scipy.sparse import csr_matrix
import io
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


# First, we retrieve the dataset directly from the British Library repository. We can also download the file in our system. The parameter name sets the fields that we will find in the CSV file.

# In[2]:


df = pd.read_csv('dblp/dblp.csv.gz', compression='gzip', header=0, sep='\t', quotechar='"', error_bad_lines=False)


# In[3]:


df


# Let's see the last year in the dataset

# In[4]:


df.YEAR.max()


# Let's see the different values of Type in the dataset.

# In[5]:


df.TYPE.unique()


# Let's filter the list of results typed as article

# In[6]:


articles = df[df['TYPE'] == "article"]


# In[ ]:


articles


# We can filter the list of results by year.

# In[ ]:


articles_2002 = articles[articles.YEAR == 2002]


# In[ ]:


articles_2002.head()


# In[ ]:


articles_2003 = articles[articles.YEAR == 2003]


# In[ ]:


articles_2003


# In[ ]:


articles_2010_2021 = articles[(articles.YEAR >= 2000) & (articles.YEAR <= 2010)]


# In[ ]:


articles_2010_2021.head()


# In[ ]:


class MPHash(object):
    # create from iterable 
    def __init__(self, terms):
        self.term = list(terms)
        self.code = {t:n for n, t in enumerate(self.term)}
    
    def __len__(self):
        return len(self.term)
    
    def get_code(self, term):
        return self.code.get(term)
    
    def get_term(self, code):
        return self.term[code]


# In[ ]:


# A sample is a collection of texts and publication dates 
# For each text, the sample stores its year and word counts. 
class Sample(object):
    pattern = pattern = r"(?:\w+[-])*\w*[^\W\d_]\w*(?:[-'’`]\w+)*"
    # Create Sample from data stored in a DataFrame with at least columns 
    # TEXT, YEAR
    # n = maximal ngram size 
    def __init__(self, data, ngram_length):
        self.size = len(data)
        self.year = data.YEAR.tolist()
        
        texts = tuple(data.TITLE)
        vectorizer = CountVectorizer(token_pattern = Sample.pattern, 
                                     stop_words=stopwords.words('english'),
                                     max_df=0.1,
                                     ngram_range=(1, ngram_length))
        matrix = vectorizer.fit_transform(texts).transpose() 
        # remove all hapax legomena to save space
        terms = vectorizer.get_feature_names()
        frequencies = matrix.sum(axis=1).A1
        selected = [m for m, f in enumerate(frequencies) if f > 1]
        hapax_rate = 1 - len(selected) / len(frequencies)
        print('Removing hapax legomena ({:.1f}%)'.format(100 * hapax_rate))
        self.matrix = matrix[selected, :]      
        self.term_codes = MPHash([terms[m] for m in selected])
        
        # store array with global term frequencies
        self.term_frequencies = self.matrix.sum(axis=1).A1
        # store doc frequencies
        self.doc_frequencies = self.matrix.getnnz(axis=1)
        # store most common capitalization of terms
        print('Obtaining most common capitalizations')
        vectorizer.lowercase = False
        matrix = vectorizer.fit_transform(texts).transpose()
        terms = vectorizer.get_feature_names()
        frequencies = matrix.sum(axis=1).A1    
        forms = dict()
        for t, f in zip(terms, frequencies):
            low = t.lower()
            if forms.get(low, (None, 0))[1] < f:
                forms[low] = (t, f)
        self.capitals = {k:v[0] for k, v in forms.items()}
        
        print('Computed stats for', len(self.term_codes), 'terms')
        
    # return the number of texts stored in this Sample
    def __len__(self):
        return self.size
    
    # return term frequency of the specified term
    def get_tf(self, term):
        code = self.term_codes.get_code(term.lower())
        
        return self.term_frequencies[code]
    
    # return document frequency of the specified term
    def get_df(self, term):
         code = self.term_codes.get_code(term.lower())
         
         return self.doc_frequencies[code]
     
    # return the most frequent capitalization form
    # (also for stopwords not in dictionary)
    def most_frequent_capitalization(self, term):
        return self.capitals.get(term.lower(), term)
    
    # return the average submission year of texts containing every term
    def average_year(self, period, tf_threshold=20, df_threshold=3):
        docs = [n for n, y in enumerate(self.year)                if period[0] <= y <= period[1]]
        tf_matrix = self.matrix[:, docs]
        tf_sum = tf_matrix.sum(axis=1).A1
        df_sum = tf_matrix.getnnz(axis=1)
        terms = [m for m, tf in enumerate(tf_sum)                 if tf >= tf_threshold and df_sum[m] >= df_threshold]
        tf_matrix = tf_matrix[terms, :]     
        rows, cols = tf_matrix.nonzero()
        df_matrix = csr_matrix((ones(len(rows)), (rows, cols)))
        year = [self.year[n] for n in docs]
        
        res = df_matrix @ year / df_matrix.getnnz(axis=1) # @ operator = matrix multiplication
        
        return {self.term_codes.get_term(terms[m]):res[m] for m in range(len(res))}

        
    # return the number of occurrences (doc frequency) for every term 
    def get_df_per_year(self, term):
        m = self.term_codes.get_code(term)
        row = self.matrix.getrow(m)
        _, docs = row.nonzero()
        c = Counter(map(self.year.__getitem__, docs))

        return c
          
    # return the number of occurrences (term frequency) for every term
    def tf_per_year(self, period=None):
        rows, cols = self.matrix.nonzero()
        res = {m:Counter() for m in rows}
        for m, n in zip(rows, cols):
            year = self.year[n]
            if period == None or period[0] <= year <= period[1]:
                res[m][year] += self.matrix[m, n]
            
        return res
    
    def plot_tf_series(self, term, period, relative=False):
        m = self.term_codes.get_code(term)
        if relative:
            norm = Counter(self.year)
        else:
            norm = Counter(set(self.year))
            
        if m:
            row = self.matrix.getrow(m)
            _, cols = row.nonzero()
            c = Counter()
            for n in cols:
                year = self.year[n]
                if period == None or period[0] <= year <= period[1]:
                    c[year] += row[0, n]
            
            X = sorted(c.keys())
            Y = [c[x] / norm[x] for x in X]
            plt.plot(X, Y, 'o-')
            plt.ylim(0, 1.2 * max(Y))
            plt.title(term)       
        else:
            raise ValueError('{} is not in store'.format(term))
             
    # return dictionary with a list of text-years per term 
    # period = pair of years (min _year, max_year) inclusive
    # keep_all = true if unlisted texts are not ignored
    def document_years(self, period=None, keep_all=True):
        rows, cols = self.matrix.nonzero()
        res = {m:list() for m in rows}
        for m, n in zip(rows, cols):
            if keep_all or self.listed[n]:
                year = self.year[n]
                print(year)
                if period == None or period[0] <= year <= period[1]:
                    res[m].append(year)
        
        return res
    
    # return dictionary with Counter of abstract-years per term
    def df_per_year(self, period=None, keep_all=True):
        doc_years = self.document_years(period, keep_all)
        
        return {m:Counter(v) for m, v in doc_years.items()}
    
    # create a plot with document frequency of terms
    def plot_df(self, terms, period, keep_all=True):
        dfs = self.df_per_year(period, keep_all)
        for term in terms:
            m = self.term_codes.get_code(term.lower())
            df = dfs[m] 
            X = range(*period)
            Y = [df.get(x, 0) for x in X]
            plt.clf()
            plt.plot(X, Y)
            plt.title(term)
            filename = 'plots/{}.png'.format(term)
            print('Saving', filename)
            plt.savefig(filename, dpi=200)
            
    # compute the average age in the specified period of documents containing 
    # each term with global term-frequency above tf_threshold
    # and annual document frequency above df_threshold (one year at least)
    # period = optional pair of years (min _year, max_year) inclusive
    def get_ages(self, period=None, 
                 tf_threshold=20, df_threshold=3, keep_all=True):
        res = dict()
        doc_years = self.document_years(period, keep_all)
        for m, values in doc_years.items():
            term = self.term_codes.get_term(m)
            if len(values) > 0:
                df = Counter(values).most_common(1)[0][1]
                tf = self.term_frequencies[m]
                #break;
                if df >= df_threshold and tf >= tf_threshold:       
                    res[term] = mean(values)
        return res
    
    # return abstract numbers containing any term in this set of terms
    def docs_with_term(self, terms, period=None):
        rows, cols = self.matrix.nonzero()
        res = set()
        for m, n in zip(rows, cols):
            term  = self.term_codes.get_term(m)
            if terms == None or term in terms:
                year = self.year[n]
                if period == None or period[0] <= year <= period[1]:
                     res.add(n)
                
        return res
            
       
    def search(self, term):
        m = self.term_codes.get_code(term)
        docs = self.matrix.getrow(m).nonzero()[1]
        
        return [(self.year[n], self.type[n], self.panel[n]) for n in docs]


# In[ ]:


data = articles_2010_2021


# In[ ]:


data = data[data.TITLE.str.len() > 40]   

print('Processing', len(data), 'texts')

s = Sample(data, 2)


# In[ ]:


with open('sample-dblp.pkl', 'wb') as f:
    pickle.dump(s, f)


# In[ ]:


with open('sample-dblp.pkl', 'rb') as f:
    s = pickle.load(f)
print('Loaded stats for', len(s), 'texts')


# In[ ]:


period = (2000, 2010)


# In[ ]:


ages = s.get_ages(period)


# In[ ]:


top = pd.DataFrame.from_dict(ages, orient='index').reset_index()
print(top)


# In[ ]:


top.columns = ['TERM', 'AGE']
#top = top.sort_values('AGE', ascending=False).head(250)   
top = top.sort_values('AGE', ascending=False).head(50)   
top['DOC FREQ'] = top.TERM.apply(s.get_df)
top['TERM FREQ'] = top.TERM.apply(s.get_tf)


# In[ ]:


# prepare to export
top['TERM'] = top.TERM.apply(s.most_frequent_capitalization)
print(top.set_index('TERM').head())


# In[ ]:


ts = pd.datetime.now().strftime("%Y-%m-%d_%H.%M")    
filename = 'output/vocabulary_{}.xlsx'.format(ts)
with pd.ExcelWriter(filename) as writer:
    top.set_index('TERM').to_excel(writer, sheet_name='terms')

print('vocabulary saved to', filename)


# ### References
# 
# https://doi.org/10.23636/1344
