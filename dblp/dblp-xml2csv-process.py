#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 06:43:58 2021

@author: rafa
"""

import xml.sax
import pandas as pd
import pickle

class DBLP_Handler(xml.sax.ContentHandler ):
    def __init__(self):
        self.key = None
        self.path = list()
        self.content = dict()
        self.text = ''
        
    def startElement(self, tag, attributes):
        self.path.append(tag)
        if 'key' in attributes.keys():
            self.key = attributes.get('key')
        if len(self.path)==2 and len(self.content) % 1e5 == 0:
            print(f'Processed {len(self.content) / 1e6:.1f} million elements')
            
    def characters(self, content):
        self.text += content
        
    def endElement(self, tag):
        if len(self.path) == 3: # author, title, year level
            if tag == 'year':
                key = (self.path[1], self.key, 'YEAR')
                if key in self.content:
                    print('Duplicated year', key, self.content[key], self.text.strip())
                self.content[key] = int(self.text)
            elif tag == 'title':
                key = (self.path[1], self.key, 'TITLE')
                if key in self.content:
                    print('Duplicated title', key, self.content[key], self.text.strip())
                self.content[key] = self.text.strip().replace('\t', ' ')
            self.text = ''
            
        self.path.pop()
        
            
    def as_df(self):
        df = pd.DataFrame()
        df['TYPE'],  df['KEY'], df['TAG'] = zip(*self.content.keys())
        df['CONTENT'] = self.content.values()
        
        return df.pivot_table(index=['TYPE', 'KEY'], columns='TAG', 
                              values='CONTENT', aggfunc=pd.Series.unique)
       
    def to_csv(self, filename):
       with open(filename, 'w') as f:
           f.write('\t'.join(['TYPE', 'KEY', 'YEAR', 'TITLE']))
           f.write('\n')
           for (entry_type, key, tag), value in self.content.items():
               if tag == 'YEAR':
                   year = value
                   title = self.content[(entry_type, key, 'TITLE')].replace('\t', ' ')
                   f.write('\t'.join([entry_type, key, str(year), title]))
                   f.write('\n')
           
    def save(self):
        res = dict()
        for (entry_type, key, tag), value in self.content.items():
            res.setdefault(key, {'TYPE':entry_type})
            res[key][tag] = value            
        with open('/tmp/output.pkl', 'bw') as f:
            pickle.dump(res, f)
        df = pd.DataFrame(res.values(), index=res.keys())     
        df.to_csv('/tmp/output.csv.gz', sep='\t', compression='gzip')
        
parser = xml.sax.make_parser()
handler = DBLP_Handler()
parser.setContentHandler(handler)
parser.parse('/tmp/dblp.xml')
handler.save()

    