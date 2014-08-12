#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, codecs
import optparse
import numpy
import ldig
import sqlite3

#sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

parser = optparse.OptionParser()
parser.add_option("-m", dest="model", help="model directory")
parser.add_option("-d", dest="database", help="database file")
(options, args) = parser.parse_args()
if not options.model:
    parser.error("need model directory (-m)")
if not options.database:
    parser.error("need database file (-d)")


class Detector(object):
    def __init__(self, modeldir):
        self.ldig = ldig.ldig(modeldir)
        self.features = self.ldig.load_features()
        self.trie = self.ldig.load_da()
        self.labels = self.ldig.load_labels()
        self.param = numpy.load(self.ldig.param)

    def detect(self, st):
        label, text, org_text = ldig.normalize_text(st)
        events = self.trie.extract_features(u"\u0001" + text + u"\u0001")
        sum = numpy.zeros(len(self.labels))

        for id in sorted(events, key=lambda id:self.features[id][0]):
            phi = self.param[id,]
            sum += phi * events[id]
        exp_w = numpy.exp(sum - sum.max())
        prob = exp_w / exp_w.sum()
        return dict(zip(self.labels, ["%0.3f" % x for x in prob]))


detector = Detector(options.model)

con = sqlite3.connect(options.database)
con.row_factory = sqlite3.Row
cur = con.cursor()

try:
    cur.execute('CREATE TABLE lang (id integer, lang string, PRIMARY KEY(id))')
except:
    pass

count = 0

cur.execute('SELECT id, text FROM tweets')
rows = cur.fetchall()

for row in rows:
    probabilities = detector.detect(row['text'])
    best_prob = 0.0
    best_lang = None
    for lang in probabilities.keys():
        prob = float(probabilities[lang])
        if prob > best_prob:
            best_prob = prob
            best_lang = lang

    try:
        cur.execute('INSERT INTO lang VALUES (?, ?)', (row['id'], best_lang))
    except:
        pass
    
    count += 1
    if count % 1000 == 0:
        con.commit()
        print 'Processed {} tweets.'.format(count)

con.commit()

