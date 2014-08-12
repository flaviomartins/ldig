#!/usr/bin/env python
# -*- coding: utf-8 -*-
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
import sys, os, codecs
import urlparse
import optparse
import json
import numpy
import ldig
import sqlite3
from pprint import pprint

#sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

parser = optparse.OptionParser()
parser.add_option("-m", dest="model", help="model directory")
parser.add_option("-d", dest="database", help="database file")
parser.add_option("-p", dest="port", help="listening port number", type="int", default=48000)
(options, args) = parser.parse_args()
if not options.model:
    parser.error("need model directory (-m)")
if not options.database:
    parser.error("need database file (-d)")


class Detector(object):
    def __init__(self, modeldir, database):
        self.ldig = ldig.ldig(modeldir)
        self.features = self.ldig.load_features()
        self.trie = self.ldig.load_da()
        self.labels = self.ldig.load_labels()
        self.param = numpy.load(self.ldig.param)
        self.cache = {}
        # sqlite3 backed storage
        self.con = sqlite3.connect(database)
        self.con.row_factory = sqlite3.Row
        # self.con.isolation_level = None
        # load cache
        cur = self.con.cursor()
        try:
            cur.execute('CREATE TABLE lang (id integer, lang string, PRIMARY KEY(id))')
        except:
            pass

        for row in cur.execute('SELECT id, lang FROM lang'):
            self.cache[row['id']] = row['lang']

        print 'Cache loaded with {} language annotations.'.format(len(self.cache))

    def get_probabilities(self, st):
        label, text, org_text = ldig.normalize_text(st)
        events = self.trie.extract_features(u"\u0001" + text + u"\u0001")
        sum = numpy.zeros(len(self.labels))

        for id in sorted(events, key=lambda id:self.features[id][0]):
            phi = self.param[id,]
            sum += phi * events[id]
        exp_w = numpy.exp(sum - sum.max())
        prob = exp_w / exp_w.sum()
        return dict(zip(self.labels, ["%0.3f" % x for x in prob]))

    def detect(self, id, st):
        if id in self.cache:
            return self.cache[id]
        else:
            probabilities = self.get_probabilities(st)
            best_prob = 0.0
            best_lang = None
            for lang in probabilities.keys():
                prob = float(probabilities[lang])
                if prob > best_prob:
                    best_prob = prob
                    best_lang = lang

            self.cache[id] = best_lang
            # save to database
            cur = self.con.cursor()
            try:
                cur.execute('INSERT INTO lang VALUES (?, ?)', (id, best_lang))
                if len(self.cache) % 1000 == 0:
                    self.con.commit()
                    print 'Cache has {} language annotations.'.format(len(self.cache))
            except:
                pass
            return best_lang


detector = Detector(options.model, options.database)


class LdigTrecServerHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        url = urlparse.urlparse(self.path)
        path = url.path
        if path == "/detect":
            params = urlparse.parse_qs(url.query)
            id = long(unicode(params['id'][0], 'utf-8'))
            text = unicode(params['text'][0], 'utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            detected = detector.detect(id, text)
            json.dump({'detected': detected}, self.wfile)
        else:
            self.send_response(404, "Not Found : " + url.path)
            self.send_header("Expires", "Fri, 31 Dec 2100 00:00:00 GMT")
            self.end_headers()


server = HTTPServer(('', options.port), LdigTrecServerHandler)
print "ready."
server.serve_forever()