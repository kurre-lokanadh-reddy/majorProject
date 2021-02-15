# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:31:44 2021

@author: lokanadh
"""

from __future__ import unicode_literals
from flask import Flask,render_template,url_for,request

from models_util import *
from spacy_summarization import text_summarizer
from gensim.summarization import summarize
from nltk_summarization import nltk_summarizer

import time
import spacy
nlp = spacy.load('en_core_web_sm')

# Web Scraping Pkg
from bs4 import BeautifulSoup
# from urllib.request import urlopen
from urllib.request import urlopen
from tensorflow.keras.models import load_model
loaded_emojifer = load_model("saved_models/model_emoji.h5",compile=False)
word_to_index = read_glove_vecs('data/words_corpus.txt')
# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Sumy 
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Reading Time
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text




app = Flask(__name__)
@app.route("/")
def tohome():
	return render_template("base.html",contentFile="home.html")

@app.route("/emojify",methods=["GET","POST"])
def emojify():
	if request.method=="POST":
		textChat = request.form["textChat"]
		sentences=np.array(list(textChat.split("."))) 
		sent_indi = sentences_to_indices(sentences , word_to_index) 
		preds = loaded_emojifer.predict(sent_indi)
		preds = np.argmax(preds,axis=1)
		res="" 
		for i,j in zip(sentences , preds):
			res = res + i+" --> "+label_to_emoji(j)+"\n"
		return render_template("base.html",contentFile="emojify.html",prediction=res)
	else:
		return render_template("base.html",contentFile="emojify.html")

@app.route("/docs")
def docs():
	return render_template("base.html",contentFile="docs.html")

@app.route("/exmp")
def exmp():
	return render_template("base.html",contentFile="exmp.html")

@app.route("/aboutus")
def aboutus():
	return render_template("base.html",contentFile="aboutus.html")

@app.route("/summerizer",methods=["GET","POST"])
def summerizer():
	start = time.time()
	rawtext = "EXAMPLE !!!--->"
	final_summary = "EXAMPLE !!!--->"
	final_time=1.00
	final_reading_time=1.30
	summary_reading_time=0.14
	if request.method == 'POST':
		rawtext = request.form.get('rawtext')
		if rawtext== None:
			rawtext = get_text(request.form.get('raw_url'))
		final_reading_time = readingTime(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('base.html',contentFile="summerizer.html",ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

'''@app.route('/summerizer',methods=['GET','POST'])
def summerizer_url():
	start = time.time()
	if request.method == 'POST':
		raw_url = request.form['raw_url']
		rawtext = get_text(raw_url)
		final_reading_time = readingTime(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('base.html',contentFile="summerizer.html",ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)
'''

@app.route('/comparer',methods=['GET','POST'])
def comparer():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary_spacy = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary_spacy)
		# Gensim Summarizer
		final_summary_gensim = summarize(rawtext)
		summary_reading_time_gensim = readingTime(final_summary_gensim)
		# NLTK
		final_summary_nltk = nltk_summarizer(rawtext)
		summary_reading_time_nltk = readingTime(final_summary_nltk)
		# Sumy
		final_summary_sumy = sumy_summary(rawtext)
		summary_reading_time_sumy = readingTime(final_summary_sumy) 

		end = time.time()
		final_time = end-start
		return render_template('base.html',contentFile="compare_summary.html",ctext=rawtext,final_summary_spacy=final_summary_spacy,final_summary_gensim=final_summary_gensim,final_summary_nltk=final_summary_nltk,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time,summary_reading_time_gensim=summary_reading_time_gensim,final_summary_sumy=final_summary_sumy,summary_reading_time_sumy=summary_reading_time_sumy,summary_reading_time_nltk=summary_reading_time_nltk)
	else:
		return render_template('base.html',contentFile="compare_summary.html")



if __name__=="__main__":
    app.run(debug=True)