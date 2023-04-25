
# Import necessary libraries
import joblib
import re
import streamlit as st
import numpy as np
import pandas as pd
import pprint
import warnings
import tempfile
from io import StringIO
from PIL import Image
from rake_nltk import Rake
import spacy
import spacy_streamlit
from collections import Counter

from nltk.tokenize import sent_tokenize
import tensorflow as tf
import en_core_web_sm

from rouge import Rouge 
rouge = Rouge()

#custom imports
import text_summarize as ts
import text_classifier as tc
import text_analysis as nlp

# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)


# Sidebar options
option = st.sidebar.selectbox('Navigation', 
["Home",
	"Text Summarizer",
 "Text Classifier",
	"N-Gram Analysis",
	"Word Cloud",
	"Summarizer-Classifier"])



if option == 'Home':

	# Describing the Web Application 

	# Title of the application 
	#st.header("Researcher's Friendly Text Analytics Tool\n", )
	st.write(
			"""
				## Researcher's Friendly Text Analytics Tool
			"""
		)

	display = Image.open('display.jpg')
	display = np.array(display)
	st.image(display)
	#st.subheader("Shwetha, Sripradha, Supreetha, Tejaswini")
	#st.subheader("RNS Institute of Technology")
	st.write(
			"""
				## Project Description
				This is a text analytics webapp.
				The various text analytics methods like text summarizer, text classifier, N-gram analyser and a Word Cloud Generator are included.\n
				Text summarizer produces a tiny and crisp version of the corpus while preserving the important information which works based on textrank algorithm.\n
				Text classifier used to classify abstract sentences into the role they play (ie. Objective, Background, Methods, Results and Conclusion) to enable researchers to skim through the literature and dive deeper when necessary.\n
				N-gram analysis identifies all continuous adjacent sequences of words in a given sentence tokens. The N-gram plots can be used to analyse the relevance in the sequence of words.\n
				Word cloud generates the most frequent words in the texts and is a text visualization tool in which words are shown in varying sizes depending on how often they appear in the corpus. An additional feature is the image mask feature where the output can defined in a particular shape based on the image that is provided to the generator.\n
				These methods can be analysed on the corpus by selecting an option from the side bar.
			"""
		)

# Text Summarizer 
elif option == "Text Summarizer": 
	st.header("Text Summarization")
	#st.subheader("Enter the corpus that you want to summarize")
	#st.write("Text summarizer produces a shortened version of the corpus while preserving the important information. The total number of sentences in the shortened version can be chosen using the slider on the left side.")
	text_input = st.text_area("Enter the corpus that you want to summarize", height=200)
	ratio = st.sidebar.slider("Select number of sentences required in summary", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
	sentence_count = len(sent_tokenize(text_input))
	ratio=int(ratio*sentence_count)
	st.write("Number of sentences:", sentence_count)
	
	if st.button("Summarize"):
		out = ts.summarize(text_input,ratio)
		st.write("**Summary Output:**", out)
		st.write("Number of output sentences:", len(sent_tokenize(out)))
		st.write("Rouge Score:",rouge.get_scores(text_input, out))

# Text Classify 
elif option == "Text Classifier": 
	st.header("Text Classification")
	#st.subheader("Enter an unstructured abstract that you want to classify")
	#st.write("Text classifier used to classify abstract sentences into the role they play (ie. Objective, Background, Methods, Results and Conclusion) to enable researchers to skim through the literature and dive deeper when necessary.")
	text_input = st.text_area("Enter an unstructured abstract that you want to classify", height=200)
	if st.button("Classify"):
		out = tc.classify(text_input)
		# st.write(out)

		st.write("**Classified Abstract:**")
		s=""
		for element in out:
			if element=='.':
				st.write(s)
				s=""
			else:
				s=s+element


# N-Gram Analysis Option 
elif option == "N-Gram Analysis":
	
	st.header("N-Gram Analysis")
	#st.subheader("This feature displays the most commonly occuring N-Grams in the entered text")
	#st.write("N-gram is a contiguous sequence of n items from a given sample of text or speech. The items can be phonemes, syllables, letters, words or base pairs according to the application. The n-grams typically are collected from a text or speech corpus. This tool identifies the most commonly occurring n-grams. ")
	# Ask for text or text file
	#st.subheader('Enter text below')
	text = st.text_area('Enter a paragraph', height=200)

	# Parameters
	n = st.sidebar.slider("N for the N-gram", min_value=1, max_value=8, step=1, value=2)
	topk = st.sidebar.slider("Top k most common phrases", min_value=10, max_value=50, step=5, value=10)

	# Add a button 
	if st.button("Generate N-Gram Plot"): 
		# Plot the ngrams
		nlp.plot_ngrams(text, n=n, topk=topk)
		st.pyplot()


# Word Cloud Feature
elif option == "Word Cloud":

	st.header("Word Cloud")
	#st.subheader("This feature generates a word cloud that contains the most frequent words in the entered text")
	#st.write("A word cloud is a data visualization method that forms an integral aspect of text analytics. It is a text representation in which words are shown in varying sizes depending on how often they appear in the corpus. The words with higher frequency are given a bigger font and stand out from the words with lower frequencies. An additional feature is the image mask feature where the output can defined in a particular shape based on the image that is provided to the generator.")
	# Ask for text or text file
	#st.header('Enter text')
	text = st.text_area('Enter a paragraph', height=200)

	# Upload mask image 
	mask = st.file_uploader('Use Image Mask', type = ['jpg'])
	
	if st.button("Generate Wordcloud"): 
		st.write(len(text))
		nlp.create_wordcloud(text, mask)
		st.pyplot()

elif option == "Summarizer-Classifier":
	st.header("Summarizer-Classifier")
	#st.write("This is an integration of text summarizer and classifier. The text summarizer first produces a shortened version of the given corpus. The number of sentences in the shortened version can be chosen as per requirements. The output of the summarizer is fed to the classifier which classifies the sentences in the summarized text into the role they play (ie. Objective, Background, Methods, Results and Conclusion)")
	#st.subheader("Enter the corpus that you want to summarize")
	text_input = st.text_area("Enter a corpus that you want to summarize and classify", height=200)
	ratio = st.sidebar.slider("Select number of sentences required in summary", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
	sentence_count = len(sent_tokenize(text_input))
	ratio=int(ratio*sentence_count)
	st.write("Number of sentences:", sentence_count)
	
	if st.button("Summarize and Classify"):
		int_out = ts.summarize(text_input,ratio)
		fin_out=tc.classify(int_out)
		st.write("**Summary Output:**", int_out)
		st.write("Number of output sentences:", len(sent_tokenize(int_out)))

		st.write("**Classified Abstract:**")
		s=""
		for element in fin_out:
			if element=='.':
				st.write(s)
				s=""
			else:
				s=s+element
