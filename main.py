import streamlit as st
import nltk
import spacy
from spacy import displacy
from polarity_pred import sentiment_scores
nltk.download('punkt')
nlp = spacy.load("en_example_pipeline")
st.title("INPUT NEWS")
h1=st.text_area('Input News:',height=20)
doc=nlp(h1)
def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)
colors = {'FIN_TERM': '#FF6A6A', 'GVT_BANK': '#6AFFCD'}

options = {'ent': ['Fin_term','gvt_bank'], 'colors':colors}
html = displacy.render(doc, style="ent",options=options)

style = "<style>mark.entity { display: inline-block }</style>"
st.title("Named entity recognation ")
st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)
a_list = nltk.tokenize.sent_tokenize(h1)
enity=[(ent.text, ent.label_) for ent in doc.ents]
importtant_sen=[]
polarity=[]
for i in range(len(enity)):
  for j in range(len(a_list)):
    if enity[i][0] in a_list[j]:
        if a_list[j] in importtant_sen:
            None
        else:
            importtant_sen.append((a_list[j]))
        polarity.append(sentiment_scores(a_list[j]))
st.title("Only important sentences Filtered by NER ")
st.write(importtant_sen,polarity)
print(len(a_list))

def final_result():
    i=1
    neg=0
    pos=0
    neu=0
    for i in range(1,len(polarity)):
        h=polarity[i]
        neg=neg+h['neg']
        pos=pos+h['pos']
        neu=neu+h['neu']

    return ("Overall negative",neg/i,"Overall positive",pos/i,"Overall neutral",neu/i)
print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
st.write(final_result())
