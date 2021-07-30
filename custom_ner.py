from __future__ import unicode_literals, print_function

import random
from pathlib import Path
import spacy
from tqdm import tqdm
print(spacy.__version__)

TRAIN_DATA = [

]
Fin_term=["IPO","capital","capital market","EPS","closing price","shareholders","Turnover","index","market capitalization","mutual fund","points","DSEX",
          "financial statements","reserve ","bonds","CSCX"]


Fin_term_txt=["The bank will raise money from the capital market to spend on the purchase of government securities and IPO costs.",
              "Shares will be allotted proportionally on Thursday at the Dhaka Stock Exchange Tower in the capital, said a press release.",
              "Therefore, South Bangla Bank will play a role in the overall development of the capital market, he added.",
              "According to the financial statement ending June 30, 2019, the company's net asset value (NAV) per share was Tk13.18, while earnings per share (EPS) were Tk0.94.",
              "During Sunday’s session, the closing price for each of Berger’s shares was Tk1,780.50.",
              "In 2020, the company offered a 295% cash dividend to its shareholders. The last time it offered more than a 375% dividend was back in 2017. The company offered a staggering 600% cash dividend that year.",
              "Turnover, another important indicator of the market, also rose to Tk1,355 core, which was 7.20% higher than the turnover of Tk1,264 crore in the previous session",
              "DSEX, the prime index of the DSE, went up by 19.17 points or 0.29% to settle at 6,424 — the highest since its inception in 2013",
              "The market capitalization of the DSE stood at Tk535,000 crore on Sunday, down from the all-time high of Tk535,200 crore of the previous session.",
              "The port city bourse traded 16.55 million shares and mutual fund units with a turnover value of Tk40.8 crore.",
              "The key index of Dhaka Stock Exchange, DSEX closed at 6,365.1 points on Sunday after gaining 0.92% during the session while CASPI advanced 0.33% to close at 18379.6 points.",
              "The Dhaka Stock Exchange currently has a market capitalization of Tk532,312 crore with the benchmark index, DSEX up by 17.83% since the beginning of this year.",
              "According to the financial statements of Oryza Agro for the period ended December 30, 2020, the earnings per share is Tk1.02 and the net asset value (NAV) per share without revaluation reserve is Tk18.09. ",
              "According to the company's financial statements for the period ended 31 December 2020, its earnings per share is Tk0.68 and the NAV per share without revaluation reserve is Tk14.08.",
              "The floor coupon rate of subordinated bonds is 7.5% and a ceiling of 10.5%, which is applicable to financial institutions, mutual funds, insurance companies, listed banks, cooperative banks, regional rural banks, organizations, trusts, autonomous corporations and other eligible investors.",
              "The port city bourse, Chittagong Stock Exchange, also ended higher with its CSE All Share Price Index (CASPI) gaining 103 points to settle at 18,673 and the Selective Categories Index (CSCX) advancing 69 points to finish at 11,216."]
label="Fin_term"
for i in range(len(Fin_term)):
  word=Fin_term[i]
  sentence=Fin_term_txt[i]
  start_index = sentence.find(word)
  end_index = start_index + len(word) # if the start_index is not -1
  print(start_index,end_index)
  f=(Fin_term_txt[i], {
        'entities': [(start_index,end_index,label)]
    })
  TRAIN_DATA.append(f)
print(TRAIN_DATA)


gvt_bank=["Sonali Bank Limited","Janata Bank","Agrani Bank Limited","Rupali Bank Limited","BASIC Bank Limited","Bangladesh Development Bank"]


gvt_bank_txt=["The bank has been converted to a Public Limited Company with 100% ownership of the government and started functioning as Sonali Bank Limited from November 15, 2007 taking over all assets",
              "Including 4 overseas branches in United Arab Emirates, Janata Bank runs its business with 904 branches across the country.",
              "There are total 921 branches of Agrani Bank Limited situated in 64 districts in Bangladesh.",
              "The state-run Rupali Bank Limited has introduced online transaction system in its 532 branches across the country. “We will introduce online banking system in the rest of 131 branches soon",
              "BASIC Bank Limited is a fully government owned bank in Bangladesh which was founded on 2 August 1988. The bank was founded to finance small enterprises.",
              "Bangladesh Development Bank was established on 16 November 2009 as a Public Limited Company by merging the state owned Bangladesh Shilpa Bank and Bangladesh"]
label="Bank"
for i in range(len(gvt_bank)):
  word=gvt_bank[i]
  sentence=gvt_bank_txt[i]
  start_index = sentence.find(word)
  end_index = start_index + len(word) # if the start_index is not -1
  print(start_index,end_index)
  f=(gvt_bank_txt[i], {
        'entities': [(start_index,end_index,label)]
    })
  TRAIN_DATA.append(f)
print(TRAIN_DATA)

model = None
#output_dir=Path("C:\\Users\\nithi\\Documents\\ner")
n_iter=100

if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')
    print("Created blank 'en' model")

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe('ner')

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            nlp.update(
                [text],
                [annotations],
                drop=0.5,
                sgd=optimizer,
                losses=losses)
        print(losses)
#nlp.tokenizer = None
nlp.to_disk("./en_example_pipeline")
nlp = spacy.load("en_example_pipeline")
