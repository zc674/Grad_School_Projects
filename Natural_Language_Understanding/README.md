# COVIDBert: A Pre-trained Language Model for COVID-19 research papers

With the spread of COVID-19 and intensivelyaccumulated  research  papers,  it  is  important for  researchers  to  extract  useful  informationand users to efficiently obtain related answerswithin  the  rapidly  growing  documents.   The  current   natural   language   processing   (NLP) models like BERT (Devlin et al., 2018) have been  proved  as  an  effective  way  to  classifyand extract valuable information among hugetext  datasets.   Thus,  based  on  the  ALBERT model  (Lan  et  al.,  2019),  our  team  proposesa COVID-Albert model. We start from the Al-bert pre-trained weights and keep pre-trainedon  COVID-19  research  papers  datasets.   We prove the efficacy of our model by comparingthe  fine-tune  performance  on  a  manually  labeled covid-19 Question & Answering dataset with ALBERT and BioBERT

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Baseline model

We  use  Albert and  SQuad v2.0 dataset to get the baseline result. The baseline model can be run on

```
Question_Answering_with_ALBERT.ipynb
albert_fine_tune.txt
```

### pre-training COVIDBert

To process the COVID-19 corpora into the format suitable for pre-traing and a sample pretraining usage on Goole Cloud, we run the following notebook.

```
Data_processing.ipynb
```



## Acknowledgments

* Thanks for @techno246 and the codes our baseline model is based on.
* Thanks for @wonjininfo and the codes for cleaning BioASQ datasets which we use for finetuning.
