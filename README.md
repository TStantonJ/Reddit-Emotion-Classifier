# DATS_6312_NLP

### Team X(?): Jonathan Schild, Thomas Stanton, Ashish Telukunta

#

## Topic Ideas:

### multiclass emotion classification:

#### https://huggingface.co/datasets/AdamCodd/emotion-balanced

#

#

### fakenews/misinformation:

#### https://www.kaggle.com/datasets/sumanthvrao/fakenewsdataset

#### https://huggingface.co/datasets/fake_news_english

#### https://huggingface.co/datasets/roupenminassian/twitter-misinformation

### Automated Resume Ranker:

#### https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

#### Building an automated resume ranking system based on job descriptions for job screening

### Idea but no dataset: Job Status Classifier

#### Scrape emails and categorize them into recruiter/non-recruiter emails then further classify the recruiter/company emails into Job application received / Pre-Screening / Technical interview / HR Interview / Offer / Reject then use the information to make a status progress bar.

#### Model is trained on approximately 90,000 tweets to a .95 validation accuracy and .95 F1 score. This model is then applied to a separate dataset of unlabeled comments. The results are filtered to only retain the reddit comments that contained an emotion with above a .6 "confidence" score. These comments and their new labels are then combined with out original twitter dataset, and used to train a new model. This second model achieves a validation accuracy of .91 and F1 of .91. The apply file loads the model and applies it to some input'ed text.
