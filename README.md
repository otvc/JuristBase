# JuristBase
This repository contain models which simplify works with text in legal field. 

At that time, because i not create huggingface dataset, all files with data are not contained in repository. But when they will in huggingface you can download they.

At the current time repository contain prepared data and learned model for 

|Model|About|Metrics|
|-----|-----|---|
|TFIDF Retriever + Linear classification score|This model contain base retriever architecture, and score model like solve of classifivation problem of matching people query and legal document | ROC AUC = 0.971|

----
## Repository structure
**[app](./app/)** contain information about api model. At that time work like a plug

**[notebooks](./notebooks/)** contain notebooks with datapreparings
and several classes for preprocessing data and classes with models, which uses to train and several instruments for training model
- **[JuristEngine](./notebooks/JuristEngine/)**: contain files with Models, Data preprocessing and several useful instruments for work
- **[Supportive](./notebooks/Supportive/)**: contain supportive functions for work with pandas or with nlp tasks
- **[Experiments](./notebooks/Experiments/)**: contain scripts for learning model and fix experiments with MLFlow