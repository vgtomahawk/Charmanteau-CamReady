# Charmanteau

Code and data for the paper

"Charmanteau: Character Embedding Models For Portmanteau Creation. EMNLP 2017. Varun Gangal*, Harsh Jhamtani*, Graham Neubig, Eduard Hovy, Eric Nyberg"

### Paper Abstract: 
Portmanteaus are a word formation phenomenon where two words are combined to form a new word. We propose character-level neural sequence-to-sequence (S2S) methods for the task of portmanteau generation that are end-to-end-trainable, language independent, and do not explicitly use additional phonetic information. We propose a noisy-channel-style model, which allows for the incorporation of unsupervised word lists, improving performance over a standard source-to-target model. This model is made possible by an exhaustive candidate generation strategy specifically enabled by the features of the portmanteau task. Experiments find our approach superior to a state-of-the-art FST-based baseline with respect to ground truth accuracy and human evaluation.

### Description
Code/ contains the code. Data/ contains the dataset.

To understand the Code, refer to Code/README_CODE.txt.

To understand the Data, refer to Data/README.txt.

You can also query our trained model on our online demo page: http://tinyurl.com/y9x6mvy

### Citation
If you use the code or data in this repository, please consider citing our work:
```
@inproceedings{gangal2017charmanteau,
  title={Charmanteau: Character embedding models for portmanteau creation},
  author={Gangal*, Varun and Jhamtani*, Harsh and Neubig, Graham and Hovy, Eduard and Nyberg, Eric},
  booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  pages={2907--2912},
  year={2017}
}
```
