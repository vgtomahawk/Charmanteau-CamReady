# Charmanteau-CamReady

Code (and accompanying data) for the short paper (To appear at EMNLP '17)

Charmanteau: Character Embedding Models For Portmanteau Creation

Abstract: Portmanteaus are a word formation phenomenon where two words are combined to form a new word. We propose character-level neural sequence-to-sequence (S2S) methods for the task of portmanteau generation that are end-to-end-trainable, language independent, and do not explicitly use additional phonetic information. We propose a noisy-channel-style model, which allows for the incorporation of unsupervised word lists, improving performance over a standard source-to-target model. This model is made possible by an exhaustive candidate generation strategy specifically enabled by the features of the portmanteau task. Experiments find our approach superior to a state-of-the-art FST-based baseline with respect to ground truth accuracy and human evaluation.

BibTex: 
@article{gangal2017charmanteau,
  title={CharManteau: Character Embedding Models For Portmanteau Creation},
  author={Gangal, Varun and Jhamtani, Harsh and Neubig, Graham and Hovy, Eduard and Nyberg, Eric},
  journal={arXiv preprint arXiv:1707.01176},
  year={2017}
}
 

Code/ contains most of the code.
Data/ contains the dataset.

To understand the Code, refer to Code/README_CODE.txt
To understand the Data, refer to Data/README.txt

You can also query our trained model on our online demo page: http://tinyurl.com/y9x6mvy

If you use our Code, please consider citing our work (https://arxiv.org/abs/1707.01176) 
If you use our dataset, please consider citing:- 
	1. Our work (https://arxiv.org/abs/1707.01176) 
	2. The earlier work on portmanteaus by (Deri and Knight, 2015) (http://www.aclweb.org/anthology/N/N15/N15-1021.pdf)
