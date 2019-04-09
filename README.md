# NLP-Quant
The purpose of NLP-quant is to replicate the idea of predicting stock prices with Twitter data in China.

The project uses users' comments from http://guba.eastmoney.com, which is one of the most popular websites for individual investors to communicate their ideas.

The project consists of two parts, crawler part and text analysis part. Craweler part can be devided into 3 sub parts, which were responsile for crawling url, post and user information respectively. Text analysis part uses bag-of-word model and random forest to fit the sentiment label. Word bag and emotion labels of train dataset were done manually.

The word bag includes 200+ related words.
