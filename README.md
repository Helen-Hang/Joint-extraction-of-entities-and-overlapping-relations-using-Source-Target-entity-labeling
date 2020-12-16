# Requirements and Installation
First, install the requirements with the following command:
```
pip install -r requirements.txt
```

# Preprocessing data
Download pre__processed_data from 
https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3 and https://drive.google.com/open?id=1zISxYa-8ROe2Zv8iRc82jY9QsQrfY1Vj 
into raw_data\nyt1\input and raw_data\webnlg1\input

Our model expects the input dataset to be in word format. To convert a dataset run the following command:
```
python3 id2Text.py
```
The generated files are stored into raw_data\nyt1\output and raw_data\webnlg1\output. 
Finally, all the data needed to run the model is stored into raw_data\nyt and raw_data\webnlg

# Result
Our paper results can be found here:
https://drive.google.com/drive/folders/16TkI-sYUPdj4Id-nh8OeK5YwOlD28yMw?usp=sharing and https://drive.google.com/drive/folders/1h28nVy3Y96J2cfDdJho4TBNhK5MHT1Vy?usp=sharing

# Evaluation
Table 1 result can run by 
```
python3 freq_static(num).py
```

fig7,fig8,fig9 results can run by
```
python3  freq_static(word).py
``` 

10 runs results of Table 3 and fig10 result can run by 
```
python3  Boxplot.py
``` 

Table 5,6,8,9 results and fig12 result can run by
```
python3  analyze.py
``` 

fig 11 results can run by
```
python3  zhexian.py
``` 
 
Training data of different proportions in Table 7 can run by
 ```
python3  split_data.py
``` 
 
