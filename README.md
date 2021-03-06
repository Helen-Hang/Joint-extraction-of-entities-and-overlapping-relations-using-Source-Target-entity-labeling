# Requirements and Installation
First, install the requirements with the following command:
```
pip install -r requirements.txt
```

# Preprocessing data
- Download pre__processed_data from

>  https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3

>  https://drive.google.com/open?id=1zISxYa-8ROe2Zv8iRc82jY9QsQrfY1Vj

into raw_data\nyt1\input and raw_data\webnlg1\input

- Our model expects the input dataset to be in word format. To convert a dataset run the following command:
```
python id2Text.py
```
The generated files are stored into raw_data\nyt1\output and raw_data\webnlg1\output.

- all the data needed to run the model is stored into raw_data\nyt and raw_data\webnlg

# Run
```
python main.py -c nyt.json -t 0 -cell lstm -g 0 -e n1 && python main.py -c nyt.json -t 1 -cell lstm -g 0 -e n1 -eve 2
python main.py -c webnlg.json -t 0 -cell lstm -g 0 -e w1 && python main.py -c webnlg.json -t 1 -cell lstm -g 0 -e w1 -eve 2
```

```
python main.py --gpu 0 --mode train --cell lstm --decoder_type multi --dataset nyt --experiment exp00 && python main.py --gpu 0 --mode test --cell lstm --decoder_type multi --dataset nyt --experiment exp00
python main.py --gpu 0 --mode train --cell lstm --decoder_type one --dataset webnlg --experiment exp00 && python main.py --gpu 0 --mode test --cell lstm --decoder_type one --dataset webnlg --experiment exp00
```

```
- NLL：
python main.py -a train -d nyt -l nll -m separatew -b 100 -tn 5 -lr 0.001 -en 50 -sf 2 -hn 1000 -n common -g 2 -cell lstm -sobm 1
- RL：
python main.py -a train -d nyt -l rl -m separatew -b 100 -tn 5 -lr 0.0005 -en 50 -sf 5 -hn 1000 -n 01 -g 0 -cell lstm -re 24 -rip 'nyt-SEPARATE_W-NLL-5-0.001-100-FixedSortedAlphabet-lstm-1000-1000'
python main.py -a valid -d nyt -l rl -m separatew -b 100 -tn 5 -lr 0.0005 -en 50 -sf 5 -hn 1000 -n 01 -g 0 -cell lstm -re 24
python main.py -a test -d nyt -l rl -m separatew -b 100 -tn 5 -lr 0.0005 -en 50 -sf 5 -hn 1000 -n 01 -g 0 -cell lstm -re 24 -eve 15
```

```
- NLL：
python main.py -a train -d webnlg -l nll -m separatew -b 100 -tn 5 -lr 0.001 -en 50 -sf 2 -hn 1000 -n common -g 1 -cell lstm -sobm 3
- RL：
python main.py -a train -d webnlg -l rl -m separatew -b 100 -tn 5 -lr 0.0005 -en 50 -sf 5 -hn 1000 -n 01 -g 0 -cell lstm -re 44 -rip 'webnlg-SEPARATE_W-NLL-5-0.001-100-FixedSortedFreq-lstm-1000-1000'
python main.py -a valid -d webnlg -l rl -m separatew -b 100 -tn 5 -lr 0.0005 -en 50 -sf 5 -hn 1000 -n 01 -g 0 -cell lstm -re 44
python main.py -a test -d webnlg -l rl -m separatew -b 100 -tn 5 -lr 0.0005 -en 50 -sf 5 -hn 1000 -n 01 -g 0 -cell lstm -re 44 -eve 10
```

# Result
Our paper results can be found here:

>  https://drive.google.com/drive/folders/16TkI-sYUPdj4Id-nh8OeK5YwOlD28yMw?usp=sharing 
>  https://drive.google.com/drive/folders/1h28nVy3Y96J2cfDdJho4TBNhK5MHT1Vy?usp=sharing

# Evaluation
- Table 1 result can be obtained by runing
```
python freq_static(num).py
```

- Table 4,5,7,9 results and fig9 result can be obtained by runing
```
python  analyze.py
``` 

- Training data of different proportions in Table 6 can be obtained by runing
 ```
python  split_data.py
``` 

 - 10 runs results of Table 3 and fig7 result can be obtained by runing
```
python  Boxplot.py
``` 

- fig 8 results can be obtained by runing
```
python  zhexian.py
``` 

- fig10,fig11,fig12 results can be obtained by runing
```
python  freq_static(word).py
``` 
then can be drawn using Origin 2018





 

