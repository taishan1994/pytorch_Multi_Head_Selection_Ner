# pytorch_Multi_Head_Selection_Ner
延申：
- 一种级联Bert用于命名实体识别，解决标签过多问题：https://github.com/taishan1994/pytorch_Cascade_Bert_Ner
- 中文命名实体识别最新进展：https://github.com/taishan1994/awesome-chinese-ner
- 信息抽取三剑客：实体抽取、关系抽取、事件抽取：https://github.com/taishan1994/chinese_information_extraction
- 一种基于机器阅读理解的命名实体识别：https://github.com/taishan1994/BERT_MRC_NER_chinese
- W2NER：命名实体识别最新sota：https://github.com/taishan1994/W2NER_predict
****
基于pytorch的多头选择方法进行中文命名实体识别。

在多头选择的基础上，设计了一个额外的任务：判断是否存在某类型的实体，即一个多标签分类的任务。目前可运行的是cner数据，其它数据可参照模板进行修改，具体步骤如下：

- 1、在raw_data下是原始数据，通过process.py获得mid_data下的数据。
- 2、通过preprocess_mhs.py获得final_data下的数据。
- 3、运行main.py进行训练、验证、测试和预测。

需要注意的是要保持每一步骤中各参数的一致性。数据和模型下载地址：链接：https://pan.baidu.com/s/1vpoDVZ875E27aLHjXK7yQA?pwd=4mcs  提取码：4mcs

# 依赖

```
pytorch==1.6.0
tensorboasX
seqeval
pytorch-crf==0.7.2
transformers==4.4.0
```

# 运行

```python
!python main.py \
--bert_dir="model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/cner/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--pair_tags=9 \
--entity_tags=8 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=150 \
--lr=3e-5 \
--crf_lr=3e-2 \
--other_lr=3e-4 \
--train_batch_size=32 \
--train_epochs=3 \
--eval_batch_size=16 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--lstm_hidden=128 \
--num_layers=1 \
--use_lstm='True' \
--dropout_prob=0.3 \
--dropout=0.3 \

```

### 结果

```python
[eval] loss:0.0000 precision=0.9410 recall=0.8931 f1_score=0.9164
Saving model checkpoint to ./checkpoints/bert_bilstm
Load ckpt from ./checkpoints/bert_bilstm/model.pt
Use single gpu in: ['0']
              precision    recall  f1-score   support

        CONT       0.91      0.91      0.91        33
         EDU       0.92      0.87      0.90       109
         LOC       0.00      0.00      0.00         2
        NAME       0.97      0.96      0.97       110
         ORG       0.93      0.91      0.92       535
         PRO       0.89      0.84      0.86        19
        RACE       0.67      0.80      0.73        15
       TITLE       0.95      0.88      0.91       748

   micro avg       0.94      0.89      0.92      1571
   macro avg       0.78      0.77      0.77      1571
weighted avg       0.94      0.89      0.91      1571

虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。
Load ckpt from ./checkpoints/bert_bilstm/model.pt
Use single gpu in: ['0']
{'NAME': [['虞兔良', 0, 3]], 'CONT': [['汉族，中国国籍', 17, 24]], 'TITLE': [['中共党员', 40, 44], ['经济师', 49, 52]], 'EDU': [['MBA', 45, 48]]}
```

# 补充

多头选择还是存在一定的问题，比如：

```python
吴重阳 NAME 0 3
大学本科 EDU 9 13
教授级高工 TITLE 14 19
教授级高工，享受国务院特殊津贴，历任邮电部侯马电缆厂仪表试制组长 TITLE 14 46
邮电部侯马电缆厂 ORG 32 40
邮电部侯马电缆厂仪表试制组长、光缆分厂 ORG 32 51
仪表试制组长 TITLE 40 46
光缆分厂 ORG 47 51
副厂长 TITLE 51 54
厂长 TITLE 52 54
研究所 ORG 55 58
副所长 TITLE 58 61
```

会提取到相对于较长的实体，这种解决方法是引入相对位置编码，如苏剑林的旋转位置编码。这里初步的解决方法是：对于头部重合的，过滤掉较长的实体；对于尾部重合的，过滤掉较短的实体。

**延申**：这里是得到一个[batchsize, seq_len, seq_len, pair_tags]对每一个字符对直接进行判断是哪一种实体，还有一种思路是得到一个[batchsize, num_tags,seq_len,seq_len]，这一种是对每一个字符对进行二分类，而标签是从第二位获取，可参考苏剑林的GlobalPoint进行命名实体识别。
