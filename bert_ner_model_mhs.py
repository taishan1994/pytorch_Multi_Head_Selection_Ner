import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_base_model import BaseModel
from torchcrf import CRF
import config


class BertNerModel(BaseModel):
    def __init__(self,
                 args,
                 **kwargs):
        super(BertNerModel, self).__init__(bert_dir=args.bert_dir, dropout_prob=args.dropout_prob)
        self.args = args
        self.num_layers = args.num_layers
        self.lstm_hidden = args.lstm_hidden
        gpu_ids = args.gpu_ids.split(',')
        device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.device = device

        out_dims = self.bert_config.hidden_size

        if args.use_lstm == 'True':
            self.lstm = nn.LSTM(out_dims, args.lstm_hidden, args.num_layers, bidirectional=True, batch_first=True, dropout=args.dropout)
            self.pair_linear = nn.Linear(args.lstm_hidden * 2 * 2, args.pair_tags)  # 字符对分类， 第一个乘以2是双向lstm，第二个乘以2是拼接
            self.entity_linear = nn.Linear(self.bert_config.hidden_size, args.entity_tags)  # 多标签分类
            self.pair_criterion = nn.CrossEntropyLoss()
            self.entity_criterion = nn.BCEWithLogitsLoss()
            init_blocks = [self.pair_linear, self.entity_linear]
            # init_blocks = [self.classifier]
            self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)
        else:
            mid_linear_dims = kwargs.pop('mid_linear_dims', 256)
            self.mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.ReLU(),
                nn.Dropout(args.dropout))
            #
            out_dims = mid_linear_dims

            # self.dropout = nn.Dropout(dropout_prob)
            # self.classifier = nn.Linear(out_dims, args.num_tags)
            self.pair_linear = nn.Linear(args.lstm_hidden * 2 * 2, args.pair_tags)  # 字符对分类
            self.entity_linear = nn.Linear(self.bert_config.hidden_size, args.entity_tags)  # 多标签分类
            self.pair_criterion = nn.CrossEntropyLoss()
            self.entity_criterion = nn.BCEWithLogitsLoss()


            init_blocks = [self.mid_linear, self.entity_linear, self.pair_linear]
            # init_blocks = [self.classifier]
            self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)



    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        c0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        return h0, c0

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                pair_labels,
                entity_labels):
        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        # 常规
        seq_out = bert_outputs[0]  # [batchsize, max_len, 768]
        seq_out_pool = bert_outputs[1] # [batchsize, 768]
        seq_len = seq_out.size(1)
        batch_size = seq_out.size(0)

        if self.args.use_lstm == 'True':
            hidden = self.init_hidden(batch_size)
            seq_out, (hn, _) = self.lstm(seq_out, hidden)
        else:
            seq_out = self.mid_linear(seq_out)  # [batchsize, max_len, 256]


        # 这里是多头选择模型的核心部分
        x1 = torch.unsqueeze(seq_out, 1)
        x2 = torch.unsqueeze(seq_out, 2)
        x1 = x1.repeat(1, seq_len, 1, 1)
        x2 = x2.repeat(1, 1, seq_len, 1)
        span_matrix = torch.cat([x2, x1], dim=-1)  # [32,150,150,256*2]

        span_matrix = span_matrix.contiguous().view(-1, span_matrix.size(3))  # [32*150*150,256*2]
        logits = self.pair_linear(span_matrix)  # [32*150*150, args.pair_tags]
        return_logits = logits
        return_logits = return_logits.view(batch_size, seq_len, seq_len, self.args.pair_tags)  # [32,150,150,args.pair_tags]
        if pair_labels is None:
            return return_logits
        # 对pad部分不进行计算
        leng = torch.sum(attention_masks, -1)
        mask = torch.tensor([(F.pad(torch.triu(torch.ones(le, le), diagonal=0), pad=(0,seq_len-le,0,seq_len-le), mode="constant", value=0) == 1).numpy().tolist() \
                  for le in leng.data], requires_grad=False)
        mask = mask.view(-1).to(logits.device)

        logits = logits.view(-1, self.args.pair_tags)
        span_logits = logits[mask]
        span_labels = pair_labels.view(-1)[mask]
        # 实体类型部分
        entity_logits = self.entity_linear(seq_out_pool)

        pair_loss = self.pair_criterion(span_logits, span_labels)
        entity_loss = self.entity_criterion(entity_logits, entity_labels)
        loss = pair_loss + entity_loss
        outputs = (loss,) + (return_logits, )
        return outputs

if __name__ == '__main__':
    args = config.Args().get_parser()
    args.pair_tags = 8
    args.entity_tags = 8
    args.use_lstm = 'True'

    model = BertNerModel(args)
    for name,weight in model.named_parameters():
        print(name)