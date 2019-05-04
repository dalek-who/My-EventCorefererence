import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

from configs import CONFIG


class DAST_SentenceTrigger(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(DAST_SentenceTrigger, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, trigger_mask_a, trigger_mask_b,
                tfidf,
                x, edge_index, edge_weight, trigger_mask):
        # bert
        tokens_embedding, sentence_embedding = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # 两个trigger的向量表示
        new_trigger_mask_a = trigger_mask_a.unsqueeze(-1).expand_as(tokens_embedding).type_as(tokens_embedding)
        trigger_embedding_a = new_trigger_mask_a.mul(tokens_embedding).mean(dim=1)  # shape: batch，token， embedding维度
        new_trigger_mask_b = trigger_mask_b.unsqueeze(-1).expand_as(tokens_embedding).type_as(tokens_embedding)
        trigger_embedding_b = new_trigger_mask_b.mul(tokens_embedding).mean(dim=1)  # shape: batch，token， embedding维度
        # 组合为总的事件向量，分类
        event_represent = torch.cat((sentence_embedding, trigger_embedding_a, trigger_embedding_b), dim=1)
        event_represent = self.dropout(event_represent)
        logits = self.classifier(event_represent)

        return logits


class DAST_Argument(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(DAST_Argument, self).__init__(config)
        self.num_labels = num_labels
        self.feature_size = 40
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.conv1 = GCNConv(self.feature_size, CONFIG.VERTEX_EMBEDDING_DIM)
        self.classifier = nn.Linear(CONFIG.VERTEX_EMBEDDING_DIM*2, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, trigger_mask_a, trigger_mask_b,
                tfidf,
                x, edge_index, edge_weight, trigger_mask):

        nodes_embedding1 = self.conv1(x, edge_index, edge_weight)
        nodes_embedding1 = F.relu(nodes_embedding1)
        nodes_embedding1 = self.dropout1(nodes_embedding1)

        nodes_embedding = nodes_embedding1
        trigger_mask = trigger_mask.type(torch.uint8)
        trigger_nodes_embedding = nodes_embedding.masked_select(trigger_mask)
        trigger_nodes_embedding = trigger_nodes_embedding.view(-1, 2 * self.conv1_size)  # 2:一个样本里有两个trigger

        logits = self.classifier(trigger_nodes_embedding)

        return logits


class DAST(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(DAST, self).__init__(config)
        # bert
        self.num_labels = num_labels
        self.bert = BertModel(config)
        # gcn
        self.feature_size = 40
        self.conv1 = GCNConv(self.feature_size, CONFIG.VERTEX_EMBEDDING_DIM)

        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        if CONFIG.CLASSIFIER_HIDDEN_SIZE > 0:
            self.final_hidden = nn.Linear(config.hidden_size*3 + CONFIG.TFIDF_PCA_DIM*2 + CONFIG.VERTEX_EMBEDDING_DIM*2,
                                          CONFIG.CLASSIFIER_HIDDEN_SIZE)
            self.classifier = nn.Linear(CONFIG.CLASSIFIER_HIDDEN_SIZE, num_labels)
        else:
            self.final_hidden = None
            self.classifier = nn.Linear(config.hidden_size*3 + CONFIG.TFIDF_PCA_DIM*2 + CONFIG.VERTEX_EMBEDDING_DIM*2,
                                        num_labels)
        self.apply(self.init_bert_weights)


    def forward(self,
                device,
                input_ids, token_type_ids, attention_mask, trigger_mask_a, trigger_mask_b,
                tfidf,
                x, edge_index, edge_weight, trigger_mask,
                use_document_feature: bool=True,
                use_sentence_trigger_feature: bool=True,
                use_argument_feature: bool=True,
                cross_document: bool=True
                ):
        # bert
        if use_sentence_trigger_feature:
            tokens_embedding, sentence_embedding = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            # 两个trigger的向量表示
            new_trigger_mask_a = trigger_mask_a.unsqueeze(-1).expand_as(tokens_embedding).type_as(tokens_embedding)
            trigger_embedding_a = new_trigger_mask_a.mul(tokens_embedding).mean(dim=1)  # shape: batch，token， embedding维度
            new_trigger_mask_b = trigger_mask_b.unsqueeze(-1).expand_as(tokens_embedding).type_as(tokens_embedding)
            trigger_embedding_b = new_trigger_mask_b.mul(tokens_embedding).mean(dim=1)  # shape: batch，token， embedding维度
        else:
            sentence_embedding = torch.zeros((input_ids.shape[0], CONFIG.BERT_EMBEDDING_SIZE)).to(device)
            trigger_embedding_a = torch.zeros((input_ids.shape[0], CONFIG.BERT_EMBEDDING_SIZE)).to(device)
            trigger_embedding_b = torch.zeros((input_ids.shape[0], CONFIG.BERT_EMBEDDING_SIZE)).to(device)


        # tfidf
        if use_document_feature and cross_document:
            using_tfidf = tfidf
        else:
            using_tfidf = torch.zeros_like(tfidf).to(device)

        # gcn
        if use_argument_feature:
            nodes_embedding = self.conv1(x, edge_index, edge_weight)
            nodes_embedding = F.relu(nodes_embedding)
            trigger_mask = trigger_mask.type(torch.uint8)
            trigger_nodes_embedding = nodes_embedding.masked_select(trigger_mask)
            trigger_nodes_embedding = trigger_nodes_embedding.view(-1, 2 * CONFIG.VERTEX_EMBEDDING_DIM)  # 2:一个样本里有两个trigger
        else:
            trigger_nodes_embedding = torch.zeros((input_ids.shape[0], 2 * CONFIG.VERTEX_EMBEDDING_DIM)).to(device)


        # 组合为总的事件向量，分类
        event_represent = torch.cat((sentence_embedding,
                                     trigger_embedding_a, trigger_embedding_b,
                                     using_tfidf,
                                     trigger_nodes_embedding),
                                    dim=1)
        event_represent = self.dropout1(event_represent)
        if CONFIG.CLASSIFIER_HIDDEN_SIZE > 0:
            event_represent = self.final_hidden(event_represent)
            event_represent = F.relu(event_represent)
            event_represent = self.dropout2(event_represent)

        logits = self.classifier(event_represent)
        return logits

