from transformers.models.bert.modeling_bert  import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertOnlyNSPHead

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss

from IPython import embed

class BertNSPHead(nn.Module):
    def __init__(self, config):
        super(BertNSPHead, self).__init__()
        self.prediction_distance = nn.Linear(config.hidden_size, 1)

    def forward(self, pooled_output):
        distance = self.prediction_distance(pooled_output)
        return distance

class BertDistHead(nn.Module):
    def __init__(self, config):
        super(BertDistHead, self).__init__()
        self.prediction_distance = nn.Linear(config.hidden_size, 1)

    def forward(self, pooled_output):
        distance = self.prediction_distance(pooled_output)
        return distance

class BertForMaskNode(BertPreTrainedModel):
    r"""

    """

    def __init__(self, config):
        super(BertForMaskNode, self).__init__(config)

        self.bert = BertModel(config)
        self.cls_MLM = BertOnlyMLMHead(config)
        self.cls_NSP = BertNSPHead(config)
        self.cls_Dist = BertDistHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls_MLM.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        label_NSP=None,
        label_dist=None,
    ):
        #Output for MLM
        outputs_mlm = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs_mlm[0]
        prediction_scores = self.cls_MLM(sequence_output)


        outputs_mlm = (prediction_scores,) + outputs_mlm[2:]

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs_mlm = (masked_lm_loss,) + outputs_mlm

        
        #Output for NSP
        outputs_NSP = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs_NSP[1]

        distance = self.cls_NSP(pooled_output)

        outputs_NSP = (distance, ) + outputs_NSP[2:]  # add hidden states and attention if they are here  
        if label_NSP is not None:
            loss_fct = BCELoss()
            sm = nn.Sigmoid()
            loss_NSP = loss_fct(sm(distance.view(-1)), label_NSP.view(-1))

        #Output for Dist
        outputs_Dist = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs_Dist[1]

        distance = self.cls_Dist(pooled_output)

        outputs_Dist = (distance, ) + outputs_Dist[2:]  # add hidden states and attention if they are here  
        if label_dist is not None:
            loss_fct = MSELoss()
            loss_Dist = loss_fct(distance, label_dist)
            
        outputs = masked_lm_loss+loss_NSP+loss_Dist
        #embed()
        outputs = (outputs, masked_lm_loss, loss_NSP, loss_Dist)
        
        return outputs  # (三個loss相加), outputs_VA[2:] -> [seq_relationship_score, (hidden_states), (attentions)]

"""
@add_start_docstrings(
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING
)
"""