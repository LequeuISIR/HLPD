# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model."""
import copy
import random
from typing import Optional, Tuple, Union
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)
from transformers.models.t5.configuration_t5 import T5Config
from models.modeling_t5 import T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from models.decoder_attention_mask_utils import *
from models.losses import ZLPR
class LabelWisePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation = nn.Tanh()
        classifier_dropout = (
            config.dropout_rate if config.dropout_rate is not None else 0.0
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.label_outputs = nn.Parameter(torch.Tensor(config.num_labels, config.d_model),
                                          requires_grad=True)
        self.label_dense = nn.Parameter(torch.Tensor(config.num_labels, config.d_model),
                                        requires_grad=True)
        self.label_biases = nn.Parameter(torch.Tensor(config.num_labels, ), requires_grad=True)
        self.label_dense.data.normal_(mean=0.0, std=0.02)
        self.label_biases.data.normal_(mean=0.0, std=0.00)
        self.label_outputs.data.normal_(mean=0.0, std=0.02)

    def forward(self,
                hidden_states: Optional[torch.FloatTensor]
                ) -> torch.Tensor:
        out = hidden_states * self.label_dense
        out = self.activation(out)
        out = self.dropout(out)
        return torch.sum(out * self.label_outputs, dim=-1) + self.label_biases


class LabelPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_outputs = nn.Parameter(torch.Tensor(config.num_labels, config.d_model),
                                          requires_grad=True)
        self.label_biases = nn.Parameter(torch.Tensor(config.num_labels, config.d_model), requires_grad=True)
        self.label_biases.data.normal_(mean=0.0, std=0.00)
        self.dense = nn.Parameter(torch.Tensor(config.num_labels, config.d_model), requires_grad=True)
        self.dense.data.normal_(mean=0.0, std=0.02)
        self.label_outputs.data.normal_(mean=0.0, std=0.02)
        self.activation = nn.Tanh()

    def forward(self,
                hidden_states: Optional[torch.FloatTensor]
                ) -> torch.Tensor:
        label_encodings = hidden_states * self.dense
        label_encodings = self.activation(label_encodings)
        return label_encodings * self.label_outputs + self.label_biases


class WeightMask(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Parameter(torch.Tensor(config.num_labels, config.num_labels),
                                  requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(config.num_labels, config.num_labels), requires_grad=True)
        self.bias.data.normal_(mean=0.0, std=0.00)
        self.dense.data.normal_(mean=1.0, std=0.02)

    def forward(self,
                mask: Optional[torch.FloatTensor]
                ) -> torch.Tensor:
        return mask * self.dense + self.bias


class LabelEncoder(nn.Module):
    def __init__(self, encoder, input_ids, attention_mask):
        super(LabelEncoder, self).__init__()
        self.encoder = encoder
        self.final_layer_norm = encoder.final_layer_norm
        self.layer_module = self.encoder.block[-1]
        self.layer_module.layer.dropout = 0.0
        self.layer_module.layer[0].dropout = torch.nn.Dropout(0.0)
        self.layer_module.layer[1].dropout = torch.nn.Dropout(0.0)
        self.layer_module.layer[1].DenseReluDense.dropout = torch.nn.Dropout(0.0)
        self.cached_hidden_states = None
        self.attention_mask = None
        self.position_bias = None
        self.cache_input(input_ids, attention_mask)

    def cache_input(self, input_ids, attention_mask=None):
        # Forward pass through all but the last encoder layer
        self.encoder.eval()
        self.attention_mask = ((1.0 - attention_mask) * torch.finfo(torch.float32).min)[:, None, None, :]
        position_bias = None
        with torch.no_grad():
            hidden_states = self.encoder.embed_tokens(input_ids)
            for i, layer_module in enumerate(self.encoder.block[:-1]):
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=self.attention_mask,
                    position_bias=position_bias
                )
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
                hidden_states, present_key_value_state = layer_outputs[:2]
                position_bias = layer_outputs[2]
            self.position_bias = position_bias
            self.cached_hidden_states = hidden_states
            self.encoder = None

    def forward(self, indices):
        if self.cached_hidden_states is None:
            raise ValueError("Cached hidden states are not available. Call `cache_input` first.")
        layer_outputs = self.layer_module(
            self.cached_hidden_states[indices, ...],
            attention_mask=self.attention_mask[indices, ...], position_bias=self.position_bias[indices, ...]
        )
        hidden_states = layer_outputs[0]
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class T5ForSequenceClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder", r"label_pooler",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder", r"label_pooler",
    ]

    def __init__(self, config: T5Config, labels_tokens=None):
        super(T5ForSequenceClassification, self).__init__(config)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(encoder_config, self.shared)
        self.train_size = config.train_size if hasattr(config, 'train_size') else 0
        self.label_train_iteration = 0
        self.use_t5_label_encoding = config.use_t5_label_encoding
        self.static_label_encoding = config.static_label_encoding
        self.model_dim = config.d_model
        self.num_labels = config.num_labels
        self.labels = config.labels
        self.parent_child_relationship = config.parent_child_relationship
        self.use_zlpr_loss = config.use_zlpr_loss
        self.weight_mask = WeightMask(config)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available. Using GPU.")
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")
        self.labels_tokens = labels_tokens.to(device)
        self.batch_size = config.batch_size
        self.labels_attention_mask = create_hiera_distance_tensor(self.labels, self.parent_child_relationship).to(
            device)
        self.use_bidirectional_attention = config.use_bidirectional_attention
        if self.use_t5_label_encoding:
            self.label_pooler = LabelPooler(config)
            self.label_encoder = None
            if config.use_bidirectional_attention:
                print(
                    "USING BI-DIRECTIONAL ATTENTION MASK !!!!!!!!!!!!!!!    ")
                self.labels_attention_mask = torch.ones(size=(len(self.labels), len(self.labels))).to(
                    device)
            if not self.static_label_encoding:
                print("Using dynamic label encoding")
                self.eval_init_done = False
                self.label_encoder = None
        self.labels_embeddings = None
        self.classifier = LabelWisePooler(config)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.output_attentions = True
        decoder_config.dropout_rate = 0.0
        self.decoder = T5Stack(decoder_config, self.shared)
        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def init_label_encoder_from_pretrained(self):
        label_encoder_dict = self.label_encoder.state_dict()
        encoder_dict = self.encoder.state_dict()
        for key in label_encoder_dict.keys():
            label_encoder_dict[key] = copy.deepcopy(encoder_dict[key])
        self.label_encoder.load_state_dict(label_encoder_dict)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.encoder_labels.set_input_embeddings(new_embeddings)
        if self.t5_enc2dec:
            self.decoder.set_input_embeddings(new_embeddings)

    def tie_weights(self):
        return

    def find_true_labels_indexes_in_batch(self, labels):
        # Use torch.nonzero to find the indices of non-zero elements
        indexes = torch.nonzero(labels, as_tuple=False)
        # Convert indexes to a list
        indexes_list = []
        [indexes_list.append(list[1]) for list in indexes.tolist()]
        return list(set(indexes_list))

    def get_encoder(self):
        return self.encoder

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.static_label_encoding and self.labels_embeddings is None:
            with torch.no_grad():
                self.encoder.eval()
                label_encoder_output = self.encoder(input_ids=self.labels_tokens['input_ids'],
                                                    attention_mask=self.labels_tokens[
                                                        'attention_mask'])
                self.labels_embeddings = label_encoder_output.last_hidden_state[:, 0, :]
                self.labels_tokens = None
                print("Label embeddings !! ")
                print(self.labels_embeddings)
                self.labels_tokens = None
                self.encoder.train()
        sequence_output = encoder_outputs[0]
        if self.use_bidirectional_attention:
            labels_attention_mask = self.labels_attention_mask.expand(
                self.batch_size, -1, -1).view(self.batch_size, self.num_labels, self.num_labels)
        else:
            labels_attention_mask = self.weight_mask(self.labels_attention_mask).expand(
                self.batch_size, -1, -1).view(self.batch_size, self.num_labels, self.num_labels)
        is_dynamic_label_encoder = (self.label_train_iteration * self.batch_size) // 2 < self.train_size
        if self.label_encoder is not None and not is_dynamic_label_encoder:
            print("End of  dynamic label encoding !!")
            self.stop_updating_label_embeddings()
        if self.use_t5_label_encoding:
            if self.use_t5_label_encoding and not self.static_label_encoding:
                if is_dynamic_label_encoder:
                    if self.training and is_dynamic_label_encoder:
                        self.eval_init_done = False
                        if self.label_train_iteration % 1 == 0:
                            self.init_label_embedding(self.find_true_labels_indexes_in_batch(labels))
                        else:
                            self.labels_embeddings = self.labels_embeddings.detach()
                        self.label_train_iteration += 1
                    elif not self.training and not self.eval_init_done and is_dynamic_label_encoder:
                        self.init_label_embedding(None)
                        self.eval_init_done = True
            if sequence_output.size()[0] != self.batch_size:
                labels_attention_mask = labels_attention_mask[:sequence_output.size()[0], :]
                labels_embeddings = self.label_pooler(self.labels_embeddings).repeat(sequence_output.size()[0],
                                                                                     1).view(
                    sequence_output.size()[0], self.num_labels, self.model_dim)

                decoder_outputs = self.decoder(
                    attention_mask=labels_attention_mask,
                    encoder_hidden_states=sequence_output,
                    encoder_attention_mask=attention_mask,
                    inputs_embeds=labels_embeddings,
                )
            else:
                labels_embeddings = self.label_pooler(self.labels_embeddings).repeat(self.batch_size, 1).view(
                    self.batch_size, self.num_labels, self.model_dim)
                decoder_outputs = self.decoder(
                    attention_mask=labels_attention_mask,
                    encoder_hidden_states=sequence_output,
                    encoder_attention_mask=attention_mask,
                    inputs_embeds=labels_embeddings, )
        else:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=sequence_output,
                encoder_attention_mask=attention_mask,
            )
        logits = self.classifier(decoder_outputs[0])

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            if self.use_zlpr_loss:
                loss_fct = ZLPR()
            loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else encoder_outputs

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def stop_updating_label_embeddings(self):
        self.labels_embeddings = self.labels_embeddings.detach()
        with torch.no_grad():
            self.label_encoder = None

    def init_label_embedding(self, indices):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder(self.encoder, self.labels_tokens['input_ids'],
                                              self.labels_tokens['attention_mask'])
            self.label_encoder.train()
            self.labels_tokens = None
        if self.labels_embeddings is None or indices is None:
            self.labels_embeddings = self.label_encoder([i for i in range(self.num_labels)])[:, 0, :]
        else:
            self.labels_embeddings = self.labels_embeddings.detach()
            self.labels_embeddings[indices, ...] = self.label_encoder(indices)[:, 0, :]


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoConfig
    import torch

    model_name = 't5-base'
    labels = ['AA', 'B'] * 73
    parent_child_relationship = {labels[i]: labels[i + 1] for i in range(len(labels) - 1)}
    config = AutoConfig.from_pretrained(model_name)
    config.dropout_rate = 0.15
    config.num_labels = len(labels)
    config.use_t5_label_encoding = True
    config.static_label_encoding = True
    config.labels = labels
    config.parent_child_relationship = parent_child_relationship
    config.batch_size = 3
    config.train_size = 3
    config.use_zlpr_loss = True
    config.use_bidirectional_attention = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False, legacy=True)
    inputs = tokenizer(['Test' * random.randint(400, 512) for _ in range(3)], truncation=True, max_length=512,
                       padding='max_length', return_tensors='pt')
    decode_inputs = tokenizer(labels, truncation=True, add_special_tokens=False,
                              padding='max_length', return_tensors='pt', max_length=64)
    model = T5ForSequenceClassification.from_pretrained(model_name, config=config, labels_tokens=decode_inputs)
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params * 1e-6}')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(7):
        optimizer.zero_grad()
        target_labels = torch.ones(len(inputs['input_ids']), len(labels))
        target_labels[0, 0] = 0
        target_labels[1, 0] = 0
        target_labels[2, 0] = 0
        out_model = model(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                          decoder_input_ids=decode_inputs['input_ids'],
                          labels=target_labels)

        out_model.loss.backward()
        # Update weights
        optimizer.step()
