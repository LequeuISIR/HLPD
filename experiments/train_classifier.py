#!/usr/bin/env python
# coding=utf-8
import glob
import logging
import os
import pickle
import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from scipy.special import expit
from sklearn.metrics import f1_score, classification_report
from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from data import DATA_DIR
from data.hiera_multilabel_bench.hiera_multilabel_bench import WOS_CONCEPTS, RCV_CONCEPTS, BGC_CONCEPTS, AAPD_CONCEPTS, \
      MANIFESTO_CONCEPTS
from data.hiera_multilabel_bench.hiera_label_descriptors import label2desc_reduced_rcv, label2desc_reduced_aapd, \
    label2desc_reduced_bgc, label2desc_reduced_manifesto
from data_collator import DataCollatorHTC
from models.t5_classifier import T5ForSequenceClassification
from models.template_label_description_temp import generate_template

logger = logging.getLogger(__name__)
from tokenizers.normalizers import NFKD
from tokenizers.pre_tokenizers import WhitespaceSplit
from optim import get_optimizer, get_lr_scheduler

normalizer = NFKD()
pre_tokenizer = WhitespaceSplit()


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="uklex-l1",
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    dataset_bench: Optional[str] = field(
        default="multilabel_bench",
        metadata={
            "help": "Directory for the HMTC dataset"
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    optim_warmup_steps: Optional[int] = field(
        default=6000,
        metadata={
            "help": ""
        },
    )
    optim_total_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": ""
        },
    )
    optim_lr_scheduler: Optional[str] = field(
        default="cosine",
        metadata={
            "help": ""
        },
    )
    optim_weight_decay: Optional[float] = field(
        default=0.001,
        metadata={
            "help": ""
        },
    )
    optim_final_cosine: Optional[float] = field(
        default=5e-5,
        metadata={
            "help": ""
        },
    )


@dataclass
class ModelArguments:
    model_name: str = field(
        default="t5-base", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_t5_label_encoding: bool = field(
        default=True,
        metadata={"help": "Whether to use T5 label encoding or not."},
    )
    static_label_encoding: bool = field(
        default=False,
        metadata={
            "help": "Whether to have static embedding of label from T5"},
    ),
    use_bidirectional_attention: bool = field(
        default=True,
        metadata={
            "help": "Whether to use_bidirectional_attention or not . If true we do not use hiera mask"},
    )
    use_zlpr_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to use ZLPR loss or not . If False BCE loss is used."},
    )
    early_stopping_patience: int = field(
        default=10,
        metadata={
            "help": "early_stopping_patience  "},
    )


def main():
    TrainingArguments.output_dir = "output"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.eval_steps = 2000
    training_args.evaluation_strategy = "steps"
    training_args.save_strategy = "steps"
    training_args.save_steps = 2000
    # training_args.eval_delay = 8000
    training_args.save_only_model = True
    training_args.logging_steps = 1000
    training_args.save_safetensors = False
    training_args.save_total_limit = 2
    print(model_args)
    print(training_args)
    print(data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print()
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    label_list = None
    label_desc2id = None
    label_id2desc = None
    train_dataset = None
    if training_args.do_train:
        print("loading MANIFESTO DATASET HEHEHEHEHE")
        print(os.path.join(DATA_DIR, data_args.dataset_bench))
        train_dataset = load_dataset(os.path.join(DATA_DIR, data_args.dataset_bench), data_args.dataset_name,
                                     split="train", num_proc=5)
        label_list = list(
            range(train_dataset.features['concepts'].feature.num_classes))
        labels_codes = train_dataset.features['concepts'].feature.names
        num_labels = len(label_list)

    if training_args.do_eval:
        eval_dataset = load_dataset(os.path.join(DATA_DIR, data_args.dataset_bench), data_args.dataset_name,
                                    split="validation", num_proc=5)
                                    
        if label_list is None:
            label_list = list(
                range(eval_dataset.features['concepts'].feature.num_classes))
            labels_codes = eval_dataset.features['concepts'].feature.names
            num_labels = len(label_list)

    if training_args.do_predict:
        predict_dataset = load_dataset(os.path.join(DATA_DIR, data_args.dataset_bench), data_args.dataset_name,
                                       split="test", num_proc=5)
        if label_list is None:
            label_list = list(
                range(predict_dataset.features['concepts'].feature.num_classes))
            labels_codes = predict_dataset.features['concepts'].feature.names
            num_labels = len(label_list)

    parent_child_relationship = None
    if 'wos' in data_args.dataset_name:
        parent_child_relationship = WOS_CONCEPTS["parent_childs"]
        label_descriptors = WOS_CONCEPTS[f'level_{data_args.dataset_name.split("-")[-1]}']
        print(label_descriptors)
        label_descs = label_descriptors
    elif 'aapd' in data_args.dataset_name:
        parent_child_relationship = AAPD_CONCEPTS["parent_childs"]
        label_descriptors = AAPD_CONCEPTS[f'level_{data_args.dataset_name.split("-")[-1]}']
        label_descs = [label2desc_reduced_aapd[key] for key in label_descriptors]
    elif 'rcv' in data_args.dataset_name:
        parent_child_relationship = RCV_CONCEPTS["parent_childs"]
        label_descriptors = RCV_CONCEPTS[f'level_{data_args.dataset_name.split("-")[-1]}']
        label_descs = [label2desc_reduced_rcv[key] for key in label_descriptors]
    elif 'bgc' in data_args.dataset_name:
        parent_child_relationship = BGC_CONCEPTS["parent_childs"]
        label_descriptors = BGC_CONCEPTS[f'level_{data_args.dataset_name.split("-")[-1]}']
        label_descs = [label2desc_reduced_bgc[key] for key in label_descriptors]

    elif 'manifesto' in data_args.dataset_name:
        print("loaded MANIFESTO CONCEPTS HEHEHEHEHE")
        parent_child_relationship = MANIFESTO_CONCEPTS["parent_childs"]
        label_descriptors = MANIFESTO_CONCEPTS[f'level_{data_args.dataset_name.split("-")[-1]}']
        print(label_descriptors)
        label_descs = label_descriptors
    else:
        raise Exception(f'Dataset {data_args.dataset_name} is not supported!')
    label_desc2id = {label_desc: idx for idx, label_desc in enumerate(labels_codes)}
    label_id2desc = {idx: label_desc for idx, label_desc in enumerate(labels_codes)}


    print(f'LabelDesc2Id: {label_desc2id}')
    print(f'Label description : {label_descs}')
    config = AutoConfig.from_pretrained(
        model_args.model_name,
        num_labels=num_labels,
        label2id=label_desc2id,
        id2label=label_id2desc,
        finetuning_task=data_args.dataset_name,
        cache_dir=None,
    )
    config.dropout_rate = 0.15
    config.use_t5_label_encoding = model_args.use_t5_label_encoding
    config.static_label_encoding = model_args.static_label_encoding
    config.use_bidirectional_attention = model_args.use_bidirectional_attention
    config.use_zlpr_loss = model_args.use_zlpr_loss
    config.batch_size = training_args.per_device_train_batch_size
    if train_dataset is not None:
        config.train_size = len(train_dataset)
    config.labels = label_descriptors
    print("LABELS")
    print(label_descriptors)
    print("Parent child")
    print(parent_child_relationship)
    config.parent_child_relationship = parent_child_relationship
    label_descs = generate_template(parent_child_relationship, label_desc2id, label_descs)
    print("HIERA TEMPLATE ")
    print(label_descs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        legacy=False
    )
    model = T5ForSequenceClassification.from_pretrained(
        model_args.model_name,
        from_tf=bool(".ckpt" in model_args.model_name),
        config=config,
        labels_tokens=tokenizer(
            label_descs, truncation=True, add_special_tokens=False,
            padding='max_length', return_tensors='pt', max_length=64
        )
    )
    print(model.config)
    padding = "max_length"

    def preprocess_function(examples):
        batch = tokenizer(
            examples["text"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
            add_special_tokens=True
        )
        decoder_inputs = tokenizer(
            [' '.join([label_id2desc[label] for label in label_id2desc]) for _ in examples['text']],
            padding=False,
            max_length=len(label_id2desc),
            truncation=True,
            add_special_tokens=False
        )
        batch['decoder_input_ids'] = decoder_inputs['input_ids']
        batch['decoder_attention_mask'] = decoder_inputs['attention_mask']
        batch["label_ids"] = [[1.0 if label in labels else 0.0 for label in label_list] for labels in
                              examples["concepts"]]
        batch['labels'] = batch['label_ids']
        return batch

    if training_args.do_train:
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['concepts', 'text'],
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
    else:
        model.eval()
    if training_args.do_eval:
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['concepts', 'text'],
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['concepts', 'text'],
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):

        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        probs = expit(logits)
        preds = (probs >= 0.5).astype('int32')
        macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
        mean_macro_micro_f1 = (macro_f1 + micro_f1) / 2
        return {'macro-f1': macro_f1, 'micro-f1': micro_f1, "macro-micro-f1": mean_macro_micro_f1}

    trainer_class = Trainer
    data_collator = DataCollatorHTC(tokenizer)
    if train_dataset is not None:
        data_args.optim_total_steps = int(
            len(train_dataset) * training_args.num_train_epochs / training_args.per_device_train_batch_size)
        model_args.early_stopping_patience = min(
            int(len(train_dataset) / (training_args.save_steps * training_args.per_device_train_batch_size)) * 7, 25)
        print("EARLY STOPPING PATIENCE")
        print(model_args.early_stopping_patience)

    optimizer = None
    lr_scheduler = None
    if training_args.optim == 'adafactor':
        optimizer = get_optimizer(model, training_args, data_args)
        lr_scheduler = get_lr_scheduler(optimizer, training_args, data_args)
    # Initialize our Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer, optimizers=(optimizer, lr_scheduler),
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience)],
    )
    # Training
    if training_args.do_train:
        
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        import time
        try:
            logger.info("*** Predict ***")
            start_time = time.time()
            predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")

            hard_predictions = (expit(predictions) >= 0.5).astype('int32')
            text_preds = [
                ', '.join(sorted([label_id2desc[idx] for idx, val in enumerate(doc_predictions) if val == 1]))
                for doc_predictions in hard_predictions]
            text_labels = [', '.join(sorted([label_id2desc[idx] for idx, val in enumerate(doc_labels) if val == 1]))
                           for doc_labels in labels]
            metrics["predict_samples"] = len(predict_dataset)
            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)
            report_predict_file = os.path.join(training_args.output_dir, "test_classification_report.txt")
            predictions_file = os.path.join(training_args.output_dir, "test_predictions.pkl")
            labels_file = os.path.join(training_args.output_dir, "test_labels.pkl")
            if trainer.is_world_process_zero():
                cls_report = classification_report(y_true=labels, y_pred=hard_predictions,
                                                   target_names=list(config.label2id.keys()),
                                                   zero_division=0, digits=4)
                with open(report_predict_file, "w") as writer:
                    writer.write(cls_report)
                with open(predictions_file, 'wb') as writer:
                    pickle.dump(text_preds, writer, protocol=pickle.HIGHEST_PROTOCOL)
                with open(labels_file, 'wb') as writer:
                    pickle.dump(text_labels, writer, protocol=pickle.HIGHEST_PROTOCOL)

                logger.info(cls_report)

        except Exception as inst:
            print(inst)

    # Clean up checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


if __name__ == "__main__":
    main()
