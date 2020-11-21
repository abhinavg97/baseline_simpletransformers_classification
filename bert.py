import torch
import json
import pandas as pd

from ast import literal_eval
from simpletransformers.classification import MultiLabelClassificationArgs
from simpletransformers.classification import MultiLabelClassificationModel

from module.metrics import class_wise_f1_scores, class_wise_precision_scores, class_wise_recall_scores,\
                             f1_score, accuracy_score, precision_score, recall_score, micro_f1_score, micro_precision_score, micro_recall_score

import time

log_file = "BERT_SMERP17_train_FIRE16_test"

dataset_train = 'smerp'
dataset_test = 'fire'


def count_parameters(model):
    param_count = sum(v.numel() for k, v in model.items())
    return param_count


def read_data():

    train_df = pd.read_csv(f'{dataset_train}/train_1.csv', index_col=0)
    val_df = pd.read_csv(f'{dataset_train}/val_1.csv', index_col=0)
    test_df = pd.read_csv(f'{dataset_test}/test_1.csv', index_col=0)

    train_df['labels'] = list(map(lambda label_list: literal_eval(label_list), train_df['labels'].tolist()))
    val_df['labels'] = list(map(lambda label_list: literal_eval(label_list), val_df['labels'].tolist()))
    test_df['labels'] = list(map(lambda label_list: literal_eval(label_list), test_df['labels'].tolist()))

    return train_df, val_df, test_df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Read data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_df, val_df, test_df = read_data()

label_id_to_label_text = {0: "L0", 1: "L1", 2: "L2", 3: "L3"}
# label_id_to_label_text = {0: "not_relevant", 1: "relevant"}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cuda_available = torch.cuda.is_available()
n_gpu = torch.cuda.device_count()

model_args = MultiLabelClassificationArgs()

# model_args.save_model_every_epoch = True
# model_args.no_save = False
model_args.n_gpu = n_gpu
model_args.dataloader_num_workers = 4
model_args.no_cache = True
model_args.save_eval_checkpoints = False
model_args.save_optimizer_and_scheduler = False
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 40
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_each_epoch = True
model_args.use_early_stopping = True
model_args.early_stopping_consider_epochs = True
model_args.early_stopping_metric = "avg_val_f1_score"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 3
model_args.early_stopping_delta = 0
model_args.train_batch_size = 60
model_args.eval_batch_size = 30
model_args.threshold = 0.5
model_args.tensorboard_dir = "lightning_logs/" + log_file
model_args.manual_seed = 23

model = MultiLabelClassificationModel('bert', 'bert-base-uncased',
                                      use_cuda=cuda_available, num_labels=4, args=model_args)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_time = time.time()
model.train_model(train_df, eval_df=val_df, output_dir="outputs", avg_val_accuracy_score=accuracy_score,
                  avg_val_f1_score=f1_score, avg_val_precision_score=precision_score, avg_val_recall_score=recall_score,
                  val_class_wise_f1_scores=class_wise_f1_scores, val_class_wise_precision_scores=class_wise_precision_scores,
                  val_class_wise_recall_scores=class_wise_recall_scores)

train_end_time = time.time()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

test_time = time.time()
result, model_outputs, wrong_predictions = model.eval_model(test_df, output_dir="outputs", f1_score=f1_score)
# result, model_outputs, wrong_predictions = model.eval_model(test_df, output_dir="outputs")
test_end_time = time.time()


mistakes_dataframe = pd.DataFrame()

mistakes_dataframe['text'] = list(map(lambda sentence: sentence.text_a, wrong_predictions))
mistakes_dataframe['labels'] = list(map(lambda sentence: sentence.label, wrong_predictions))


mistakes_dataframe['preds'] = list(map(lambda sentence: model.predict([sentence.text_a])[0][0], wrong_predictions))

mistakes_dataframe.to_csv(f'misclassifications_{log_file}.csv')

# preds, model_outputs, all_embedding_outputs, all_layer_hidden_states = model.predict(['This thing is entirely different from the other thing. '])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calculate metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

labels = torch.Tensor(test_df['labels'].tolist())

avg_micro_f1_score = micro_f1_score(labels, model_outputs)
avg_micro_precision_score = micro_precision_score(labels, model_outputs)
avg_micro_recall_score = micro_recall_score(labels, model_outputs)

avg_test_f1_score = f1_score(labels, model_outputs)
avg_test_precision_score = precision_score(labels, model_outputs)
avg_test_recall_score = recall_score(labels, model_outputs)
avg_test_accuracy_score = accuracy_score(labels, model_outputs)

test_class_f1_scores = class_wise_f1_scores(labels, model_outputs)
test_class_precision_scores = class_wise_precision_scores(labels, model_outputs)
test_class_recall_scores = class_wise_recall_scores(labels, model_outputs)

test_class_f1_scores_dict = {label_id_to_label_text[i]: test_class_f1_scores[i] for i in range(len(label_id_to_label_text))}
test_class_recall_scores_dict = {label_id_to_label_text[i]: test_class_recall_scores[i] for i in range(len(label_id_to_label_text))}
test_class_precision_scores_dict = {label_id_to_label_text[i]: test_class_precision_scores[i] for i in range(len(label_id_to_label_text))}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Log metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_val_metrics = pd.read_csv('outputs/training_progress_scores.csv', index_col=0)
avg_train_loss = train_val_metrics['train_loss'].tolist()
epochs = len(avg_train_loss)

rd = {}

rd['accuracy'] = {'unnormalize': avg_test_accuracy_score}

rd['f1'] = {'macro': avg_test_f1_score, 'micro': avg_micro_f1_score, 'classes': test_class_f1_scores}
rd['recall'] = {'macro': avg_test_recall_score, 'micro': avg_micro_recall_score, 'classes': test_class_recall_scores}
rd['precision'] = {'macro': avg_test_precision_score, 'micro': avg_micro_precision_score, 'classes': test_class_precision_scores}

rd['train_time'] = train_end_time - train_time
rd['test_time'] = (test_end_time - test_time) / (1.0*len(labels))

rd['train_epochs'] = epochs

model = torch.load('outputs/pytorch_model.bin', map_location=torch.device('cpu'))

rd['params'] = count_parameters(model)

with open(log_file+'.json', 'w') as f:
    json.dump(rd, f)
