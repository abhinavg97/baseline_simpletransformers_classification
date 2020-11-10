import torch
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger

from ast import literal_eval
from simpletransformers.classification import MultiLabelClassificationArgs
from simpletransformers.classification import MultiLabelClassificationModel

from module.metrics import class_wise_f1_scores, class_wise_precision_scores, class_wise_recall_scores,\
                             f1_score, accuracy_score, precision_score, recall_score

from module.utils import TextProcessing


text_processor = TextProcessing()

log_file = "baseline_nepal_train_q_test"

def process_text(df):

    return list(map(lambda text: text_processor.process_text(text), df['text'].tolist()))


def read_data():

    train_df = pd.read_csv('nepal/final_train2.csv', index_col=0)
    val_df = pd.read_csv('nepal/final_val2.csv', index_col=0)
    test_df = pd.read_csv('q/final_test2.csv', index_col=0)

    train_df['labels'] = list(map(lambda label_list: literal_eval(label_list), train_df['labels'].tolist()))
    val_df['labels'] = list(map(lambda label_list: literal_eval(label_list), val_df['labels'].tolist()))
    test_df['labels'] = list(map(lambda label_list: literal_eval(label_list), test_df['labels'].tolist()))

    #train_df['text'] = process_text(train_df)
    #val_df['text'] = process_text(val_df)
    #test_df['text'] = process_text(test_df)


    #train_df.to_csv("q/final_train2")
    #val_df.to_csv("q/final_val2")
    #test_df.to_csv("q/final_test2")

    return train_df, val_df, test_df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Read data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_df, val_df, test_df = read_data()

label_id_to_label_text = {0: "not_relevant", 1: "relevant"}

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


model = MultiLabelClassificationModel('distilbert', 'distilbert-base-uncased-distilled-squad',
                                              use_cuda=cuda_available, num_labels=2, args=model_args)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.train_model(train_df, eval_df=val_df, output_dir="outputs", avg_val_accuracy_score=accuracy_score,
                  avg_val_f1_score=f1_score, avg_val_precision_score=precision_score, avg_val_recall_score=recall_score,
                  val_class_wise_f1_scores=class_wise_f1_scores, val_class_wise_precision_scores=class_wise_precision_scores,
                  val_class_wise_recall_scores=class_wise_recall_scores)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

result, model_outputs, wrong_predictions = model.eval_model(test_df, output_dir="outputs", f1_score=f1_score)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calculate metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

labels = torch.Tensor(test_df['labels'].tolist())
train_val_metrics = pd.read_csv('outputs/training_progress_scores.csv', index_col=0)

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

avg_train_loss = train_val_metrics['train_loss'].tolist()
avg_val_loss = train_val_metrics['eval_loss'].tolist()
avg_val_f1_score = train_val_metrics['avg_val_f1_score'].tolist()
avg_val_precision_score = train_val_metrics['avg_val_precision_score'].tolist()
avg_val_recall_score = train_val_metrics['avg_val_recall_score'].tolist()
avg_val_accuracy_score = train_val_metrics['avg_val_accuracy_score'].tolist()

val_class_f1_scores_list = train_val_metrics['val_class_wise_f1_scores'].tolist()
val_class_precision_scores_list = train_val_metrics['val_class_wise_precision_scores'].tolist()
val_class_recall_scores_list = train_val_metrics['val_class_wise_recall_scores'].tolist()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Logger initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

logger = TensorBoardLogger("lightning_logs", name=log_file)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Log metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

epochs = len(avg_train_loss)

for epoch in range(epochs):

    val_class_f1_scores = literal_eval(val_class_f1_scores_list[epoch])
    val_class_precision_scores = literal_eval(val_class_precision_scores_list[epoch])
    val_class_recall_scores = literal_eval(val_class_recall_scores_list[epoch])

    val_class_f1_scores_dict = {label_id_to_label_text[i]: val_class_f1_scores[i] for i in range(len(label_id_to_label_text))}
    val_class_recall_scores_dict = {label_id_to_label_text[i]: val_class_recall_scores[i] for i in range(len(label_id_to_label_text))}
    val_class_precision_scores_dict = {label_id_to_label_text[i]: val_class_precision_scores[i] for i in range(len(label_id_to_label_text))}

    logger.log_metrics(metrics={'avg_train_loss': avg_train_loss[epoch]}, step=epoch)
    logger.log_metrics(metrics={'avg_val_loss': avg_val_loss[epoch]}, step=epoch)
    logger.log_metrics(metrics={'avg_val_f1_score': avg_val_f1_score[epoch]}, step=epoch)
    logger.log_metrics(metrics={'avg_val_precision_score': avg_val_precision_score[epoch]}, step=epoch)
    logger.log_metrics(metrics={'avg_val_recall_score': avg_val_recall_score[epoch]}, step=epoch)
    logger.log_metrics(metrics={'avg_val_accuracy_score': avg_val_accuracy_score[epoch]}, step=epoch)
    logger.experiment.add_scalars('val_class_f1_scores', val_class_f1_scores_dict, global_step=epoch)
    logger.experiment.add_scalars('val_class_recall_scores', val_class_recall_scores_dict, global_step=epoch)
    logger.experiment.add_scalars('val_class_precision_scores', val_class_precision_scores_dict, global_step=epoch)

logger.log_metrics(metrics={'avg_test_loss': result['eval_loss']}, step=0)
logger.log_metrics(metrics={'avg_test_f1_score': avg_test_f1_score}, step=0)
logger.log_metrics(metrics={'avg_test_precision_score': avg_test_precision_score}, step=0)
logger.log_metrics(metrics={'avg_test_recall_score': avg_test_recall_score}, step=0)
logger.log_metrics(metrics={'avg_test_accuracy_score': avg_test_accuracy_score}, step=0)
logger.experiment.add_scalars('test_class_f1_scores', test_class_f1_scores_dict, global_step=0)
logger.experiment.add_scalars('test_class_recall_scores', test_class_recall_scores_dict, global_step=0)
logger.experiment.add_scalars('test_class_precision_scores', test_class_precision_scores_dict, global_step=0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Use your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# predictions, raw_outputs = model.predict(['This thing is entirely different from the other thing. '])
