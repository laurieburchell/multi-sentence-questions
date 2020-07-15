import json, csv
import nltk
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.utils.multiclass import unique_labels

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from nltk.tokenize import TreebankWordTokenizer

import features

MSQ_TYPES = [
    'NONE', # not a question
    'UNK', # question, but not sure of type
    'SEPARABLE',
    'REFORMULATION',
    'ELABORATIVE',
    # '2ELAB',
    'DISJUNCTIVE',
    'CONDITIONAL'
]


CLASS_REQUIREMENTS = {
    'SEPARABLE': {
        'pro_q2': True,
        'or': False,
        'if': False,
        'elab_cue': False,
        'question_then_statement': False,
        'polar_q1': False,
        'double_pro_q2': False
    },
    'REFORMULATION': {
        'pro_q2': False,
        'vp_ell_q2': False,
        'or': False,
        'if': False,
        'elab_cue': False,
        'question_then_statement': False,
        'semantic_overlap': True,
        'sep_cue': False
    },
    'DISJUNCTIVE': {
        'polar_q1': True,
        'polar_q2': True,
        'or': True,
        'if': False,
        'elab_cue': False,
        'wh_q1': False,
        'wh_q2': False,
        'question_then_statement': False,
        'sep_cue': False
    },
    'CONDITIONAL': {
        'polar_q1': True,
        'if': True,
        'elab_cue': False,
        'wh_q1': False,
        'sep_cue': False
    },
    'ELABORATIVE': {
        'if': False,
        'elab_cue': True,
        'sep_cue': False,
        'or': False
    }
    ,
    'ELABORATIVE2': {
        'if': False,
        'question_then_statement': True,
        'semantic_overlap': True,
        'sep_cue': False
    }
}


class ParsedExample:
    def __init__(self, row):
        self.q1 = row['q1'].strip()
        self.q2 = row['q2'].strip()

        self.q1_toks = q_toks = [tok.lower() for tok in TreebankWordTokenizer().tokenize(self.q1)] # lower after tokenising as case info is useful
        self.q2_toks = q_toks = [tok.lower() for tok in TreebankWordTokenizer().tokenize(self.q2)] # lower after tokenising as case info is useful
        
        



def classify(example):
    feature_dict = features.get_all_feats(example)

    # print(feature_dict)

    pred_class = feats_to_class(feature_dict)

    return 'ELABORATIVE' if pred_class is 'ELABORATIVE2' else pred_class


def feats_to_class(feats):

    for msq_class, mask in CLASS_REQUIREMENTS.items():
        matched = True
        for feat_name, req in mask.items():
            matched = matched and feats[feat_name] == req
        if matched:
            return msq_class
    
    return 'UNK'


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    cls_ixs = unique_labels(y_true, y_pred)
    classes = [classes[ix] for ix in cls_ixs]
    # if normalize:
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black", fontsize=16, fontweight='bold')
    fig.tight_layout()
    return ax



if __name__ == "__main__":


    
    for split in ['test']: #['train', 'dev', 'test']:
        with open(f'./{split}_msq.csv') as in_f, open(f'./{split}_msq_autolabels.csv','w') as out_f:
            # data = json.load(f)
            csv_reader = csv.DictReader(in_f)
            for ix, row in tqdm(enumerate(csv_reader), desc=f"Labelling {split}"):
                if ix >= 0:
                    pred = classify(ParsedExample(row))

                    # print(pred)
                    # print(row['q1'], row['q2'])
                    # print(ParsedExample(row))
                    # exit()

                    
                    out_f.write(f"{pred}\n")

    


    