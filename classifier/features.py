

import spacy
import numpy as np
import nltk

from collections import defaultdict

from nltk.tokenize import TreebankWordTokenizer

nlp = spacy.load("en_core_web_sm")

SEPARABLE_MARKERS = ['also', 'secondly', 'next', 'related', 'relatedly', 'similarly', 'furthermore']

ELABORATIVE_MARKERS = [
    'for instance',
    'for example',
    'e.g.',
    'specifically',
    'particularly',
    'in particular',
    'more specifically',
    'more precisely',
    'therefore'
]

CONDITIONAL_MARKERS = [
    'if so',
    'accordingly',
    'then',
    'as a result',
    'it follows',
    'subsequently',
    'consequently',
    'if yes',
    'if not',
    'if the answer is yes',
    'if the answer is no'
]

PRONOUNS = [
    'she',
    'he',
    'it',
    'they',
    'her',
    'his',
    'its',
    'their',
    'them',
    'it\'s' # because typos!
]

VERB_ELLIPSIS_MARKERS = [
    'do so',
    'did so',
    'does so',
    'do it',
    'do too',
    'does too',
    'did too',
    'did it too',
    'do it too',
    'does it too'
]

WH_WORDS = [
    'who',
    'what',
    'where',
    'when',
    'why',
    'how',
    'which'
]

POLAR_MARKERS = ["do",
	"does",
	"did",
	"didn’t",
	"will",
	"won’t",
	"would",
	"is",
	"are",
	"were",
	"weren’t",
	"wasn’t",
	"can",
	"can’t",
	"could",
	"must",
	"have",
	"has",
	"had",
	"hasn’t",
	"haven’t",
	"should",
	"shouldn’t",
	"may",
	"might",
	"shall",
	"ought"]

# What's the threshold semantic cosine similarity to distinguish related qs?
SIMILARITY_THRESHOLD = 0.8

def find_sublist(sub, bigger):
    if not bigger:
        return -1
    if not sub:
        return 0
    first, rest = sub[0], sub[1:]
    pos = 0
    try:
        while True:
            pos = bigger.index(first, pos) + 1
            if not rest or bigger[pos:pos+len(rest)] == rest:
                return pos
    except ValueError:
        return -1

def get_all_feats(example):
    feats = defaultdict(lambda: None)

    feats['pro_q2'] = pro_q2(example)
    feats['double_pro_q2'] = double_pro_q2(example)
    feats['vp_ell_q2'] = vp_ell_q2(example)
    feats['polar_q1'] = polar_q1(example)
    feats['polar_q2'] = polar_q2(example)
    feats['wh_q1'] = wh_q1(example)
    feats['wh_q2'] = wh_q2(example)
    feats['elab_cue'] = elab_cue(example)
    feats['sep_cue'] = sep_cue(example)
    feats['or'] = get_or(example)
    feats['if'] = get_if(example)
    feats['question_then_statement'] = get_question_then_statement(example)
    feats['question_markers'] = polar_q1(example) | polar_q2(example) | wh_q1(example) | wh_q2(example)
    
    feats['semantic_overlap'] = get_semantic_overlap(example)

    return feats



def pro_q2(example):
    q_toks = example.q2_toks # lower after tokenising as case info is useful
    for marker in PRONOUNS:
        marker_toks = TreebankWordTokenizer().tokenize(marker)
        if find_sublist(marker_toks, q_toks) > 0:
            return True
    return False

def double_pro_q2(example):
    count = 0
    q_toks = example.q2_toks # lower after tokenising as case info is useful
    for marker in PRONOUNS:
        marker_toks = TreebankWordTokenizer().tokenize(marker)
        if find_sublist(marker_toks, q_toks) > 0:
            count += 1

    if count > 1:
        return True
    return False

def vp_ell_q2(example):
    q_toks = example.q2_toks # lower after tokenising as case info is useful
    for marker in VERB_ELLIPSIS_MARKERS:
        marker_toks = TreebankWordTokenizer().tokenize(marker)
        
        if find_sublist(marker_toks, q_toks) > 0:
            return True
    return False

def polar_q1(example):
    q_toks = example.q1_toks # lower after tokenising as case info is useful

    for marker in POLAR_MARKERS:
        marker_toks = TreebankWordTokenizer().tokenize(marker)
        if find_sublist(marker_toks, q_toks) > 0:
            return True
    return False

def polar_q2(example):
    q_toks = example.q2_toks # lower after tokenising as case info is useful
    for marker in POLAR_MARKERS:
        marker_toks = TreebankWordTokenizer().tokenize(marker)
        
        if find_sublist(marker_toks, q_toks) > 0:
            return True
    return False


def wh_q1(example):
    q_toks = example.q1_toks # lower after tokenising as case info is useful
    for marker in WH_WORDS:
        marker_toks = TreebankWordTokenizer().tokenize(marker)
        
        if find_sublist(marker_toks, q_toks) > 0:
            return True
    return False

def wh_q2(example):
    q_toks = example.q2_toks # lower after tokenising as case info is useful
    for marker in WH_WORDS:
        marker_toks = TreebankWordTokenizer().tokenize(marker)
        
        if find_sublist(marker_toks, q_toks) > 0:
            return True
    return False

def get_question_then_statement(example):
    if example.q1.lower()[-1] == '?' and example.q2.lower()[-1] != '?':
        return True
    else:
        return False

def elab_cue(example):
    q_toks = example.q2_toks # lower after tokenising as case info is useful
    for marker in ELABORATIVE_MARKERS:
        marker_toks = TreebankWordTokenizer().tokenize(marker)
        
        if find_sublist(marker_toks, q_toks) > 0:
            return True
    return False

def get_if(example):
    q_toks = example.q2_toks # lower after tokenising as case info is useful
    for marker in CONDITIONAL_MARKERS:
        marker_toks = TreebankWordTokenizer().tokenize(marker)
        
        if find_sublist(marker_toks, q_toks) > 0:
            return True
    return False
    
def get_or(example):
    if example.q2.lower()[:3] == 'or ' or example.q2.lower()[:3] == 'or,':
        return True
    else:
        return False

def sep_cue(example):
    q_toks = example.q2_toks # lower after tokenising as case info is useful
    for marker in SEPARABLE_MARKERS:
        marker_toks = TreebankWordTokenizer().tokenize(marker)
        
        if find_sublist(marker_toks, q_toks) > 0:
            return True
    return False

def get_semantic_overlap(example):
    q1 = nlp(example.q1)
    q2 = nlp(example.q2)

    similarity = np.dot(q1.vector, q2.vector)/(q1.vector_norm * q2.vector_norm)

    return similarity > SIMILARITY_THRESHOLD