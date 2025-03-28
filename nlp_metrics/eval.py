"""This script defines two classes (MIMICEvalCap and COCOEvalCap) designed for evaluating image captioning models. These 
classes calculate evaluation metrics such as BLEU, METEOR, ROUGE, CIDEr, and SPICE, which are common metrics used to 
evaluate the quality of generated captions in comparison to ground truth captions."""
import string # for string manipulation
import pandas as pd # for data processing

from builtins import dict 
from nltk import sent_tokenize, word_tokenize # sent_tokenize and word_tokenize are used for tokenizing sentences and words respectively
# 
from .tokenizer.ptbtokenizer import PTBTokenizer # used for tokenizing based on the Penn Treebank tokenization style
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice

# This class evaluates captions given two CSV files containing predicted and ground truth captions.
class MIMICEvalCap:
    """he constructor takes the paths to two CSV files:
true_df_csv: Contains the true captions (ground truth).
pred_df_csv: Contains the predicted captions.
It loads these CSVs into pandas DataFrames and converts them to numpy arrays (self.pred_df and self.true_df).
Initializes dictionaries: self.eval for storing evaluation scores and self.imgToEval for storing image-level evaluations."""
    def __init__(self, true_df_csv, pred_df_csv):

        self.pred_df = pd.read_csv(pred_df_csv, header=None).values
        self.true_df = pd.read_csv(true_df_csv, header=None).values

        self.eval = dict()
        self.imgToEval = dict()
"""This method preprocesses the input string s by:
Removing newline characters (\n).
Removing the <s> and </s> tags, which are often used in tokenized captions (indicating sentence boundaries)."""
    def preprocess(self, s):
        s = s.replace('\n', '')
        s = s.replace('<s>', '')
        s = s.replace('</s>', '')
        # s = s.translate(str.maketrans('', '', '0123456789'))
        # s = s.translate(str.maketrans('', '', string.punctuation))
        return s
"""This function performs the evaluation.
gts (ground truth) and res (predictions) are dictionaries that will store tokenized ground truth and 
predicted captions, respectively."""
    def evaluate(self):

        gts = dict()
        res = dict()

        # Sanity Checks
        assert self.pred_df.shape == self.true_df.shape

        # =================================================
        # Pre-process sentences
        # =================================================
        print('tokenization...')
        """The captions are preprocessed and tokenized using word_tokenize.
The tokenized captions are stored in gts and res dictionaries."""
        for i in range(self.pred_df.shape[0]):
            pred_text = ' '.join(word_tokenize(self.preprocess(self.pred_df[i][0])))
            true_text = ' '.join(word_tokenize(self.preprocess(self.true_df[i][0])))

            res[i] = [pred_text]
            gts[i] = [true_text]

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        """The scoring functions (e.g., BLEU, METEOR, ROUGE, CIDEr, SPICE) are initialized.
Bleu(4) refers to BLEU score with n-grams up to 4.
Meteor(), Rouge(), Cider(), and Spice() are classes for computing the respective metrics."""
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        """The compute_score function is called to compute the score for the given predictions (res) and ground truth captions (gts).
If the method is a list (as in the case of BLEU), scores are computed for each n-gram (BLEU_1, BLEU_2, etc.).
The setEval function stores the score in self.eval, and setImgToEvalImgs stores the image-level scores in self.imgToEval."""
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()
    """etEval: Stores the score for the overall evaluation (e.g., BLEU score, METEOR score).
setImgToEvalImgs: Stores the score for each individual image.
setEvalImgs: Compiles the evaluation for each image into a list (self.evalImgs)."""
    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = dict()
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

#This class evaluates captioning models using the COCO dataset.
class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = dict()
        self.imgToEval = dict()
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = dict()
        res = dict()
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = dict()
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
