import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
import q1

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        # add bias to perceptron
        self.w = np.zeros(len(xFeat[0]))
        for epoch in range(self.mEpoch):
            num_mistakes = 0
            preds = self.predict(xFeat)
            for i in range(len(preds)):
                y_true = y[i][0]
                if preds[i] != y_true:
                    num_mistakes += 1
                    if y_true > 0:
                        self.w += xFeat[i]
                    else:
                        self.w -= xFeat[i]
            stats[epoch] = num_mistakes
        return stats


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        yHat = []
        for i in range(len(xFeat)):
            pred = np.dot(self.w, xFeat[i])
            if pred >= 0:
                yHat.append(1) # predict plus one
            else:
                yHat.append(0)
        return yHat


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    err = 0
    for i in range(len(yHat)):
        if yHat[i] != yTrue[i][0]:
            err += 1
    return err

def opt_epochs(xFeat, y):
    epochs = [1, 10, 20, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    mistakes = []
    for epoch in epochs:
        num_mistakes = 0
        kf = KFold(n_splits=10, shuffle=True)
        for train_idx, test_idx in kf.split(xFeat):
            # determine splits
            xTrain, xTest = xFeat[train_idx, :], xFeat[test_idx, :]
            yTrain, yTest = y[train_idx], y[test_idx]

            # fit and train model
            model = Perceptron(epoch)
            model.train(xTrain, yTrain)
            yHat = model.predict(xTest)

            # calculate mistakes
            num_mistakes += calc_mistakes(yHat, yTest)
        mistakes.append(num_mistakes/10)

    df = pd.DataFrame()
    df['Epoch'] = epochs
    df['Num_Mistakes'] = mistakes
    print(df)
    return df[df['Num_Mistakes'] == df['Num_Mistakes'].min()]

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    # yHatTrain = model.predict(xTrain)
    yHatTest = model.predict(xTest)
    # print out the number of mistakes
    # print("Number of mistakes on the train dataset")
    # print(calc_mistakes(yHatTrain, yTrain))
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHatTest, yTest))
    # print optimal epochs
    # print(opt_epochs(xTrain, yTrain))

    # vocab map from q1 imported here
    if True:
        vocab_map = {'wrote': 1219, 'on': 12654, 'the': 50367, 'receiv': 1916, 'side': 196, 'my': 3338, 'email': 3260, 'client': 466,
         'between': 437, 'messag': 2243, 'that': 11982, 'ar': 6560, 'read': 1100, 'and': 26258, 'not': 5780, 'like': 2234,
         'to': 34874, 'mark': 232, 'or': 6621, 'save': 694, 'particularli': 80, 'import': 397, 'me': 1813, 'even': 1187,
         'if': 4795, 'make': 2341, 'point': 663, 'delet': 244, 'materi': 160, 'immedi': 298, 'upon': 245, 'it': 13013,
         'might': 496, 'leav': 218, 'an': 3940, 'interest': 895, 'kind': 322, 'of': 24942, 'trace': 56, 'machin': 278,
         'you': 15994, 'choos': 197, 'have': 6231, 'your': 8689, 'do': 3356, 'don': 1819, 'short': 201, 'can': 4524,
         'whatev': 165, 'want': 1646, 'with': 7783, 'byte': 81, 'hold': 207, 'includ': 1377, 'll': 977, 'bui': 551,
         'for': 14090, 'anyon': 584, 'who': 1802, 'show': 592, 'otherwis': 103, 'attack': 283, 'be': 7941, 'abl': 404,
         'verifi': 100, 'by': 5268, 'see': 1366, 'edit': 261, 'copi': 512, 'but': 3821, 'haven': 189, 'wa': 3437,
         'could': 1155, 'put': 571, 'line': 1075, 'befor': 823, 'download': 586, 'mail': 4511, 'instal': 748, 'run': 973,
         'in': 16763, 'onli': 2259, 'except': 173, 'rule': 308, 'belong': 40, 'is': 14540, 'mayb': 281, 'httpaddr': 11316,
         'thi': 10522, 'multi': 298, 'part': 782, 'mime': 338, 'format': 446, 'nextpart': 564, 'number': 72687,
         'content': 1712, 'type': 1206, 'text': 1146, 'plain': 400, 'charset': 523, 'iso': 328, 'transfer': 724,
         'encod': 541, 'quot': 1087, 'printabl': 245, 'get': 3471, 'anoth': 577, 'fix': 472, 'ireland': 92, 'newspap': 50,
         'larg': 320, 'break': 168, 'new': 3376, 'evil': 103, 'staff': 81, 'html': 588, 'nbsp': 12246, 'some': 1860,
         'patch': 182, 'seem': 612, 'avoid': 164, 'problem': 1006, 'now': 2433, 'amaz': 133, 'what': 2492, 'achiev': 126,
         'when': 1966, 'main': 199, 'power': 608, 'fail': 231, 'long': 654, 'enough': 341, 'up': 2486, 'out': 2855,
         'all': 4241, 'left': 221, 'oper': 447, 'laptop': 118, 'batteri': 58, 'suppli': 357, 'first': 1259, 'defens': 116,
         'code': 819, 'into': 1317, 'area': 325, 'where': 909, 'occur': 97, 'so': 2768, 'exmh': 649, 'attempt': 154,
         'ani': 2269, 'reason': 433, 'expand': 132, 'sequenc': 251, 'isn': 281, 'either': 329, 'rang': 152, 'list': 4223,
         'such': 804, 'thing': 961, 'will': 4744, 'simpli': 386, 'ignor': 201, 'rather': 268, 'than': 1641, 'give': 816,
         'solv': 91, 'initi': 181, 'mh': 105, 'tcl': 53, 'aug': 411, 'set': 941, 'split': 67, 'string': 147, 'dollar': 831,
         'just': 2609, 'anyth': 375, 'continu': 407, 'amend': 56, 'which': 2253, 'error': 496, 'assum': 199, 'we': 4895,
         'should': 1105, 'probabl': 403, 'sure': 643, 'true': 314, 'issu': 673, 'instead': 426, 'think': 1136, 'need': 1609,
         'allow': 555, 'creat': 571, 'similar': 237, 'place': 597, 'fight': 109, 'fire': 109, 'thought': 333, 'more': 3364,
         'about': 2641, 'brent': 47, 'suggest': 307, 'select': 404, 'actual': 554, 'valid': 141, 'input': 89, 'would': 1941,
         'mean': 561, 'know': 1306, 'decid': 311, 'right': 1278, 'caus': 360, 'stuck': 33, 'catch': 104, 'around': 422,
         'process': 531, 'user': 1739, 'ha': 2916, 'wai': 1391, 'later': 260, 'treat': 85, 'as': 5936, 'legal': 533,
         'name': 1477, 'next': 637, 'els': 386, 'handl': 244, 'there': 2450, 'without': 706, 'someon': 427, 'rememb': 325,
         'thei': 3024, 'go': 1506, 'while': 763, 'plai': 343, 'notic': 375, 'someth': 615, 'never': 587, 'chang': 1512,
         'each': 835, 'suspect': 66, 'perhap': 193, 'bound': 60, 'differ': 687, 'function': 252, 'us': 5963, 'current': 658,
         'target': 316, 'folder': 315, 'mode': 104, 'let': 611, 'found': 601, 'nice': 266, 'why': 674, 'well': 1012,
         'limit': 360, 'bug': 262, 'then': 1452, 'wonder': 230, 'contain': 419, 'special': 560, 'charact': 110, 'no': 2963,
         'those': 956, 'from': 6313, 'keyboard': 107, 'two': 770, 'restrict': 70, 'begin': 367, 'pain': 124, 'anywai': 183,
         'doubt': 100, 'bother': 54, 'possibl': 483, 'empti': 57, 'less': 467, 'happi': 251, 'live': 634, 'order': 1580,
         'them': 1592, 'exist': 395, 'sinc': 702, 'featur': 575, 'alwai': 420, 'normal': 207, 'kei': 385, 've': 1206,
         'highlight': 91, 'here': 2457, 'remain': 205, 'discov': 223, 'extern': 76, 'world': 1154, 'bind': 74, 'same': 863,
         'work': 2308, 'cours': 446, 'got': 525, 'start': 1040, 'also': 1596, 'made': 682, 'clear': 194, 'statu': 132,
         'entri': 126, 'previous': 101, 'prompt': 78, 'sit': 89, 'until': 375, 'appear': 276, 'obviou': 71, 'had': 1162,
         'usual': 170, 'case': 715, 'follow': 872, 'doe': 1036, 'believ': 413, 'appli': 341, 'abov': 355, 'hide': 59,
         'realli': 640, 'invok': 178, 'complet': 625, 'previou': 107, 'ok': 222, 'info': 359, 'msg': 200, 'global': 323,
         'return': 408, 'match': 156, 'configur': 324, 'state': 1387, 'disabl': 105, 'focu': 101, 'worker': 160,
         'emailaddr': 4612, 'etc': 530, 'cf': 224, 'file': 1697, 'method': 303, 'load': 248, 'averag': 174, 'anymor': 71,
         'look': 1359, 'config': 83, 'regard': 377, 'adam': 146, 'origin': 548, 'behalf': 148, 'joe': 85, 'sent': 695,
         'sundai': 69, 'august': 322, 'pm': 302, 'subject': 879, 'razor': 645, 'reduc': 135, 'impact': 79, 'ton': 39,
         'server': 978, 'pc': 415, 'fun': 142, 'freebsd': 58, 'quit': 271, 'support': 770, 'wife': 92, 'web': 1367,
         'site': 1403, 'few': 633, 'other': 2084, 'servic': 1373, 'jabber': 79, 'variou': 157, 'member': 397, 'our': 3659,
         'famili': 537, 'fair': 110, 'amount': 271, 'come': 878, 'base': 768, 'isp': 108, 'di': 126, 'hour': 511,
         'internet': 1287, 'huge': 176, 'flow': 137, 'filter': 160, 'via': 316, 'result': 605, 'sometim': 158, 'over': 1443,
         'at': 4969, 'time': 2807, 'after': 916, 'slow': 128, 'argument': 115, 'associ': 302, 'call': 1228, 'check': 1096,
         'log': 399, 'though': 427, 'ad': 853, 'home': 1151, 'down': 622, 'cannot': 222, 'comment': 340, 'whether': 292,
         'paragraph': 129, 'mai': 1360, 'real': 665, 'descript': 122, 'rate': 782, 'suffer': 79, 'lot': 539, 'idea': 472,
         'welcom': 339, 'thank': 740, 'aim': 62, 'yahoo': 299, 'sf': 509, 'net': 1194, 'sponsor': 620, 'osdn': 116,
         'tire': 169, 'old': 600, 'cell': 200, 'phone': 821, 'free': 2796, 'chat': 86, 'how': 1966, 'been': 1818,
         'try': 822, 'almost': 244, 'everyth': 345, 'lose': 274, 'weight': 283, 'feel': 382, 'diet': 88, 'pill': 46,
         'exercis': 108, 'equip': 108, 'help': 1186, 'pound': 80, 'lost': 186, 'heard': 155, 'extrem': 189, 'plu': 326,
         're': 1320, 'yourself': 282, 'oh': 151, 'skeptic': 67, 'said': 961, 'her': 371, 'week': 809, 'told': 169,
         'noth': 327, 'tell': 545, 'best': 859, 'decis': 153, 'ever': 497, 'period': 149, 'six': 138, 'month': 855,
         'write': 749, 'gone': 108, 'routin': 40, 'ye': 393, 'still': 765, 'eat': 80, 'contact': 567, 'manufactur': 106,
         'permiss': 112, 'resel': 58, 'discount': 109, 'peopl': 2061, 'did': 577, 'becaus': 1192, 'much': 1046, 'self': 185,
         'mention': 201, 'health': 192, 'am': 955, 'person': 919, 'absolut': 246, 'monei': 1410, 'back': 876,
         'guarante': 464, 'frustrat': 45, 'product': 1314, 'success': 441, 'were': 1034, 'promis': 131, 'recommend': 278,
         'ask': 470, 'stuff': 370, 'fat': 116, 'scientif': 55, 'proven': 101, 'increas': 467, 'rapid': 40, 'loss': 194,
         'these': 1362, 'bottom': 133, 'per': 559, 'natur': 288, 'dai': 1369, 'confid': 124, 'gain': 168, 'fast': 290,
         'bonu': 192, 'ship': 366, 'bottl': 130, 'secur': 1078, 'click': 1827, 'link': 1033, 'custom': 661, 'visit': 565,
         'inform': 2197, 'test': 519, 'studi': 175, 'pleas': 1842, 'send': 1454, 'request': 377, 'remov': 1527,
         'apolog': 120, 'inconveni': 51, 'rohit': 52, 'crap': 46, 'unsolicit': 93, 'commerci': 164, 'heaven': 383,
         'captur': 51, 'post': 545, 'numberk': 150, 'articl': 287, 'invit': 80, 'wait': 270, 'didn': 328, 'fork': 161,
         'deliv': 218, 'address': 1898, 'wast': 70, 'org': 194, 'spamassassin': 650, 'enabl': 220, 'spam': 1040,
         'figur': 218, 'miss': 231, 'big': 372, 'deal': 353, 'pick': 286, 'repli': 590, 'flag': 117, 'word': 493,
         'guess': 184, 'fashion': 42, 'everi': 804, 'protocol': 87, 'strategi': 208, 'emerg': 108, 'york': 166,
         'newest': 36, 'david': 168, 'sort': 258, 'sever': 375, 'he': 1399, 'british': 67, 'offic': 358, 'celebr': 54,
         'reveal': 92, 'sai': 1167, 'public': 564, 'crack': 47, 'univers': 383, 'group': 1247, 'dvd': 305, 'join': 291,
         'unsubscrib': 703, 'john': 226, 'hall': 52, 'opinion': 171, 'spend': 191, 'too': 682, 'regul': 138, 'pretti': 225,
         'liber': 68, 'polit': 291, 'geeg': 61, 'hello': 115, 'unlimit': 70, 'intern': 435, 'telephon': 139, 'market': 1151,
         'flat': 70, 'domest': 77, 'minut': 329, 'trial': 106, 'local': 382, 'access': 691, 'design': 545, 'date': 967,
         'dial': 99, 'page': 773, 'toll': 78, 'within': 639, 'switch': 150, 'hear': 200, 'voic': 123, 'must': 607,
         'digit': 387, 'distanc': 98, 'exampl': 399, 'www': 342, 'com': 1334, 'listen': 103, 'pai': 483, 'monthli': 103,
         'fee': 160, 'again': 545, 'tax': 186, 'awai': 301, 'final': 254, 'enter': 310, 'speed': 202, 'easier': 160,
         'thu': 303, 'question': 633, 'addit': 295, 'enhanc': 177, 'portabl': 71, 'card': 647, 'tool': 445, 'network': 979,
         'promot': 256, 'offer': 1215, 'their': 2023, 'program': 1396, 'cent': 80, 'great': 573, 'sign': 442, 'below': 649,
         'septemb': 291, 'high': 653, 'dr': 231, 'citizen': 107, 'former': 107, 'militari': 163, 'worth': 221,
         'himself': 35, 'taken': 182, 'last': 656, 'mani': 1026, 'board': 175, 'class': 208, 'section': 192, 'began': 59,
         'air': 140, 'move': 426, 'mr': 382, 'describ': 138, 'man': 262, 'refus': 43, 'hi': 1371, 'lead': 422, 'role': 84,
         'latest': 392, 'action': 230, 'him': 253, 'attend': 53, 'oblig': 208, 'find': 1184, 'alreadi': 392, 'end': 804,
         'readi': 188, 'pull': 94, 'demand': 151, 'seri': 173, 'stand': 146, 'arm': 98, 'extend': 144, 'danger': 107,
         'shot': 76, 'most': 1265, 'terrorist': 148, 'battl': 54, 'pressur': 47, 'senior': 75, 'judg': 109, 'jame': 120,
         'court': 185, 'common': 193, 'year': 1604, 'die': 61, 'husband': 63, 'she': 331, 'worri': 148, 'speak': 156,
         'english': 95, 'therefor': 98, 'understand': 318, 'threat': 134, 'land': 153, 'half': 212, 'grab': 56,
         'doctor': 67, 'behind': 128, 'good': 984, 'polic': 99, 'station': 75, 'straight': 58, 'retir': 54, 'major': 348,
         'done': 334, 'wrong': 276, 'sole': 72, 'skin': 63, 'held': 129, 'three': 368, 'releas': 665, 'charg': 297,
         'offici': 211, 'tri': 260, 'eventu': 73, 'identifi': 112, 'report': 1450, 'administr': 213, 'depart': 154,
         'declin': 109, 'discuss': 328, 'detail': 404, 'explan': 48, 'watch': 229, 'close': 281, 'becom': 379, 'arrest': 60,
         'head': 260, 'yesterdai': 70, 'life': 768, 'kill': 127, 'lawyer': 63, 'american': 429, 'civil': 107, 'liberti': 68,
         'union': 60, 'feder': 272, 'govern': 947, 'take': 1163, 'men': 200, 'women': 219, 'freedom': 227, 'countri': 531,
         'hettinga': 86, 'bearer': 54, 'underwrit': 59, 'corpor': 300, 'farquhar': 45, 'street': 230, 'boston': 85,
         'ma': 109, 'usa': 203, 'howev': 461, 'deserv': 104, 'respect': 182, 'antiqu': 62, 'predict': 125, 'agreeabl': 42,
         'experi': 362, 'edward': 59, 'gibbon': 44, 'fall': 174, 'roman': 103, 'empir': 86, 'url': 757, 'numbertnumb': 265,
         'came': 197, 'beauti': 71, 'odd': 58, 'color': 990, 'light': 114, 'obvious': 75, 'todai': 903, 'jim': 83,
         'explain': 160, 'forc': 348, 'veri': 999, 'bit': 428, 'saw': 75, 'photo': 125, 'busi': 1824, 'accept': 318,
         'credit': 652, 'consult': 195, 'applic': 670, 'retail': 184, 'low': 413, 'merchant': 71, 'account': 540,
         'cancel': 60, 'beat': 85, 'anybodi': 97, 'easi': 619, 'afford': 91, 'approv': 172, 'obtain': 133, 'resid': 105,
         'mon': 172, 'sep': 286, 'rick': 48, 'sorri': 127, 'wasn': 119, 'inbox': 70, 'recent': 423, 'command': 260,
         'sh': 48, 'outlook': 89, 'hate': 87, 'indic': 153, 'sender': 142, 'size': 858, 'column': 65, 'instantli': 58,
         'reader': 171, 'provid': 941, 'face': 674, 'fact': 428, 'industri': 383, 'standard': 330, 'unix': 161,
         'connect': 401, 'traffic': 141, 'bad': 272, 'compar': 186, 'modern': 67, 'mailer': 108, 'hei': 93, 'given': 249,
         'pentium': 52, 'numbermb': 124, 'floppi': 81, 'drive': 425, 'funni': 51, 'soon': 232, 'stop': 402, 'mess': 54,
         'advis': 87, 'small': 394, 'serial': 54, 'port': 209, 'irish': 592, 'linux': 1485, 'un': 630, 'subscript': 783,
         'maintain': 655, 'numberp': 210, 'ascii': 159, 'cv': 174, 'ed': 184, 'unseen': 182, 'blame': 47, 'multipl': 182,
         'chri': 265, 'garrigu': 73, 'vircio': 42, 'congress': 135, 'suit': 206, 'austin': 50, 'tx': 64, 'war': 308,
         'iii': 110, 'doer': 81, 'vs': 113, 'pgp': 394, 'signatur': 425, 'version': 856, 'gnupg': 105, 'vnumber': 282,
         'gnu': 225, 'anumb': 1826, 'deathtospamdeathtospamdeathtospam': 117, 'thinkgeek': 253, 'geek': 311, 'sight': 164,
         'mount': 69, 'sourc': 607, 'medium': 40, 'packag': 682, 'suse': 105, 'impress': 83, 'mike': 75, 'mobil': 163,
         'devic': 254, 'talk': 465, 'dynam': 83, 'system': 1614, 'build': 660, 'own': 867, 'basic': 237, 'dollarnumb': 3678,
         'specif': 350, 'licens': 436, 'cover': 147, 'singl': 260, 'advanc': 176, 'ftp': 252, 'brian': 101, 'requir': 667,
         'minimum': 75, 'fund': 301, 'add': 480, 'piec': 170, 'serious': 63, 'fundament': 61, 'unusu': 36, 'situat': 178,
         'redhat': 251, 'debian': 62, 'option': 380, 'familiar': 35, 'distribut': 284, 'altern': 224, 'involv': 249,
         'better': 614, 'mainten': 39, 'logic': 44, 'through': 844, 'compani': 1657, 'imagin': 142, 'cd': 808,
         'hardwar': 231, 'matthew': 35, 'sport': 75, 'music': 233, 'chart': 44, 'off': 669, 'hit': 284, 'button': 161,
         'anywher': 143, 'young': 88, 'co': 246, 'exclus': 137, 'video': 218, 'hundr': 284, 'model': 224, 'love': 279,
         'bigger': 72, 'plenti': 47, 'stabl': 90, 'water': 88, 'hand': 306, 'shoot': 52, 'qualiti': 278, 'girl': 111,
         'door': 67, 'wild': 43, 'duncan': 37, 'beberg': 79, 'search': 608, 'setup': 145, 'spamd': 109, 'balanc': 96,
         'devel': 81, 'protect': 400, 'financi': 417, 'purchas': 407, 'auto': 105, 'warranti': 105, 'car': 204,
         'troubl': 126, 'happen': 401, 'worst': 71, 'expens': 195, 'onc': 551, 'vehicl': 64, 'mile': 96, 'direct': 318,
         'price': 1018, 'claim': 362, 'plan': 489, 'assist': 295, 'benefit': 296, 'trip': 81, 'interrupt': 32,
         'easili': 251, 'across': 169, 'journal': 132, 'electron': 224, 'publish': 317, 'januari': 66, 'decemb': 63,
         'paper': 209, 'avail': 829, 'onlin': 672, 'appar': 113, 'confer': 137, 'relev': 127, 'directli': 161,
         'valuabl': 137, 'gener': 903, 'top': 547, 'research': 297, 'led': 68, 'technolog': 677, 'among': 126,
         'intellectu': 44, 'especi': 157, 'context': 50, 'document': 357, 'languag': 210, 'develop': 800, 'implement': 190,
         'histori': 189, 'prepar': 133, 'richard': 54, 'excel': 129, 'author': 348, 'engin': 444, 'alsa': 140,
         'mpnumber': 134, 'sound': 279, 'front': 115, 'wed': 221, 'matthia': 232, 'saou': 121, 'stori': 432, 'sub': 52,
         'driver': 199, 'kernel': 371, 'modul': 140, 'doc': 68, 'depend': 236, 'rpm': 889, 'gordon': 47, 'further': 332,
         'spec': 102, 'aren': 123, 'inumb': 221, 'rebuild': 64, 'clean': 192, 'red': 290, 'hat': 228, 'valhalla': 46,
         'ac': 337, 'tip': 333, 'larger': 83, 'path': 180, 'jul': 243, 'apt': 281, 'coupl': 159, 'late': 205, 'night': 147,
         'updat': 491, 'http': 107, 'object': 207, 'appreci': 100, 'icq': 53, 'gnumber': 57, 'resolv': 66, 'resolut': 54,
         'suppos': 128, 'trade': 499, 'abil': 186, 'lift': 55, 'whole': 257, 'societi': 167, 'fridai': 115, 'unit': 605,
         'comfort': 74, 'human': 303, 'race': 55, 'neither': 67, 'nor': 116, 'correct': 156, 'realiz': 99, 'earth': 123,
         'polici': 420, 'bear': 56, 'nation': 576, 'terror': 135, 'insid': 182, 'futur': 615, 'struggl': 51, 'america': 316,
         'surpris': 86, 'radio': 199, 'channel': 89, 'practic': 200, 'near': 142, 'trust': 257, 'past': 386, 'share': 549,
         'broadcast': 182, 'hard': 385, 'term': 402, 'clearli': 81, 'relat': 301, 'act': 365, 'platform': 236, 'africa': 90,
         'aid': 110, 'fear': 70, 'replac': 185, 'sens': 163, 'view': 357, 'presid': 304, 'vast': 58, 'educ': 198,
         'engag': 53, 'heck': 41, 'map': 94, 'carrier': 66, 'hotel': 71, 'librari': 143, 'rest': 158, 'influenc': 46,
         'doesn': 513, 'bush': 221, 'pre': 125, 'strike': 58, 'permit': 82, 'principl': 88, 'loos': 46, 'full': 631,
         'entitl': 44, 'transmit': 50, 'declar': 126, 'centuri': 122, 'enterpris': 169, 'twenti': 74, 'commit': 147,
         'econom': 249, 'potenti': 282, 'assur': 89, 'everywher': 36, 'children': 106, 'male': 142, 'femal': 90,
         'properti': 230, 'enjoi': 134, 'valu': 524, 'against': 421, 'enemi': 91, 'ag': 409, 'posit': 268, 'strength': 74,
         'keep': 615, 'press': 233, 'advantag': 178, 'seek': 170, 'favor': 75, 'condit': 157, 'themselv': 127, 'reward': 53,
         'challeng': 139, 'defend': 83, 'preserv': 56, 'encourag': 100, 'open': 649, 'task': 94, 'dramat': 56,
         'capabl': 167, 'individu': 336, 'bring': 233, 'cost': 626, 'organ': 276, 'turn': 373, 'law': 461, 'enforc': 57,
         'intellig': 108, 'cut': 204, 'financ': 91, 'reach': 233, 'cooper': 133, 'togeth': 186, 'deni': 71, 'li': 54,
         'weapon': 94, 'mass': 193, 'evid': 97, 'determin': 126, 'effort': 245, 'succe': 59, 'deliveri': 123, 'acquir': 118,
         'matter': 330, 'fulli': 152, 'form': 816, 'friend': 472, 'hope': 291, 'safeti': 62, 'histor': 71, 'opportun': 484,
         'commun': 794, 'chanc': 168, 'rise': 57, 'compet': 86, 'ourselv': 53, 'partner': 266, 'chines': 71, 'leader': 105,
         'wealth': 83, 'social': 202, 'both': 532, 'foundat': 81, 'stabil': 57, 'strongli': 61, 'aggress': 57,
         'cultur': 105, 'moment': 164, 'activ': 305, 'corner': 52, 'event': 259, 'strong': 138, 'poor': 146, 'yet': 331,
         'institut': 113, 'vulner': 238, 'drug': 124, 'border': 102, 'besid': 42, 'entir': 286, 'region': 189, 'grow': 274,
         'greater': 92, 'invest': 560, 'diseas': 54, 'guid': 366, 'respons': 506, 'prevent': 149, 'spread': 73, 'wise': 34,
         'spent': 74, 'expect': 333, 'alon': 96, 'perman': 82, 'symbol': 73, 'ideal': 48, 'attain': 44, 'non': 406,
         'throughout': 50, 'threaten': 69, 'west': 85, 'june': 108, 'faith': 40, 'vision': 57, 'equal': 82, 'translat': 68,
         'decad': 86, 'goal': 135, 'progress': 114, 'growth': 230, 'infrastructur': 70, 'center': 342, 'meet': 265,
         'ii': 76, 'somehow': 63, 'circumst': 48, 'father': 127, 'mother': 84, 'await': 35, 'secret': 300, 'speech': 91,
         'privat': 234, 'met': 76, 'constitut': 95, 'serv': 124, 'successfulli': 44, 'core': 132, 'independ': 188,
         'arriv': 96, 'central': 137, 'europ': 106, 'republ': 89, 'elect': 47, 'evolv': 58, 'tradit': 109, 'belief': 41,
         'foreign': 220, 'resourc': 240, 'bodi': 305, 'violat': 64, 'vote': 81, 'ensur': 130, 'toward': 115, 'step': 382,
         'oppos': 46, 'answer': 350, 'rid': 43, 'washington': 127, 'motiv': 80, 'legitim': 57, 'seen': 270, 'hunt': 73,
         'al': 99, 'thousand': 330, 'train': 273, 'north': 113, 'south': 145, 'middl': 99, 'east': 60, 'prioriti': 80,
         'destroi': 50, 'control': 438, 'effect': 451, 'campaign': 98, 'particular': 175, 'necessari': 141, 'finish': 79,
         'block': 188, 'asset': 133, 'abus': 93, 'element': 77, 'harm': 60, 'convinc': 50, 'win': 236, 'behavior': 89,
         'ground': 91, 'risk': 341, 'recogn': 78, 'propos': 210, 'largest': 119, 'comprehens': 84, 'level': 365,
         'sector': 66, 'manag': 789, 'medic': 132, 'improv': 444, 'effici': 108, 'reli': 76, 'beyond': 69, 'capac': 76,
         'pursu': 43, 'neighbor': 57, 'forget': 87, 'ultim': 94, 'quick': 183, 'relationship': 113, 'concern': 224,
         'critic': 100, 'minim': 55, 'restor': 69, 'anticip': 39, 'approach': 119, 'mind': 276, 'count': 172, 'parti': 275,
         'bank': 301, 'establish': 170, 'monitor': 134, 'truli': 74, 'reject': 61, 'ident': 129, 'prior': 57, 'consist': 73,
         'committe': 67, 'resum': 63, 'crucial': 40, 'india': 151, 'gave': 72, 'construct': 85, 'becam': 62, 'choic': 225,
         'earlier': 86, 'took': 140, 'minor': 79, 'emploi': 58, 'western': 68, 'canada': 116, 'integr': 172, 'adjust': 54,
         'economi': 121, 'illeg': 118, 'european': 104, 'environ': 140, 'zone': 44, 'arrang': 57, 'focus': 54,
         'nigeria': 107, 'attent': 130, 'essenti': 80, 'primari': 65, 'basi': 92, 'present': 304, 'rout': 60, 'often': 282,
         'along': 196, 'intent': 33, 'caught': 48, 'cold': 68, 'produc': 207, 'mutual': 38, 'reduct': 58, 'none': 231,
         'complex': 67, 'displai': 262, 'proof': 51, 'acquisit': 52, 'agent': 321, 'partnership': 60, 'innov': 84,
         'collect': 290, 'analysi': 114, 'detect': 94, 'export': 54, 'agreement': 99, 'billion': 260, 'forward': 352,
         'consequ': 59, 'respond': 186, 'desir': 122, 'longer': 203, 'far': 280, 'consid': 296, 'whose': 111, 'convent': 56,
         'superior': 42, 'concept': 99, 'death': 121, 'adapt': 60, 'warn': 208, 'popul': 135, 'demonstr': 52, 'gather': 52,
         'accur': 57, 'conduct': 56, 'precis': 58, 'purpos': 191, 'elimin': 127, 'measur': 117, 'privileg': 50,
         'revenu': 128, 'capit': 201, 'march': 89, 'job': 479, 'higher': 117, 'incom': 345, 'pro': 120, 'lower': 160,
         'margin': 67, 'skill': 75, 'heavi': 62, 'sake': 38, 'structur': 98, 'perform': 271, 'loan': 246, 'regular': 109,
         'seven': 81, 'adopt': 58, 'rais': 119, 'exchang': 160, 'sell': 514, 'launch': 123, 'novemb': 60, 'china': 157,
         'agre': 190, 'prefer': 482, 'ahead': 53, 'mix': 66, 'execut': 291, 'pass': 227, 'conclud': 52, 'under': 532,
         'competit': 130, 'knowledg': 152, 'nearli': 86, 'equival': 101, 'medicin': 37, 'extraordinari': 38, 'commerc': 62,
         'scienc': 156, 'farm': 47, 'dump': 51, 'healthi': 35, 'energi': 165, 'ga': 74, 'concentr': 96, 'overal': 41,
         'rel': 109, 'percent': 118, 'regist': 300, 'conserv': 69, 'total': 383, 'sum': 114, 'million': 851, 'budget': 49,
         'safe': 176, 'massiv': 59, 'wors': 62, 'typic': 70, 'recipi': 103, 'failur': 74, 'imposs': 71, 'signific': 126,
         'doubl': 76, 'project': 342, 'care': 265, 'revers': 99, 'trend': 60, 'contribut': 92, 'grant': 711, 'insist': 34,
         'evalu': 69, 'portion': 65, 'debt': 212, 'scale': 70, 'honest': 66, 'combin': 176, 'twice': 55, 'learn': 410,
         'least': 429, 'yield': 57, 'numberth': 345, 'littl': 431, 'itself': 153, 'carri': 96, 'field': 180, 'highli': 96,
         'membership': 178, 'appropri': 74, 'domin': 53, 'dure': 304, 'perspect': 38, 'pattern': 130, 'shape': 82,
         'realiti': 59, 'euro': 129, 'extens': 108, 'record': 377, 'repres': 132, 'quarter': 89, 'road': 108, 'assembl': 52,
         'annual': 107, 'agenc': 162, 'touch': 83, 'creativ': 80, 'joint': 43, 'session': 59, 'highest': 62, 'presenc': 41,
         'host': 220, 'length': 80, 'remot': 177, 'space': 237, 'inspir': 50, 'proper': 77, 'director': 130,
         'interact': 198, 'investig': 180, 'inquiri': 34, 'crimin': 72, 'complic': 49, 'mechan': 77, 'intend': 200,
         'citi': 343, 'apart': 47, 'uniqu': 160, 'tue': 176, 'tom': 124, 'solicit': 52, 'blood': 70, 'random': 86,
         'viru': 71, 'summari': 90, 'slightli': 60, 'data': 691, 'header': 263, 'fals': 194, 'percentag': 73, 'won': 340,
         'ti': 78, 'went': 178, 'neg': 122, 'tim': 89, 'databas': 367, 'unknown': 48, 'token': 135, 'everybodi': 72,
         'ten': 109, 'second': 403, 'multipart': 89, 'boundari': 110, 'basenumb': 96, 'insur': 431, 'prove': 69,
         'expert': 146, 'luck': 60, 'numberbit': 166, 'worldwid': 101, 'volum': 70, 'distributor': 42, 'fax': 388,
         'black': 186, 'ibm': 128, 'hp': 156, 'white': 143, 'numberf': 159, 'lnumber': 71, 'lcd': 48, 'numbercnumb': 125,
         'pack': 214, 'pnumber': 56, 'panel': 107, 'numberxnumb': 68, 'websit': 464, 'shop': 189, 'compon': 65,
         'reserv': 271, 'img': 41, 'repositori': 79, 'fan': 67, 'bought': 58, 'bai': 56, 'upgrad': 235, 'pleasur': 37,
         'edg': 77, 'usb': 97, 'box': 427, 'room': 101, 'cool': 125, 'fit': 107, 'sold': 131, 'separ': 129, 'advertis': 565,
         'vari': 60, 'kit': 48, 'silent': 38, 'justin': 109, 'mondai': 176, 'ilug': 64, 'god': 80, 'xp': 199, 'cheap': 72,
         'dublin': 53, 'softwar': 1210, 'profession': 519, 'util': 216, 'virus': 44, 'comput': 989, 'third': 159,
         'wish': 532, 'chip': 87, 'vipul': 64, 'te': 61, 'razornumb': 89, 'ebai': 101, 'proprietari': 91, 'comparison': 92,
         'feedback': 86, 'buyer': 74, 'unfortun': 85, 'incred': 80, 'spot': 91, 'ago': 265, 'submit': 207, 'item': 232,
         'due': 198, 'calcul': 80, 'surviv': 63, 'imho': 49, 'cloth': 40, 'argu': 67, 'scheme': 63, 'tend': 74, 'wide': 152,
         'henc': 58, 'famou': 45, 'perfectli': 79, 'crazi': 42, 'craig': 61, 'conclus': 33, 'burn': 158, 'cycl': 44,
         'stupid': 97, 'optim': 67, 'cheer': 146, 'numberpm': 237, 'score': 148, 'ie': 144, 'ved': 33, 'artist': 90,
         'william': 91, 'mac': 164, 'thread': 191, 'excess': 61, 'cnet': 328, 'weekli': 93, 'newslett': 370, 'dell': 138,
         'sat': 84, 'snumber': 93, 'canon': 117, 'cnumber': 1903, 'popular': 223, 'graphic': 140, 'spin': 35, 'ram': 54,
         'perfect': 117, 'favorit': 79, 'game': 286, 'difficult': 103, 'memori': 269, 'lowest': 142, 'numberx': 102,
         'smart': 112, 'pioneer': 74, 'writer': 55, 'interfac': 237, 'profit': 253, 'tech': 422, 'tv': 173, 'faq': 97,
         'review': 389, 'copyright': 374, 'inc': 395, 'gt': 93, 'toner': 248, 'laser': 93, 'printer': 127, 'cartridg': 240,
         'school': 201, 'stock': 348, 'hewlett': 52, 'packard': 50, 'numbera': 404, 'numberb': 667, 'numberc': 1640,
         'numberm': 90, 'es': 82, 'numberd': 1625, 'enumb': 919, 'numberanumb': 84, 'ps': 60, 'imag': 268, 'appl': 152,
         'mnumber': 74, 'nt': 93, 'amp': 334, 'ms': 168, 'satisfact': 44, 'fill': 347, 'zip': 127, 'bill': 299,
         'brand': 112, 'decor': 213, 'font': 562, 'arial': 221, 'helvetica': 147, 'san': 245, 'anytim': 47, 'mortgag': 228,
         'holidai': 39, 'travel': 148, 'book': 489, 'vacat': 78, 'pictur': 146, 'daili': 305, 'subscrib': 420, 'degre': 84,
         'dark': 91, 'width': 143, 'strip': 88, 'visual': 69, 'somewhat': 31, 'sun': 256, 'elsewher': 58, 'recal': 62,
         'hint': 48, 'observ': 76, 'eugen': 51, 'exactli': 197, 'definit': 126, 'throw': 61, 'emot': 44, 'walk': 108,
         'encount': 59, 'although': 175, 'procedur': 145, 'stage': 70, 'disk': 253, 'rom': 123, 'confidenti': 177,
         'entiti': 44, 'notifi': 43, 'necessarili': 41, 'confirm': 151, 'gmt': 59, 'heart': 76, 'laugh': 51, 'weblog': 238,
         'wire': 63, 'pop': 106, 'st': 76, 'announc': 248, 'numberst': 79, 'virtual': 144, 'cabl': 135, 'modem': 90,
         'rare': 55, 'crash': 85, 'hot': 115, 'lifetim': 87, 'tabl': 124, 'stai': 123, 'unless': 148, 'angl': 58,
         'solut': 279, 'attach': 140, 'plug': 86, 'modifi': 158, 'manner': 35, 'browser': 184, 'candid': 61,
         'collector': 52, 'feed': 86, 'refer': 279, 'round': 74, 'certainli': 137, 'flash': 86, 'boot': 193, 'backup': 140,
         'compress': 46, 'storag': 141, 'os': 289, 'blue': 90, 'fresh': 71, 'feet': 69, 'decent': 36, 'mous': 73,
         'locat': 226, 'comp': 52, 'style': 159, 'letter': 342, 'hmm': 53, 'anim': 135, 'repeat': 50, 'woman': 79,
         'finger': 48, 'palm': 107, 'em': 38, 'dozen': 52, 'shall': 64, 'domain': 282, 'oct': 106, 'autom': 68,
         'presum': 47, 'usr': 155, 'freshrpm': 67, 'yeah': 82, 'voyag': 32, 'saturdai': 54, 'everyon': 229, 'sex': 147,
         'msn': 63, 'greet': 33, 'brought': 63, 'gari': 130, 'lawrenc': 134, 'murphi': 110, 'teledynam': 60, 'useless': 91,
         'pablo': 42, 'picasso': 59, 'rob': 54, 'skip': 134, 'directori': 367, 'lib': 170, 'perlnumb': 45, 'perl': 362,
         'archiv': 181, 'startup': 69, 'dave': 116, 'opt': 236, 'chosen': 42, 'inkjet': 62, 'super': 98, 'track': 216,
         'privaci': 181, 'dedic': 37, 'consum': 169, 'media': 387, 'contract': 256, 'coverag': 87, 'cach': 152,
         'randomli': 37, 'sophist': 46, 'bandwidth': 51, 'store': 247, 'pnumberp': 79, 'sa': 153, 'window': 1025,
         'juli': 434, 'broker': 68, 'commiss': 218, 'tm': 300, 'ext': 76, 'honor': 98, 'payment': 195, 'suck': 77,
         'thoma': 38, 'buck': 55, 'ah': 59, 'spain': 78, 'kid': 120, 'ride': 36, 'truth': 105, 'adult': 115, 'survei': 60,
         'paid': 226, 'movi': 161, 'onto': 54, 'porn': 68, 'simpl': 393, 'note': 459, 'banner': 51, 'nb': 44,
         'password': 136, 'en': 230, 'de': 661, 'da': 152, 'ne': 55, 'dear': 214, 'homeown': 46, 'lender': 91, 'refin': 105,
         'equiti': 60, 'lock': 113, 'pocket': 128, 'forev': 109, 'gui': 220, 'hang': 63, 'joke': 47, 'five': 216,
         'spring': 38, 'surround': 37, 'tree': 112, 'tune': 68, 'player': 197, 'consol': 49, 'wednesdai': 115, 'mo': 36,
         'snapshot': 37, 'desktop': 192, 'drop': 200, 'screen': 186, 'background': 218, 'menu': 118, 'dan': 87,
         'googl': 162, 'lack': 75, 'zdnet': 158, 'swap': 52, 'boost': 58, 'team': 192, 'solari': 83, 'park': 54,
         'microsoft': 471, 'steve': 87, 'seriou': 123, 'ceo': 71, 'acknowledg': 81, 'wi': 71, 'fi': 61, 'roll': 87,
         'jump': 69, 'wireless': 234, 'hasn': 45, 'kept': 80, 'lucki': 38, 'robert': 153, 'earli': 137, 'inch': 79,
         'debat': 44, 'java': 262, 'remedi': 82, 'firewal': 123, 'built': 165, 'hole': 65, 'embed': 62, 'queri': 75,
         'ip': 136, 'handi': 41, 'fine': 170, 'print': 257, 'dot': 76, 'pda': 70, 'builder': 46, 'career': 111, 'mine': 87,
         'town': 49, 'relax': 45, 'prospect': 70, 'qualifi': 202, 'scientist': 72, 'defin': 128, 'tomorrow': 58,
         'schedul': 69, 'sincer': 118, 'specialist': 68, 'earn': 295, 'pr': 60, 'tag': 156, 'bnumber': 1995, 'manual': 154,
         'index': 116, 'retriev': 61, 'revis': 67, 'phrase': 74, 'consolid': 83, 'nationwid': 35, 'owner': 166, 'va': 38,
         'beach': 53, 'ca': 403, 'guido': 35, 'inde': 89, 'alan': 45, 'unlik': 68, 'se': 100, 'default': 240, 'certifi': 40,
         'complain': 54, 'rock': 80, 'classic': 52, 'outsid': 117, 'fuck': 49, 'owen': 50, 'remark': 49, 'dev': 132,
         'beta': 100, 'quickli': 139, 'bulk': 157, 'compil': 253, 'blank': 92, 'remind': 66, 'automat': 226,
         'instruct': 370, 'admin': 52, 'cat': 132, 'camera': 142, 'affili': 95, 'glad': 42, 'fastest': 127, 'audienc': 59,
         'awar': 117, 'meant': 64, 'label': 73, 'statist': 51, 'blog': 284, 'biz': 72, 'folk': 145, 'goe': 167,
         'termin': 75, 'signal': 71, 'script': 219, 'somewher': 67, 'exit': 65, 'ran': 60, 'reboot': 53, 'hospit': 53,
         'dn': 86, 'gnome': 144, 'exact': 64, 'mirror': 66, 'usag': 80, 'physic': 111, 'edificio': 43, 'nort': 43,
         'planta': 91, 'barcelona': 43, 'programm': 138, 'titl': 160, 'xml': 227, 'python': 86, 'php': 155, 'numbernd': 51,
         'draw': 75, 'interview': 88, 'paul': 94, 'express': 190, 'topic': 77, 'fly': 42, 'gold': 99, 'confus': 97,
         'aspect': 63, 'attract': 100, 'written': 156, 'experienc': 52, 'technic': 132, 'four': 194, 'categori': 109,
         'award': 113, 'hash': 47, 'advic': 93, 'wave': 53, 'blow': 34, 'er': 47, 'bitbitch': 34, 'ban': 78, 'damag': 71,
         'deploi': 39, 'song': 75, 'mere': 55, 'audio': 136, 'hacker': 58, 'hidden': 62, 'extra': 166, 'cash': 412,
         'easiest': 49, 'insert': 64, 'po': 52, 'et': 149, 'ng': 43, 'fri': 163, 'niall': 41, 'strictli': 49, 'impli': 36,
         'wouldn': 132, 'lie': 40, 'processor': 81, 'interpret': 50, 'app': 130, 'kevin': 103, 'happier': 55, 'rh': 85,
         'prescript': 58, 'ey': 106, 'visa': 71, 'firm': 137, 'student': 82, 'sale': 475, 'transact': 264, 'dead': 108,
         'satellit': 113, 'introduc': 135, 'gatewai': 70, 'faster': 132, 'intel': 105, 'encrypt': 79, 'cio': 50,
         'biggest': 72, 'jack': 53, 'revok': 71, 'algorithm': 56, 'spammer': 133, 'id': 322, 'stream': 127, 'aol': 124,
         'proxi': 68, 'dice': 45, 'investor': 219, 'wall': 150, 'analyst': 84, 'advisor': 56, 'bar': 188, 'dream': 172,
         'ugli': 42, 'excit': 95, 'ticket': 88, 'junk': 52, 'root': 307, 'bell': 62, 'van': 68, 'numberam': 151, 'hair': 74,
         'kate': 56, 'repair': 51, 'colleg': 105, 'child': 63, 'employe': 108, 'april': 44, 'hack': 82, 'estim': 86,
         'thursdai': 111, 'draft': 45, 'giant': 44, 'chicago': 55, 'summer': 99, 'correspond': 81, 'sweet': 106, 'soni': 94,
         'stick': 67, 'clue': 82, 'pipe': 107, 'smoke': 188, 'extract': 97, 'plant': 97, 'substanti': 103, 'planet': 70,
         'depress': 41, 'sexual': 108, 'sleep': 85, 'ratio': 100, 'factor': 120, 'master': 83, 'solid': 80, 'hous': 221,
         'oil': 55, 'seed': 68, 'whom': 59, 'knew': 85, 'muscl': 76, 'cheaper': 50, 'sampl': 109, 'unabl': 43, 'boi': 80,
         'liter': 51, 'prohibit': 68, 'tast': 49, 'bless': 46, 'drink': 40, 'satisfi': 47, 'inexpens': 42, 'art': 99,
         'fantasi': 58, 'accord': 193, 'bone': 42, 'significantli': 36, 'instanc': 74, 'maximum': 58, 'reg': 106,
         'pure': 52, 'deep': 67, 'gift': 104, 'conveni': 113, 'weekend': 68, 'render': 58, 'fairli': 60, 'trick': 103,
         'dont': 70, 'gotten': 34, 'fortun': 123, 'numberdnumb': 407, 'numberenumb': 117, 'numberrd': 37, 'california': 95,
         'notif': 32, 'uk': 128, 'anti': 110, 'tuesdai': 250, 'entertain': 97, 'catalog': 68, 'maker': 70, 'ventur': 157,
         'televis': 56, 'vice': 55, 'deposit': 88, 'corp': 52, 'affect': 87, 'appeal': 61, 'gai': 63, 'morn': 84,
         'compat': 128, 'loonei': 32, 'myself': 107, 'whitelist': 88, 'latter': 37, 'bunch': 54, 'joseph': 44, 'babi': 81,
         'src': 93, 'brain': 98, 'bin': 87, 'georg': 74, 'duplic': 46, 'im': 120, 'tape': 81, 'scan': 117, 'toni': 73,
         'la': 205, 'known': 229, 'greg': 65, 'profil': 184, 'michael': 135, 'octob': 99, 'son': 75, 'shock': 53,
         'somebodi': 62, 'currenc': 57, 'estat': 129, 'cross': 85, 'lai': 48, 'anonym': 69, 'nobodi': 82, 'statement': 246,
         'settl': 38, 'mid': 62, 'felt': 48, 'grand': 87, 'headlin': 82, 'smaller': 55, 'rich': 119, 'texa': 71, 'wear': 43,
         'flaw': 52, 'annoi': 60, 'assumpt': 44, 'foot': 60, 'orient': 43, 'rush': 35, 'hadn': 35, 'opposit': 38,
         'king': 143, 'couldn': 104, 'push': 56, 'theori': 76, 'virginia': 47, 'attornei': 100, 'parent': 92, 'crime': 63,
         'discoveri': 99, 'ill': 60, 'roger': 53, 'ex': 65, 'ow': 48, 'frequent': 55, 'relai': 40, 'convers': 98,
         'expos': 43, 'shown': 62, 'rent': 47, 'despit': 78, 'dealer': 36, 'penni': 38, 'victim': 40, 'french': 107,
         'binari': 95, 'broken': 54, 'silli': 41, 'hurt': 34, 'patent': 75, 'wrinkl': 41, 'receipt': 57, 'urgent': 71,
         'okai': 36, 'dog': 63, 'lo': 98, 'season': 74, 'particip': 158, 'pudg': 73, 'btw': 65, 'mozilla': 75, 'curiou': 34,
         'correctli': 66, 'club': 63, 'convert': 80, 'strang': 55, 'plugin': 51, 'ham': 163, 'francisco': 57, 'island': 327,
         'employ': 101, 'winner': 108, 'invent': 54, 'techniqu': 125, 'teach': 82, 'approxim': 58, 'frame': 66,
         'disappear': 58, 'hire': 76, 'partit': 90, 'mason': 46, 'conf': 82, 'dig': 40, 'boss': 74, 'variabl': 81,
         'formula': 60, 'alter': 71, 'editor': 234, 'null': 80, 'sir': 71, 'consider': 66, 'rose': 51, 'delai': 94,
         'alert': 59, 'mistak': 83, 'loop': 51, 'wipe': 36, 'mailbox': 72, 'procmail': 90, 'certain': 155, 'harlei': 49,
         'cc': 308, 'architectur': 77, 'unwant': 42, 'reliabl': 101, 'disclaim': 54, 'ebook': 61, 'examin': 45,
         'carefulli': 57, 'rank': 59, 'trademark': 60, 'disposit': 121, 'inlin': 99, 'shouldn': 49, 'tremend': 47,
         'rss': 92, 'viagra': 62, 'occasion': 43, 'anthoni': 41, 'angel': 45, 'varieti': 81, 'star': 105, 'film': 119,
         'martin': 37, 'zero': 91, 'tel': 68, 'vendor': 114, 'regardless': 46, 'franc': 86, 'kick': 38, 'ass': 39,
         'hell': 60, 'mlm': 75, 'properli': 82, 'admit': 42, 'shell': 82, 'pilot': 52, 'combo': 32, 'llc': 36, 'ben': 40,
         'green': 69, 'il': 49, 'registr': 73, 'classifi': 79, 'wrap': 36, 'postal': 68, 'fl': 59, 'guidelin': 66, 'pa': 54,
         'ny': 36, 'transmiss': 55, 'warm': 72, 'retain': 50, 'explor': 101, 'peter': 74, 'mostli': 70, 'welch': 35,
         'var': 110, 'el': 110, 'brows': 55, 'mastercard': 47, 'int': 134, 'steal': 44, 'theft': 43, 'compens': 62,
         'lab': 41, 'le': 164, 'brother': 67, 'entrepreneur': 67, 'food': 67, 'stephen': 41, 'output': 90, 'weird': 39,
         'su': 48, 'florida': 50, 'instant': 102, 'rapidli': 40, 'older': 59, 'chief': 116, 'filenam': 53, 'england': 52,
         'cite': 41, 'brief': 61, 'est': 54, 'ct': 35, 'raw': 40, 'whatsoev': 38, 'ing': 40, 'txt': 48, 'smtp': 94,
         'magazin': 77, 'bb': 501, 'congratul': 47, 'expir': 82, 'exclud': 59, 'xnumber': 72, 'damn': 53, 'dnumber': 1756,
         'complianc': 52, 'py': 63, 'fnumber': 883, 'ce': 230, 'th': 57, 'specifi': 92, 'eas': 41, 'numer': 46, 'navig': 52,
         'premium': 113, 'colleagu': 58, 'outstand': 47, 'usdollarnumb': 78, 'friendli': 50, 'exce': 43, 'electr': 53,
         'shift': 45, 'authent': 59, 'bet': 34}

    word_score = list()
    for idx, word in enumerate(vocab_map):
        word_score.append([word, model.w[idx]])

    word_score.sort(key=lambda x:x[1])
    positive=word_score[:15]
    negative=word_score[-15:]
    positive = [word[0] for word in positive]
    negative = [word[0] for word in negative]
    print("Top 15 Positive Weights: ", positive)
    print("Top 15 Negative Weights: ", negative)


if __name__ == "__main__":
    main()