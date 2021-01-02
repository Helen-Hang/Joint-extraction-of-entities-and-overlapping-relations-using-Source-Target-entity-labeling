
import json
import codecs
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import LayerNormalization
from bert4keras.tokenizer import Tokenizer
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam, ExponentialMovingAverage
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import *
from keras.models import Model
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

maxlen = 100
batch_size =16
config_path = './pretrained_model/cased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './pretrained_model/cased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './pretrained_model/cased_L-12_H-768_A-12/vocab.txt'



def load_data(filename):
    D = []
    with codecs.open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'sentText': l['sentText'],
                'relationMentions': [
                    (spo['em1Text'], spo['label'], spo['em2Text'])
                    for spo in l['relationMentions']
                ]
            })
    return D


# 加载数据集
train_data = load_data('./raw_data/webnlg/train.json')
valid_data = load_data('./raw_data/webnlg/valid.json')
test_data = load_data('./raw_data/webnlg/test.json')
predicate2id, id2predicate = {}, {}

with codecs.open('./raw_data/webnlg/all_246_schemas.json', encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=False)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for i in idxs:
            d = self.data[i]
            #直接对应到原始序列，在原始序列中切片输出
            token_ids, segment_ids = tokenizer.encode(d['sentText'], max_length=maxlen)
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['relationMentions']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(batch_subject_labels, padding=np.zeros(2))
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_object_labels, padding=np.zeros((len(predicate2id), 2)))
                    yield [
                        batch_token_ids, batch_segment_ids,
                        batch_subject_labels, batch_subject_ids, batch_object_labels
                    ], None
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


def extrac_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    subject_ids = K.cast(subject_ids, 'int32')
    #支持对张量的批量索引,根据索引获取数组特定下标元素的任务
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]


# 补充输入
subject_labels = Input(shape=(None, 2), name='Subject-Labels')
subject_ids = Input(shape=(2, ), name='Subject-Ids')
object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels')

# 加载预训练模型
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

# 预测subject
output = Dense(units=2,
               activation='sigmoid',
               kernel_initializer=bert.initializer)(bert.model.output)
subject_preds = Lambda(lambda x: x**2)(output)

subject_model = Model(bert.model.inputs, subject_preds)

# 传入subject，预测object
# 通过Conditional Layer Normalization将subject融入到object的预测中
output = bert.model.layers[-2].get_output_at(-1)
subject = Lambda(extrac_subject)([output, subject_ids])
output = LayerNormalization(conditional=True)([output, subject])
output = Dense(units=len(predicate2id) * 2,
               activation='sigmoid',
               kernel_initializer=bert.initializer)(output)
output = Reshape((-1, len(predicate2id), 2))(output)
object_preds = Lambda(lambda x: x**4)(output)

object_model = Model(bert.model.inputs + [subject_ids], object_preds)

# 训练模型
train_model = Model(bert.model.inputs + [subject_labels, subject_ids, object_labels],
                    [subject_preds, object_preds])

mask = bert.model.get_layer('Sequence-Mask').output_mask

subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
subject_loss = K.mean(subject_loss, 2)
subject_loss = K.sum(subject_loss * mask) / K.sum(mask)

object_loss = K.binary_crossentropy(object_labels, object_preds)
object_loss = K.sum(K.mean(object_loss, 3), 2)
object_loss = K.sum(object_loss * mask) / K.sum(mask)

train_model.add_loss(subject_loss + object_loss)
train_model.compile(optimizer=Adam(2e-5))


def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, max_length=maxlen)
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    # 抽取subject
    subject_preds = subject_model.predict([[token_ids], [segment_ids]])
    start = np.where(subject_preds[0, :, 0] > 0.4)[0]
    end = np.where(subject_preds[0, :, 1] > 0.4)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            if (j-i<12):
                subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat([token_ids], len(subjects), 0)
        segment_ids = np.repeat([segment_ids], len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        for subject, object_pred in zip(subjects, object_preds):
            #输出满足条件 (即非0) 元素的坐标
            start = np.where(object_pred[:, :, 0] > 0.4)
            end = np.where(object_pred[:, :, 1] > 0.4)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append((subject, predicate1, (_start, _end)))
                        break
        return [
            (
                tokenizer.decode(token_ids[0, s[0]:s[1] + 1], tokens[s[0]:s[1] + 1]),
                id2predicate[p],
                tokenizer.decode(token_ids[0, o[0]:o[1] + 1], tokens[o[0]:o[1] + 1])
            ) for s, p, o in spoes
        ]
    else:
        return []


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z, K = 1e-10, 1e-10, 1e-10, 1e-10

    f = codecs.open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['sentText'])])
        T = set([SPO(spo) for spo in d['relationMentions']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        K += np.max([len(R),len(T)])
        f1, precision, recall, acc = 2 * X / (Y + Z), X / Y, X / Z, X / K
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f, acc: %.5f' %
                             (f1, precision, recall, acc))
        #对数据进行编码
        s = json.dumps(
            {
                'sentText': d['sentText'],
                'relationMentions': list(T),
                'relationMentions_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            },
            ensure_ascii=False,
            indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall, acc

def evaluate_mse_acc(data):
    mse=[]
    acc=[]
    for d in data:
        X, Y, Z, K = 1e-10, 1e-10, 1e-10, 1e-10
        R = set([SPO(spo) for spo in extract_spoes(d['sentText'])])
        T = set([SPO(spo) for spo in d['relationMentions']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        K = np.max([Y,Z])
        acc_ = X / K
        mse_ = (np.max([Y,Z])-X)**2
        mse.append(mse_)
        acc.append(acc_)
    return np.mean(mse),np.mean(acc)

class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        val_f1, val_precision, val_recall, val_acc = evaluate(valid_data)
        if val_f1 >= self.best_val_f1:
            self.best_val_f1 = val_f1
            train_model.save_weights('best_model.weights')
        EMAer.reset_old_weights()
        print('val_f1: %.5f, val_precision: %.5f, val_recall: %.5f, best_val_f1: %.5f, val_acc: %.5f\n' % (val_f1, val_precision, val_recall, self.best_val_f1, val_acc))


if __name__ == '__main__':

    # train_mse = []
    # test_mse = []
    # train_acc_ave = []
    # test_acc_ave = []
    # train_acc_sum = []
    # test_acc_sum = []
    #
    # # starttime = datetime.datetime.now()
    #
    # for i in range(1001, 5019, 1000):
    #     train_generator = data_generator(train_data[:i], batch_size)
    #     valid_generator = data_generator(valid_data, batch_size)
    #     test_generator = data_generator(test_data, batch_size)
    #     evaluator = Evaluator()
    #
    #     EMAer = ExponentialMovingAverage(0.999)
    #     early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='min')
    #
    #     history = train_model.fit_generator(train_generator.forfit(),
    #                                         steps_per_epoch=len(train_generator),
    #                                         epochs=20,
    #                                         validation_data=valid_generator.forfit(),
    #                                         validation_steps=len(valid_generator),
    #                                         callbacks=[evaluator, EMAer])
    #
    #     mse, acc_ave = evaluate_mse_acc(train_data[:i])
    #     val_mse, val_acc_ave = evaluate_mse_acc(valid_data)
    #     f1, precision, recall, acc_sum = evaluate(train_data[:i])
    #     val_f1, val_precision, val_recall, val_acc_sum = evaluate(valid_data)
    #
    #     train_mse.append(mse)
    #     test_mse.append(val_mse)
    #     train_acc_ave.append(acc_ave)
    #     test_acc_ave.append(val_acc_ave)
    #     train_acc_sum.append(acc_sum)
    #     test_acc_sum.append(val_acc_sum)
    #
    # print(train_mse, test_mse, train_acc_sum, test_acc_sum, train_acc_ave, test_acc_ave)


    # train_mse=[1.2267732267732268, 0.42128935532233885, 0.15361546151282907, 0.07273181704573857, 0.04579084183163367]
    # test_mse=[1.89, 0.97, 0.686, 0.482, 0.382]
    # train_acc_ave=[0.7942441283493287, 0.923814011938424, 0.9582149621212126, 0.9754404145077723, 0.9817324185248715]
    # test_acc_ave=[0.6112430720507038, 0.7619047619047812, 0.8323793949305125, 0.8679562657695653, 0.8901384083045077]
    # train_acc_sum=[0.8403501260689651, 0.9474998611818194, 0.9703269545081175, 0.9784717511110534, 0.9806217327975599]
    # test_acc_sum=[0.5887500000255165, 0.7436333333495302, 0.8425214285798291, 0.8678000000074912, 0.8885666666733535]

    # plt.plot([i for i in range(1001, 5019, 1000)], train_mse, label='train')
    # plt.plot([i for i in range(1001, 5019, 1000)], test_mse, label='test')
    # plt.xlabel('Number of training samples', fontsize=16)
    # plt.ylabel('RMSE', fontsize=16)
    # plt.ylim(0,2)
    # plt.yticks(np.linspace(0,2,5))
    # plt.title(r'Learning curve on the WebNLG dataset', fontsize=16)
    # plt.legend()
    # plt.savefig("learning curves1.png")
    # plt.show()

    # plt.plot([i for i in range(1001, 5019, 1000)], train_acc_sum, label='train')
    # plt.plot([i for i in range(1001, 5019, 1000)], test_acc_sum, label='test')
    # plt.xlabel('Number of training samples', fontsize=16)
    # plt.ylabel('Accuracy_sum', fontsize=16)
    # plt.ylim(0, 1)
    # plt.title(r'Learning curve on the WebNLG dataset', fontsize=16)
    # plt.legend()
    # plt.savefig("learning curves2.png")
    # plt.show()
    #
    # plt.plot([i for i in range(1001, 5019, 1000)], train_acc_ave, label='train')
    # plt.plot([i for i in range(1001, 5019, 1000)], test_acc_ave, label='test')
    # plt.xlabel('Number of training samples', fontsize=16)
    # plt.ylabel('Accuracy_ave', fontsize=16)
    # plt.ylim(0, 1)
    # plt.title(r'Learning curve on the WebNLG dataset', fontsize=16)
    # plt.legend()
    # plt.savefig("learning curves3.png")
    # plt.show()

    # epochs = len(history.history['loss'])
    # plt.plot(range(0, epochs, 1), history.history['loss'], label='train_loss')
    # plt.plot(range(0, epochs, 1), history.history['val_loss'], label='val_loss')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.savefig("loss.png")
    # plt.show()

    # train_model.load_weights('best_model.weights')
    # print(u'final test f1: %.5f, precision: %.5f, recall: %.5f\n' % (evaluate(test_data)))
    #
    # endtime = datetime.datetime.now()
    # print(endtime - starttime)


