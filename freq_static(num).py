import json
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def sorted_dict(data, reverse=False):
    """返回 keys 的列表,根据container中对应的值排序"""
    aux = [(data[k], k) for k in data.keys()]
    aux.sort()
    if reverse:
        aux.reverse()
    return [(k, v) for v, k in aux]


def get_rel_frequency_from_file(dataset):
    """
    统计数据集中句子关系数的频数。按照句子中关系的数量，将关系分为[1,2,3,4,>=5]共5个等级，统计所有等级的频数。

    :param dataset: 数据集文件名
    :return: 直接在控制台输出信息，无返回结果
    """
    with open(dataset) as f:
        # 读取json
        content = json.load(f)
        rels = content[-1]  # json文件中的第二大块（关系集）
        print('--------- ', dataset, ' ---------')
        print('relation count:', len(rels))
        # 统计关系最长长度
        max_len = 0
        for rel in rels:
            if max_len < int(len(rel) / 3):
                max_len = int(len(rel) / 3)
        print('max relation length:', max_len)
        # 为每个长度的关系统计频数
        freq = [0 for _ in range(max_len + 1)]  # 关系数最大为max_len
        for rel in rels:
            freq[int(len(rel) / 3)] += 1
        # print(freq)
        # 写成最多5个的形式
        freq_adj = freq[1:5]
        freq_adj.append(sum(freq[5:]))
        print(freq_adj)
        # x_values = list(range(1, 6))
        # plt.bar(x_values, freq_adj)
        # x_major_locator = MultipleLocator(1)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # plt.xlim(0.5, 5.5)
        # plt.show()


def count_seo_frequency(word_freq, rels, ids):
    """
    重叠实体频数统计

    :param word_freq: 重叠实体频数统计字典
    :param rels: 关系集
    :param ids: 句中实体标识符集
    :return: 返回更新后的word_freq和存在重叠实体的句子编号
    """
    word_freq = copy.deepcopy(word_freq)
    sentences = set()
    original_word_set = set()  # 数据集中的不同实体总数（包含有重叠关系的和没有重叠关系的）
    # 重叠实体统计
    for k in range(len(rels)):
        # 取出一个句子中所有实体对
        pairs = set([(rels[k][i], rels[k][i + 1]) for i in range(0, len(rels[k]), 3)])  # 先把重复的EPO去除
        heads = [pair[0] for pair in pairs]
        tails = [pair[1] for pair in pairs]
        # 生成一个句子中重复的实体的字典统计频数
        word_set = set()
        for i in range(len(heads)):
            # 如果在其他实体对中的头尾实体包含此实体，表明重叠
            if heads[i] in heads[:i] or heads[i] in heads[i+1:] or heads[i] in tails[:i] or heads[i] in tails[i+1:]:
                word_set.add(heads[i])
            if tails[i] in heads[:i] or tails[i] in heads[i+1:] or tails[i] in tails[:i] or tails[i] in tails[i+1:]:
                word_set.add(tails[i])
        # 找到重复实体对应的标识符
        for word in word_set:
            # 加入频数统计哈希表
            identity = ids[k][word]
            if identity not in word_freq:
                word_freq[identity] = 1
            else:
                word_freq[identity] += 1
            # 将此句标记为存在重叠
            sentences.add(k)
        # 将所有实体加入original_word_set
        for word in heads:
            identity = ids[k][word]
            original_word_set.add(identity)
        for word in tails:
            identity = ids[k][word]
            original_word_set.add(identity)
    return word_freq, sentences, original_word_set


def count_epo_frequency(pair_freq, rels, ids):
    """
    重叠实体对频数统计

    :param pair_freq: 重叠实体对频数统计字典
    :param rels: 关系集
    :param ids: 句中实体标识符集
    :return: 返回更新后的pair_freq和存在重叠实体对的句子编号
    """
    pair_freq = copy.deepcopy(pair_freq)
    sentences = set()
    original_pair_set = set()  # 数据集中的不同实体总数（包含有重叠关系的和没有重叠关系的）
    # 重叠实体统计
    for k in range(len(rels)):
        # 取出一个句子中所有实体对
        pairs = [(rels[k][i], rels[k][i+1]) for i in range(0, len(rels[k]), 3)]
        # 生成一个句子中重复实体对的字典统计频数
        pair_set = set()
        for i in range(len(pairs)):
            if pairs[i] in pairs[:i] or pairs[i] in pairs[i+1:]:
                pair_set.add(pairs[i])
        # 找到重复实体对对应的标识符
        for pair in pair_set:
            # 加入频数统计哈希表
            identity = (ids[k][pair[0]], ids[k][pair[1]])
            if identity not in pair_freq:
                pair_freq[identity] = 1
            else:
                pair_freq[identity] += 1
            # 将此句标记为存在重叠
            sentences.add(k)
        # 将所有实体加入original_word_set
        for pair in pairs:
            identity = (ids[k][pair[0]], ids[k][pair[1]])
            original_pair_set.add(identity)
    return pair_freq, sentences, original_pair_set


def get_seo_frequency_from_file(dataset, words2id, number_of_interest):
    """
    统计数据集中重叠实体出现的频数。按照最频繁重叠的实体排序，输出前number_of_interest个实体和频数。

    :param dataset: 数据集文件名
    :param words2id: 实体标识符映射文件
    :param number_of_interest: 需要查看的频数统计信息数量
    :return: 返回按频数排序的五个数组，分别为词名、标识符、频数、频率、存在重叠的句子，并直接在控制台输出前number_of_interest个
    """
    with open(dataset) as f:
        content = json.load(f)
        ids = content[-2]  # 句中实体标识符集
        rels = content[-1]  # 关系集
        word_freq = {}  # 重叠实体频数统计字典
        sentence_list = set()
        # 重叠实体统计
        word_freq, sentences, original_word_set = count_seo_frequency(word_freq, rels, ids)
        sentence_list.update(sentences)
        word_freq = sorted_dict(word_freq, True)
        # 实体标识符映射文件导入
        with open(words2id) as fmap:
            mapping = json.load(fmap)
            word_map = {}
            for p in mapping.items():
                word_map[p[1]] = p[0]
            # 形成三个数组，分别为词名、标识符、频数
            word_list = [word_map[item[0]] for item in word_freq]
            id_list = [item[0] for item in word_freq]
            freq_list = [item[1] for item in word_freq]
            percent_list = [float(i/sum(freq_list)) for i in freq_list]
        # 输出前number_of_interest个实体和频数
        print('========== SEO', dataset, ' ==========')
        print('total sentence count:', len(rels))
        print('overlapping sentence count:', len(sentence_list))
        print('total word count:', len(original_word_set))
        print('overlapping word count:', len(word_freq))
        print('word:     ', word_list[:number_of_interest])
        print('identity: ', id_list[:number_of_interest])
        print('frequency:', freq_list[:number_of_interest])
        print('percent:  ', percent_list[:number_of_interest])
        return word_list, id_list, freq_list, percent_list, sentence_list


def get_epo_frequency_from_file(dataset, words2id, number_of_interest):
    """
    统计数据集中重叠实体对出现的频数。按照最频繁重叠的实体对排序，输出前number_of_interest个实体和频数。

    :param dataset: 数据集文件名
    :param words2id: 实体标识符映射文件
    :param number_of_interest: 需要查看的频数统计信息数量
    :return: 返回按频数排序的四个数组，分别为实体对名、标识符、频数、频率，并直接在控制台输出前number_of_interest个
    """
    with open(dataset) as f:
        content = json.load(f)
        ids = content[-2]  # 句中实体标识符集
        rels = content[-1]  # 关系集
        pair_freq = {}  # 重叠实体频数统计字典
        sentence_list = set()
        # 重叠实体对统计
        pair_freq, sentences, original_pair_set = count_epo_frequency(pair_freq, rels, ids)
        sentence_list.update(sentences)
        pair_freq = sorted_dict(pair_freq, True)
        # 实体标识符映射文件导入
        with open(words2id) as fmap:
            mapping = json.load(fmap)
            word_map = {}
            for p in mapping.items():
                word_map[p[1]] = p[0]
            # 形成三个数组，分别为词名、标识符、频数
            pair_list = [(word_map[item[0][0]], word_map[item[0][1]]) for item in pair_freq]
            id_list = [(item[0][0], item[0][1]) for item in pair_freq]
            freq_list = [item[1] for item in pair_freq]
            percent_list = [float(i/sum(freq_list)) for i in freq_list]
        # 输出前number_of_interest个实体和频数
        print('========== EPO', dataset, ' ==========')
        print('total sentence count:', len(rels))
        print('overlapping sentence count:', len(sentence_list))
        print('total pair count:', len(original_pair_set))
        print('overlapping pair count:', len(pair_freq))
        print('entity pair:', pair_list[:number_of_interest])
        print('identity:   ', id_list[:number_of_interest])
        print('frequency:  ', freq_list[:number_of_interest])
        print('percent:    ', percent_list[:number_of_interest])
        return pair_list, id_list, freq_list, percent_list, sentence_list


if __name__ == "__main__":
    get_rel_frequency_from_file('/home/htt/Desktop/NYT-Multi/raw_data/original data/train.json')
    get_rel_frequency_from_file('/home/htt/Desktop/NYT-Multi/raw_data/original data/test.json')

    number_of_interest = 11
    get_seo_frequency_from_file('/home/htt/Desktop/NYT-Multi/raw_data/original data/train.json', '/home/htt/Desktop/NYT-Multi/raw_data/original data/words2id.json', number_of_interest)

    print()
    get_epo_frequency_from_file('/home/htt/Desktop/NYT-Multi/raw_data/original data/train.json', '/home/htt/Desktop/NYT-Multi/raw_data/original data/words2id.json', number_of_interest)