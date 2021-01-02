import json


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
    :return: 返回关系数的频数/频率统计结果，从0开始
    """
    # 取出所有样本
    with open(dataset) as f:
        samples = f.readlines()
        if samples[-1] == '':
            samples = samples[:-1]
    assert len(samples) > 0
    # 统计关系数
    rel_freq = dict()
    count_triples = 0
    for sample in samples:
        freq = len(json.loads(sample)['relationMentions'])
        count_triples += freq
        if freq not in rel_freq:
            rel_freq[freq] = 1
        else:
            rel_freq[freq] += 1
    # 转换成数组
    res_rel = [0 for _ in range(5 + 1)]
    for k,v in rel_freq.items():
        if k >= 5:
            res_rel[5] += v
        else:
            res_rel[k] += v
    res_ratio = [v/sum(res_rel[1:]) for v in res_rel[1:]]
    print('-------------------- 关系频数统计:', dataset, ' --------------------')
    print('>>>>>>>>>> 共', count_triples, '个三元组，其中关系有[1,2,3,4,>=5]的频数/频率为', res_rel[1:], res_ratio)
    return res_rel[1:], res_ratio


def get_seo_frequency_from_file(dataset, number_of_interest):
    """
    统计数据集中重叠实体出现的频数。按照最频繁重叠的实体排序，输出前number_of_interest个实体和频数。

    :param dataset: 数据集文件名
    :param number_of_interest: 需要查看的频数统计信息数量
    :return: 返回按频数排序的四个数组，分别为词名、频数、频率、存在重叠的句子编号，并直接在控制台输出前number_of_interest个
    """
    # 取出所有样本
    with open(dataset) as f:
        samples = f.readlines()
        if samples[-1] == '':
            samples = samples[:-1]
    assert len(samples) > 0
    # 对每个样本进行统计
    sent_list = set()  # 存在重叠关系的句子编号
    word_freq = {}  # 重叠实体频数
    word_orig = set()  # 实体统计
    for sample_id, sample in enumerate(samples):
        triples = json.loads(sample)['relationMentions']
        # 获取去重后的实体对（去除epo）
        pairs = set([(s['em1Text'], s['em2Text']) for s in triples])
        # 获取实体
        words = []
        for pair in pairs:
            words += list(set(pair))
        # 生成一个句子中重复的实体的字典统计频数
        word_dict = {}
        for word in words:
            # 记录所有的实体
            word_orig.add(word)
            # 统计存在重叠的词频
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 0
        # 加入重复实体统计字典
        for word, freq in word_dict.items():
            if freq > 0:
                # 将存在重叠的实体加入
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
                # 将重叠句加入统计字典
                sent_list.add(sample_id)
    # 词频排序
    word_freq = sorted_dict(word_freq, reverse=True)
    word_list = [item[0] for item in word_freq]
    freq_list = [item[1] for item in word_freq]
    ratio_list = [float(i/sum(freq_list)) for i in freq_list]
    # 打印输出
    print('-------------------- 重叠实体统计:', dataset, ' --------------------')
    print('句子总数:', len(samples))
    print('重叠句子数:', len(sent_list))
    print('实体总数:', len(word_orig))
    print('重叠实体数:', len(word_freq))
    print('前', number_of_interest, '个实体名称:', word_list[:number_of_interest])
    print('前', number_of_interest, '个实体频数:', freq_list[:number_of_interest])
    print('前', number_of_interest, '个实体频率:  ', ratio_list[:number_of_interest])
    return word_list, freq_list, ratio_list, sent_list


def get_epo_frequency_from_file(dataset, number_of_interest):
    """
    统计数据集中重叠实体对出现的频数。按照最频繁重叠的实体对排序，输出前number_of_interest个实体和频数。

    :param dataset: 数据集文件名
    :param number_of_interest: 需要查看的频数统计信息数量
    :return: 返回按频数排序的四个数组，分别为实体对名、频数、频率、存在重叠的句子编号，并直接在控制台输出前number_of_interest个
    """
    # 取出所有样本
    with open(dataset) as f:
        samples = f.readlines()
        if samples[-1] == '':
            samples = samples[:-1]
    assert len(samples) > 0
    # 对每个样本进行统计
    sent_list = set()  # 存在重叠关系的句子编号
    pair_freq = {}  # 重叠实体频数
    pair_orig = set()  # 实体统计
    for sample_id, sample in enumerate(samples):
        triples = json.loads(sample)['relationMentions']
        # 获取实体对
        pairs = [(s['em1Text'], s['em2Text']) for s in triples]
        # 生成一个句子中重复的实体对的字典统计频数
        pair_dict = {}
        for pair in pairs:
            # 记录所有的实体对
            pair_orig.add(pair)
            # 统计存在重叠的实体对数量
            if pair in pair_dict:
                pair_dict[pair] += 1
            else:
                pair_dict[pair] = 0
        # 加入重复实体对统计字典
        for pair, freq in pair_dict.items():
            if freq > 0:
                # 将存在重叠的实体对加入
                if pair in pair_freq:
                    pair_freq[pair] += 1
                else:
                    pair_freq[pair] = 1
                # 将重叠句加入统计字典
                sent_list.add(sample_id)
    # 词频排序
    pair_freq = sorted_dict(pair_freq, reverse=True)
    pair_list = [item[0] for item in pair_freq]
    freq_list = [item[1] for item in pair_freq]
    ratio_list = [float(i/sum(freq_list)) for i in freq_list]
    # 打印输出
    print('-------------------- 重叠实体对统计:', dataset, ' --------------------')
    print('句子总数:', len(samples))
    print('重叠句子数:', len(sent_list))
    print('实体对总数:', len(pair_orig))
    print('重叠实体对数:', len(pair_freq))
    print('前', number_of_interest, '个实体对名称:', pair_list[:number_of_interest])
    print('前', number_of_interest, '个实体对频数:', freq_list[:number_of_interest])
    print('前', number_of_interest, '个实体对频率:  ', ratio_list[:number_of_interest])
    return pair_list, freq_list, ratio_list, sent_list


if __name__ == "__main__":
    #fig10
    get_rel_frequency_from_file('/home/htt/Desktop/NYT-Multi/raw_data/nyt1/output/train.json')
    get_rel_frequency_from_file('/home/htt/Desktop/NYT-Multi/raw_data/nyt1/output/test.json')
    
    #fig11
    get_epo_frequency_from_file('/home/htt/Desktop/NYT-Multi/raw_data/nyt1/output/train.json', number_of_interest=5)
    
    #fig12
    get_seo_frequency_from_file('/home/htt/Desktop/NYT-Multi/raw_data/nyt1/output/train.json', number_of_interest=10)
