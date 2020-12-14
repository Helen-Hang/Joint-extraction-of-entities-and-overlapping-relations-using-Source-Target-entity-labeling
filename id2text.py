import json


def transform(file, w2i, r2i, out):
    # load words2id
    with open(w2i) as f:
        w2i = json.load(f)
        assert w2i != None
        i2w = dict([(v, k) for k, v in w2i.items()])
    # load relations2id
    with open(r2i) as f:
        r2i = json.load(f)
        assert r2i != None
        i2r = dict([(v, k) for k, v in r2i.items()])
    # load dataset.json
    with open(file) as f:
        content = json.load(f)
        assert content != None
        sen_ids = content[-2]
        tri_ids = content[-1]
        assert len(sen_ids) == len(tri_ids)
    # generate samples
    samples = []
    for i in range(len(sen_ids)):
        # get sentence
        sen_list = []
        for w_id in sen_ids[i]:
            sen_list.append(i2w[w_id])
        sentence = sen_list[0]
        for k in range(1, len(sen_list)):
            # write rules
            if sen_list[k] == '.':
                item = sen_list[k]
            elif sen_list[k] == '\'\'':
                item = '\"'
            elif sen_list[k] == ',':
                item = sen_list[k]
            elif sen_list[k] == '\'':
                item = sen_list[k]
            elif sen_list[k] == '\'s':
                item = sen_list[k]
            elif sen_list[k] == ':':
                item = sen_list[k]
            elif sen_list[k] == ';':
                item = sen_list[k]
            elif sen_list[k] == ')':
                item = sen_list[k]
            else:
                item = ' ' + sen_list[k]
            if sentence[-1] == '(':
                item = sen_list[k]
            elif sentence[-2:] == '``':
                sentence = sentence[:-2] + '\"'
                item = sen_list[k]
            sentence += item
        # get relation_mentions
        tri_list = []
        for idx in range(0, len(tri_ids[i]), 3):
            triple = {'em1Text': i2w[sen_ids[i][tri_ids[i][idx]]], 'label': i2r[tri_ids[i][idx+2]], 'em2Text': i2w[sen_ids[i][tri_ids[i][idx+1]]]}
            tri_list.append(triple)
        # generate sample
        sample = {'sentText': sentence, 'relationMentions': tri_list}
        samples.append(sample)
    with open(out, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    # webnlg
    transform(file='webnlg/input/train.json', w2i='webnlg/input/words2id.json', r2i='webnlg/input/relations2id.json', out='webnlg/output/train.json')
    transform(file='webnlg/input/valid.json', w2i='webnlg/input/words2id.json', r2i='webnlg/input/relations2id.json', out='webnlg/output/valid.json')
    transform(file='webnlg/input/dev.json', w2i='webnlg/input/words2id.json', r2i='webnlg/input/relations2id.json', out='webnlg/output/dev.json')
    # nyt
    transform(file='nyt1/input/train.json', w2i='nyt1/input/words2id.json', r2i='nyt1/input/relations2id.json', out='nyt1/output/train.json')
    transform(file='nyt1/input/valid.json', w2i='nyt1/input/words2id.json', r2i='nyt1/input/relations2id.json', out='nyt1/output/valid.json')
    transform(file='nyt1/input/test.json', w2i='nyt1/input/words2id.json', r2i='nyt1/input/relations2id.json', out='nyt1/output/test.json')
