import json

relation_sum='./result/dev_pred.json'
relation1='./result/relation1.json'
relation2='./result/relation2.json'
relation3='./result/relation3.json'
relation4='./result/relation4.json'
relation5='./result/relation5.json'
relation_normal='./result/normal.json'
relation_entityoverlap='./result/entityoverlap.json'
relation_entitypairoverlap='./result/entitypairoverlap.json'

def different_relation():
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = open(relation_sum, 'r')
    fr = f.read().replace('\n', '').split('}')
    for line in fr:
        if line == '':
            continue
        sent = json.loads(line + '}')
        relations = sent['relationMentions']
        output = []
        for i in relations:
            output.append(i)
        if len(output) >= 1:
            count_relations = sum((1 if i != '' else 0 for i in output))
        if count_relations==1:
            a += 1
            f = open('./relation1.json', 'a')
            f.write(line+'}')
        if count_relations==2:
            b += 1
            f = open('./relation2.json', 'a')
            f.write(line + '}')
        if count_relations==3:
            c += 1
            f = open('./relation3.json', 'a')
            f.write(line + '}')
        if count_relations==4:
            d += 1
            f = open('./relation4.json', 'a')
            f.write(line+'}')
        if count_relations>=5:
            e += 1
            f = open('./relation5.json', 'a')
            f.write(line+'}')
    print("数据集总数为：" + str(a+b+c+d+e))
    print("1个数为：" + str(a))
    print("2个数为：" + str(b))
    print("3个数为：" + str(c))
    print("4个数为：" + str(d))
    print(">5个数为：" + str(e))

def different_overlap():
    a = 0
    b = 0
    c = 0
    f = open(relation_sum, 'r')
    fr = f.read().replace('\n', '').split('}')
    for line in fr:
        if line == '':
            continue
        sent = json.loads(line + '}')
        relations = sent['relationMentions']
        output = []
        output1 = []
        output2 = []
        output3 = []
        for i in relations:
            output.append(i[0])
            output.append(i[2])
        output1=set(output)
        if len(output1)==len(output):
            a+=1
            f = open('./normal.json', 'a')
            f.write(line + '}')
        else:
            b+=1
            f = open('./entityoverlap.json', 'a')
            f.write(line + '}')
        for i in relations:
            output2.append((i[0],i[2]))
        output3 = set(output2)
        if len(output3)!=len(output2):
            c+=1
            f = open('./entitypairoverlap.json', 'a')
            f.write(line + '}')
    print("normal个数为：" + str(a))
    print("entityoverlap个数为：" + str(b))
    print("entitypairoverlap个数为：" + str(c))

def evaluate(data):
    total_predict_right = 1e-10
    total_predict = 1e-10
    total_right = 1e-10
    X = 1e-10
    Y = 1e-10
    Z = 1e-10
    f = open(data, 'r')
    fr = f.read().replace('\n', '').split('}')
    for line in fr:
        if line == '':
            continue
        data = json.loads(line + '}')
        R= set()
        T= set()
        for spo in data['relationMentions_pred']:
            spo=tuple(spo)
            R.add(spo)
        for spo in data['relationMentions']:
            spo=tuple(spo)
            T.add(spo)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        total_predict_right += X
        total_predict += Y
        total_right += Z
    P = total_predict_right / float(total_predict)
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R)
    print('f1: %.5f, precision: %.5f, recall: %.5f' %(F, P, R))

def evaluate_entity(data):
    total_predict_right = 1e-10
    total_predict = 1e-10
    total_right = 1e-10
    X = 1e-10
    Y = 1e-10
    Z = 1e-10
    f = open(data, 'r')
    fr = f.read().replace('\n', '').split('}')
    for line in fr:
        if line=='':
            continue
        data = json.loads(line+'}')
        output=[]
        output1=[]
        R= []
        T= []
        for spo in data['relationMentions_pred']:
            output.append(spo[0])
            output.append(spo[2])
        R = set(output)
        for spo in data['relationMentions']:
            output1.append(spo[0])
            output1.append(spo[2])
        T = set(output1)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        total_predict_right += X
        total_predict += Y
        total_right += Z
    P = total_predict_right / float(total_predict)
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R)
    # print(total_right)
    print('f1: %.5f, precision: %.5f, recall: %.5f' %(F, P, R))

def static_tribution(data):
    """计算卡方分布，统计出四种不同的情况的个数
    （1）实体关系都正确a1
    （2）实体关系都不正确a4
    （3）实体正确关系错误a3
    （4）实体错误关系正确a2
        """
    sum_a1 = 0
    sum_a2 = 0
    sum_a3 = 0
    sum_a4 = 0
    f = open(data, 'r')
    fr = f.readlines()
    for line in fr:
        a1 = 0
        a2 = 0
        a3 = 0
        a4 = 0
        data = json.loads(line.strip('\r\n'))
        # split ground truth into two parts, namely {(s,o)} and {p}
        gt_entities = []
        gt_relations = []
        for spo in data['relationMentions']:
            gt_entities.append((spo[0], spo[2]))
            gt_relations.append(spo[1])
        # split predictions into two parts
        pd_entities = []
        pd_relations = []
        for spo in data['relationMentions_pred']:
            pd_entities.append((spo[0], spo[2]))
            pd_relations.append(spo[1])

        # analyze each prediction
        for entity, relation in zip(pd_entities, pd_relations):
            indexes = [i for i, x in enumerate(gt_entities) if x == entity]
            if len(indexes):
                flag = True
                for index in indexes:
                    if relation == gt_relations[index]:
                        a1 += 1
                        flag = False
                        break
                if flag:
                    a3 += 1
            else:
                if relation in gt_relations:
                    a2 += 1
                else:
                    a4 += 1

        sum_a1 += a1
        sum_a2 += a2
        sum_a3 += a3
        sum_a4 += a4
    return (sum_a1, sum_a2, sum_a3, sum_a4)

def evaluate_E1(data):
    total_predict_right = 1e-10
    total_predict = 1e-10
    total_right = 1e-10
    X = 1e-10
    Y = 1e-10
    Z = 1e-10
    f = open(data, 'r')
    fr = f.read().replace('\n', '').split('}')
    for line in fr:
        if line == '':
            continue
        data = json.loads(line + '}')
        output = []
        output1 = []
        R = []
        T = []
        S = []
        for spo in data['relationMentions_pred']:
            output.append(spo[0])
        R = set(output)
        for spo in data['relationMentions']:
            output1.append(spo[0])
        T = set(output1)
        S = [x for x in R if x in T]
        X += len(S)
        Y += len(R)
        Z += len(T)
        total_predict_right += X
        total_predict += Y
        total_right += Z
    P = total_predict_right / float(total_predict)
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R)
    print('precision: %.5f, recall: %.5f, f1: %.5f' %(P, R, F))

def evaluate_E2(data):
    total_predict_right = 1e-10
    total_predict = 1e-10
    total_right = 1e-10
    X = 1e-10
    Y = 1e-10
    Z = 1e-10
    f = open(data, 'r')
    fr = f.read().replace('\n', '').split('}')
    for line in fr:
        if line == '':
            continue
        data = json.loads(line + '}')
        output = []
        output1 = []
        R = []
        T = []
        S = []
        for spo in data['relationMentions_pred']:
            output.append(spo[2])
        R = set(output)
        for spo in data['relationMentions']:
            output1.append(spo[2])
        T = set(output1)
        S = [x for x in R if x in T]
        X += len(S)
        Y += len(R)
        Z += len(T)
        total_predict_right += X
        total_predict += Y
        total_right += Z
    P = total_predict_right / float(total_predict)
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R)
    print('precision: %.5f, recall: %.5f, f1: %.5f' %(P, R, F))

def evaluate_relation(data):
    total_predict_right = 1e-10
    total_predict = 1e-10
    total_right = 1e-10
    X = 1e-10
    Y = 1e-10
    Z = 1e-10
    f = open(data, 'r')
    fr = f.read().replace('\n', '').split('}')
    for line in fr:
        if line == '':
            continue
        data = json.loads(line + '}')
        output = []
        output1 = []
        R =[]
        T = []
        S=[]
        for spo in data['relationMentions_pred']:
            output.append(spo[1])
        R = set(output)
        # R=set(output)
        for spo in data['relationMentions']:
            output1.append(spo[1])
        T = set(output1)
        # T = set(output1)
        S=[x for x in R if x in T]
        X += len(S)
        # X += len(R & T)
        Y += len(R)
        Z += len(T)
        total_predict_right += X
        total_predict += Y
        total_right += Z
    P = total_predict_right / float(total_predict)
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R)
    print('f1: %.5f, precision: %.5f, recall: %.5f' % (F, P, R))

def evaluate_E1R(data):
    total_predict_right = 1e-10
    total_predict = 1e-10
    total_right = 1e-10
    X = 1e-10
    Y = 1e-10
    Z = 1e-10
    f = open(data, 'r')
    fr = f.read().replace('\n', '').split('}')
    for line in fr:
        if line == '':
            continue
        data = json.loads(line + '}')
        output = []
        output1 = []
        R = []
        T = []
        S = []
        for spo in data['relationMentions_pred']:
            output.append((spo[0], spo[1]))
        R = set(output)
        for spo in data['relationMentions']:
            output1.append((spo[0], spo[1]))
        T = set(output1)
        S = [x for x in R if x in T]
        X += len(S)
        Y += len(R)
        Z += len(T)
        total_predict_right += X
        total_predict += Y
        total_right += Z
    P = total_predict_right / float(total_predict)
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R)
    print('precision: %.5f, recall: %.5f, f1: %.5f' % (P, R, F))

def evaluate_RE2(data):
    total_predict_right = 1e-10
    total_predict = 1e-10
    total_right = 1e-10
    X = 1e-10
    Y = 1e-10
    Z = 1e-10
    f = open(data, 'r')
    fr = f.read().replace('\n', '').split('}')
    for line in fr:
        if line == '':
            continue
        data = json.loads(line + '}')
        output = []
        output1 = []
        R = []
        T = []
        S = []
        for spo in data['relationMentions_pred']:
            output.append((spo[1], spo[2]))
        R = set(output)
        for spo in data['relationMentions']:
            output1.append((spo[1], spo[2]))
        T = set(output1)
        S = [x for x in R if x in T]
        X += len(S)
        Y += len(R)
        Z += len(T)
        total_predict_right += X
        total_predict += Y
        total_right += Z
    P = total_predict_right / float(total_predict)
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R)
    print('precision: %.5f, recall: %.5f, f1: %.5f' % (P, R, F))

def evaluate_entitypair(data):
    total_predict_right = 1e-10
    total_predict = 1e-10
    total_right = 1e-10
    X = 1e-10
    Y = 1e-10
    Z = 1e-10
    f = open(data, 'r')
    fr = f.read().replace('\n', '').split('}')
    for line in fr:
        if line == '':
            continue
        data = json.loads(line + '}')
        output=[]
        output1=[]
        R= []
        T= []
        S = []
        for spo in data['relationMentions_pred']:
            output.append((spo[0],spo[2]))
        R = set(output)
        for spo in data['relationMentions']:
            output1.append((spo[0],spo[2]))
        T = set(output1)
        S = [x for x in R if x in T]
        X += len(S)
        Y += len(R)
        Z += len(T)
        total_predict_right += X
        total_predict += Y
        total_right += Z
    P = total_predict_right / float(total_predict)
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R)
    print('precision: %.5f, recall: %.5f, f1: %.5f' %(P, R, F))

if __name__ == "__main__":
     #different_relation()
     #different_overlap()

     #fig 12
     evaluate(relation_normal)
     evaluate(relation_entitypairoverlap)
     evaluate(relation_entityoverlap)

     #table 5
     evaluate_entity(relation_sum)
     evaluate_relation(relation_sum)

     # table 6
    (a1, a2, a3, a4) = static_tribution(relation_sum)
     print(a1, a2, a3, a4)

     # table 8
     evaluate(relation1)
     evaluate(relation2)
     evaluate(relation3)
     evaluate(relation4)
     evaluate(relation5)

     #table 9
     evaluate_E1(relation_sum)
     evaluate_E2(relation_sum)

     evaluate_E1R(relation_sum)
     evaluate_RE2(relation_sum)
     evaluate_entitypair(relation_sum)

