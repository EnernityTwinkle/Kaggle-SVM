# import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import math

N = 5000

# 判断类别0和1的比例
def get_01_percentage(file_name):
    file_read = open(file_name, 'r', encoding='utf-8')
    num_0 = 0.0
    num_1 = 0.0
    lines = file_read.readlines()
    i = 1
    while i < len(lines):
        if (lines[i][len(lines[i]) - 2] == '0'):
            num_0 += 1
        else:
            num_1 += 1
        i += 1
    print("0 is " + str(num_0) + " " + str(num_0 / (num_0 + num_1)))
    print("1 is " + str(num_1) + " " + str(num_1 / (num_0 + num_1)))
    file_read.close()
    return int(num_0 / num_1)
# 进行英文分词
def cut_word(sentence):
    sentence = sentence.replace(',', '')
    sentence = sentence.replace('.', '')
    sentence = sentence.replace('\"', '')
    sentence = sentence.replace('(', '')
    sentence = sentence.replace(')', '')
    sentence = sentence.replace('?', ' ?')
    sentence = sentence.replace('!', ' !')
    sentence = sentence.replace('/', ' /')
    # print(sentence)
    words = sentence.split(' ')
    return words
# 得到特征词
def get_features():
    fp_read = open('../input/train.csv', 'r', encoding='utf-8')
    lines = fp_read.readlines()
    idf = {}
    tf = {}
    words_num = 0.0
    for line in lines[1:]:
        sentence = line[21:len(line) - 3]
        # print(sentence)
        words = cut_word(sentence)
        temp_dic = {}
        for word in words:
            words_num += 1.0
            if temp_dic.__contains__(word):  # 统计词出现的次数
                temp_dic[word] += 1.0
            else:
                temp_dic[word] = 1.0
        # print(temp_dic)
        for temp in temp_dic:
            # print(temp)
            if idf.__contains__(temp):
                idf[temp] += 1.0
                tf[temp] += temp_dic[temp]
            else:
                idf[temp] = 1.0
                tf[temp] = 1.0
    # break
    for temp in tf:  # 得出词语的频率
        tf[temp] /= words_num
        idf[temp] /= len(lines)
    for temp in idf:  # 得出
        idf[temp] = math.log(1.0 / idf[temp])
    tf_idf = {}
    for temp in tf:
        tf_idf[temp] = tf[temp] * idf[temp]
    sort_words_list = sorted(tf_idf.items(), key=lambda item: item[1], reverse=True)
    fp_write = open('../words_list', 'w', encoding='utf-8')
    for temp in sort_words_list:
        fp_write.write(str(temp[0]) + ' ' + str(temp[1]) + '\n')
    fp_read.close()
    fp_write.close()
    return sort_words_list
# 对句子进行向量化表示，并写出到文件中
def generate_vec():
    file_train = '../input/train.csv'
    bei = get_01_percentage(file_train)
    print(bei)

    features = get_features()
    features_dic = {}
    i = 0
    while i < N and i < len(features):
        features_dic[features[i][0]] = i
        i += 1
    print('特征维度为：', end='')
    print(len(features_dic))
    fp_read = open('../input/train.csv', 'r', encoding='utf-8')
    fp_write = open('../digit_feature', 'w', encoding='utf-8')
    lines = fp_read.readlines()
    digit_feature = []
    i = 0
    count_bei = 0
    for line in lines[1:]:
        i += 1
        if i % 10000 == 0:
            print(i)
        flag = line[len(line) - 2]
        sentence = line[21:len(line) - 3]
        # print(line)
        words = cut_word(sentence)
        temp = ['0'] * N
        for word in words:
            if features_dic.__contains__(word):
                temp[features_dic[word]] = '1'
        # 删除多的进行数据均衡化
        if flag == '1':
            fp_write.write(','.join(temp) + ',' + str(len(sentence)) + ',' + flag + '\n')
        elif flag == '0':
            count_bei += 1
            if count_bei % bei == 0:
                fp_write.write(','.join(temp) + ',' + str(len(sentence)) + ',' + flag + '\n')
        # 重复少的进行数据均衡化
        # if flag == '1':
        #     for j in range(bei):
        #         fp_write.write(','.join(temp) + ',' + flag + '\n')
        # elif flag == '0':
        #     fp_write.write(','.join(temp) + ',' + flag + '\n')
    fp_write.close()
    fp_read.close()
# 对测试集的句子进行向量化表示
def generate_vec_test():
    file_train = '../input/train.csv'
    bei = get_01_percentage(file_train)
    print(bei)

    features = get_features()
    features_dic = {}
    i = 0
    while i < N and i < len(features):
        features_dic[features[i][0]] = i
        i += 1
    print('特征维度为：', end='')
    print(len(features_dic))
    fp_read = open('../input/test.csv', 'r', encoding='utf-8')
    fp_write = open('../digit_feature_test', 'w', encoding='utf-8')
    lines = fp_read.readlines()
    digit_feature = []
    i = 0
    for line in lines[1:]:
        i += 1
        if i % 10000 == 0:
            print(i)
        flag = line[len(line) - 2]
        sentence = line[21:len(line) - 1]
        # print(line)
        words = cut_word(sentence)
        temp = ['0'] * N
        for word in words:
            if features_dic.__contains__(word):
                temp[features_dic[word]] = '1'
        fp_write.write(','.join(temp) + ',' + str(len(sentence)) + '\n')
    fp_write.close()
    fp_read.close()
if __name__ == "__main__":

    generate_vec_test()

    # file_train = '../input/train.csv'
    # get_01_percentage(file_train)
    # 生成向量化表达
    generate_vec()
    path = u'../digit_feature'
    data = np.loadtxt(path, dtype = float, delimiter = ',')
    x, y = np.split(data, (N + 1,), axis=1)
    print(len(x))
    print(len(x[0]))

    # 随机划分训练子集和测试子集
    #test_size为测试集的比例， random_state是随机数种子，0为每次产生随机数不同
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    # 开始训练
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())
    joblib.dump(clf, "svm.model")
    # 加载模型
    print('开始加载模型。。。')
    model = joblib.load('svm.model')
    print('加载模型结束')

    print(model.score(x_train, y_train))
    train_res = model.predict(x_train)
    acc_train = accuracy_score(y_train, train_res)
    recall_train = recall_score(y_train, train_res, average='macro')
    print('训练集：', end='')
    print(2.0 * acc_train * recall_train / (acc_train + recall_train))

    test_res = model.predict(x_test)
    acc_test = accuracy_score(y_test, test_res)
    recall_test = recall_score(y_test, test_res, average='macro')
    print('测试集：', end='')
    print(2.0 * acc_test * recall_test / (acc_test + recall_test))
    count1 = 0
    for temp in test_res:
        if (temp == 1.0):
            count1 += 1
    print('测试集大小：', end='')
    print(len(x_test))
    print(count1)
    print(model.score(x_test, y_test))

    # 对测试集进行预测
    path = u'../digit_feature_test'
    data = np.loadtxt(path, dtype=float, delimiter=',')
    result = model.predict(data)
    fp_out = open('../submission.csv', 'w', encoding='utf-8')
    fp_out.write('qid,prediction\n')
    fp_in = open('../input/test.csv', 'r', encoding='utf-8')
    lines = fp_in.readlines()
    i = 1
    count1 = 0
    print('lines length:',end='')
    print(len(lines))
    print('result length:', end='')
    print(len(result))
    while i < len(lines):
        if(len(lines[i]) > 20):
            fp_out.write(lines[i][0:20] +  str(i) + ',' + str(int(result[i - 1])) + '\n')
        if (result[i - 1] == 1.0):
            count1 += 1
        i += 1
    print('count1:', end='')
    print(count1)
