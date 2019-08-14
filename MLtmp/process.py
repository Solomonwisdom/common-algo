import json
import data
from naive_bayes import GNB
from lda import LDA
from qda import QDA





def train_acc(data_path, algorithm_name):
    print(data_path)
    x, y, test_x, test_y = data.run(data_path)
    clf = None
    if algorithm_name == "gnb":
        clf = GNB()
        print("gnb instance.")
    elif algorithm_name == "lda":
        clf = LDA()
        print("lda instance.")
    elif algorithm_name == "qda":
        clf = QDA()
        print("qda instance.")
    else:
        print("NO Implement")
        return "NO Implement"

    num = 0
    clf.fit(x, y)
    train_result = clf.predict(x)
    for i in range(len(train_result)):
        if train_result[i] == y[i]:
            num += 1

    return num / len(y)


def test_acc(data_path, algorithm_name):
    print(data_path)
    x, y, test_x, test_y = data.run(data_path)
    clf = None
    if algorithm_name == "gnb":
        clf = GNB()
        print("gnb instance.")
    elif algorithm_name == "lda":
        clf = LDA()
        print("lda instance.")
    elif algorithm_name == "qda":
        clf = QDA()
        print("qda instance.")
    else:
        print("NO Implement")
        return "NO Implement"

    num = 0
    clf.fit(x, y)
    train_result = clf.predict(test_x)
    for i in range(len(train_result)):
        if train_result[i] == test_y[i]:
            num += 1

    return num / len(test_y)


def ml_handle(event, context):
    data = event['data']
    print(data)
    print(type(data))

    if type(data) == bytes:
        data = json.loads(data)
    data_path = data['path']
    algorithm_name = data['algorithm']
    mode = data['mode']
    result = "incorrect parameters"
    if mode == 'train':
        result = str(train_acc(data_path, algorithm_name))
    elif mode == 'test':
        result = str(test_acc(data_path, algorithm_name))
    return result.encode('utf-8')
