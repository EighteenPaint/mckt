
# 加了五倍交叉验证的DKT
import os
import datetime
import random
import time
import csv
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, accuracy_score
# sklearn.metrics:包含了许多模型评估指标，例如决定系数R2、准确度等;
# roc_curve:roc曲线；mean_squared_error：均方差；mean_absolute_error：平均绝对误差；accuracy_score:准确率
import argparse
# 用于从 sys.argv 中解析命令项选项与参数的模块
import numpy as np

# ### 定义相关函数

# In[10]:


# In[2]:


# 是否适用gpu，cuda()用于将变量传输到GPU上，gpu版本是torch.cuda.FloatTensor,cpu版本是torch.FloatTensor
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.cuda.set_device(0)
# 定义cuda()函数：参数为o，如果use_cuda为真返回o.cuda(),为假返回o
cuda = lambda o: o.cuda() if use_cuda else o
# torch.Tensor是默认的tensor类型（torch.FlaotTensor）的简称。
tensor = lambda o: cuda(torch.tensor(o))
# 生成对角线全1，其余部分全0的二维数组,函数原型：torch.eye(n, m=None, out=None)，m (int) ：列数.如果为None,则默认为n。
eye = lambda d: cuda(torch.eye(d))
# 返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor。
zeros = lambda *args: cuda(torch.zeros(*args))

# 截断反向传播的梯度流,返回一个新的Variable即tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
# 不同之处只是它的requires_grad是false，也就是说这个Variable永远不需要计算其梯度，不具有grad。
detach = lambda o: o.cpu().detach().numpy().tolist()


# In[3]:


def set_seed(seed=0):
    # seed()方法改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数。random:随机数生成器，seed:种子
    random.seed(seed)
    # 为CPU设置种子用于生成随机数
    torch.manual_seed(seed)
    # 为当前GPU设置随机种子,如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    torch.cuda.manual_seed(seed)
    '''
    置为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，
    应该可以保证每次运行网络的时候相同输入的输出是固定的。（说人话就是让每次跑出来的效果是一致的）
    '''
    torch.backends.cudnn.deterministic = True
    '''
     置为True的话会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，
    其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
    '''
    torch.backends.cudnn.benchmark = False
# 数据集情况：题目序列的长度 题目序列 答对的情况
class Data:

    def __init__(self, file, length, q_num, is_test=False, index_split=None, is_train=False):
        '''
        len: 4
        q: 53,54,53,54
        y: 0,0,0,0
        t1: 0,1,2,0
        t2: 0,1,3,5
        t3: 3,1,2,1
        '''
        # 读取csv文件，delimiter说明分割字段的字符串为逗号
        rows = csv.reader(file, delimiter=',')
        # rows为:[[题目个数], [题目序列], [答对情况]……]
        rows = [[int(e) for e in row if e != ''] for row in rows]

        q_rows, r_rows = [], []

        student_num = 0
        # zip()将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象,注意：不是列表
        if is_test:
            # 双冒号：实质为lis[start:end:step]，end没写就是到列表的最后，意思就是从索引为start到end，步长为step进行切片，每个step取一次
            # q_row, r_row：题目序号列表，答对情况列表
            for q_row, r_row in zip(rows[1::3], rows[2::3]):
                num = len(q_row)
                n = num // length
                for i in range(n + 1):
                    q_rows.append(q_row[i * length: (i + 1) * length])
                    r_rows.append(r_row[i * length: (i + 1) * length])
        else:
            if is_train:
                for q_row, r_row in zip(rows[1::3], rows[2::3]):

                    if student_num not in index_split:

                        num = len(q_row)

                        n = num // length

                        for i in range(n + 1):
                            q_rows.append(q_row[i * length: (i + 1) * length])

                            r_rows.append(r_row[i * length: (i + 1) * length])
                    student_num += 1
            # 验证集
            else:
                for q_row, r_row in zip(rows[1::3], rows[2::3]):

                    if student_num in index_split:

                        num = len(q_row)

                        n = num // length

                        for i in range(n + 1):
                            q_rows.append(q_row[i * length: (i + 1) * length])

                            r_rows.append(r_row[i * length: (i + 1) * length])
                    student_num += 1

        q_rows = [row for row in q_rows if len(row) > 2]

        r_rows = [row for row in r_rows if len(row) > 2]

        # q_min = min([min(row) for row in q_rows])

        # q_rows = [[q - q_min for q in row] for row in q_rows]

        self.r_rows = r_rows

        # self.q_num = max([max(row) for row in q_rows]) + 1
        self.q_num = q_num
        self.q_rows = q_rows

    # 获取[[题号,答对]，[题号,答对]，……]列表
    def __getitem__(self, index):
        return list(
            zip(self.q_rows[index], self.r_rows[index]))

    # 批次大小
    def __len__(self):
        return len(self.q_rows)


def collate(batch, q_num):
    # print("1",batch) # 列表：[[(题目，答案)，(题目，答案)，(题目，答案)……][(题目，答案)，……]……],32个包含一定数量(题目，答案)的列表
    lens = [len(row) for row in batch]
    # 最大题目数量
    max_len = max(lens)
    # 第三维的第三个数0或1表示数据是否有效，列表、元组前加星号是将其元素拆解为独立参数
    batch = tensor([[[*e, 1] for e in row] + [[0, 0, 0]] * (max_len - len(row)) for row in batch])
    # print("2",batch) # torch.size([32,200,3])
    Q, Y, S = batch.T  # Q:问题，Y:预测，S:padding,样本数据缺失或者说不够时填充[[0,0,0]]张量
    Q, Y, S = Q.T, Y.T, S.T  # torch.size([32,200])
    # X:题目矩阵加上q_num * (1 - Y)，为什么处理成X这样???
    X = Q + q_num * (1 - Y)
    # print("4",X,X.shape,X.dtype) # shape:torch.size([32,200]),X.dtype:torch.int64
    return X, Y, S, Q

class DKVMN(nn.Module):
    def __init__(self, q_num, hidden_num, concept_num):
        super(DKVMN, self).__init__()

        self.hidden_num, self.concept_num = hidden_num, concept_num

        self.q_embedding = nn.Embedding(q_num, hidden_num)

        self.key_linear = nn.Linear(hidden_num, concept_num, bias = False)

        self.p_fnn = nn.Sequential(
            nn.Linear(2 * hidden_num, hidden_num),
            nn.Tanh(),
            nn.Linear(hidden_num, 1),
            nn.Sigmoid(),
        )

        self.x_embedding = nn.Embedding(2 * q_num, hidden_num)

        self.e_fnn = nn.Sequential(
            nn.Linear(hidden_num, hidden_num),
            nn.Sigmoid(),
        )

        self.a_fnn = nn.Sequential(
            nn.Linear(hidden_num, hidden_num),
            nn.Tanh(),
        )

    def forward(self, X, Q):
        batch_size, length = X.shape

        hidden_num, concept_num = self.hidden_num, self.concept_num

        Q = self.q_embedding(Q)

        K = F.softmax(self.key_linear(Q), 2)

        X = self.x_embedding(X)

        memory = zeros(batch_size, concept_num, hidden_num)

        P = []

        for t in range(length):

            Qt = Q[:, t, :].unsqueeze(1)

            Kt = K[:, t, :].unsqueeze(1)

            Xt = X[:, t, :].unsqueeze(1)

            Rt = torch.bmm(Kt, memory)

            Pt = self.p_fnn(torch.cat((Qt, Rt), 2))

            P.append(Pt.reshape(batch_size, 1))

            Et = self.e_fnn(Xt)

            last_memory = memory * (1 - torch.bmm(Kt.transpose(1, 2), Et))

            At = self.a_fnn(Xt)

            memory = last_memory + torch.bmm(Kt.transpose(1, 2), At)

            # print(detach(memory.max()), detach(memory.min()), detach(memory.mean()))

        P = torch.cat(P, 1)

        return P

def train(model, data, optimizer, batch_size):
    model.train(mode = True)

    criterion = nn.BCELoss()

    for X, Y, S, Q in DataLoader(
        dataset = data,
        batch_size = batch_size,
        collate_fn = lambda batch: collate(batch, data.q_num),
        shuffle = True
        ):
        P = model(X, Q)

        P, Y, S = P[:, 1:], Y[:, 1:], S[:, 1:]

        index = S == 1

        loss = criterion(P[index], Y[index].float())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

def evaluate(model, data, batch_size):
    model.eval()

    criterion = nn.BCELoss()

    y_pred, y_true = [], []

    loss = 0.0

    for X, Y, S, Q in DataLoader(
        dataset = data,
        batch_size = batch_size,
        collate_fn = lambda batch: collate(batch, data.q_num)
        ):
        P = model(X, Q)

        P, Y, S = P[:, 1:], Y[:, 1:], S[:, 1:]

        index = S == 1

        Y = Y[index].float()

        P = P[index]

        y_pred += detach(P)

        y_true += detach(Y)

        loss += detach(criterion(P, Y) * P.shape[0])

        # fpr:假阳性率;tpr:真阳性率;thres:减少了用于计算fpr和tpr的决策函数的阈值.
    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    mse_value = mean_squared_error(y_true, y_pred)
    mae_value = mean_absolute_error(y_true, y_pred)
    bi_y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
    acc_value = accuracy_score(y_true, bi_y_pred)
    # auc, loss, mse, acc
    return auc(fpr, tpr), loss / len(y_true), mse_value, mae_value, acc_value

def experiment(data_path, dataset, m, n, learning_rate, length, kernel_size, epochs, batch_size, seed, q_num,
               cv_num, ffn_h_num, opt, d_model, encoder_out, dropout, channel_size, d_ff, hidden_num=128,
               model_type='lskt',
               n_pid=None, th=100, ):
    set_seed(seed)
    path = './result_dkt_cross/%s' % ('{0:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now()))
    os.makedirs(path)
    info_file = open('%s/info.txt' % path, 'w+')
    params_list = (
        'dataset = %s\n' % dataset,
        'm = %d\n' % m,
        'n = %d\n' % n,
        'learning_rate = %f\n' % learning_rate,
        'length = %d\n' % length,
        'kernel_size = %d\n' % kernel_size,
        'batch_size = %d\n' % batch_size,
        'seed = %d\n' % seed,
        'q_num = %d\n' % q_num
    )
    info_file.write('file_name = allxt-onehot no norm + weight decay 5e-4')
    info_file.write('%s%s%s%s%s%s%s%s%s' % params_list)
    model_list = []

    for cv in [cv_num]:
        random.seed(cv + 1000)
        random.seed(0)

        train_data = Data(open('%s/%s/%s_train%d.csv' % (data_path, dataset, dataset, cv), 'r'), length, q_num,
                          is_test=True)
        valid_data = Data(open('%s/%s/%s_valid%d.csv' % (data_path, dataset, dataset, cv), 'r'), length, q_num,
                          is_test=True)
        max_auc = 0.0
        # DKT模型实例化：model
        model = cuda(DKVMN(train_data.q_num, hidden_num,q_num))
        if opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        early_stop = 0
        for epoch in range(1, epochs + 1):
            time_start = time.time()
            train(model, train_data, optimizer, batch_size)
            train_auc, train_loss, train_mse, train_mae, train_acc = evaluate(model, train_data, batch_size)
            valid_auc, valid_loss, valid_mse, valid_mae, valid_acc = evaluate(model, valid_data, batch_size)
            time_end = time.time()
            if max_auc < valid_auc:
                early_stop = epoch
                max_auc = valid_auc
                torch.save(model.state_dict(), '%s/model_%s' % ('%s' % path, '%d' % cv))
                current_max_model = model
            if epoch - early_stop > 30:
                break

            print_list = (
                'cv:%-3d' % cv,
                'epoch:%-3d' % epoch,
                'max_auc:%-8.4f' % max_auc,
                'valid_auc:%-8.4f' % valid_auc,
                'valid_loss:%-8.4f' % valid_loss,
                'valid_mse:%-8.4f' % valid_mse,
                'valid_mae:%-8.4f' % valid_mae,
                'valid_acc:%-8.4f' % valid_acc,
                'train_auc:%-8.4f' % train_auc,
                'train_loss:%-8.4f' % train_loss,
                'train_mse:%-8.4f' % train_mse,
                'train_mae:%-8.4f' % train_mae,
                'train_acc:%-8.4f' % train_acc,
                'time:%-6.2fs' % (time_end - time_start)
            )

            print('%s %s %s %s %s %s %s %s %s %s %s %s %s %s' % print_list)
            info_file.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % print_list)
        model_list.append(current_max_model)

    train_list = []
    auc_list = []
    mse_list = []
    mae_list = []
    acc_list = []
    loss_list = []
    for _, model_item in enumerate(model_list):
        train_data = Data(open('%s/%s/%s_train%d.csv' % (data_path, dataset, dataset, cv_num), 'r'), length, q_num,
                          is_test=True)
        test_data = Data(open('%s/%s/%s_test%d.csv' % (data_path, dataset, dataset, cv_num), 'r'), length, q_num,
                         is_test=True)
        train_auc, train_loss, train_mse, train_mae, train_acc = evaluate(model_item, train_data, batch_size)
        test_auc, test_loss, test_mse, test_mae, test_acc = evaluate(model_item, test_data, batch_size)

        train_list.append(train_auc)
        auc_list.append(test_auc)
        mse_list.append(test_mse)
        mae_list.append(test_mae)
        acc_list.append(test_acc)
        loss_list.append(test_loss)
        print_list_test = (
            'cv:%-3d' % cv_num,
            'train_auc:%-8.4f' % train_auc,
            'test_auc:%-8.4f' % test_auc,
            'test_mse:%-8.4f' % test_mse,
            'test_mae:%-8.4f' % test_mae,
            'test_acc:%-8.4f' % test_acc,
            'test_loss:%-8.4f' % test_loss
        )

        print('%s %s %s %s %s %s %s\n' % print_list_test)
        info_file.write('%s %s %s %s %s %s %s\n' % print_list_test)

    # info_file.write('%s %s %s %s %s %s\n' % print_result)


# ### 运行程序-使用命令行参数传入相关参数


# In[17]:


# 数据集：assist2009, synthetic, assist2015, STATICS，assist2012，eanalyst
# 创建解析步骤
parser = argparse.ArgumentParser(description='Script to test DKT.')
# 添加参数步骤
parser.add_argument('--hidden_num', type=int, default=128, help='')
# 参数中加入args=[]
parser.add_argument('--m', type=int, default=2, help='')
parser.add_argument('--n', type=int, default=2, help='')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='')
parser.add_argument('--kernel_size', type=int, default=2, help='')
parser.add_argument('--length', type=int, default=200, help='')
parser.add_argument('--epochs', type=int, default=500, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cv_num', type=int, default=1, help='')
parser.add_argument('--data_path', type=str, default='./dataset', help='')
parser.add_argument('--opt', type=str, default='adam', help='')
parser.add_argument('--d_model', type=int, default=256, help='')
parser.add_argument('--encoder_out', type=int, default=256, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='')
parser.add_argument('--d_ff', type=int, default=1024, help='')
# set according to dataset
parser.add_argument('--model_type', type=str, default='lskt', help='')
parser.add_argument('--q_num', type=int, default=112, help='')
parser.add_argument('--dataset', type=str, default='assist2009', help='')
parser.add_argument('--ffn_h_num', type=int, default=112, help='')
parser.add_argument('--n_pid', type=int, default=None, help='')
parser.add_argument('--channel_size', type=int, default=256, help='')
parser.add_argument('--device', type=int, default=0, help='')
parser.add_argument('--th', type=int, default=100, help='')
params = parser.parse_args()
dataset = params.dataset
if dataset == 'junyi_pid':
    params.q_num = 40
    params.n_pid = 721
    params.ffn_h_num = 40
    params.channel_size = 40
if dataset == 'poj3':
    params.q_num = 4055
    params.ffn_h_num = 4055
    params.channel_size = 4055
if dataset == 'eanalyst_math':
    params.q_num = 2696
    params.ffn_h_num = 2696
    params.channel_size = 2696
if dataset == 'assist2015':
    params.q_num = 104
    params.ffn_h_num = 104
    params.channel_size = 104
    params.cv_num = 1
if dataset == 'assist2009':
    params.q_num = 112
    params.ffn_h_num = 112
    params.channel_size = 112
    params.cv_num = 4
if dataset == 'STATICS':
    params.q_num = 1224
    params.ffn_h_num = 1224
    params.channel_size = 1224
    params.learning_rate = 1e-5
    params.cv_num = 3
if dataset == 'statics':
    params.q_num = 1224
    params.ffn_h_num = 1224
    params.channel_size = 1224
    params.learning_rate = 1e-5
    params.cv_num = 3
if dataset in {"assist2009_pid"}:
    params.q_num = 112
    params.n_pid = 16891
    params.ffn_h_num = 112
    params.channel_size = 112
    params.cv_num = 4

if dataset in {"assist2017_pid"}:
    params.q_num = 104
    params.n_pid = 3162
    params.ffn_h_num = 104
    params.channel_size = 104
    params.cv_num = 2

if dataset in {"assist2012_pid"}:
    params.q_num = 246
    params.n_pid = 50988
    params.ffn_h_num = 246
    # channel size should divide by 8
    params.channel_size = 248

if dataset in {"assist2012"}:
    params.q_num = 246
    # params.n_pid = 50988
    params.ffn_h_num = 246
    # channel size should divide by 8
    params.channel_size = 248

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    for cv in [4]:
        # params.cv_num = cv
        print(params)
        experiment(
            data_path=params.data_path,
            dataset=params.dataset,
            m=params.m,
            n=params.n,
            learning_rate=params.learning_rate,
            length=params.length,
            epochs=params.epochs,
            batch_size=params.batch_size,
            seed=params.seed,
            q_num=params.q_num,
            cv_num=params.cv_num,
            kernel_size=params.kernel_size,
            ffn_h_num=params.ffn_h_num,
            opt=params.opt,
            d_model=params.d_model,
            encoder_out=params.encoder_out,
            dropout=params.dropout,
            channel_size=params.channel_size,
            d_ff=params.d_ff,
            model_type=params.model_type,
            n_pid=params.n_pid,
            th=params.th,
            hidden_num=params.hidden_num
        )


