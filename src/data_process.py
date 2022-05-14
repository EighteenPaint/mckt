def f23(dir, dataset):
    file_name = []
    file_name.extend([dataset + '_train' + str(i) for i in range(1, 6)])
    file_name.extend([dataset + '_valid' + str(i) for i in range(1, 6)])
    file_name.extend([dataset + '_test' + str(i) for i in range(1, 6)])
    new_file_name = []
    if dataset == 'assist2009_pid':
        new_file_name.extend(['assist2009_train' + str(i) for i in range(1, 6)])
        new_file_name.extend(['assist2009_valid' + str(i) for i in range(1, 6)])
        new_file_name.extend(['assist2009_test' + str(i) for i in range(1, 6)])
    if dataset == 'assist2017_pid':
        new_file_name.extend(['assist2017_train' + str(i) for i in range(1, 6)])
        new_file_name.extend(['assist2017_valid' + str(i) for i in range(1, 6)])
        new_file_name.extend(['assist2017_test' + str(i) for i in range(1, 6)])
    for j in range(15):
        with open(dir + dataset + '/' + file_name[j] + '.csv', 'r') as data:
            lines = data.readlines()
            i = 0
            with open(dir + dataset + '/' + new_file_name[j] + '.csv', 'a') as new:
                for line in lines:
                    if i == 1:  # discard
                        i += 1
                        continue
                    elif i == 3:
                        i = 0
                        new.write(line)
                    else:
                        i += 1
                        new.write(line)


if __name__ == '__main__':
    dir = '../dataset/'
    dataset = 'assist2009_pid'
    f23(dir, dataset)
