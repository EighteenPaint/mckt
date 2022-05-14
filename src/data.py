import csv


class Data:

    def __init__(self, file, length, q_num, is_test=True, index_split=None, is_train=False):
        rows = csv.reader(file, delimiter=',')
        rows = [[int(e) for e in row if e != ''] for row in rows]

        q_rows, r_rows = [], []

        student_num = 0
        if is_test:
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

        self.r_rows = r_rows

        self.q_num = q_num
        self.q_rows = q_rows

    def __getitem__(self, index):
        return list(
            zip(self.q_rows[index], self.r_rows[index]))

    def __len__(self):
        return len(self.q_rows)


class PData:
    # data format
    # id, true_student_id
    # pid1, pid2, ...
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def __init__(self, file, length, q_num, is_test=True, index_split=None, is_train=False):
        rows = csv.reader(file, delimiter=',')
        rows = [[int(e) for e in row if e != ''] for row in rows]

        p_rows, q_rows, r_rows = [], [],[]

        student_num = 0
        if is_test:
            for p_row, q_row, r_row in zip(rows[1::4], rows[2::4], rows[3::4]):
                num = len(q_row)
                n = num // length
                for i in range(n + 1):
                    p_rows.append(p_row[i * length: (i + 1) * length])
                    q_rows.append(q_row[i * length: (i + 1) * length])
                    r_rows.append(r_row[i * length: (i + 1) * length])
        else:
            if is_train:
                for p_row, q_row, r_row in zip(rows[1::4], rows[2::4], rows[3::4]):

                    if student_num not in index_split:

                        num = len(q_row)

                        n = num // length

                        for i in range(n + 1):
                            p_rows.append(p_row[i * length: (i + 1) * length])
                            q_rows.append(q_row[i * length: (i + 1) * length])
                            r_rows.append(r_row[i * length: (i + 1) * length])
                    student_num += 1
            # 验证集
            else:
                for p_row, q_row, r_row in zip(rows[1::4], rows[2::4], rows[3::4]):

                    if student_num in index_split:

                        num = len(q_row)

                        n = num // length

                        for i in range(n + 1):
                            q_rows.append(q_row[i * length: (i + 1) * length])
                            p_rows.append(p_row[i * length: (i + 1) * length])
                            r_rows.append(r_row[i * length: (i + 1) * length])
                    student_num += 1

        q_rows = [row for row in q_rows if len(row) > 2]
        p_rows = [row for row in p_rows if len(row) > 2]
        r_rows = [row for row in r_rows if len(row) > 2]

        self.r_rows = r_rows
        self.p_rows = p_rows
        self.q_num = q_num
        self.q_rows = q_rows

    def __getitem__(self, index):
        return list(
            zip(self.p_rows[index], self.q_rows[index], self.r_rows[index]))

    def __len__(self):
        return len(self.q_rows)
