import csv
import torch
import random

from .loader import Loader


class AssistDataLoader(Loader):

    def __init__(self, args):
        super(AssistDataLoader, self).__init__(args)

        self.args = args

        self.max_step = 0
        self.n_questions = 0

        self.read_data()

        
    def _load_student(self, f_obj):
        n_steps_str = f_obj.readline().strip() #총 몇 문제
        question_id_str = f_obj.readline().strip() #q 시퀀스
        correct_str = f_obj.readline().strip() #r 시퀀스
        if n_steps_str == "" or question_id_str == "" or correct_str == "":
            return None

        n = int(n_steps_str)
        if self.max_step == 0: 
            self.max_step = n
        
        student = {}
        student['questionId'] = torch.zeros(n, dtype=torch.int64) # id -> int
        question_ids = [int(x) for x in question_id_str.strip(",").split(",")]
        for i, q_id in enumerate(question_ids):
            student['questionId'][i] = int(q_id) # id 0~123, no padding #['questionId']에 q 시퀀스 저장
        if q_id not in self.questions.keys():
            self.questions[q_id] = True #새로운 q_id 가 나올 때마다 추가
            self.n_questions += 1

        student['correct'] = torch.zeros(n, dtype=torch.int64) # answer -> int 
        corrects = [int(x) for x in correct_str.strip(",").split(",")]
        for i, c in enumerate(corrects):
            student['correct'][i] = c #['correct']에 r 시퀀스 저장

        student['n_answers'] = n

        return student


    def _load_csv_data(self, file_name):
        self.questions = {}
        csv_file_path = self.args.ast_path + file_name

        data = []
        longest = 0
        total_n_answers = 0

        with open(csv_file_path, 'r', encoding='utf8') as f:
            #i = 0
            while True:
                #i += 1
                student = self._load_student(f)
                if student is None: break
                #if i == 4 : break

                # add trainind instance (if n_answers > 2)
                if student['n_answers'] >= 2: #2문제 이상 푼 학생에 대하여
                    data.append(student)
                #if len(data) % 100 == 0: print(len(data))
                
                # check max length
                if student['n_answers'] > longest:
                    longest = student['n_answers'] #가장 많이 푼 문제 개수

                total_n_answers += student['n_answers']

        # find max question id
        q_ids = list(sorted(self.questions.keys())) #문제 id sort
        n_questions = max(q_ids) + 1
        return data, longest, total_n_answers, n_questions #각 student 정보를 담은 dict, 가장 많이 푼 문제 개수, 전체 문제 개수, 문제 개수


    def read_data(self):
        print('Loading Assisment...')

        # load training data
        train  = self._load_csv_data("builder_train.csv")
        self.train_data, train_max_len, train_n_answers, n_questions = train
        print('training data:')
        print(' n_students:', len(self.train_data))
        print(' n_questions:', n_questions)
        print(' total answers:', train_n_answers)
        print(' longest:', train_max_len)
        
        # load test data
        test = self._load_csv_data("builder_test.csv")
        self.test_data, test_max_len, text_n_answers, n_questions = test
        print('test data:')
        print(' n_students:', len(self.test_data))
        print(' n_questions:', n_questions)
        print(' total answers:', text_n_answers)
        print(' longest:', test_max_len)
