import sys
from sys import argv
import pandas as pd
import numpy as np
import time

start = time.time()
columns = ['user_id','item_id','rating','time_stamp']
train_path = argv[1]
test_path = argv[2]
train_data = pd.read_csv(train_path, sep='\t',names=columns)
test_data = pd.read_csv(test_path, sep='\t',names=columns)

def BuildRatingMatrix(data):
    '''
    user_id를 행, item_id를 열로 하는 rating matrix를 얻기 위해
    unique한 user_id, item_id를 데이터에서 추출
    '''
    users = list(np.unique(train_data['user_id']))
    items = list(np.unique(train_data['item_id']))
    
    # train과 test의 item columns 를 맞추기 위해 test의 item column을 추출하여
    # train과 test의 차집합을 구하고 그것을 train에 append한다.
    test_items = np.unique(test_data['item_id'])
    items += list(np.setdiff1d(test_items,items))
    
    # adjacency matrix 생성 but empty matrix
    df = pd.DataFrame(columns = items, index = users).fillna(0)

    # data grouping using user_id
    user_group = data.groupby(train_data['user_id'])

    # extract item_list from df
    item_set = list(df.columns)

    # build rating matrix
    '''
    1. user_id를 기준으로 grouping한 데이터(sub_data)를 grouping한 순서대로
       user_id, sub_data로 looping
    2. 만일 user 1 이라면, user 1로 grouping 된 데이터에서
       item_id들을 하나하나 꺼내면서 그 item이 받은 rating 점수를
       rating matrix에서 user 1행과 item_id가 만나는 셀에 넣는다.
    '''
    for user, sub_data in user_group:
        for i in list(sub_data.index):
            item = sub_data.loc[i, 'item_id']
            df.loc[user,item] = sub_data.loc[i,'rating']
    return df

class Perform_MF():
    def __init__(self, R, K, Alpha, Beta, numOFiteration):
        """
        Perform matrix factorization to fill sparse
        entries in a matrix.
        
        Arguments
        - R (matrix)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - Alpha (float) : learning rate
        - Beta (float)  : regularization parameter
        """
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.Alpha = Alpha
        self.Beta = Beta
        self.iterations = numOFiteration

    def Train(self):
        '''
         Initialize user and item latent feature matrice
         scale : 표준편차, size : matrix의 크기
         initialization of U(User * latent K variables)
         initialization of I(Items * latent K variables)
         users × k로 만든 matrix
         items × k로 만든 matrix
        '''
        self.User = np.random.normal(scale=1/self.K, size=(self.num_users, self.K))
        self.Item = np.random.normal(scale=1/self.K, size=(self.num_items, self.K))

        # Initialize the biases by zero at the first time
        self.B = np.mean(self.R[np.where(self.R != 0)]) # R의 value중 0이 아닌 값의 전체 평균
        self.B_U = np.zeros(self.num_users) # num_user size의 리스트에 0 채우기
        self.B_I = np.zeros(self.num_items) # num_items size의 리스트에 0 채우기
        

        # Create a list of training samples
        # (index(i), column(j), iXj) 이 세 값들을 하나의 set으로 하는 리스트 생성
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples) # samples를 random으로 섞기
            self.Perform_SGD()
            MSE = self.Check_MSE()
            training_process.append((i, MSE))
        return training_process

    def Check_MSE(self):
        """
        A function to compute the total mean square error
        """
        X, Y = self.R.nonzero()
        predicted = self.Build_full_matrix()
        error = 0
        for x, y in zip(X, Y):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def Perform_SGD(self):
        """
        https://mangkyu.tistory.com/62
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Compute prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            # 편미분 진행
            # https://analytics4everything.tistory.com/104
            self.B_U[i] += self.Alpha * (e - self.Beta * self.B_U[i])
            self.B_I[j] += self.Alpha * (e - self.Beta * self.B_I[j])

            # Update user and item latent feature matrices
            self.User[i, :] += self.Alpha * (e * self.Item[j, :] - self.Beta * self.User[i,:])
            self.Item[j, :] += self.Alpha * (e * self.User[i, :] - self.Beta * self.Item[j,:])

    def get_rating(self, i, j):
        
        """
        Get the predicted rating of user i and item j
        B : 전체 평균
        B_I : 아이템의 편차
        B_U : 사용자의 평균
        User[i, :] : User에서 사용자의 벡터
        Item[j, :] : Item에서 아이템에 대한 벡터
        """
        prediction = self.B + self.B_U[i] + self.B_I[j] + self.User[i, :].dot(self.Item[j, :].T)
        return prediction

    def Build_full_matrix(self):
        """
        Compute the full matrix using the resultant biases, User and Item
        """
        return self.B + self.B_U[:,np.newaxis] + self.B_I[np.newaxis:,] + self.User.dot(self.Item.T)
def Execute_recommendation(result_data, test_data):
    output = ''
    for i in test_data.index:
        user = test_data['user_id'][i]
        item = test_data['item_id'][i]
        output += str(user)+'\t'+ str(item) + '\t' + str(int(result[item][user])) + '\n'
    return output

if __name__ == '__main__':
    '''-------------Building Matrix------------'''
    print("\nBuilding sparse Matrix...")
    Matrix = BuildRatingMatrix(train_data)
    print()
    mXn = np.array(Matrix.values)
    #print(mXn)
    #print("execution time to build Rating Matrix :", str(round((time.time() - start),1)) +'s')
    print("Build done")
    print()
    print("Performing Matrix Factorization...\n")
    
    '''---------------Perform Matrix Factorization-----------------'''
    '''
    alpha : 기울기에 alpha(학습률 or Learning Rate)라고 하는 스칼라를 곱해서 다음 지점을 결정
            ex : 기울기가 2.5이고 alpha가 0.01이면 현재점에서 0.025 떨어진 곳에 다음 점을 정한다.
    beta : 규제화 상수. to avoid overfitting, make a bit change for result
    '''
    MF = Perform_MF(mXn, K=15, Alpha=0.002, Beta=0.02, numOFiteration = 80)
    MF.Train()

    trained_matrix = MF.Build_full_matrix()
    trained_matrix = np.around(trained_matrix)
    print(trained_matrix)
    print()
    
    
    '''-----------------Execute recommendation-----------------'''
    result = pd.DataFrame(trained_matrix, index = Matrix.index, columns = Matrix.columns)
    output_file = Execute_recommendation(result, test_data)
    print("execution time :", str(round((time.time() - start),1)) +'s')
    output_path = '_prediction.txt'
    output_path = train_path + output_path
    f = open(output_path, "w")
    f.write(output_file)
    f.close