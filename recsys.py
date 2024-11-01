'''import pandas as pd
import numpy as np
import sys


def sgd(S, m, n, k, alpha, lamb, epochs):
    P =  np.random.uniform(low=0, high=5, size=(m, k))
    Q = np.random.uniform(low=0, high=5, size=(k, n))
    
    P = P/np.sqrt(5)
    Q = Q/np.sqrt(5)
    
    for _ in epochs:
        for row in S.itertuples():
            user_id = row.UserId
            item_id = row.ItemId
            error = row.NormRating - np.inner(P[user_id, :], Q[:, item_id])
            newP = P[user_id, :] + alpha(error*Q[:,item_id] - lamb*P[user_id, :])
            newQ = Q[:,item_id] + alpha(error*P[user_id, :] - lamb*Q[:,item_id])
            P[user_id, :] = newP
            Q[:, item_id] = newQ

    
    return P, Q
    
    

def data_input_treatment(rating_file):
    #Tratamento de arquivo de entrada um dataframe
    temp = pd.read_csv(rating_file)
    new = temp['UserId:ItemId'].str.split(pat = ":", expand = True)
    new.columns = ['UserId', 'ItemId']
    rating_df = pd.concat([new, temp['Rating']], axis=1)
    return rating_df

def find_target_rating(target_df, users, items, P, Q):
    df = target_df['UserId:ItemId'].str.split(pat = ":", expand = True)
    df.columns = ['UserId', 'ItemId']
    ratings = pd.DataFrame()
    for row in df.itertuples():
        user_index =  np.where(users == row['UserId'])
        item_index =  np.where(users == row['ItemId'])
        rating = np.dot(P[user_index,:], Q[:, item_index])
        ratings.loc[row.Index] = [rating]
    return ratings
    
def main():
    
    ratings_file = sys.argv[1]
    targets_file = sys.argv[2]
    
    df = data_input_treatment(ratings_file)
    target_df = pd.read_csv(targets_file)

    users = df['UserId'].unique()
    items = df['ItemId'].unique()

    users2 = np.array(df['UserId'].unique())
    items2 = np.array(df['ItemId'].unique())

    print(users.size)
    
    factor_matrix = np.zeros((users2.size, items2.size)) 
    ##PREENCHER MATRIZ COM OS RATING QUE JA TEMOS
    for row in df.iterrows:
        user_index =  np.where(users == row['UserId'])
        item_index =  np.where(users == row['ItemId'])
        factor_matrix[user_index][item_index] = row['Rating']

    P, Q = sgd(factor_matrix, users.size, items.size, 200, 0.005,0.002,30)

    scores = find_target_rating(target_df, users, items, P, Q)

    output = pd.concat([target_df, scores], axis=1)
    
    output.to_csv('output_file.csv', index=False)


if __name__ == '__main__': 
    main()
'''
import pandas as pd
import numpy as np
import sys

def sgd(df, users, items, k, alpha, lamb, epochs):
    # Número de usuários e itens únicos
    m, n = len(users), len(items)
    
    # Inicializa as matrizes P e Q com valores aleatórios entre 0 e 5
    P = np.random.uniform(0, 5, size=(m, k)) / np.sqrt(5)
    Q = np.random.uniform(0, 5, size=(k, n)) / np.sqrt(5)
    
    # Mapeia IDs de usuários e itens para índices (para acesso eficiente)
    user_map = {user_id: idx for idx, user_id in enumerate(users)}
    item_map = {item_id: idx for idx, item_id in enumerate(items)}
    
    # Treinamento do SGD por número de épocas
    for _ in range(epochs):
        for row in df.itertuples():
            # Obtém os índices do usuário e do item a partir do mapeamento
            user_id = user_map[row.UserId]
            item_id = item_map[row.ItemId]
            
            # Calcula o erro entre a avaliação real e a avaliação prevista
            error = row.Rating - np.inner(P[user_id, :], Q[:, item_id])
            
            # Atualiza os vetores de usuário e item com gradiente descendente
            P[user_id, :] += alpha * (error * Q[:, item_id] - lamb * P[user_id, :])
            Q[:, item_id] += alpha * (error * P[user_id, :] - lamb * Q[:, item_id])
    
    return P, Q

def data_input_treatment(rating_file):
    # Lê o arquivo CSV e separa UserId e ItemId
    temp = pd.read_csv(rating_file)
    new = temp['UserId:ItemId'].str.split(pat=":", expand=True)
    new.columns = ['UserId', 'ItemId']
    rating_df = pd.concat([new, temp['Rating']], axis=1)
    return rating_df

def find_target_rating(target_df, users, items, P, Q):
    # Converte users e items para arrays NumPy, se ainda não estiverem
    users = np.array(users)
    items = np.array(items)
    
    # Separa UserId e ItemId no DataFrame de alvo
    df = target_df['UserId:ItemId'].str.split(pat=":", expand=True)
    df.columns = ['UserId', 'ItemId']
    
    # Calcula ratings previstos
    ratings = []
    for row in df.itertuples():
        # Encontra o índice do usuário e do item nas listas users e items usando np.where
        user_index = np.where(users == row.UserId)[0][0]
        item_index = np.where(items == row.ItemId)[0][0]
        
        # Calcula a previsão de rating usando as matrizes P e Q
        rating = np.dot(P[user_index, :], Q[:, item_index])
        ratings.append(rating)
    
    # Constrói DataFrame com ratings previstos
    ratings_df = pd.DataFrame({'PredictedRating': ratings})
    return ratings_df

def main():
    ratings_file = sys.argv[1]
    targets_file = sys.argv[2]
    
    # Tratamento dos dados de entrada
    df = data_input_treatment(ratings_file)
    target_df = pd.read_csv(targets_file)

    # Obtém listas de IDs únicos de usuários e itens
    users = df['UserId'].unique()
    items = df['ItemId'].unique()

    # Treina as matrizes P e Q com SGD
    P, Q = sgd(df, users, items, k=200, alpha=0.005, lamb=0.002, epochs=30)

    # Calcula os ratings previstos para os alvos
    scores = find_target_rating(target_df, users, items, P, Q)

    # Salva o resultado em um arquivo CSV
    output = pd.concat([target_df, scores], axis=1)
    output.to_csv('output_file.csv', index=False)

if __name__ == '__main__': 
    main()

    
