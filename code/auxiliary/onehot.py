from sklearn.preprocessing import OneHotEncoder

data = [[1], [3], [5]]

encoder = OneHotEncoder(sparse=False)

encoder.fit(data)

# print(encoder.transform([
#             [0],
#             [1],
#             [2],
#             [3],
#             [4],
#             [5]
#         ]))


print(encoder.fit_transform(data))