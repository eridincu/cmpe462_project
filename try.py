# from os import listdir
# from os.path import isfile, join

# mypath = "./reviews"
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# i = 0
# for fileName in onlyfiles:
#     i += 1
#     print(i)
#     with open(join(mypath, fileName), "r") as f:
#         lines = f.readlines()
#         print(lines)
#         print()
# import numpy as np
# from sklearn.naive_bayes import MultinomialNB
# rng = np.random.RandomState(1)
# X = rng.randint(5, size=(6, 100))
# print(X)
# y = np.array([1, 2, 3, 4, 5, 6])
# X_test = rng.randint(5, size=(10, 100))
# clf = MultinomialNB()
# print(clf.fit(X, y))
# print(clf.predict(X_test))
def write_to_txt(name, _list):
    txt_file = open(name, "w")
    for element in _list:
        txt_file.write(element + "\n")
    txt_file.close()


def read_from_txt(name):
    txt_file = open(name, "r")
    content_list = txt_file.readlines()
    _list = list()
    for element in content_list:
        _list.append(element.strip())
    return _list

a = ["cagri ve oyku proje yapiyorlar1",
     "cagri ve oyku proje yapiyorlar2",
     "cagri ve oyku proje yapiyorlar3",
     "cagri ve oyku proje yapiyorlar4",
     "cagri ve oyku proje yapiyorlar5",
     "cagri ve oyku proje yapiyorlar6",
     "cagri ve oyku proje yapiyorlar7",
     "cagri ve oyku proje yapiyorlar8",
     "cagri ve oyku proje yapiyorlar9",]
write_to_txt('try.txt', a)
print(read_from_txt('try.txt'))
