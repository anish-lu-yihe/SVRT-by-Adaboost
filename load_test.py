from load_svrt import load_svrt_parsing

for i in range(23):
    X, y = load_svrt_parsing(i+1)
    print(X[0])
    print(X[-1])
