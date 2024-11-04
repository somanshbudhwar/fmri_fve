from dataset_creation import make_linear_df

df = make_linear_df(100,10, hidden_layers=(50,50))


print(df.shape)
print(df[:,-1].var())