from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

data = {"Year" :[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
        "House_Price" :[500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]}

df = pd.DataFrame(data)
print(df)

x = df[["Year"]]
y = df["House_Price"]

model_obj = LinearRegression()
model_obj.fit(x,y)

with open("uptor_linear_trained_model.pkl", "wb") as file_obj:
    pickle.dump(model_obj, file_obj)


