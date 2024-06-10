from fastapi import FastAPI, UploadFile, File
import pickle
import pandas as pd
import shutil
from fastapi.responses import FileResponse
from sklearn.preprocessing import LabelEncoder

# app = FastAPI()
# moedl = pickle.load(open("text_clf.pkl", "rb"))
# tfidf_vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))
# label_encoder = pickle.load(open("text_label_encoder.pkl", "rb"))


# # @app.get("/")
# # def processor():
# #     return "HELLO WORD!"
# @app.get("/")
# def processor():

#     input_df = pd.read_csv("test.csv")

#     features = tfidf_vectorizer.transform(input_df["body"])
#     predictions = model.predict(features)
#     input_df["category"] = label_encoder.inverse_transform(predictions)
#     outpit_df = input_df[["id", "category"]]
#     outpit_df.to_csv("result.csv", index=False)
#     return FileResponse("result.csv")


# @app.post("/upload")
# async def uploadFile(file: UploadFile = File(...)):
#     with open("test.csv", "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     return {"file_name": file.filename}
from fastapi import FastAPI, UploadFile, File
import pickle
import pandas as pd
import shutil
from fastapi.responses import FileResponse
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# 加載模型和其他相關物件
model = pickle.load(open("text_clf.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("text_label_encoder.pkl", "rb"))


@app.get("/")
def processor():
    # 讀取測試文件
    input_df = pd.read_csv("test.csv")

    # 提取特徵並進行預測
    features = tfidf_vectorizer.transform(input_df["body"])
    predictions = model.predict(features)

    # 使用正確的 inverse_transform 方法將編碼轉回原本的標籤
    input_df["category"] = label_encoder.inverse_transform(predictions)
    output_df = input_df[["id", "category"]]

    # 將結果保存為 CSV 文件
    output_df.to_csv("result.csv", index=False)
    return FileResponse("result.csv")


@app.post("/upload")
async def uploadFile(file: UploadFile = File(...)):
    # 保存上傳的文件
    with open("test.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"file_name": file.filename}
