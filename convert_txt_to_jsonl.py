import os
import tqdm
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

if __name__ == "__main__":

    # 您可以依需求調整 chunk_size, chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=200
    )

    # 【輸入資料夾】放原始 .txt 檔
    fdir = "txt_files"          
    # 【輸出資料夾】儲存切好的 .jsonl 檔
    outdir = "chunk"

    # 如果輸出資料夾不存在，則創建它
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 取得指定資料夾中所有檔案名稱
    fnames = sorted(os.listdir(fdir))


    for fname in tqdm.tqdm(fnames):
        fpath = os.path.join(fdir, fname)

        # 如果是資料夾或其他非檔案，則跳過
        if not os.path.isfile(fpath):
            continue
        
        # 1. 指定編碼 + 處理無法解碼的字元
        with open(fpath, 'r', encoding='utf-8', errors='replace') as fin:
            file_text = fin.read()

        # 2. 使用 text_splitter 分段
        texts = text_splitter.split_text(file_text.strip())

        # 3. 依每個 chunk 建立 JSON
        saved_text = []
        for i, chunk in enumerate(texts):
            # 移除多餘空白
            chunk_clean = re.sub(r"\s+", " ", chunk)

            # 組合出 ID、標題、內容
            doc_id = f"{fname.replace('.txt', '')}_{i}"
            doc_title = fname.replace(".txt", "")
            doc_content = chunk_clean
            doc_contents = concat(doc_title, chunk_clean)

            data = {
                "id": doc_id,
                "title": doc_title,
                "content": doc_content,
                "contents": doc_contents
            }

            # 確保輸出時中文不會變成 Unicode 編碼
            saved_text.append(json.dumps(data, ensure_ascii=False))

        # 4. 寫入到對應的 .jsonl 檔
        #    檔名: `file1.txt` -> `file1.jsonl`
        out_path = os.path.join(outdir, fname.replace(".txt", ".jsonl"))
        with open(out_path, 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(saved_text))
