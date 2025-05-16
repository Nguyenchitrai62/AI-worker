import os
import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()
password = os.getenv('MONGO_PASSWORD')

# Kết nối MongoDB Atlas
uri = f"mongodb+srv://trainguyenchi30:{password}@cryptodata.t2i1je2.mongodb.net/?retryWrites=true&w=majority&appName=CryptoData"
client = MongoClient(uri, server_api=ServerApi('1'))

# Truy cập DB và Collection
db = client['my_database']
collection = db['my_collection']

# Đọc và xử lý dữ liệu từ CSV
data = pd.read_csv('FLOW_data.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Convert về dict
data_dict = data.to_dict(orient='records')

# ✅ Cập nhật từng dòng theo Date nếu có thay đổi thực sự
updated, inserted = 0, 0
for doc in data_dict:
    existing_doc = collection.find_one({'Date': doc['Date']})

    if existing_doc is None:
        collection.insert_one(doc)
        inserted += 1
    else:
        # So sánh từng field, bỏ qua _id
        different = any(existing_doc.get(k) != doc.get(k) for k in doc.keys() if k != '_id')
        if different:
            collection.update_one({'Date': doc['Date']}, {'$set': doc})
            updated += 1

print(f"✅ Đã cập nhật {updated} dòng, chèn mới {inserted} dòng vào MongoDB.")
