# AI-worker

## Giới thiệu

**AI-worker** là một hệ thống backend worker tự động thu thập dữ liệu thị trường crypto, tính toán các chỉ báo kỹ thuật, dự đoán xu hướng bằng AI và cập nhật kết quả vào MongoDB. Hệ thống hỗ trợ việc phân tích, nghiên cứu và triển khai chiến lược giao dịch thông minh.

Repo này đóng vai trò là **backend worker**, đồng hành với một backend server chính nằm tại: [BE\_WEB\_Crypto](https://github.com/Nguyenchitrai62/BE_WEB_Crypto). Toàn bộ hệ thống đã được triển khai tại: [https://nguyenchitrai.id.vn](https://nguyenchitrai.id.vn)

## Các chức năng chính

* **Crawl dữ liệu thị trường** từ các sàn giao dịch thông qua CCXT.
* **Thêm chỉ báo kỹ thuật** như EMA, MACD, RSI, Bollinger Bands, v.v.
* **Dự đoán xu hướng thị trường** bằng mô hình AI (Transformer).
* **Cập nhật dữ liệu dự đoán và chỉ báo vào MongoDB** để sử dụng cho các hệ thống trực quan hóa hoặc cảnh báo.

## Cài đặt

### Yêu cầu hệ thống

* Python >= 3.8
* MongoDB Atlas hoặc local MongoDB server

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Khởi chạy hệ thống

Chạy file `app.py` để thực hiện toàn bộ luồng:

* Crawl dữ liệu
* Tính toán chỉ báo kỹ thuật
* Dự đoán bằng mô hình AI (`transformer_model_balanced.keras`)
* Cập nhật kết quả vào MongoDB

```bash
python app.py
```
