# AI-worker

## Giới thiệu

**AI-worker** là một hệ thống tự động thu thập dữ liệu thị trường crypto, tính toán các chỉ báo kỹ thuật, dự đoán xu hướng bằng AI và cập nhật kết quả vào MongoDB. Hệ thống hỗ trợ việc phân tích, nghiên cứu và triển khai chiến lược giao dịch thông minh.

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

Để khởi chạy toàn bộ luồng xử lý từ thu thập dữ liệu đến cập nhật kết quả vào MongoDB, chạy file `app.py`:

```bash
python app.py
```

## Cấu trúc dự án

```
AI-worker/
├── app.py                       # Điểm khởi đầu của pipeline: crawl -> indicator -> dự đoán -> MongoDB
├── crawl.py                    # Crawl dữ liệu mới cho từng cặp coin
├── crawl_all_volume.py        # Crawl tất cả volume thị trường
├── add_indicator.py           # Tính toán các chỉ báo kỹ thuật
├── model.py                   # Load và dự đoán bằng mô hình Transformer
├── transformer_model_balanced.keras  # Trọng số của mô hình đã huấn luyện
├── update_data_mongoDB.py     # Cập nhật kết quả vào MongoDB
├── requirements.txt           # Danh sách thư viện cần cài đặt
```

## Đóng góp

Hãy tạo pull request hoặc mở issue nếu bạn muốn đóng góp mã nguồn hoặc báo lỗi.

---

**Người phát triển**: [Nguyenchitrai62](https://github.com/Nguyenchitrai62)
