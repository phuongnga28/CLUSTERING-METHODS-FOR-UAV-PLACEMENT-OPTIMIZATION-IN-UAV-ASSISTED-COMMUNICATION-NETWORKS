# Code khoá luận tốt nghiệp

## 1. Khung chương trình

### 1.1. Mô tả

- Chương trình sử dụng ngôn ngữ lập trình Python, được chuyển thể từ chương trình matlab của bài báo gốc

### 1.2. Cài đặt

- Cài đặt các thư viện cần thiết: numpy, matplotlib, scipy, pandas, sklearn

```bash
pip install -r requirements.txt
```

### 1.3. Sử dụng

- Chạy chương trình

```bash
python main.py
```

### 1.4. Cấu trúc file main

- Gồm có 4 phần chính:

  - Phần 1: Khởi tạo dữ liệu, Thiết lập các tham số đầu vào
  - Phần 2: Sử dụng các thuật toán từ thư viện để tìm vị trí cho các UAV
  - Phần 3: Tính toán các thông số cần thiết dựa theo công thức trong bài báo
  - Phần 4: Hiển thị kết quả ra terminal và vẽ đồ thị

### 1.5. Các file chính

- main.py: File chính chứa toàn bộ chương trình
- generate_uniform_data.py: Tạo dữ liệu ngẫu nhiên theo phân phối đều
- k_means_centroids.py: Tìm vị trí các UAV bằng thuật toán KMeans
- k_medoids_centroids.py: Tìm vị trí các UAV bằng thuật toán KMedoids
- minibatch_k_means_centroids.py: Tìm vị trí các UAV bằng thuật toán MiniBatchKMeans
- optimize_pow_height_cluster.py: Tìm công suất và chiều cao và một số tham số tối ưu khác của UAV
- plot_all_UAV_ranges.py: Vẽ đồ thị biểu diễn vị trí các UAV và phạm vi hoạt động của chúng
- plot_range_with_density.py: Vẽ đồ thị biểu diễn phạm vi hoạt động của các UAV và mật độ của các điểm dữ liệu

## 2. Chú thích

- File uniform_data.npy: Dữ liệu ngẫu nhiên theo phân phối đều để đánh giá khách quan hơn về hiệu suất của các thuật toán. Tránh trường hợp dữ liệu ngẫu nhiên
- File minibatch_draw.py để vẽ các bước thực hiện phân cụm và tìm vị trí UAV bằng thuật toán MiniBatchKMeans trên tập dữ liệu mẫu
- Các bước thực hiện đã được trình bày kỹ trong khoá luận tốt nghiệp

## Made with passion by [Nga Phuong Nguyen]
