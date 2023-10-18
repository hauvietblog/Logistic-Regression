# [Machine Learning cơ bản](https://machinelearningcoban.com/about/)
## [Logistic Regression](https://machinelearningcoban.com/2017/01/27/logisticregression/)
### 1. Bài toán
Xét ánh xạ $$f: \mathbb{R}^m \to \mathbb{R},(x_1,x_2,\dots,x_m) \mapsto f(x_1,x_2,\dots,x_m)= 1 \vee 0$$
Giả sử có n điểm dữ liệu trong không gian m-chiều $(x_1^{(i)},x_2^{(i)},\dots,x_m^{(i)}), i=1 \dots n$, và $f(x_1^{(i)},x_2^{(i)},\dots,x_m^{(i)}) = y^{(i)}$, chúng ta cần tìm một hàm số $f$ sao cho 
$$f(x_1,x_2,\dots,x_m)=\theta(\mathbf{w^T x}) \approx y$$
Giả sử xác suất để một điểm dữ liệu $\mathbf{x}$ rơi vào lớp 1 là $\theta(\mathbf{w^T x})$ và rơi vào lớp 0 là $1-\theta(\mathbf{w^T x})$  
Ký hiệu $z_i = \theta(\mathbf{w^T x_i})$ ta có:
$$P(y_i| \mathbf{x}_i; \mathbf{w}) = z_i^{y_i}(1 - z_i)^{1- y_i}$$  
$$\prod_{i=1}^n$$
