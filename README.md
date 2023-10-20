# [Machine Learning cơ bản](https://machinelearningcoban.com/about/)
## [Logistic Regression](https://machinelearningcoban.com/2017/01/27/logisticregression/)
### 1. Bài toán
Xét ánh xạ  

$$\begin{aligned}
g: \mathbb{R}^m &\to \mathbb{R} \\\\ (x_1,x_2,\dots,x_m) &\mapsto g(x_1,x_2,\dots,x_m)= y
\end{aligned}$$

Giả sử có n điểm dữ liệu trong không gian m-chiều $(x_1^{(i)},x_2^{(i)},\dots,x_m^{(i)}), i=1 \dots n$, và $g(x_1^{(i)},x_2^{(i)},\dots,x_m^{(i)}) = y^{(i)}=0 \vee 1$, chúng ta cần tìm một hàm số $f$ sao cho  

$$\begin{cases} f(x_1,x_2,\dots,x_m)=\theta(\mathbf{w^T x}) \approx g(x_1,x_2,\dots,x_m) \\ 
\lim_{x\rightarrow -\infty } \theta(\mathbf{w^T x})=0 & \\ 
\lim_{x\rightarrow +\infty } \theta(\mathbf{w^T x})=1 &  \end{cases}$$  

Giả sử xác suất để một điểm dữ liệu $\mathbf{x}$ rơi vào lớp 1 là $\theta(\mathbf{w^T x})$ và rơi vào lớp 0 là $1-\theta(\mathbf{w^T x})$  
Ký hiệu $z_i = \theta(\mathbf{w^T x_i})$ ta có:
$$P(y_i| \mathbf{x}_i; \mathbf{w}) = z_i^{y_i}(1 - z_i)^{1- y_i}$$
Xét toàn bộ dữ liệu với $\mathbf{X=(x_1,x_2,\dots,x_n)}$ và $\mathbf{y}=(y_1,y_2,\dots,y_n),$ chúng ta cần tìm $w$ để cho biểu thức sau đạt giá trị lớn nhất:
$$P(\mathbf{y}|\mathbf{X}; \mathbf{w})$$
Giả sử rằng các điểm dữ liệu là ngẫu nhiên độc và lập với nhau, ta có thể viết:

$$P(\mathbf{y}|\mathbf{X}; \mathbf{w}) =\prod_{i=1}^n P(y_i| \mathbf{x}_i; \mathbf{w}) = \prod\_{i=1}^n z_i^{y_i}(1 - z_i)^{1- y_i} $$  
Trực tiếp tối ưu hàm số này theo $\mathbf{w}$ không đơn giản, do đó ta sẽ tối ưu hàm số sau

$$J(\mathbf{w}) = -\log P(\mathbf{y}|\mathbf{X}; \mathbf{w})= -\sum\_{i=1}^n(y_i \log {z}_i + (1-y_i) \log (1 - {z}_i))$$  
Hàm mất mát với chỉ một điểm dữ liệu $\mathbf{(x_i,y_i)}$ là:  

$$J(\mathbf{w}; \mathbf{x}_i, y_i) = -(y_i \log {z}_i + (1-y_i) \log (1 - {z}_i))$$
 
$$\frac{\partial J(\mathbf{w}; \mathbf{x}_i, y_i)}{\partial \mathbf{w}} = -(\frac{y_i}{z_i} - \frac{1- y_i}{1 - z_i} ) \frac{\partial z_i}{\partial \mathbf{w}}=\frac{z_i - y_i}{z_i(1 - z_i)} \frac{\partial z_i}{\partial \mathbf{w}}$$  
Sau nhiều phép biến đổi ta tìm được 

$$z=\frac{1}{1+e^{-\mathbf{w^T x}}} = \theta(\mathbf{w^T x})$$  

Vậy công thức cập nhật (theo thuật toán SGD) cho logistic regression là:
$$\mathbf{w} = \mathbf{w} + \eta(y_i - z_i)\mathbf{x}_i$$
