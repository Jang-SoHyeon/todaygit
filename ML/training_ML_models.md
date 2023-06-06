## 목차  
1. [Part 1](#part-1)  
    1-1 [선형 회귀(Linear Regression)](#선형-회귀linear-regression)  
    1-2 [경사 하강법(Gradient Descent)](#경사-하강법gradient-descent)  
    1-3 [다항 회귀(Polynomial Regression)](#다항-회귀polynomial-regression)
2. Part 2  


### Part 1
* 훈련 세부 진행 과정 및 원리를 이해하고 있으면 적절한 모델, 올바른 훈련 알고리즘, 작업에 적합한 하이퍼파라미터를 찾는데 도움이 됨    

다항회귀  
: 비선형 데이터셋을 학습하는데 활용 가능  
: 선형 회귀보다 모델 파라미터가 많아 훈련 데이터에 과대적합되기 더 쉬움

학습 곡선(learning curve)을 사용해 모델 과대적합 여부를 감지하는 방법  
과대적합을 완화하기 위한 규제 기법  
분류(classification) 작업에 사용 가능한 회귀 모델인 로지스틱 회귀와 소프트맥스 회귀   
<hr>

#### 선형 회귀(Linear Regression)

<b>선형 회귀 모델</b>  
: n개의 입력 특성을 사용하여 주어진 샘플의 타깃/레이블을 에측하는 선형 회귀 모델  
$\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n$

- 입력 특성 가중치 합(weighted sum)과 편향(bias term)이라는 상수를 더해 예측을 만듦  
- $\hat{y}$ : 주어진 샘플에 대한 (결과) 예측 값  
- n : 각 샘플의 특성 수 
- $x_i (1<= i <= n)$ : 샘플의 i번째 특성 값
- $\theta_j (0<=j<=n)$ : j번째 모델 파라미터
    - $\theta_1\cdots\theta_n$ : 대응되는 입력 특성 $x_i$에 곱해지는 가중치(weight)
    - $\theta_0$ : bias term(편향)


<b>선형 회귀 모델 : 벡터 형태 표현</b>  
$\hat{y} = h_\theta(x) = \theta \cdot x$

- $h_\theta()$ : 선형 회귀 모델의 예측 함수  
- $\theta : \theta_0 ~ \theta_n$으로 이루어진 모델
- $x$ : 주어진 샘플의 입력 특성 벡터($x_0$부터 $x_n$으로 구성되며, $x_0=1$이 추가 됨. (n+1)x1의 벡터)
- $\theta\cdot x = \theta_0x_0 + \theta_1x_1 + \cdots + \theta_nx_n = \theta^tx(\theta^t와 x의 행렬 곱셈)$
    - $\theta^t : \theta$의 transpose => 1x(n+1)의 row 벡터가 됨
    - x는 (n+1)x1의 열 벡터


- 넘파이 2차원 배열 표현
  
||표기법|shape|
|:-----:|:----:|:---:|
|타깃/레이블, 예측값|$y, \hat{y}$|(m,1)|
|모델 파라미터 (벡터)|$\theta$|(n+1,1)|
|전체 훈련셋 (행렬)|$X$|(m,n+1)|
|i번째 훈련 샘플|$x^{(i)}$|(n+1,1)|
m : 훈련 샘플들의 총 개수, n : 샘플 당 특성 수(타깃 특성은 제외)  
  
**선형 회귀 모델 훈련**
- 모델을 훈련시킨다는 것은 모델이 훈련셋에 가장 잘 맞도록 모델 파라미터를 설정하는 것
- 이를 위해 모델이 훈련 데이터에 얼마나 잘 맞는지를 측정할 수 있는 지표가 필요함
(회귀에 가장 널리 사용되는 성능 측정 지표는 평균 제곱근 오차(RMSE))  
- 최종 목표는 훈련 데이터에 대한 RMSE를 최소화하는 모델 파라미터 벡터 $\theta$를 찾는 것임
- 실제로는 RMSE보다 MSE를 최소화하는 모델 파라미터 $\theta$를 찾는 것이 더 간단해 MSE가 많이 사용 됨
- MSE를 최소화하는 동일한 모델 파라미터를 사용하면 RMSE 또한 최소라는 것이 보장 됨  
  
**선형 회귀 모델의 MSE 비용 함수**

MSE($X,h_\theta$) = $\left(\frac{1}{3}\right)\displaystyle\sum_{i=1}^{m} (\theta^Tx^{(i)} - y^{(i)})^2$  
: m개 훈련 샘플들에 대한 모델의 예측이 타깃(정답)과 평균적으로 얼마나 차이가 나는지를 보여주는 지표
  
- m : 훈련셋에 샘플들의 총 개수
- $\theta^T$ : 모델 파라미터 벡터 $\theta$의 transpose => 1x(n+1)의 행 벡터  
- $x^{(i)}$ : 훈련셋 내 i번째 샘플의 특성 벡터. (n+1)x1의 열 벡터  
- $y^{(i)}$ : i번쨰 훈련 샘플의 타깃/레이블  

RMSE$(X, h)$  
 $ = \sqrt{(\displaystyle\sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}))^2}$


#### 경사 하강법(Gradient Descent)




#### 다항 회귀(Polynomial Regression)