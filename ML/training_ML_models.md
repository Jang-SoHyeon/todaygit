## 목차  
1. [Part 1](#part-1)  
    1-1 [선형 회귀(Linear Regression)](#선형-회귀linear-regression)  
    1-2 [경사 하강법(Gradient Descent)](#경사-하강법gradient-descent)  
    1-3 [다항 회귀(Polynomial Regression)](#다항-회귀polynomial-regression)
2. [Part 2](#part-2)  
    2-1 [규제를 사용하는 선형 회귀 모델](#규제regularization를-사용하는-선형-회귀-모델)  
    2-2 [로지스틱 회귀](#로지스틱-회귀logistic-regression)  
    2-3 [결정 경계](#결정-경계)  
    2-4 [소프트맥스 회귀](#소프트맥스-회귀softmax-regression)  


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
(즉, 훈련셋에 대해 비용 함수를 최소화 하는 (최적의) 모델 파라미터를 계산해내는 방법)  
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

RMSE($X, h) = \sqrt{\displaystyle\sum_{i=1}^{m} ((h(x^{(i)}) - y^{(i)}))^2}$

**선형 회귀 모델을 훈련시키는 2가지 방법**  

방식 1 : 정규방정식 또는 특이값 분해(SVD) 활용  
- 드물지만 수학적으로 비용함수를 최소화하는 $\theta$ 값을 직접 계산할 수 있는 경우 활용
- 계산복잡도가 O($n^2$) 이상인 행렬 연산을 수행해야 함.
- 따라서 특성 수(n)가 많은 경우 메모리 관리 및 시간복잡도 문제 때문에 비효율적임  
  
정규 방정식  
 $\hat{\theta} = (X^TX)^-1X^Ty$  
문제점 : 행렬 곱셈과 역행렬 계산은 계싼 복잡도가 O(n^2.4)이상이고 항상 역행렬 계산이 가능한 것도 아님  
  
SVD(특이값 분해) 활용    
- 특이값 분해를 활용하여 얻어지는 유사 역행렬(e.g., 무어-펜로즈(Moore-Penrose) 역행렬) X+ 계산이 보다 효율적임. 계산 복잡도는 O(n^2).
$\hat{\theta} = X^+y$  
  
방식 2 : 경사하강법    

#### 경사 하강법(Gradient Descent)

> 훈련셋을 이용한 모델 훈련 중에 모델의 비용 함수의 크기가 줄어드는 방향으로 모델 프로미터를 조금씩 반복적으로 조정해가는 기법   
  
(훈련 시 보용함수(MSE)에 입력으로 주어지는 훈련 데이터는 고정값으로 간주되며, 반면에 비용함수의 모델 파라미터가 변수에 해당함)
  
MSE($X,h_0)= \left(\frac{1}{m}\right)\displaystyle\sum_{i=1}^{m} ((\theta^Tx^{(i)}-y^{(i)})^2)$  
최종목표 : 비용 함수의 minimum에 도달할 수 있는 모델 파라미터 값 조합 찾기  

  
최적 학습 모델  
: 주어진 훈련 데이터에 대해 비용함수(cost function, e.g., MSE)를 최소화 하는 모델 파라미터로 설정된 모델  

모델 파라미터  
- 주어진 인스턴스에 대한 예측값을 리턴하는 함수로 구현되는 머신러닝 모델에 포함된 파라미터  
- 예제 : 다음과 같은 선형 회귀 모델에 사용되는 편향과 가중치 파라미터  
$\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n$

**경사 하강법 관련 주요 개념**
1. 비용함수  
: 머신러닝 모델의 예측값이 실제 타깃값과 얼마나 차이가 나는지를 계산해주는 함수  
(e.g., 선형 회귀 모델의 평균 제곱 오차(MSE, mean squared error))
: 선형 회귀 모델의 성능 평가를 위해서는 RMSE가 일반적으로 많이 사용됨
(훈련 과정에서는 MSE가 많이 사용됨 - 계산이 간단하기 때문)
(모델 성능 평가에서는 RMSE가 사용됨)
(MSE를 최소화하는 동일한 모델 파라미터를 사용하면 RMSE 또한 최소화 된다는 것이 보장됨)

2. 전역 최소값
: 모델 파라미터 조정을 통해 도달할 수 있는 비용함수의 가능한 최소값  
<img src = "../image/전역 최소값.png" width = 50%>

전역 최솟값 vs 지역 최솟값   
<img src = "../image/전역 최소값 vs 지역 최소값.png" width = 50%>

그레디언트 벡터(Gradient Vector)  
: 비용함수(MSE)를 $\theta_0 ~ \theta_n$ 각 모델 파라미터로 편미분해서 구해진 (n+1)개 편도함수(partial derivative)들로 이루어진 벡터  
: (n+1)-차원 공간 상의 한 point를 나타내며, 방향과 크기에 대한 정보를 제공  
: 그레이언트 벡터가 가리키는 방향의 반대 방향으로 모델 파라미터 벡터 $\theta$를 조정함으로써 비용함수의 최소값에 접근해갈 수 있음 
<gradient descent step>  

<img src = "../image/Gradient descent step.png" width = 50%> 

$\eta$(eta)는 학습률(learning rate)  

: 다음 식을 통해 주어진 (고정값) 훈련 데이터(X,y)와 특정 모델 파라미터 값 $\theta$ 일 때 그레디언트 벡터를 계산할 수 있다.  
  
<img src = "../image/그레디언트 벡터.png" width = 50%>
 
학습률(lerning rate, $eta$)  
: 매 gradient descent step마다 모델 파라미터 $\theta$ 조정 폭을 결정

e.g., 경사 하강법에 의한 선형 회귀 모델 파라미터 조정 과정
- 모델 파라미터 벡터 $\theta$를 랜덤하게 초기화하여 훈련 시작 
- 현재 설정된 모델 파라미터 벡터 $\theta$와 (batch size로 지정된 수의)훈련 샘플들을 이용하여 그레디언트 벡터를 계산
- 그레디언트 벡터의 크기(norm)이 허용오차(tolerance)보다 작은지 확인  
-> 만약 작다면, 비용함수의 최소값에 근접했음을 의미하며 최적의 모델 파라미터를 찾은 것이므로 훈련 과정을 멈춤  
-> 아니라면, 

<img src = "../image/Gradient descent step.png" width = 50%>

경사 하강법에서 학습률이 너무 작으면 비용함수의 전역 최소값에 도달하기까지 
시간이 오래 걸림   

<img src = "../image/학습률이 작을 경우 비용함수.png" width = 50%>

너무 크면 비용함수의전역 최소값에 도달하지 못 할 수 있음  

<img src = "../image/학습률이 클 경우 비용함수.png" width = 50%>

모델 비용함수 곡선이 아래와 같으면 경사 하강법 알고리즘 실행 결과로 전역 최소값이 아닌 지역 최소값으로 귀결될 수 있음

<img src = "../image/독특한 모델 비용함수 곡선.png" width = 50%>  

* 선형 회귀 모델의 비용함수는 convex function에 해당함
- 오직 전역 최소값만 존재(지역 최소값 없음)
- 학습률이 너무 크지 않다면 결국엔 전역 최소값에 수렴 가능함  

**특성 스케일링의 중요성**  

데이터셋에 포함된 특성들의 스케일을 통일시키면 학습에 걸리는 시간이 단축됨  

**경사 하강법의 하이퍼파라미터**  
:경사 하강법을 통한 모델 훈련 과저잉 어떻게 진행되도록 할 것인지를 설정하기 위한 파라미터.  
배치(batch) 사이즈 : 현재 모델 파라미터 벡터 설정에 대한 그레디언트 벡터를 계산에 사용되는 훈련 샘플의 수  
- 배치 경사 하강법 : 훈련셋 전체를 사용하여 그레디언트 벡터를 계산  
- 확률적 경사 하강법 : 랜덤하게 선택된 훈련 샘플 하나를 사용하여 그레디언트 벡터를 계산  
- 미니 배치 경사 하강법 : 훈련셋의 랜덤 서브셋을 사용하여 그레디언트 벡터를 계산  

에포크(epoch) 
: 총 훈련 샘플 수 m개만큼의 훈련 데이터에 대한 학습이 이루어지는 주기  
(m개만큼의 훈련 샘플들을 이용해 모델 파라미터 벡터 조정이 이루어지는 주기)  
: 1 에포크 동안 1번 이상의 gradient descent step들이 수행된다.  

스텝(graidnet descent step)  
: 지정된 배치 사이즈만큼의 훈련 샘플들에 대해 그레디언트 벡터를 계산하고 모델 파라미터 벡터 조정을 수행하는 주기  
- 1 에포크 당 총 스텝 수 = 훈련 샘플 수 / 배치 크기  
- 예 : 훈련셋의 크기가 2000이고 배치 사이즈 10이면, 1 에포크 동안 총 200번의 스텝이 실행됨  

허용오차(tolerance) : 그레디언트 벡터의 크기(norm)가 허용오차보다 작아지면 비용함수의 최소값에 근접했음을 의미하므로 경사 하강법 알고리즘의 반복 과정을 종료  

**경사 하강법의 종류**
1. 배치 경사 하강법(Batch gradient descent)  
배치 크기 : m(=total number of training samples)  
매 gradient descent step 마다 훈련셋 전체를 사용하여 그레디언트 벡터를 계산  

2. 확률적 경사 하강법(stochastic gradient descent)
배치 크기 : 1  
매 gradient descent step마다 랜덤하게 선택된 훈련 샘플 하나를 사용하여 그레디언트 벡터를 계산하고 모델 파라미터 벡터를 조정한다.  
- 1 에포크 동안 위 과정을 m 번 반복해서 수행 (m : 총 훈련 샘플 수)  
- 1 에포크 동안 중복 선택되거나 한번도 선택되지 않는 훈련 샘플이 존재할 수 있음  

3. 미니배치 경사 하강법(Mini-batch gradient descent)
2 <= 배치 크기 < m  
매 gradient descent step마다 훈련셋의 랜덤 서브셋을 사용하여 그레디언트 벡터를 계산

1. 배치 경사 하강법  
  
- 진행 과정 알고리즘  
초기화 : 모델 파라미터 벡터 $\theta$ 를 랜덤하게 초기화 하여 훈련 시작  
매 에포크/스텝마다 전체 훈련셋을 사용하고 그레디언트 벡터를 계산하여 모델 파라미터를 조정한다.  
(배치 사이즈가 전체 훈련셋의 크기와 같고, 따라서 에포크 당 스텝 수 ((훈련 샘플 수)/(배치사이즈))는 1이 된다.)  
  
- 허용 오차()와 에포크 수 간 관계  
: 요구되는 허용 오차가 작을수록 비용 함수의 최소값에 더 근접함을 의미  
: 허용 오차와 (그에 도달하기 위해) 필요한 에포크 수는 상호 반비례 관계임  
(e.g., 허용 오차를 1/10로 줄이려면 에포크 수를 10배 늘려야 함)  
  
- 단점   
: 훈련셋이 클 경우 전체 훈련셋을 사용하여 반복적으로 그레디언트 벡터를 계산하는데 많은 시간과 메모리가 필요함 -> 이런 이유로 사이킷런은 배치 경사 하강법을 지원하지 않음  

- 학습률($\etha$)과 경사 하강법의 관계  
: 학습률에 따라 선형 회귀 모델이 최적의 모델로 수렴하는지 여부 수렴 속도가 달라진다.  
: 최적의 학습률은 그리드 탐색 등을 통해 찾아볼 수 있다.  
  
2. 확률적 경사 하강법  
장점  
- 스텝 당 계산량이 상대적으로 적어 매우 큰 훈련셋을 다룰 수 있으며, 외부 메모리(out-of-core) 학습을 활용할 수 있음  
- 모델 파라미터 조정이 불규칙한 패턴으로 이루어질 수 있으며, 이런 이유로 지역 최소값에는 상대적으로 덜 민감함
  
단점  
- 전역 최소값에 수렴하지 못하고 주변을 맴돌 수 있음  
-> 이러한 딜레마에 대한 해결책으로 훈련이 진행되감에 따라 학습률을 점진적으로 줄여나가는 방법을 취할 수 있음  
(주의 사항)  
- 학습률이 너무 빨리 줄어들면, 지역 최소값에 갇힐 수 있음  
- 반대로 학습률이 너무 천천히 줄어들면 전역 최소값에 제대로 수렴하지 못하기 맴돌 수 있음  
  
**학습 스케줄(learning schedule)**    
: 모델 훈련이 진행되는 동안 학습률을 조금씩 줄여나가는 방법  
: 일반적으로 훈련 과정이 얼마나 진행되었는지에 대해 에포크 및 스텝 카운트 값을 이용하여 매 스템(스텝?)마다 적용할 학습률을 계산  

확률적 경사 하강법 적용 예 : SGDRegressor  
```
from sklearn.linear_model import SGDRegressor  
# 모델 파라미터 조정 과정을 언제까지 반복할 것인지 설정하는 파라미터

sgd_reg = SGDRegressor(max_iter = 1000, tol = 1e-5, penalty=None, eta0=0.01, n_iter_no_change = 100, random_state=42)
# max_iter = 1000 : 최대 1000 에포크동안 실행  
# 또는 n_iter_no_change = 100 : 100 에포크동안 향상되는 정도가 $10^{-5}$(=tol)보다도 적으면 종료  
# eta = 0.01 : 학습률 0.01로 시작되며 default learning schedule 사용 됨
sgd_reg.fit(X, y.ravel()) # y.rauel() because fit() expects 1D
 targets
 ```

 3. 미니배치 경사 하강법
 장점  
 - 배치 사이즈를 어느 정도 크게하면 확률적 경사 하강법(SGD)보다 모델 파라미터의 움직임의 불규칙성을 완화할 수 있음  
- 반면에 배치 경사 하강법보다 빠르게 학습
- 학습 스케줄을 잘 설정하면 최소값에 수렴 가능함  
  
단점  
- SGD에 비해 지역 최소값에 수렴할 위험도는 보다 높아짐  
<img src = "../image/미니배치 경사 하강법.png" width=50%>
선형 회귀 모델 훈련을 위한 각 알고리즘 별 주요 특징 및 관련 사이킷런 클래스  
<img src = "../image/선형회귀모델 훈련을 위한 각 알고리즘 별 주요 특징.png" width = 50%>


#### 다항 회귀(Polynomial Regression) 
: 선형 회귀 모델을 이용하여 비선형 데이터를 학습하는 기법  
(즉, 비선형 데이터를 학습하는데 선형 모델 사용을 가능하게 함)  

기본 아이디어  
: 훈련셋에 포함된 각 특성의 power(e.g., 제곱)를 새로운 특성으로 추가   
: 새로운 특성 추가를 통해 확장된 훈련셋을 이용하여 선형 회귀 모델을 훈련함  

**선형 회귀 vs 다항 회귀**
  
<img src = "../image/1차 선형 회귀 모델.png" width=60%>
L 선형 회귀 : 1차 선형 회귀 모델  
  
<img src = "../image/다항회귀 2차 다항식 모델.png" width=60%>
L 다항 회귀 : 2차 다항식 모델


**사이킷런의 PolynomialFeatures 변환기**  
: 주어진 훈련셋에 포함된 특성들 각각의 거듭제곱과 특성들 간의 곱셈을 실행하여 새로운 특성을 추가하는 기능 제공  
degree = d
: 몇 차 다항식 모델에 해당하는 새로운 특성들을 추가 생성할지를 지정하는 하이퍼파라미터
> 예시
>- 훈련셋 특성 수 = 1, degree=2인 경우
 기존 특성 :  $x_1$  
 새로운 특성 : $x_1^2$  
>- 훈련셋 특성 수 = 2, degree=2인 경우  
기존 특성 : $x_1, x_2$  
새로운 특성 : $x_1^2, x_1x_2, x_2^2$  
>- 훈련셋 특성 수 = 2, degree=3인 경우  
기존 특성 : $x_1, x_2$
새로운 특성 : $x_1^2, x_1x_2, x_2^2, x_1^3, x_1^2x_2, x_1x_2^2, x_2^3$  

```
from sklearn.preprocessing import PolynomailFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X) # 새로운 특성 추가를 통해 확장된 훈련셋

X[0]
x_poly[0]
```
```
# 결과
array([-0.75275929])
array([-0.75275929, 0.56664654])
```

```
#확장된 훈련셋을 이용하여 선형 회귀 모델 훈련
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
```
```
# 결과
(array([1.78134581]), array([0.93366893, 0.56456263]))
```

#### 학습 곡선 (Learning Curve) 
다항 회귀 모델의 차수에 따라 훈련된 모델이 훈련셋에 과소 또는 과대 적합 할 수 있다. 
  
교차 검증 vs 학습 곡선   
- 교차 검증
    - 과소 적합 : 훈련 세트와 교차 검증 점수 모두 낮은 경우
    - 과대 적합 : 훈련 세트에 대한 성능은 우수하지만 교차 검증 점수가 낮은 경우
  
- 학습 곡선  
: 모델이 학습한 훈련 샘플 수가 조금씩 증가함에 따라 훈련 세트와 검증 세트에 대한 모델 성능을 비교하는 그래프
(훈련이 진행되는 동안 주기적으로 훈련셋과 검증셋에 대한 모델의 성능을 추천)  
:학습 곡선 모양에 따라 과소 적합/ 과대 적합 판정 가능  

**과소 적합 모델의 학습 곡선 특징**  

<img src="../image/%EA%B3%BC%EC%86%8C%EC%A0%81%ED%95%A9%20%EB%AA%A8%EB%8D%B8%EC%9D%98%20%ED%95%99%EC%8A%B5%20%EA%B3%A1%EC%84%A0.png" width = 60%>

red line (훈련 샘플들에 대한 training error)  
: 모델이 1 or 2개 훈련 샘플만을 학습했을 때 training error(e.g., RMSE)는 0에서 출발  
-> 훈련 샘플들이 추가되면서 training error가 커짐  
-> 훈련 세트가 어느 정도 커지면 새로운 훈련 샘플이 추가되더라도 더 이상 training error가 향상되지 않음  

blue line (검증 데이터에 대한 validation error)  
: 모델이 적은 수의 훈련 샘플들만을 학습한 상태일 때는 상당히 높은 valid error가 관찰됨
-> 훈련셋 사이즈가 일정 수준에 도달하면 valid error가 더 이상 나아지지 않음   
-> 검증 세트에 대한 성능이 훈련 세트에 대한 성능과 거의 비슷해짐  

**과대적합 모델의 학습 곡선 특징**  

<img src = "../image/과대적합 모델.png" width = 50%>   
  
red line (훈련 샘플들에 대한 training error)  
: 훈련셋에 대한 RMSE가 과소적합 모델 경우보다 상대적으로 매우 낮음  

blue line (검증 세트에 대한 성능)
: 훈련셋에 대한 성능과 차이가 크게 벌어짐  

과대적합 모델 개선법 : 훈련셋 사이즈를 훨씬 더 크게 키우면 두 커브가 더 가까워 질 수 있어 보임  

**모델 일반화 오차 : 편향 / 분산 간 trade off**
모델 일반화 오차는 다음 세 가지 종류 오차들의 합으로 표현될 수 있음
- 편향(bias), 분산(variance), 줄일 수 없는 오차(irreducible error)  
> 편향 : 데이터에 대한 잘못된 가정에서 기인하는 오차  
(e.g., 데이터가 실제로는 2차원인데 선형으로 잘못 가능함)
(일반적으로 편향이 큰 모델이 과소적합이 될 가능성이 높음)

분산 : 훈련 데이터에 있는 작은 변동에 모델이 과도하게 민감하게 반응함에서 기인하는 오차
(고차 다항 회귀 모델같이 자유도가 높은 모델일 수록 분산 오차가 커지며, 과대적합이 될 가능성도 높음)  

축소 불가능 오차 : 훈련 데이터 자체에 존재하는 노이즈때문에 발생하는 오차  
(노이즈를 제거해야만 오차를 줄일 수 있음)  


편향 - 분산 간 trade off(반비례)  
: 모델의 복잡도가 커지지면 보통 분산 오차가 늘고 편향을 줄어듦  
반대로, 복잡도를 줄이면 분산 오차가 작아지고 편향은 커짐.

### part 2  
#### 규제(regularization)를 사용하는 선형 회귀 모델  
규제  
: 모델이 훈련셋에 과대적합 되는 것을 방지하기 위한 방법  
: 훈련 과정에서 모델의 자유도 제한  
: 선형 회귀 모델에 대해서는 모델의 가중치가 가능한 작게 유지되도록 하는 방향으로 규제가 이루어짐  
: 다항 회귀 모델를 규제하는 간단한 방법은 다항식의 차수를 작게 하는 것 
   
자유도(degree of freedom) : 훈련 과정을 통해 도출되는 최종 모델 결정에 영향을 주는 요소들 (e.g., 특성 수)  
-> 단순 선형 회귀의 경우 : 특성 수  
    : 데이터셋의 특성 수가 선형 회귀 모델의 파라미터 수를 결정하며, 각 모델 파라미터가 훈련 과정에서 조정될 수 있는 모델의 자유도 해당한다.   
-> 다중 (선형) 회귀의 경우 : 차수(degree)  
    
가중치 규제가 적용되는 선형 회귀 모델    
- 릿지 회귀(Ridge regression)   
- 라쏘 회귀(Lasso regression)  
- 엘라스틱넷(Elasitc Net regression)  

>규제 적용 시 주의사항  
>- 선형 회귀 모델 훈련 과정에서 비용함수에 규제항이 추가된다.  
>- 규제항은 훈련 과정에서만 비용함수에 추가 적용 되며, 테스트 과정에는 다른 기준으로(e.g., RMSE) 성능을 평가한다.  
>- 훈련 과정 : 주어진 훈련셋에 대해 비용 함수 최소화 목표  
>- 테스트 과정 : 최종 목표에 따른 성능 평가  
>(e.g., 선형 회귀 모델의 경우 RMSE 사용하여 성능 평가)  

- 릿지 회귀(Ridge regression)  
릿지 회귀의 비용함수 
<img src = "../image/릿지 회귀의 비용함수.png" width = 50%>

- MSE()에 규제항이 추가된 형태  
- 모델 파라미터 중 bias term $\theta_0$에 대해서는 규제를 적용하지 않음(summation에서 i=1부터 시작되기 때문)  
- 규제항은 모델의 가중치 벡터 w에 대한 $l_2$ norm의 제곱에 해당함 $\left(\frac{\alpha}{m}\right)\displaystyle\sum_{i=1}^{n}(\theta_i^2) $
- 이런 이유에서 릿지 회귀의 규제를 $l_2$ regularization 이라고 부름  
  
$\alpha$: 규제 강도 지정  
($\alpha$ = 0이면 규제가 전혀 없는 기본 선형 회귀의 비용 함수와 동일)  
($\alpha$가 커질 수록 가중치의 역할이 줄어듦.-> 릿지회귀의 비용을 줄이기 위해 가중치를 작게 유지하는 방향으로 학습이 이루어짐)    
  
(주의사항 : 릿지 회귀는 입력 특성의 스케일에 민감하기 때문에 훈련 전의 특성들의 스케일을 통일 시키는 것이 중요함 (e.g., StandardScaler를 이용해서) )

일반적으로 n 요소들을 포함하는 벡터 v의 벡터값(norm) $l_k$ ($v_i, 1<= i <= n$)    
$\parallel v\parallel_k = (|v_1|^k + |v_2|^k + \cdots + |v_n|^k)^{\left(\frac{1}{k}\right)} $  

$l_2$ norm of a vector v  
$\parallel v\parallel_2 = \sqrt{(|v_1|^2 + |v_2|^2 + \cdots + |v_n|^2)} $    
$l_2$ norm of a vector v  
$\parallel v\parallel_1 = |v_1| + |v_2| + \cdots + |v_n| $   
<img src = "../image/다양한 규제 강도를 사용한 단순 릿지 모델.png" width=50%>
(L 선형적인 형태로 예측을 만드는 단순 릿지 모델에 다양한 규제 강도 $\alpha$를 적용)  
<img src = "../image/다양한 규제 강도를 적용한 릿지 규제.png" width=50%>  
(L 릿지 규제가 적용되는 다항 회귀 모델에 다양한 규제 강도 $\alpha$를 적용)  
- PolnomialFeatures(degree=10)으로 데이터 확장 후 StandardScaler를 사용해 특성 스케일 후 릿지 모델 적용  

=> $\alpha$가 커질수록 모델의 variance는 줄어들고 bias는 커지는 경향  
  
- 라쏘 회귀 
<img src = "../image/라쏘 회귀 비용함수.png" width=50%>  
- 규제항은 모델의 가중치 벡터 w에 대한 $l_1$ norm에 해당함 (2$\alpha\parallel w \parallel_1$)  
- 이런 이유로 라쏘 회귀의 규제를 $l_1$ regularization이라고도 부름
    
  $\alpha$ = 0이면 규제가 전혀 없는 기본 선형 회귀의 비용 함수와 동
  일
  
주요 특징 
: 중요도가 낮은 특성에 대한 가중지 $\theta_i$가 0이 되도록 학습이 유도되는 경향  
: 릿지 회귀보다 더 과감한 규제가 가해지는 경향이 있음

<img src = "../image/라쏘 회귀.png" width = 50%>  
오른쪽 그래프에서 $\alpha$ = 0.01 경우는 거의 3차 방정식 곡선이며, 이는 높은 차수(higher degree)의 다항 특성에 대응되는 가중치 파라미터가 모두 0으로 설정되었음을 의미  
   
- 엘라스틱넷  
: 릿지 회귀와 라쏘 회귀를 절충한 모델  
비용 함수 
<img src = "../image/엘라스틱넷의 비용함수.png" width = 50%>
- 릿지/라쏘 회귀의 규제항이 모두 포함됨    
- 릿지/라쏘 규제 적용 비율은 r에 의해 조절.  
(r = 0이면 릿지 회귀와 동일, r = 1이면 라쏘 회귀와 동일)  


**규제 적용의 일반적인 원칙**
: 적어도 약간의 규제가 적용되는 것이 대부분의 경우 바람직함. (규제가 없는 평범한 선형 회귀는 피해야 함)  
: 릿지 규제를 적용하는 것이 기본
  
-> 예측에 활용되는 (중요한) 특성이 많지 않다고 판단되는 경우
: 라쏘규제나 엘라스틱넷 활용이 바람직(불필요한 특성의 가중치를 0으로 만들어주기 때문)  
-> 특성 수가 훈련 샘플 수보다 많은 경우나 몇 가지 특성들이 강하게 연관되어 있는 경우  
: 라쏘 규제는 적절치 않으며, 엘라스틱넷이 추천됨  

<br>

조기 종료(Early Stopping)  
: 모델이 훈련셋에 과대 적합되는 것을 방지하기 위해 훈련을 적절한 시기에 중단시키는 기법    
: 검증 데이터에 대한 예측 에러가 줄어 들다가 다시 커지는 순간 훈련 종료  
: 경사 하강법같은 iterative 학습 알고리즘을 규제하는 또 다른 방법    
: 모델 훈련셋에 과대 적합되는 것을 방지하기 위해 지정된 에포크까지 진행하지 않고 중간에 적절한 시기에 훈련을 중단시키는 기법  
: 검증셋에 대한 비용함수 값이 줄어 들다가 다시 커지는 순간 훈련 종료    

<img src = "../image/조기종료.png" width = 50%>
확률적 경사 하강법이나 미니 배치 경사 하강법의 경우 예측 에러 곡선이 그리 매끄럽지 않아 최소값에 도달했는지 판단이 어려울 수 있음  
( 해결 방안 ) 검증 에러가 기존에 도달했던 최소값보다 한동안 높게 유지될 때(즉, 모델이 더 나아지지 않는다는 확신이 들 때) 학습을 멈추고 기억해둔 최적의(최소 검증 에러에 대응되는) 모델 파라미터로 되돌린다. 

```
#PolynomialFeatures와 StandardScaler 변환기로 구성된 파이프랑니 생성
preprocessing = make_pipeline(PolynomialFeatures(degree=0, include_bias=False), StandardScaler())

# X_train과 X_valid에 변환 파이프라인 적용
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)

#SGDRegressor 생성
sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf') # 양의 무한대로 초기화

train_errors, val_errors = [], [] # extra code - it's for the figure below

"""
n_epochs 동안 sgd_reg에 대한 훈련 과정을 진행하면서 best validation error에 해당하는 모델을 찾는다. 매 epoch마다 검증셋에 대한 RMSE 측정하여 best_valid_rmse보다 작은지 비교
"""
for epochin range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train) #sgd_reg.partial_fit() to perform incremental_learning
    y_valid_predict = sgd_reg.predict(X_valid_prep) 
    val_error = mean_squared_error(y_valid, y_valid_predict, squared = False)
    if val_error < best_valid_rmse :
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)

    # extra code - we evaluate the train error and save it for the figure
    y_train_predict = sgd_reg.predict(X_train_prep)
    train_error = mean_squared_error(y_train, y_train_predict, squared=False)
    val_errors.append(val_error)
    train_errors.append(train_error)
```
  
 > 회귀 모델을 분류 작업에 활용
이진 분류 : 로지스틱 회귀  
다중 클래스 분류 : 소프트맥스 회귀  
#### 로지스틱 회귀(Logistic Regression)  


#### 결정 경계   

로지스틱 회귀 모델에 대한 규제  
: 사이킷런의 로지스틱 회귀 모델의 하이퍼파라미터 penalty와 C를 이용하여 규제의 방식과 규제의 강도를 지정함  
  
penalty
- $l$1(라쏘 규제), $l$2(릿지 규제), 엘라스틱넷 3가지 중 하나 지정
- 기본값 $l$2, 즉 $l_2$ 릿지 규제를 기본 적용 함
- 엘라스틱넷을 선택한 경우에는 $l$1_ratio 옵션 값을 함께 지정해야 함  
  
C 
- 릿지 또는 라쏘 규제 강도를 지정하는 $a$의 inverse에 해당  
- 0에 가까울수록 강한 규제, 1에 가까울수록 약한 규제  

> (e.g., 붓꽃(Iris) 데이터셋)  
꽃받침(sepal)과 꽃입(petal)에 관한 다음 4가지 특성들로 이루어짐  
꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비  

분류 클래스 : 총 3가지 품종  
0 : Setosa(세토사) 1 : Versicolor(버시컬러) 2 : Virginica(버지니카)  


: 로지스틱 회귀 모델을 이용하여 Virginica 품종 여부를 판정하는 이진 분류기를 다음과 같이 훈련시킨다. (단, 데이터셋의 4가지 특성 중 꽃잎 너비 특성 하나만 이용함)  
```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
```
<img src = "../image/로지스틱 회귀 적용 예.png" width = 50%>
로지스틱 회귀 모델의 예측 확률값이 0.5에 해당하는 꽃잎 너비 약 1.6cm 가 결정 경계에 해당함  


#### 소프트맥스 회귀(Softmax Regression)
: 로지스틱 회귀 모델을 일반화하여 다중 클래스 분류를 지원하도록 만든 회귀 모델   
: 다항 로지스틱 회귀로도 불림  
(주의 사항)  
소프트맥스 회귀는 다중 출력 분류는 지원하지 못함.  
(e.g., 하나의 사진에서 여러 사람의 얼굴 인식 불가능)  