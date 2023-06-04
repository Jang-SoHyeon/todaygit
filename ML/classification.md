# 기계학습 기말
## 목차  
1. [Part 1](#part-1)    
    1-1 [MNIST 데이터 셋](#mnist-데이터-셋)  
    1-2 [이진 분류기 훈련](#이진-분류기-훈련)  
    1-3 [분류기 성능 측정](#분류기-성능-측정)  


### Part 1
#### MNIST 데이터 셋  
<b>MNIST 데이터 셋</b>  
: 미국 고등학생과 인구 조사국 직원들이 손으로 쓴 70,000개의 숫자 이미지로 구성된 데이터 셋   
(0~9까지 숫자들에 대한 손글씨 이미지)     
  
  <img src = "../image/MNIST 데이터 셋.png" width ="150%">
  
이미지  
: 각 이미지는 28x28=784 개의 픽셀들로 구성된 이미지 데이터  
: 각 이미지는 2차원 배열이 아닌 길이가 784인 1차원 배열로 제공  
: 이미지 데이터 셋의 shape은 (70000,784) 
   
레이블   
: 총 70,000개의 사진 샘플들 각각이 어떤 숫자를 나타내는지에 대한 레이블링이 되어 있음  
: 레이블 데이터 셋의 shape은 (70000,)  

> 문제 정의  
>- 지도 학습 (각 이미지가 어떤 숫자를 나타내는지에 대한 레이블이 지정되어 있음)  
>- 분류 : 주어진 이미지 데이터가 0~9 중 어떤 숫자에 해당하는지 예측  
>    - 0~9까지의 숫자 각각이 하나의 클래스에 해당  
>    - 주어진 이미지 데이터가 총 10개의 클래스 중 어느 클래스에 해당하는지 분류  -> 다중 클래스 분류(multiclass classification) 또는 다항 분류(multinomial classification)    
>- 배치 또는 온라인 학습 : 둘 다 가능
>   - 확률적 경사하강법(SGD) 분류기 : 배치와 온라인 학습 모두 가능
>   -  랜덤 포레스트 분류기 : 배치 학습  

#### 이진 분류기 훈련 
<b>- 숫자 5 감지기 -</b> 

-> 다중 클래스 분류 모델을 훈련하기에 아펏 주어진 샘플이 숫자 5에 해당하는지 여부를 판단하는 이진 분류기(binary classifier)를 훈련시키고자 함  
-> 이를 통해 분류기의 기본 훈련 과정과 성능 평가 방법을 알아보고자 함  
   
각 이미지에 대한 레이블은 0 또는 1로 수정되어야 함  
(레이블 0 : 해당 숫자 X, 레이블 1 : 해당 숫자 O)  
  
```
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
```

* 이진 분류기로 SGD 분류기 활용  
> SGD Classifier이란?
> - 확률적 경사 하강법(stochastic gradient descent) 분류기
> - 한번에 하나씩 훈련 샘플을 이용하여 학습한 후 파라미터를 조정
> - 매우 큰 데이터 셋 처리에 효율적이며 온라인 학습에도 적합  
> - 사이킷런의 SGDClassifier 클래스 활용 
  
```
from sklearn.linear_model import SGDClassifier 

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_Clf.predict([some_digit]) #predict() 메소드 호출을 통해 주어진 이미지가 0 또는 1 중 어느 클래스에 해당하는지 예측  
```
  
#### 분류기 성능 측정  
분류기 성능 평가는 회귀 모델 평가보다 고민할 점들이 많음  
  
 > 분류기 성능 측정 기준  
 > - 정확도(accuracy)  
 > - 정밀도(precision) / 재현율(recall)  
 > - ROC curve의 AUC

 <b> 정확도(accuracy) </b>  
 (옳게 분류된 샘플 수) / (전체 샘플 수) -> 분류기에 의해 올바르게 분류된 샘플들의 비율  
     
 교차 검증을 사용한 정확도 측정  
: k-fold 교차 검증(cross validation) 기법을 이용하여 SGD 분류기의 성능 평가  
: 사이킷 런의 cross_val_score() 함수 활용  
: cross_val_score() 반환 값은 성능 지표 측정값을 배열 형태로 반환 함. 개수는 cv(폴드 수)만큼 들어있다. 일반저긍로 이를 평균한 값을 평가 수치로 사용 함.

```
from sklearn.model_selection import cross_val_score  

# cross_val_score(모델 명,훈련데이터, 타깃,cv,평가지표)
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") 
```
```
# 결과  
array([0.95035, 0.96035, 0.9604]) #95%가 넘는 정확도를 보임
```
  
무조건 '5가 아님'으로 찍는 DummyClassifier 분류기를 생성하여 정확도 측정  
```
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)  

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy") 
```
```
# 결과
array([0.90965, 0.90965, 0.90965])
```
- 무조건 '5가 아님'을 찍는 경우에도 90% 넘는 정확도를 보임
(훈련셋의 10%만 숫자 5의 이미지이고 나머지 90%는 5가 아닌 이미지이기 때문)  
    
결론  
: 이처럼 훈련셋 샘플들의 <b>클래스 불균형이 심한 경우</b>에는 분류기의 성능 측정 지표로 <b>정확도는 부적합</b>
  
오차 행렬(Confusion Matrix)  
: 클래스 별 에측 결과를 정리한 행렬
: 행은 실제 클래스를(true label), 열은 분류기에 의해 에측된 클래스(predicted label)를 의미
  
> if. 아래 오차 행렬에서 cat에 해당하는 이미지 샘플을 dog으로 잘못 분류한 횟수를 알고 싶다면 (cat 행, dog 열)에 위치한 값을 확인  
>     
>  <img src = "../image/오차 행렬.png" width ="50%">

숫자 5-감지기에 대한 오차 행렬
- cross_val_predict() : k-fold 교차 검증 수행, 각 validation fold의 샘플들에 대해 분류기가 에측한 결과(y_train_pred)를 반환  
- confusion_matrix() : 타깃 클래스(y_train_5)와 예측된 클래스(y_train_pred) 정보를 이용하여 숫자 5-감지기에 대한 오차 행렬 생성

```
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
cm
```
```
# 결과
array([[53892,  687],
       [1891,  3530]])
# 행은 실제 클래스 열은 분류기에 의해 에측된 클래스이므로
-------------------------------
|            | 0      |  1    | 
-------------------------------            
|5가 아닌 숫자| 538892 | 687   |
-------------------------------
|5인 숫자     | 1891   | 3530  |
-------------------------------  -> 2x2행렬
 
---------------------------------------------------------
|            |   0                |   1                 | 
---------------------------------------------------------            
|5가 아닌 숫자| True Negative(TN)  | False Positive(FP)  |
---------------------------------------------------------
|5인 숫자     | False Negative(FN) | True Positive(TP)   |
--------------------------------------------------------- 
```
TN : 실제 5가 아닌 이미지를 5가 아닌거(N)로 옳게(T) 예측한 경우  
TP : 실제 5 이미지를 5로 예측(P)을 옳게(T) 경우  
FN : 실제 5인 이미지를 5가 아닌거(N)로 틀리게(F) 예측한 경우  
FP : 실제 5가 아닌 이미지를 5로 예측을(P) 틀리게(F) 경우  
     
-> 오차 행렬이 분류 결과에 대한 많은 정보를 제공하지만 더 요약된 지표가 필요함 ( 정밀도, 재현율 )  
  
<b> 정밀도(Precision), 재현율(Recall) </b>  
<hr>

- 정밀도
: positive 예측의 정확도를 나타내는 지표   
(positive라고 예측한 샘플(instance) 중 실제 positive인 인스턴스의 비율)  
<image src = "../image/정밀도.png" width = 50%>

정밀도(precision)가 높다는 것은 분류기가 주어진 샘플(instance)을 <b>positive라고 예측하면 그 예측이 맞을 확률이 높다</b>는 의미  

<br> 

앞서 살펴본 SGD classifier의 정밀도 
array([[53892,  687],  
&emsp;&emsp;[1891,  3530]])  
: 3530 / 3530 + 687 = 0.83708..  

> 정밀도 100%를 달성할 수 있는 간단한 방법  
기본적으로 항상 non-5로 예측하다가 5가 확실한 이미지 샘플 하나만 5로 예측  
(FP=0 & TP=1 -> Precision = 1)    
-> 이러한 분류기는 숫자 5인 이미지들 중 오직 하나만 5로 예측한 것이기 때문에 전혀 유용하지 x  
-> 결과적으로 정밀도만으로는 좋은 분류기를 가려낼 수 없으며, 정밀도를 보완할 성능 지표가 추가로 필요함  



- 재현율   
: positive 샘플에 대한 예측 정확도  
(데이터셋에 포함된 모든 positive instance들 중에서 분류기가 positive라고 분류한 비율)    

<img src = "../image/재현율.png" width=55%>

재현율이 높다는 것은 분류기가 positive instance를 놓치지 않고 잘 감지해낸다는 의미  
array([[53892,  687],  
&emsp;&emsp;[1891,  3530]])  
: 3530 / 3530 + 1891 = 0.6511..  
<br>

- 정밀도와 재현율의 조화 평균(harmonic mean)(F1 Score)  
: 서로 다른 종류의 두 분류기 모델의 성능을 하나의 metric으로 비교하고자 할 때 유용  
<img src = "../image/정밀도와 재현율의 조화 평균.png" width=55%>

```
from sklearn.metrics import f1_score  

f1_score(y_train_5, y_train_pred)  
```
```
결과
0.7325171197343846
```
> F1 Score 계산 시 정밀도와 재현율에 대한 가중치 설정  
위 계산 식은 정밀도와 재현율을 동일한 중요도로 반영하여 F1 Score 계산하는 경우  
경우에 따라서는 정밀도와 재현율 중 하나에 더 높은 가중치를 두어야 할 때도 있음  


- 정밀도(Precision) vs 재현율(Recall)  
: 모델 사용의 목적에 따라 정밀도와 재현율의 중요도가 상이할 수 있음  
> 정밀도 < 재현율 (재현율이 더 중요한 경우)    
> : e.g., 암 진단 분류기   
(암인 케이스를 놓치지 않고 암으로 예측하는 것이 중요)  
정밀도 : 암(즉, positive)으로 진단된 경우 중에 실제로 암인 경우의 비율  
(암으로 진단한 경우에 대한 신뢰도/정확도를 의미)  
재현율 : 실제로 암인 경우들 중에서 놓치지 않고 암으로 진단한 경우의 비율  
 

> 정밀도 > 재현율 (정밀도가 더 중요한 경우)  
> : e.g.,아이가 시청해도 되는 안전한 동영상 분류기  
(안전한 동영상으로 예측된 샘플 중에 실제로 안전하지 않은 동영상이 포함된 경우 문제가 됨)  
정밀도 : 안전(즉, positive)하다고 판단된 동영상 중에 실제로 안전한 동영상 비율  
재현율 : 실제 안전한 동영상 중 안전하다고 예측된 영상 경우의 비율  



