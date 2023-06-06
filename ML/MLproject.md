## 목차  
1. Part 1
2. Part 2
3. [Part 3](#part-3)  
    3-1. [머신러닝 모델 선택과 훈련](#머신러닝-모델-선택과-훈련)  
    3-2. 모델 세부 튜닝

    
### part 3
#### 머신러닝 모델 선택과 훈련<br>
><b>진행 중인 프로젝트 개요</b> 
>    
>목표 : 캘리포니아 주 district(구역) 별 중간 주택 가격 예측 모델 
>    
> 결정해야 할 사항  
> 
> - 주어진 구역의 중간 주택 가격 예측을 위해 어떤 머신러닝 모델을 사용할 것인지  
> - 머신러닝 모델 성능 측정 지표로 무엇을 사용할 것인지   
>   
> 머신러닝 모델 : <b> 회귀 모델</b>  
> 회귀 모델 성능 측정 지표 : <b>평균 제곱근 오차(RMSE)</b> 를 사용  
> 이번에 할 일  
> - 예측 모델 선택 후 훈련시키기  


예측 모델  
1. 선형 회귀 모델 (Linear Regression)  
: 사이킷런의 LinearRegression 클래스를 활용하여 선형 회귀 모델 생성  
: 앞서 구현한 데이터 전처리 파이프라인(preprocessing 변환기)에 LinearRegression 예측기 추가  

``` 
from sklearn.linear_model import LinearRegression 

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
# housing : 훈련 샘플들에 대한 입력 특성(predictors)
# housing_labels : 훈련 샘플들에 대한 타깃(레이블) 특성
```
