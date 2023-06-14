## 목차  
1. Part 1
2. Part 2
    2-? 사용자 정의 변환기 클래스
3. [Part 3](#part-3)  
    3-1. [머신러닝 모델 선택과 훈련](#머신러닝-모델-선택과-훈련)  
    3-2. 모델 세부 튜닝

### part 2  
#### 사용자 전환 변환기 클래스  
: 결측치를 채우기 위한 SimpleImputer 변환기 경우처럼 fit() 메서드 실행을 통해 각 특성 별 mean, median 등 학습 후 transform() 메서드 적용이 가능한 변환기를 직접 구현하고자 할 경우 변환기 클래스를 직접 구현해야 함.  
: 사이킷런의 다른 변환기와의 호환을 위해 fit(), transform() 등 메서드 구현 필요  
```
# e.g., 캘리포니아 주택 가격 데이터에서 서로 근접한 구역들의 클러스터를 확인하기 위한 변환기 클래스  
# -> fit() 메서드에서 K-means 클러스터링 알고리즘을 이용하여 훈련셋 내 근접한 구역들의 군집을 알아냄  
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1, random_state=45)
similarities = cluster_simil.fit_transform(housing[["latitude", "longtitude"]], sample_weight=housing_labels)  


from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_cluster=10, gamma=1.0, random_state = None) :
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    def fit(self,X, y=None, sample_weight=None) : 
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self # always return self

    def transform(self, X): 
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None) : 
        return [f"Cluster (i) similarity" for i in range(self.n_clusters)]


```
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

### 교차 검증(Cross Validation) 

k-fold 교차 검증  
: 사이킷런의 cross_val_score() 함수 활용
훈련셋(training data)을 k개의 서브셋으로 랜덤하게 나눔  (서브셋을 fold라고 칭함)  
-> 모델 훈련 및 검증 과정을 k번 반복
(k번 각각에 대해서 서로 다른 1개 폴드를 선택하여 검증) (데이터셋(validation set)으로 활용)  
-> 나머지 (k-1)개 폴드를 활용하여 모델 훈련 후 선택된  검증용 폴드를 이용하여 모델 성능 측정(e.g., RMSE 측정)  
최종적으로 K개의 성능 측정 결과치가 저장된 배열이 생성됨
  
<img src="../image/k-fold 교차 검증.png" width=50%>



### 모델 세부 튜닝
(지금까지 살펴 본 모델 중 랜덤 포레스트 회귀 모델이 성능이 가장 우수했음)  
가능성이 높은 모델 선정 후 모델 훈련 과정에 대한 세부 설정(하이퍼파라미터)를 튜닝해야 함
  
하이퍼파라미터 튜닝을 위한 2가지 방식  
- 그리드 탐색(Grid Search)
- 랜덤 탐색(Randomized Search)  

그리드 탐색    
: 랜덤 포레스트 회귀 모델에 대해
 탐색하고자 하는 하이퍼파라미터 값 조합들의 경우의 수가 적은 경우에 적합  
랜덤 탐색  
: 탐색하고자 하는 하이퍼파라미터 값 조합들의 경우의 수가 굉장히 많은 경우에 적합  
: 사이킷런의 RandomizedSearchCV 클래스 활용  

앙상블(Ensemble) 기법  
: 결정트리(decision tree) 모델 하나보다 랜덤 포레스트처럼 여러 결정트리 모델들로 이루어진 모델이 일반적으로 보다 좋은 성능을 낸다.  
: 여러 개별 모델들을 함께 구성하여(ensemble) 학습시킨 후 각 모델의 예측값들의 평균값을 사용하면 보다 좋은 성능을 내는 모델을 얻을 수 있다.  

그리드 또는 랜덤 탐색을 통해 얻어진 best 모델을 분석하면 문제 해결에 대한 좋은 insight를 얻는 경우가 많음   



### 론칭, 모니터링, 시스템 유지 보수  
- 실제 시스템에 모델 론칭  
: 훈련된 최고 성능 모델(전처리 파이프라인 및 RandomForestRegressor 예측기 포함)을 파일로 저장  
- joblib 라이브러리 사용하여 가능  
: 저장된 파일을 production 환경으로 옮긴 다음 파일로부터 모델을 로드해서 시스템에 적용  

e.g., 웹 서비스 형태로 모델 배포  
: 사용자는 웹 사이트에서 특정 구역에 관한 정보를 입력한 후 "가격 예측" 버튼을 클릭  
: 사용자가 입력한 정보가 웹 서버로 전송되어 웹 어플리케이션에 전달됨  
: 웹 어플리케이션 REST API를 통해 사용자가 입력한 정보를 웹 서비스(회귀 모델)로 전달하고 모델은 예측 결과값을 reply  
장점 : 확장에 용이   

클라우드를 통한 모델 론칭  
e.g., Google's Vertex AI같은 클라우드를 통해 머신러닝 모델 배포 가능  
: 저장된 모델 파일을 Google Cloud Storage(GCS)에 업로드  
: Vertex AI에서 새로운 모델 버전을 생성한 다음 GCS에 업로드 한 모델 파일을 지정해줌  
: Vertex AI에서 모델에 대한 웹 서비스를 생성해주고 확장 및 로드 밸런싱 등을 알아서 처리해줌  

- 성능 모니터링을 위한 시스템 구축  
: 다양한 원인으로 머신러닝 시스템 성능 저하가 발생할 수 있음  
: 시스템 성능 저하를 감지하기 위한 모니터링 시스템 필요  
: 모델의 예측 성능이 일정 수준 이하로 저하될 경우 어떻게 대비할 것인지에 관한 프로세스 준비 또한 필요함  

- 데이터 변화에 다른 모델 업데이트  
: 시간이 지남에 따라 진화하는 성격의 데이터라면 정기적으로 데이터셋을 업데이트하고 업데이트된 데이터셋으로 모델을 다시 훈련해야 함  
-> 필요한 작업 및 자동화  
: 정기적으로 새로운 데이터 수집 & 레이블링
: 모델을 다시 훈련 & 하이퍼파라미터 튜닝 과정 자동화를 위한 파이프라인 구현  
: 업데이트 된 테스트셋으로 새로운 모델과 기존 모델을 평가하는 스크립트 작성 -> 새로운 모델의 서능이 더 낫다면 새로운 모델로 교체  

- 데이터셋 및 모델 백업  
: 새로운 모델이 문제가 있을 경우 이전 모델로 신속하게 롤백할 수 있도록 훈련을 마친 완성된 모델은 하상 백업해두는 것이 바람직함  
: 새로운 버전의 데이터셋이 어떤 이유로 오염(e.g., 이상치가 굉장히 많음)되었다면 롤백 할 수 있도록 모든 버전의 데이터셋 또한 백업하는 것이 바람직함  