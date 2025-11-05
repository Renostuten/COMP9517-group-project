세가지 정도로 생각해 봤음. (2개의 정확도 최적화, 1개의 속도 최적화.속도 최적화가 필요한건지 사실 잘 모르지만 그래도 넣었어)

1. 학습 전략 최적화 (Hard Negative Mining)

2.파라미터 최적화

3.모델 성능 (속도) 최적화


1. 학습 전략 최적화

지금 현재는 곤충에 대한 것만 학습하는데, 배경에 대한 학습이 필요하다고 해.(AI의 조언) 이 모델로 SlidingWindowDetector를 실행하면, '나뭇잎'을 보고도 가장 비슷하게 생긴 'Ants'라고 예측(False Positive)하게 됨.(이 역시 AI의 조언임. 읽어보고 논리적으로 타당한지 평가해 줘)

   1. 문제점

   [ML]Status doc.md에서 지적하듯, 현재 코드는 'ground-truth crop' 접근 방식을 사용함. Status doc.md`].

   load_yolo_dataset 함수는 YOLO 라벨을 기반으로 '곤충'(Positive) 이미지만 잘라내어 학습.

   결과: 현재 SVMClassifier는 곤충이 뭔지는 구별할 수 있지만, 곤충과 나뭇잎을 구분하는 방법은 배운 적이 없음.

   이 모델로 SlidingWindowDetector를 실행하면, '나뭇잎'을 보고도 가장 비슷하게 생긴 'Ants'라고 예측(False Positive)하게 됨.

   2. 해결책: Hard Negative Mining (HNM)

   HNM은 "SVM이 곤충으로 착각하기 쉬운, 어려운(Hard) 배경(Negative) 샘플"들을 체계적으로 찾아내어 SVM을 재학습시키는 전략.

   1단계: Random Negatives 수집

   목표: 1차 모델(v1)을 학습시키기 위해 "명백한 배경" 샘플을 수집.

   train/images/의 원본 이미지를 불러옵니다.

   train/labels/의 YOLO 라벨을 읽어, '곤충'이 있는 실제 박스(Ground-Truth) 영역을 확인.

   HOGFeatureExtractor의 image_size(예: 128x128)와 동일한 크기로, 이미지의 무작위 위치에서 패치(patch)를 잘라냄.

   이 무작위 패치가 '곤충' 박스 영역과 겹치지 않는지 (예: IoU < 0.1) 확인.

   겹치지 않으면, 이 패치를 'Random Negative' 샘플로 저장.

   2단계: 1차 모델(v1) 학습

   목표: '곤충'과 '명백한 배경'을 구별하는 기본 모델을 만듭니다.

   Positive 샘플: 기존 load_yolo_dataset로 수집한 '곤충' 이미지 (라벨: 'Ants', 'Bees' 등 12개)

   Negative 샘플: 1단계에서 수집한 'Random Negative' 이미지

   이 'Random Negative' 샘플에 **'Background'**라는 13번째 새 클래스 라벨을 붙여줍니다.

   이 두 데이터를 합쳐(Positive + Negative) SVMClassifier를 **1차 학습(v1)**시킵니다. (총 13개 클래스 분류)

   3단계: Hard Negatives 수집 (Mining)

   목표: 1차 모델(v1)이 "배경인데 곤충으로 착각하는" **'어려운' 샘플(False Positives)**을 수집합니다.

   학습된 SVMClassifier (v1)을 SlidingWindowDetector에 장착합니다.

   이 탐지기를 train/images/의 원본 이미지 전체(1차 학습에 사용했던 이미지들)에 대해 다시 실행합니다.

   탐지기가 'Background'가 아닌 것(즉, 12종류 곤충 중 하나)으로 예측한 모든 윈도우 박스를 수집합니다.

   이 박스들이 '곤충'의 실제 정답(Ground-Truth) 박스와 겹치지 않는다면, 이것이 바로 SVM(v1)을 속인 'Hard Negative' 샘플입니다.

   4단계: 2차 모델(v2) 재학습

   목표: '어려운' 샘플까지 학습하여 최종 탐지 모델을 완성합니다.

   2단계에서 사용한 학습 데이터(Positive + Random Negative)에

   3단계에서 수집한 'Hard Negative' 샘플 (라벨은 동일하게 'Background')을 추가합니다.

   이 "확장된" 최종 데이터셋으로 SVMClassifier를 **다시 학습(v2)**시킵니다.




2.파라미터 튜닝.

3단계로 구성해 보았음.(요약)

   1. 스크리닝. 일단 6~10개의 간단한 조합을 수동으로 돌려서, 각 파라미터의 중요도와 우선순위를 정함.

   2. 실제 최적화 메소드 사용.

      1순위 (Best): 베이지안 최적화 (Bayesian Optimization)

      방법: Optuna 사용

      이유: 과거 시도를 학습하여 지능적으로 다음을 탐색합니다. 가장 적은 시도 횟수(예: 30~50회)로 최적값을 찾고"가지치기(Pruning)" 기능도 지원.

      2순위 (Good): 랜덤 서치 + 해빙 (Random Search + Halving)

      방법: HalvingRandomSearchCV를 사용.

      이유: 성능 나쁜 후보를 "작은 예산"으로 조기 탈락시켜 자원을 아낍니다.

   3. 정밀 미세조정

      방법: 2단계에서 찾은 Top 1~3개 조합의 주변 값을 핀포인트로 탐색.
      이때는 GridSearchCV가 유용. (예: step={6, 8, 10}, nms={0.45, 0.5, 0.55}처럼 좁은 범위의 그리드를 돌려 마지막 1%의 성능을 확보합니다.)

1.스크리닝
   일단 6~10개의 간단한 조합을 수동으로 돌려서, 각 파라미터의 중요도와 우선순위를 정함. 조정값 별 속도와 정확도를 확인해서 최적의 조합을 찾고, 해당 조합에 대해 한번 돌리는데 걸리는 시간측정.



이 작업을 통해 핵심 파라미터를 선별.
핵심 파라미터들만 가지고, 파라미터 최적화 알고리즘을 돌리면 됨.
ex) 단계(베이지안 최적화)에 넘길 파라미터가10개에서 [classifier_type='LinearSVC', step_size] 2개로 줄어듬.

   파라미터 리스트
      1. HOG 특징 추출기 파라미터
      (HOGFeatureExtractor 클래스의 __init__에서)
      image_size (현재: (128, 128))
      HOG 계산 전 이미지 크기를 통일합니다. 속도/성능에 모두 영향을 줍니다.
      orientations (현재: 9)
      기울기 방향을 몇 개의 "바구니"로 나눌지 결정합니다. (성능 영향)
      pixels_per_cell (현재: (8, 8))
      HOG의 기본 단위인 "셀"의 크기입니다. 속도/성능에 가장 큰 영향을 주는 파라미터 중 하나입니다.
      cells_per_block (현재: (2, 2))
      밝기 정규화를 위한 "블록"의 크기입니다. (성능 영향)

      2. SVM 분류기 파라미터
      (SVMClassifier 클래스의 __init__에서)
      kernel (현재: 'rbf')
      SVM의 커널을 결정합니다. 'rbf' (비선형) 또는 'linear' (혹은 LinearSVC 클래스) 중에서 선택해야 합니다. 속도/성능에 결정적인 영향을 줍니다.
      C (현재: 1.0)
      SVM의 규제(Regularization) 강도입니다. (성능 영향)
      gamma (현재: 'scale')
      kernel='rbf'일 때만 유효하며, RBF 커널의 영향 범위를 결정합니다. (성능 영향)

      3. 슬라이딩 윈도우 탐지기 파라미터
      (SlidingWindowDetector 클래스의 __init__에서)
      scales (현재: [0.5, 0.75, 1.0, 1.25, 1.5])
      다양한 크기의 곤충을 찾기 위해 이미지를 리사이즈할 비율 목록입니다. (mAP/속도 영향)
      step_size (현재: 32)
      윈도우가 한 번에 몇 픽셀씩 건너뛸지(보폭) 결정합니다. 속도에 가장 큰 영향을 줍니다.
      confidence_threshold (현재: 0.5)
      SVM 예측 확률이 이 값(50%)보다 높아야만 탐지된 것으로 인정합니다. (mAP 영향)
      nms_threshold (현재: 0.3)
      겹친 박스를 제거(NMS)할 때 사용할 IoU 임계값입니다. (mAP 영향)

      4. (논의된) 파이프라인 파라미터
      (HNM 및 속도 최적화 논의에서 나온, 코드에 아직 없는 파라미터)
      HNM 비율 (Positive vs. Negative)
      1차/2차 학습 시 곤충 샘플 1개당 배경 샘플을 몇 개나 사용할지 비율을 정해야 합니다.
      PCA n_components
      속도 최적화를 위해 PCA를 쓴다면, 몇 차원으로 압축할지(예: 128, 256) 결정해야 합니다.

이제 이렇게 뽑아낸 파라미터 항목들에 대해 
베이지안 최적화 (TPE 포함)
추천 도구: Optuna 또는 Hyperopt
이유 (Why):
압도적인 효율성 (Trial 비용): 이 방법의 존재 이유입니다. 이전 시도(예: C=10일 때 85% 성능)를 학습하여 다음 시도(예: C=12 시도)를 지능적으로 결정합니다. 비싼 SVM 학습 횟수를 최소화(20~30회)하면서도 최적점에 가장 빠르게 도달할 수 있습니다.
복합 파라미터 처리: Optuna의 TPE 알고리즘은 SVM의 kernel('rbf' vs 'linear') 같은 범주형 파라미터와 C 값 같은 연속형 파라미터가 섞여 있을 때 최고의 성능을 보입니다.
방법을 통해 최적 파라미터를 찾는다.


3.모델 성능 (속도) 최적화
HOG + SlidingWindowDetector의 가장 큰 단점은 끔찍하게 느린 속도입니다. detect 함수는 이미지 한 장에 수천 개의 윈도우를 스캔하고, 매번 HOG 추출 + SVM predict를 호출해야 합니다.
리서치 목표: detect 함수의 속도를 높일 수 있는 방법을 조사해야 합니다.
핵심 리서치 주제:
특징 벡터 차원 축소 (PCA):
이유: HOGFeatureExtractor가 생성하는 특징 벡터는 차원이 매우 높습니다 (수천~수만). StandardScaler를 거친 이 벡터에 sklearn.decomposition.PCA를 적용해 (예: 128, 256 차원)로 줄이면, SVM의 predict 속도가 획기적으로 빨라집니다.
조사할 내용: PCA를 적용하는 위치, 유지할 주성분(차원) 개수에 따른 성능/속도 트레이드오프.
Linear SVM 사용:
이유: SVMClassifier는 현재 kernel='rbf'를 기본값으로 사용합니다. RBF 커널은 강력하지만 predict 속도가 매우 느립니다. 반면 kernel='linear' (또는 LinearSVC 모델)는 predict 속도가 비교할 수 없을 정도로 빠릅니다. HOG 같은 고차원 특징에서는 선형(Linear) 커널로도 충분히 좋은 성능이 나오는 경우가 많습니다.
조사할 내용: LinearSVC와 SVC(kernel='rbf')의 성능 및 예측 속도 비교 벤치마크.
(Advanced) Sliding Window 대안:
SlidingWindowDetector [cell 17] 대신, "객체가 있을 법한 영역"만 제안(proposal)하는 Selective Search 기법을 조사하여 탐지 속도 자체를 개선하는 방법을 리서치합니다.



