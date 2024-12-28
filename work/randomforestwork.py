from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# 데이터셋 로드 및 전처리
wine = load_wine()
X = wine.data
y = wine.target

# 데이터셋 나누기 (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 두 단계로 나누어 파라미터 탐색
# 1단계: 넓은 범위에서 대략적인 최적 지점 찾기
param_dist_1 = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

random_search_1 = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist_1,
    n_iter=20,  # 20개의 랜덤 조합만 시도
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 1단계 탐색 수행
random_search_1.fit(X_train, y_train)

# 1단계 결과를 바탕으로 좁은 범위 설정
best_params = random_search_1.best_params_
print("1단계 최적 파라미터:", best_params)

# 2단계: 좁은 범위에서 세밀한 탐색
param_dist_2 = {
    'n_estimators': np.linspace(
        max(50, best_params['n_estimators'] - 50),
        best_params['n_estimators'] + 50,
        5, dtype=int),
    'max_depth': ([None] if best_params['max_depth'] is None 
                 else np.linspace(
                     max(5, best_params['max_depth'] - 5),
                     best_params['max_depth'] + 5,
                     4, dtype=int)),
    'min_samples_split': np.linspace(
        max(2, best_params['min_samples_split'] - 2),
        best_params['min_samples_split'] + 2,
        3, dtype=int),
    'min_samples_leaf': np.linspace(
        max(1, best_params['min_samples_leaf'] - 1),
        best_params['min_samples_leaf'] + 1,
        3, dtype=int)
}

random_search_2 = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist_2,
    n_iter=15,  # 15개의 랜덤 조합 시도
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 2단계 탐색 수행
random_search_2.fit(X_train, y_train)

# 최종 결과 출력
print("\n2단계 최적 파라미터:", random_search_2.best_params_)
print("최고 교차 검증 점수:", random_search_2.best_score_)

# 최적의 모델로 예측
y_pred = random_search_2.predict(X_test)

# 평가 결과 출력
print("\n테스트 세트 정확도:", accuracy_score(y_test, y_pred))
print("\n분류 보고서:\n", classification_report(y_test, y_pred))