from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
from time import time

class ClassifierComparer:
    def __init__(self):
        # 다양한 분류기와 각각의 파라미터 정의
        self.classifiers = {
            'SVM': (
                SVC(random_state=42),
                {
                    'C': np.logspace(-3, 3, 7),
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 0, 4))
                }
            ),
            'Logistic Regression': (
                LogisticRegression(random_state=42, max_iter=1000),
                {
                    'C': np.logspace(-3, 3, 7),
                    'solver': ['lbfgs', 'newton-cg'],
                    'multi_class': ['multinomial']
                }
            ),
            'KNN': (
                KNeighborsClassifier(),
                {
                    'n_neighbors': [3, 5, 7, 9, 11, 13],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # Manhattan or Euclidean distance
                }
            ),
            'Gradient Boosting': (
                GradientBoostingClassifier(random_state=42),
                {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [2, 4, 6]
                }
            )
        }
        
    def prepare_data(self):
        # 데이터 로드 및 전처리
        wine = load_wine()
        X = wine.data
        y = wine.target
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # 특성 스케일링
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        results = {}
        
        for name, (classifier, param_dist) in self.classifiers.items():
            print(f"\n{'-'*50}\n훈련 시작: {name}")
            start_time = time()
            
            # RandomizedSearchCV를 사용한 하이퍼파라미터 튜닝
            random_search = RandomizedSearchCV(
                classifier,
                param_distributions=param_dist,
                n_iter=20,
                cv=5,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            
            # 모델 학습
            random_search.fit(X_train, y_train)
            
            # 예측 및 평가
            y_pred = random_search.predict(X_test)
            training_time = time() - start_time
            
            # 결과 저장
            results[name] = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'test_accuracy': accuracy_score(y_test, y_pred),
                'training_time': training_time,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"\n{name} 결과:")
            print(f"최적 파라미터: {results[name]['best_params']}")
            print(f"최고 교차 검증 점수: {results[name]['best_score']:.4f}")
            print(f"테스트 정확도: {results[name]['test_accuracy']:.4f}")
            print(f"훈련 시간: {results[name]['training_time']:.2f}초")
            print("\n분류 보고서:")
            print(results[name]['classification_report'])
        
        return results

# 실행
comparer = ClassifierComparer()
X_train, X_test, y_train, y_test = comparer.prepare_data()
results = comparer.train_and_evaluate(X_train, X_test, y_train, y_test)

# 최종 비교
print("\n=== 모델 간 성능 비교 ===")
for name, result in results.items():
    print(f"\n{name}:")
    print(f"교차 검증 점수: {result['best_score']:.4f}")
    print(f"테스트 정확도: {result['test_accuracy']:.4f}")
    print(f"훈련 시간: {result['training_time']:.2f}초")