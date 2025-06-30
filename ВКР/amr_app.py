import traceback
import sys
import os
import psycopg2
import bcrypt
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from fpdf import FPDF
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QTabWidget,
    QFileDialog, QTextEdit
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf

# Конфигурация БД
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'dbname': 'amr_db',
    'user': 'postgres',
    'password': 'postpass'
}

# Настройка логирования
logging.basicConfig(
    filename='amr_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class EnhancedModelHandler:
    models = {
        'RandomForest': None,
        'XGBoost': None,
        'LogisticRegression': None,
        'SVM': None,
        'NeuralNetwork': None
    }

    @classmethod
    def init_models(cls):
        """Инициализация моделей с оптимизированными параметрами"""
        try:
            os.makedirs("models", exist_ok=True)  # Создаем папку если не существует

            # Проверка наличия всех файлов моделей
            model_files_exist = all([
                os.path.exists('models/rf_model.pkl'),
                os.path.exists('models/xgb_model.pkl'),
                os.path.exists('models/lr_model.pkl'),
                os.path.exists('models/svm_model.pkl'),
                os.path.exists('models/nn_model.h5'),
                os.path.exists('models/nn_scaler.pkl')
            ])

            if model_files_exist:
                cls.models = {
                    'RandomForest': joblib.load('models/rf_model.pkl'),
                    'XGBoost': joblib.load('models/xgb_model.pkl'),
                    'LogisticRegression': joblib.load('models/lr_model.pkl'),
                    'SVM': joblib.load('models/svm_model.pkl'),
                    'NeuralNetwork': {
                        'model': tf.keras.models.load_model('models/nn_model.h5'),
                        'scaler': joblib.load('models/nn_scaler.pkl')
                    }
                }
                logging.info("Модели успешно загружены из файлов")
            else:
                cls._initialize_new_models()
                logging.info("Инициализированы новые модели")

        except Exception as e:
            logging.error(f"Ошибка загрузки моделей: {str(e)}\n{traceback.format_exc()}")
            cls._initialize_new_models()

    @classmethod
    def _initialize_new_models(cls):
        """Создание новых моделей с оптимальными параметрами"""
        cls.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                scale_pos_weight=1,
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            ),
            'LogisticRegression': make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    class_weight='balanced',
                    C=0.1,
                    random_state=42,
                    max_iter=1000
                )
            ),
            'SVM': make_pipeline(
                MinMaxScaler(),
                SVC(
                    kernel='rbf',
                    C=1.0,
                    class_weight='balanced',
                    probability=True,
                    random_state=42
                )
            ),
            'NeuralNetwork': {
                'model': None,
                'scaler': StandardScaler()
            }
        }

    @classmethod
    def train_models(cls, X_train, y_train):
        """Обучение моделей с обработкой данных и улучшенной обработкой ошибок"""
        try:
            # Проверка входных данных
            if X_train is None or y_train is None:
                raise ValueError("Отсутствуют данные для обучения")

            if len(np.unique(y_train)) < 2:
                raise ValueError("Целевая переменная содержит только один класс")

            logging.info(f"Начало обучения моделей. Форма данных: {X_train.shape}")

            # Обработка дисбаланса классов
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            # Обучение RandomForest
            cls.models['RandomForest'].fit(X_res, y_res)
            joblib.dump(cls.models['RandomForest'], 'models/rf_model.pkl')

            # Обучение XGBoost
            cls.models['XGBoost'].fit(X_res, y_res)
            joblib.dump(cls.models['XGBoost'], 'models/xgb_model.pkl')

            # Обучение LogisticRegression
            cls.models['LogisticRegression'].fit(X_res, y_res)
            joblib.dump(cls.models['LogisticRegression'], 'models/lr_model.pkl')

            # Обучение SVM
            cls.models['SVM'].fit(X_res, y_res)
            joblib.dump(cls.models['SVM'], 'models/svm_model.pkl')

            # Обучение нейронной сети
            cls._train_neural_network(X_train, y_train)  # Используем оригинальные данные

            logging.info("Все модели успешно обучены")
            return True

        except Exception as e:
            logging.error(f"Ошибка обучения моделей: {str(e)}\n{traceback.format_exc()}")
            return False

    @classmethod
    def _train_neural_network(cls, X_train, y_train):
        """Обучение нейронной сети с улучшенной обработкой ошибок"""
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.5),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )

            early_stop = EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True
            )

            history = model.fit(
                X_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )

            cls.models['NeuralNetwork'] = {
                'model': model,
                'scaler': scaler
            }

            model.save('models/nn_model.h5')
            joblib.dump(scaler, 'models/nn_scaler.pkl')

        except Exception as e:
            logging.error(f"Ошибка обучения нейронной сети: {str(e)}\n{traceback.format_exc()}")
            raise

    @classmethod
    def calculate_metrics(cls, y_true, y_pred, y_proba):
        """Расчет метрик качества с проверкой входных данных"""
        try:
            if len(y_true) != len(y_pred) or len(y_true) != len(y_proba):
                raise ValueError("Размеры y_true, y_pred и y_proba должны совпадать")

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'sensitivity': recall_score(y_true, y_pred),
                'specificity': tn / (tn + fp),
                'precision': precision_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_proba),
                'f1': f1_score(y_true, y_pred),
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }

        except Exception as e:
            logging.error(f"Ошибка расчета метрик: {str(e)}\n{traceback.format_exc()}")
            raise


class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'AMR Prediction Report', ln=True, align='C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def generate(self, patient_data, prediction, probability):
        self.add_page()
        self.set_font("Arial", size=12)
        for k, v in patient_data.items():
            self.cell(0, 10, f"{k}: {v}", ln=True)
        self.cell(0, 10, f"Prediction: {'High Risk' if prediction else 'Low Risk'}", ln=True)
        self.cell(0, 10, f"Probability: {probability:.2%}", ln=True)
        os.makedirs("reports", exist_ok=True)
        filename = f"reports/report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        self.output(filename)
        return filename


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Вход в систему AMR")
        self.setGeometry(100, 100, 300, 150)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Имя пользователя")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Пароль")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        login_button = QPushButton("Войти")
        login_button.clicked.connect(self.handle_login)
        layout.addWidget(QLabel("Имя пользователя:"))
        layout.addWidget(self.username_input)
        layout.addWidget(QLabel("Пароль:"))
        layout.addWidget(self.password_input)
        layout.addWidget(login_button)
        self.setLayout(layout)

    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        user_id = self.verify_user(username, password)
        if user_id:
            QMessageBox.information(self, "Успех", f"Добро пожаловать, {username}!")
            self.hide()
            self.main = MainWindow(user_id)
            self.main.show()
        else:
            QMessageBox.critical(self, "Ошибка", "Неверное имя пользователя или пароль")

    def verify_user(self, username, password):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row and bcrypt.checkpw(password.encode(), row[1].encode()):
                return row[0]
            return None
        except Exception as e:
            print("DB error:", e)
            return None


class HistoryGraph(QWidget):
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id
        self.setWindowTitle("Графики истории")
        self.setGeometry(200, 200, 800, 600)
        layout = QVBoxLayout()
        self.canvas = FigureCanvas(plt.Figure(figsize=(10, 6)))
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.plot_history()

    def plot_history(self):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            df = pd.read_sql(
                "SELECT created_at, probability FROM predictions WHERE user_id = %s ORDER BY created_at",
                conn, params=(self.user_id,)
            )
            conn.close()

            if df.empty:
                return

            ax = self.canvas.figure.subplots()
            ax.clear()
            ax.plot(df['created_at'], df['probability'], marker='o', linestyle='-', color='teal')
            ax.set_title("Динамика вероятности AMR")
            ax.set_xlabel("Дата")
            ax.set_ylabel("Вероятность")
            ax.grid(True)
            self.canvas.draw()
        except Exception as e:
            print("Ошибка при построении графика:", e)


class EnhancedTrainTestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обучение моделей")
        self.setGeometry(300, 200, 800, 600)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.init_ui()
        EnhancedModelHandler.init_models()

    def init_ui(self):
        layout = QVBoxLayout()
        self.train_button = QPushButton("Загрузить данные и обучить модели")
        self.test_button = QPushButton("Оценить точность моделей")
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.train_button.clicked.connect(self.train_models)
        self.test_button.clicked.connect(self.test_models)
        layout.addWidget(self.train_button)
        layout.addWidget(self.test_button)
        layout.addWidget(self.results_display)
        self.setLayout(layout)

    def train_models(self):
        """Улучшенный метод обучения с обработкой данных"""
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV файл", "", "CSV Files (*.csv)")
            if not path:
                return

            df = pd.read_csv(path)

            # Проверка данных
            if 'AMR' not in df.columns:
                raise ValueError("CSV должен содержать столбец 'AMR'")

            # Предобработка
            df = self._preprocess_data(df)

            # Разделение данных
            X = df.drop(columns=['AMR'])
            y = df['AMR']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42)

            # Обучение моделей
            if EnhancedModelHandler.train_models(self.X_train, self.y_train):
                self._show_training_results()
            else:
                QMessageBox.critical(self, "Ошибка", "Ошибка при обучении моделей")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка: {str(e)}")
            logging.error(f"Training error: {str(e)}")

    def _preprocess_data(self, df):
        """Предобработка данных"""
        # Заполнение пропусков
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                # Новый рекомендуемый способ
                df.loc[:, col] = df[col].fillna(df[col].median())
            else:
                # Новый рекомендуемый способ
                df.loc[:, col] = df[col].fillna(df[col].mode()[0])

        # Кодирование категориальных признаков
        if 'C4d' in df.columns:
            c4d_map = {"Нет": 0, "Слабо": 1, "Умеренно": 2, "Сильно": 3}
            df['C4d'] = df['C4d'].map(c4d_map)

        return df

    def _show_training_results(self):
        """Отображение результатов обучения"""
        result_text = "Результаты обучения:\n\n"
        metrics = []

        for name, model in EnhancedModelHandler.models.items():
            if name == 'NeuralNetwork':
                train_metrics, test_metrics = self._evaluate_nn(model)
            else:
                train_metrics = self._evaluate_sklearn(model, self.X_train, self.y_train)
                test_metrics = self._evaluate_sklearn(model, self.X_test, self.y_test)

            metrics.append((name, train_metrics, test_metrics))

        # Сортировка по AUC-ROC на тесте
        metrics.sort(key=lambda x: x[2]['roc_auc'], reverse=True)

        for name, train_metrics, test_metrics in metrics:
            result_text += self._format_metrics(name, train_metrics, test_metrics)

        self.results_display.setPlainText(result_text)
        QMessageBox.information(self, "Успех", "Обучение завершено!\nМодели отсортированы по AUC-ROC.")

    def _evaluate_sklearn(self, model, X, y):
        """Оценка sklearn-моделей"""
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'sensitivity': recall_score(y, y_pred),
            'specificity': recall_score(y, y_pred, pos_label=0),
            'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
            'f1': f1_score(y, y_pred),
            'cm': confusion_matrix(y, y_pred)
        }

    def _evaluate_nn(self, model_dict):
        """Оценка нейронной сети"""
        scaler = model_dict['scaler']
        model = model_dict['model']

        # Оценка на обучающих данных
        X_train_scaled = scaler.transform(self.X_train)
        y_train_pred = (model.predict(X_train_scaled) > 0.5).astype(int)
        y_train_proba = model.predict(X_train_scaled)

        train_metrics = {
            'accuracy': accuracy_score(self.y_train, y_train_pred),
            'sensitivity': recall_score(self.y_train, y_train_pred),
            'specificity': recall_score(self.y_train, y_train_pred, pos_label=0),
            'roc_auc': roc_auc_score(self.y_train, y_train_proba),
            'f1': f1_score(self.y_train, y_train_pred),
            'cm': confusion_matrix(self.y_train, y_train_pred)
        }

        # Оценка на тестовых данных
        X_test_scaled = scaler.transform(self.X_test)
        y_test_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
        y_test_proba = model.predict(X_test_scaled)

        test_metrics = {
            'accuracy': accuracy_score(self.y_test, y_test_pred),
            'sensitivity': recall_score(self.y_test, y_test_pred),
            'specificity': recall_score(self.y_test, y_test_pred, pos_label=0),
            'roc_auc': roc_auc_score(self.y_test, y_test_proba),
            'f1': f1_score(self.y_test, y_test_pred),
            'cm': confusion_matrix(self.y_test, y_test_pred)
        }

        return train_metrics, test_metrics

    def _format_metrics(self, name, train_metrics, test_metrics):
        """Форматирование метрик для вывода"""
        cm = test_metrics['cm']
        return (
            f"=== {name} ===\n"
            "Обучающая выборка:\n"
            f"Accuracy: {train_metrics['accuracy']:.2%} | "
            f"AUC-ROC: {train_metrics['roc_auc']:.3f} | "
            f"F1: {train_metrics['f1']:.3f}\n"
            f"Sensitivity: {train_metrics['sensitivity']:.2%} | "
            f"Specificity: {train_metrics['specificity']:.2%}\n"
            "Тестовая выборка:\n"
            f"Accuracy: {test_metrics['accuracy']:.2%} | "
            f"AUC-ROC: {test_metrics['roc_auc']:.3f} | "
            f"F1: {test_metrics['f1']:.3f}\n"
            f"Sensitivity: {test_metrics['sensitivity']:.2%} | "
            f"Specificity: {test_metrics['specificity']:.2%}\n"
            "Confusion Matrix (Test):\n"
            f"TP: {cm[1, 1]} | FP: {cm[0, 1]}\n"
            f"FN: {cm[1, 0]} | TN: {cm[0, 0]}\n\n"
        )

    def test_models(self):
        if self.X_test is None:
            QMessageBox.warning(self, "Ошибка", "Сначала необходимо обучить модели")
            return

        result_text = "Результаты тестирования:\n\n"
        for name, model in EnhancedModelHandler.models.items():
            if model is None:
                continue

            if name == 'NeuralNetwork':
                X_test_scaled = model['scaler'].transform(self.X_test)
                y_pred = (model['model'].predict(X_test_scaled) > 0.5).astype(int).flatten()
                y_proba = model['model'].predict(X_test_scaled).flatten()
            else:
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)[:, 1]

            metrics = EnhancedModelHandler.calculate_metrics(self.y_test, y_pred, y_proba)
            result_text += (
                f"=== {name} ===\n"
                f"Accuracy: {metrics['accuracy']:.2%}\n"
                f"Sensitivity: {metrics['sensitivity']:.2%}\n"
                f"Specificity: {metrics['specificity']:.2%}\n"
                f"AUC-ROC: {metrics['roc_auc']:.3f}\n"
                f"Confusion Matrix:\n"
                f"TP: {metrics['tp']} | FP: {metrics['fp']}\n"
                f"FN: {metrics['fn']} | TN: {metrics['tn']}\n\n"
            )

        self.results_display.setText(result_text)


class PredictionTab(QWidget):
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id
        self.init_ui()
        EnhancedModelHandler.init_models()

    def init_ui(self):
        layout = QVBoxLayout()
        font = QFont("Segoe UI", 10)

        self.age_input = QSpinBox()
        self.age_input.setRange(18, 80)
        self.dsa_input = QDoubleSpinBox()
        self.dsa_input.setRange(0.0, 10000.0)
        self.pra_input = QDoubleSpinBox()
        self.pra_input.setRange(0.0, 100.0)
        self.c4d_input = QComboBox()
        self.c4d_input.addItems(["Нет", "Слабо", "Умеренно", "Сильно"])
        self.troponin_input = QDoubleSpinBox()
        self.troponin_input.setRange(0.0, 100.0)
        self.creatinine_input = QDoubleSpinBox()
        self.creatinine_input.setRange(0.0, 5.0)
        self.bnp_input = QDoubleSpinBox()
        self.bnp_input.setRange(0.0, 10000.0)
        self.model_input = QComboBox()
        self.model_input.addItems(list(EnhancedModelHandler.models.keys()) + ["Комбинированное решение"])

        predict_button = QPushButton("Сделать прогноз")
        graph_button = QPushButton("Показать график истории")
        predict_button.clicked.connect(self.handle_prediction)
        graph_button.clicked.connect(self.show_graph)

        for label, widget in [
            ("Возраст", self.age_input), ("DSA", self.dsa_input), ("PRA (%)", self.pra_input),
            ("C4d позитивность", self.c4d_input), ("Тропонин", self.troponin_input),
            ("Креатинин", self.creatinine_input), ("BNP", self.bnp_input),
            ("Модель", self.model_input)
        ]:
            lbl = QLabel(label)
            lbl.setFont(font)
            layout.addWidget(lbl)
            layout.addWidget(widget)

        layout.addWidget(predict_button)
        layout.addWidget(graph_button)
        self.setLayout(layout)

    def handle_prediction(self):
        try:
            age = self.age_input.value()
            dsa = self.dsa_input.value()
            pra = self.pra_input.value()
            c4d_map = {"Нет": 0, "Слабо": 1, "Умеренно": 2, "Сильно": 3}
            c4d = c4d_map[self.c4d_input.currentText()]
            troponin = self.troponin_input.value()
            creatinine = self.creatinine_input.value()
            bnp = self.bnp_input.value()

            input_data = {
                'Age': age, 'DSA': dsa, 'PRA': pra, 'C4d': c4d,
                'Troponin': troponin, 'Creatinine': creatinine, 'BNP': bnp
            }
            input_df = pd.DataFrame([input_data])

            selected_model = self.model_input.currentText()

            if selected_model == "Комбинированное решение":
                votes = []
                probabilities = []
                for name, model in EnhancedModelHandler.models.items():
                    if model is None:
                        continue

                    if name == 'NeuralNetwork':
                        X_scaled = model['scaler'].transform(input_df)
                        prob = model['model'].predict(X_scaled)[0][0]
                    else:
                        prob = model.predict_proba(input_df)[0][1]

                    pred = int(prob >= 0.5)
                    votes.append(pred)
                    probabilities.append(prob)

                final = int(sum(votes) >= 3)  # Голосование большинством
                probability = np.mean(probabilities)
            else:
                model = EnhancedModelHandler.models[selected_model]
                if model is None:
                    raise ValueError(f"Модель {selected_model} не загружена")

                if selected_model == 'NeuralNetwork':
                    X_scaled = model['scaler'].transform(input_df)
                    probability = model['model'].predict(X_scaled)[0][0]
                else:
                    probability = model.predict_proba(input_df)[0][1]

                final = int(probability >= 0.5)

            pdf = PDFReport()
            filename = pdf.generate(input_data, final, probability)
            self.save_to_db(final, probability, input_data)

            QMessageBox.information(
                self, "Результат",
                f"Прогноз: {'Риск AMR' if final else 'Нет риска'}\n"
                f"Вероятность: {probability:.2%}\n"
                f"PDF сохранён: {filename}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при прогнозировании: {str(e)}")

    def save_to_db(self, result, probability, data):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO predictions 
                (user_id, age, dsa, pra, c4d, troponin, creatinine, bnp, result, probability)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                self.user_id, data['Age'], data['DSA'], data['PRA'], data['C4d'],
                data['Troponin'], data['Creatinine'], data['BNP'], bool(result), float(probability)
            ))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print("Ошибка при сохранении в БД:", e)

    def show_graph(self):
        self.graph_window = HistoryGraph(self.user_id)
        self.graph_window.show()


class MainWindow(QWidget):
    def __init__(self, user_id):
        super().__init__()
        self.setWindowTitle("Система прогноза AMR")
        self.setGeometry(150, 150, 800, 600)
        tabs = QTabWidget()
        tabs.addTab(PredictionTab(user_id), "Прогноз")
        tabs.addTab(EnhancedTrainTestWindow(), "Обучение и тест")
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)


if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec())
