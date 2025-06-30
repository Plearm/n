import sys
import os
import psycopg2
import bcrypt
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from keras.wrappers.scikit_learn import KerasClassifier

# Конфигурация БД
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'dbname': 'amr_db',
    'user': 'postgres',
    'password': 'postpass'
}


# Класс для работы с моделями
class ModelHandler:
    models = {
        'RandomForest': None,
        'XGBoost': None,
        'LogisticRegression': None,
        'SVM': None,
        'NeuralNetwork': None
    }

    @classmethod
    def init_models(cls):
        """Инициализация моделей с загрузкой из файлов"""
        try:
            cls.models['RandomForest'] = joblib.load('models/random_forest_model.pkl')
            cls.models['XGBoost'] = joblib.load('models/xgboost_model.pkl')
            cls.models['LogisticRegression'] = joblib.load('models/logistic_regression_model.pkl')
            cls.models['SVM'] = joblib.load('models/svm_model.pkl')

            # Загрузка нейронной сети
            nn_model = Sequential([
                Dense(64, activation='relu', input_shape=(7,)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            nn_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            nn_model.load_weights('models/neural_network_model.weights.h5')
            cls.models['NeuralNetwork'] = {
                'model': nn_model,
                'scaler': joblib.load('models/neural_network_scaler.pkl')
            }
        except:
            pass

    @classmethod
    def calculate_metrics(cls, y_true, y_pred, y_proba):
        """Расчет метрик качества"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred),
            'specificity': tn / (tn + fp),
            'precision': precision_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }


# Класс для отчетов PDF
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


# Классы интерфейса
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


class TrainTestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обучение моделей")
        self.setGeometry(300, 200, 800, 600)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.init_ui()
        ModelHandler.init_models()

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
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV файл", "", "CSV Files (*.csv)")
        if not path:
            return

        try:
            df = pd.read_csv(path)
            if 'AMR' not in df.columns:
                raise ValueError("CSV должен содержать столбец 'AMR'")

            X = df.drop(columns=['AMR'])
            y = df['AMR']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            # SMOTE для балансировки
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)

            os.makedirs("models", exist_ok=True)
            result_text = "Результаты обучения:\n\n"

            # RandomForest
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
            rf.fit(X_resampled, y_resampled)
            joblib.dump(rf, 'models/random_forest_model.pkl')
            ModelHandler.models['RandomForest'] = rf
            result_text += self._calculate_model_metrics(rf, "RandomForest")

            # XGBoost
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=(y_resampled == 0).sum() / (y_resampled == 1).sum(),
                random_state=42
            )
            xgb.fit(X_resampled, y_resampled)
            joblib.dump(xgb, 'models/xgboost_model.pkl')
            ModelHandler.models['XGBoost'] = xgb
            result_text += self._calculate_model_metrics(xgb, "XGBoost")

            # LogisticRegression
            lr = LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                solver='liblinear',
                random_state=42
            )
            lr.fit(X_resampled, y_resampled)
            joblib.dump(lr, 'models/logistic_regression_model.pkl')
            ModelHandler.models['LogisticRegression'] = lr
            result_text += self._calculate_model_metrics(lr, "LogisticRegression")

            # SVM
            svm = make_pipeline(
                StandardScaler(),
                SVC(kernel='rbf', C=1.2, probability=True, class_weight='balanced', random_state=42)
            )
            svm.fit(X_resampled, y_resampled)
            joblib.dump(svm, 'models/svm_model.pkl')
            ModelHandler.models['SVM'] = svm
            result_text += self._calculate_model_metrics(svm, "SVM")

            # Нейронная сеть
            scaler = StandardScaler()
            X_res_scaled = scaler.fit_transform(X_resampled)
            X_test_scaled = scaler.transform(self.X_test)

            nn = Sequential([
                Dense(128, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l2(0.001)),
                Dropout(0.3),
                Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            nn.compile(optimizer=Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])
            nn.fit(X_res_scaled, y_resampled, epochs=70, batch_size=32, verbose=0,
                   callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

            nn.save_weights('models/neural_network_model.weights.h5')
            joblib.dump(scaler, 'models/neural_network_scaler.pkl')
            ModelHandler.models['NeuralNetwork'] = {'model': nn, 'scaler': scaler}
            result_text += self._calculate_nn_metrics(nn, scaler, "NeuralNetwork")

            self.results_display.setText(result_text)
            QMessageBox.information(self, "Успех", "Все модели обучены и сохранены.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обучении: {str(e)}")

    def _calculate_model_metrics(self, model, name):
        """Вычисление метрик для классических моделей"""
        # Метрики на обучающей выборке
        train_pred = model.predict(self.X_train)
        train_proba = model.predict_proba(self.X_train)[:, 1]
        train_metrics = ModelHandler.calculate_metrics(self.y_train, train_pred, train_proba)

        # Метрики на тестовой выборке
        test_pred = model.predict(self.X_test)
        test_proba = model.predict_proba(self.X_test)[:, 1]
        test_metrics = ModelHandler.calculate_metrics(self.y_test, test_pred, test_proba)

        return (
            f"=== {name} ===\n"
            "Обучающая выборка:\n"
            f"Accuracy: {train_metrics['accuracy']:.2%} | "
            f"AUC-ROC: {train_metrics['roc_auc']:.3f} | "
            f"Sensitivity: {train_metrics['sensitivity']:.2%} | "
            f"Specificity: {train_metrics['specificity']:.2%}\n"
            "Тестовая выборка:\n"
            f"Accuracy: {test_metrics['accuracy']:.2%} | "
            f"AUC-ROC: {test_metrics['roc_auc']:.3f} | "
            f"Sensitivity: {test_metrics['sensitivity']:.2%} | "
            f"Specificity: {test_metrics['specificity']:.2%}\n"
            f"Confusion Matrix (Test):\n"
            f"TP: {test_metrics['tp']} | FP: {test_metrics['fp']}\n"
            f"FN: {test_metrics['fn']} | TN: {test_metrics['tn']}\n\n"
        )

    def _calculate_nn_metrics(self, model, scaler, name):
        """Вычисление метрик для нейронной сети"""
        # Масштабирование данных
        X_train_scaled = scaler.transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        # Метрики на обучающей выборке
        train_pred = (model.predict(X_train_scaled) > 0.5).astype(int).flatten()
        train_proba = model.predict(X_train_scaled).flatten()
        train_metrics = ModelHandler.calculate_metrics(self.y_train, train_pred, train_proba)

        # Метрики на тестовой выборке
        test_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()
        test_proba = model.predict(X_test_scaled).flatten()
        test_metrics = ModelHandler.calculate_metrics(self.y_test, test_pred, test_proba)

        return (
            f"=== {name} ===\n"
            "Обучающая выборка:\n"
            f"Accuracy: {train_metrics['accuracy']:.2%} | "
            f"AUC-ROC: {train_metrics['roc_auc']:.3f} | "
            f"Sensitivity: {train_metrics['sensitivity']:.2%} | "
            f"Specificity: {train_metrics['specificity']:.2%}\n"
            "Тестовая выборка:\n"
            f"Accuracy: {test_metrics['accuracy']:.2%} | "
            f"AUC-ROC: {test_metrics['roc_auc']:.3f} | "
            f"Sensitivity: {test_metrics['sensitivity']:.2%} | "
            f"Specificity: {test_metrics['specificity']:.2%}\n"
            f"Confusion Matrix (Test):\n"
            f"TP: {test_metrics['tp']} | FP: {test_metrics['fp']}\n"
            f"FN: {test_metrics['fn']} | TN: {test_metrics['tn']}\n\n"
        )

    def test_models(self):
        if self.X_test is None:
            QMessageBox.warning(self, "Ошибка", "Сначала необходимо обучить модели")
            return

        result_text = "Результаты тестирования:\n\n"
        for name, model in ModelHandler.models.items():
            if model is None:
                continue

            if name == 'NeuralNetwork':
                X_test_scaled = model['scaler'].transform(self.X_test)
                y_pred = (model['model'].predict(X_test_scaled) > 0.5).astype(int).flatten()
                y_proba = model['model'].predict(X_test_scaled).flatten()
            else:
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)[:, 1]

            metrics = ModelHandler.calculate_metrics(self.y_test, y_pred, y_proba)
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
        ModelHandler.init_models()

    def init_ui(self):
        layout = QVBoxLayout()
        font = QFont("Segoe UI", 10)

        self.age_input = QSpinBox();
        self.age_input.setRange(18, 80)
        self.dsa_input = QDoubleSpinBox();
        self.dsa_input.setRange(0.0, 10000.0)
        self.pra_input = QDoubleSpinBox();
        self.pra_input.setRange(0.0, 100.0)
        self.c4d_input = QComboBox();
        self.c4d_input.addItems(["Нет", "Слабо", "Умеренно", "Сильно"])
        self.troponin_input = QDoubleSpinBox();
        self.troponin_input.setRange(0.0, 100.0)
        self.creatinine_input = QDoubleSpinBox();
        self.creatinine_input.setRange(0.0, 5.0)
        self.bnp_input = QDoubleSpinBox();
        self.bnp_input.setRange(0.0, 10000.0)
        self.model_input = QComboBox()
        self.model_input.addItems(list(ModelHandler.models.keys()) + ["Комбинированное решение"])

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
                for name, model in ModelHandler.models.items():  # З
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
            else:  # Этот else должен быть на том же уровне, что и if selected_model == ...
                model = ModelHandler.models[selected_model]
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
        tabs.addTab(TrainTestWindow(), "Обучение и тест")  # Заменили здесь
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
