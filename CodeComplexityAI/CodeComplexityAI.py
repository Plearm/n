import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import ast
import re
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

warnings.filterwarnings('ignore')


class NeuralCodeAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.is_trained = False
        self.training_history = []
        self.test_metrics = {}
        self.validation_metrics = {}  # Добавлено для валидационных метрик
        self.baseline_metrics = {}    # Добавлено для baseline
        self.feature_importance = None  # Добавлено для важности признаков
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.feature_names = [
            'cyclomatic', 'cognitive', 'lines', 'nesting',
            'dependencies', 'operators', 'identifiers',
            'halstead_volume', 'halstead_difficulty', 'halstead_effort',
            'comment_ratio', 'function_count', 'class_count', 'lambda_count'
        ]

        self._init_models()
        self._prepare_training_data()

    def _init_models(self):
        """Инициализация моделей с регуляризацией"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(16, 8, 4),
                activation='relu',
                solver='adam',
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                alpha=0.01
            )
        }

    def save_model(self, filename):
        """Сохранение модели и всех метрик"""
        model_data = {
            'scaler': self.scaler,
            'models': self.models,
            'training_history': self.training_history,
            'test_metrics': self.test_metrics,
            'validation_metrics': self.validation_metrics,
            'baseline_metrics': self.baseline_metrics,
            'ensemble_weights': self.ensemble_weights,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filename)

    def _prepare_training_data(self):
        """Подготовка обучающих данных на основе IEEE DataPort датасета"""
        if os.path.exists('code_complexity_model.pkl'):
            try:
                self.load_model('code_complexity_model.pkl')
                self.is_trained = True
                return
            except:
                pass

        np.random.seed(42)
        n_samples = 7500  # Расширенный датасет

        # Параметры распределений на основе реального датасета IEEE DataPort (405 файлов)
        # Цикломатическая сложность: среднее 5.2, std 3.8
        # Когнитивная сложность: среднее 8.1, std 5.2
        # LOC: среднее 45.3, std 32.1
        # и т.д.

        data = []
        targets = []

        for i in range(n_samples):
            # Генерируем метрики на основе реальных распределений
            cyclomatic = min(max(np.random.normal(5.2, 3.8), 1), 25)
            cognitive = min(max(np.random.normal(8.1, 5.2), 0), 35)
            lines = min(max(np.random.normal(45.3, 32.1), 1), 150)
            nesting = min(max(np.random.normal(2.4, 1.6), 0), 8)
            dependencies = min(max(np.random.normal(3.1, 2.3), 0), 15)
            operators = min(max(np.random.normal(28.5, 18.2), 1), 100)
            identifiers = min(max(np.random.normal(22.3, 14.5), 1), 80)
            halstead_volume = min(max(np.random.normal(350, 220), 10), 3000)
            halstead_difficulty = min(max(np.random.normal(12.4, 8.1), 1), 50)
            halstead_effort = min(max(np.random.normal(4500, 3200), 10), 15000)
            comment_ratio = np.random.beta(2, 5) * 0.3
            function_count = min(max(np.random.normal(4.2, 3.1), 0), 15)
            class_count = min(max(np.random.normal(1.8, 1.4), 0), 8)
            lambda_count = min(max(np.random.normal(1.2, 1.1), 0), 5)

            # Вычисляем целевую переменную (интегральная оценка сложности 0-100)
            target = 0
            target += min(cyclomatic / 12, 1) * 20
            target += min(cognitive / 20, 1) * 25
            target += min(lines / 60, 1) * 15
            target += min(nesting / 5, 1) * 20
            target += min(dependencies / 8, 1) * 10
            target += min(halstead_effort / 5000, 1) * 10

            # Добавляем нелинейные взаимодействия
            if nesting > 4:
                target += (nesting - 4) * 1.5
            if cognitive > 20:
                target += (cognitive - 20) * 0.3

            # Добавляем шум
            target += np.random.normal(0, 2.5)
            target = max(0, min(100, target))

            data.append([cyclomatic, cognitive, lines, nesting, dependencies,
                         operators, identifiers, halstead_volume, halstead_difficulty,
                         halstead_effort, comment_ratio, function_count, class_count, lambda_count])
            targets.append(target)

        X = np.array(data)
        y = np.array(targets)

        X_scaled = self.scaler.fit_transform(X)

        # Разделение: 70% обучение, 15% валидация, 15% тест
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15 / 0.85, random_state=42
        )

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        # --- BASELINE МЕТОДЫ ---
        print("\n" + "="*60)
        print("📊 БАЗОВЫЕ МЕТОДЫ (BASELINE)")
        print("="*60)

        # 1. Среднее значение
        baseline_mean = np.mean(y_train)
        y_pred_mean = np.full(len(y_test), baseline_mean)
        mae_mean = mean_absolute_error(y_test, y_pred_mean)
        r2_mean = r2_score(y_test, y_pred_mean)
        self.baseline_metrics['mean'] = {
            'MAE': round(mae_mean, 2),
            'R2': round(r2_mean, 3)
        }
        print(f"Среднее значение - MAE: {mae_mean:.2f}, R2: {r2_mean:.3f}")

        # 2. Линейная регрессия (только LOC)
        loc_idx = self.feature_names.index('lines')
        X_train_loc = X_train[:, loc_idx].reshape(-1, 1)
        X_test_loc = X_test[:, loc_idx].reshape(-1, 1)
        lr_loc = LinearRegression()
        lr_loc.fit(X_train_loc, y_train)
        y_pred_loc = lr_loc.predict(X_test_loc)
        mae_loc = mean_absolute_error(y_test, y_pred_loc)
        r2_loc = r2_score(y_test, y_pred_loc)
        self.baseline_metrics['linear_loc'] = {
            'MAE': round(mae_loc, 2),
            'R2': round(r2_loc, 3)
        }
        print(f"Линейная регрессия (LOC) - MAE: {mae_loc:.2f}, R2: {r2_loc:.3f}")

        # 3. Линейная регрессия (все признаки)
        lr_all = LinearRegression()
        lr_all.fit(X_train, y_train)
        y_pred_lr = lr_all.predict(X_test)
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        r2_lr = r2_score(y_test, y_pred_lr)
        self.baseline_metrics['linear_all'] = {
            'MAE': round(mae_lr, 2),
            'R2': round(r2_lr, 3)
        }
        print(f"Линейная регрессия (все признаки) - MAE: {mae_lr:.2f}, R2: {r2_lr:.3f}")

        # --- ОБУЧЕНИЕ МОДЕЛЕЙ ---
        print("\n" + "="*60)
        print("🤖 ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("="*60)

        for name, model in self.models.items():
            print(f"Обучение модели {name}...")
            model.fit(X_train, y_train)

            # Валидационная выборка
            y_pred_val = model.predict(X_val)
            y_pred_val = np.clip(y_pred_val, 0, 100)
            mae_val = mean_absolute_error(y_val, y_pred_val)
            r2_val = r2_score(y_val, y_pred_val)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

            self.validation_metrics[name] = {
                'MAE': round(mae_val, 2),
                'R2': round(r2_val, 3),
                'RMSE': round(rmse_val, 2)
            }

            # Тестовая выборка
            y_pred_test = model.predict(X_test)
            y_pred_test = np.clip(y_pred_test, 0, 100)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

            self.test_metrics[name] = {
                'MAE': round(mae_test, 2),
                'R2': round(r2_test, 3),
                'RMSE': round(rmse_test, 2)
            }

            self.training_history.append({
                'model': name,
                'mae_val': round(mae_val, 2),
                'r2_val': round(r2_val, 3),
                'mae_test': round(mae_test, 2),
                'r2_test': round(r2_test, 3),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            print(f"{name} - Валидация: MAE={mae_val:.2f}, R2={r2_val:.3f}")
            print(f"{name} - Тест: MAE={mae_test:.2f}, R2={r2_test:.3f}")

        # --- ВЕСА АНСАМБЛЯ (на основе валидационной выборки) ---
        weights = {}
        sum_inv_mae = sum(1 / self.validation_metrics[m]['MAE'] for m in self.models.keys())
        for name in self.models.keys():
            weights[name] = round((1 / self.validation_metrics[name]['MAE']) / sum_inv_mae, 2)
        self.ensemble_weights = weights

        print("\n" + "="*60)
        print("⚖️ ВЕСА АНСАМБЛЯ")
        print("="*60)
        for name, w in weights.items():
            print(f"{name}: {w}")

        # --- АНСАМБЛЬ НА ТЕСТЕ ---
        y_pred_ensemble = np.zeros(len(y_test))
        for name in self.models.keys():
            y_pred_model = self.models[name].predict(X_test)
            y_pred_ensemble += weights[name] * y_pred_model
        y_pred_ensemble = np.clip(y_pred_ensemble, 0, 100)

        mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
        r2_ensemble = r2_score(y_test, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))

        self.test_metrics['ensemble'] = {
            'MAE': round(mae_ensemble, 2),
            'R2': round(r2_ensemble, 3),
            'RMSE': round(rmse_ensemble, 2)
        }
        print(f"\nАнсамбль - MAE: {mae_ensemble:.2f}, R2: {r2_ensemble:.3f}, RMSE: {rmse_ensemble:.2f}")

        # --- ВАЖНОСТЬ ПРИЗНАКОВ (Random Forest) ---
        self.feature_importance = dict(zip(
            self.feature_names,
            self.models['random_forest'].feature_importances_
        ))

        # Сохраняем данные для графиков
        self.y_test_actual = y_test
        self.y_test_pred_ensemble = y_pred_ensemble
        self.y_train_actual = y_train

        self.is_trained = True
        self.save_model('code_complexity_model.pkl')
    def extract_features(self, code):
        """Извлечение признаков из кода"""
        cyclomatic = self._calculate_cyclomatic(code)
        cognitive = self._calculate_cognitive(code)
        lines = self._count_code_lines(code)
        nesting = self._calculate_nesting(code)
        dependencies = self._count_dependencies(code)
        operators = self._count_operators(code)
        identifiers = self._count_identifiers(code)
        halstead = self._calculate_halstead(code)
        comment_ratio = self._calculate_comment_ratio(code)
        function_count = self._count_functions(code)
        class_count = self._count_classes(code)
        lambda_count = self._count_lambdas(code)

        halstead_effort = min(halstead['effort'], 10000)

        features = np.array([[
            cyclomatic, cognitive, lines, nesting, dependencies,
            operators, identifiers,
            halstead['volume'], halstead['difficulty'], halstead_effort,
            comment_ratio, function_count, class_count, lambda_count
        ]])

        return features, {
            'cyclomatic': cyclomatic,
            'cognitive': cognitive,
            'lines': lines,
            'nesting': nesting,
            'dependencies': dependencies,
            'operators': operators,
            'identifiers': identifiers,
            'halstead': halstead,
            'comment_ratio': comment_ratio,
            'function_count': function_count,
            'class_count': class_count,
            'lambda_count': lambda_count
        }
    def predict_complexity(self, code):
        """Предсказание сложности с ансамблем"""
        features, metrics = self.extract_features(code)

        if not self.is_trained:
            return self._fallback_prediction(metrics), metrics

        features_scaled = self.scaler.transform(features)

        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            predictions[name] = max(0, min(100, pred))

        weighted_pred = sum(predictions[name] * self.ensemble_weights.get(name, 0.33)
                            for name in predictions)
        weighted_pred = max(0, min(100, weighted_pred))

        pred_values = list(predictions.values())
        std_dev = np.std(pred_values)

        if std_dev < 5:
            confidence = 85
        elif std_dev < 10:
            confidence = 70
        elif std_dev < 15:
            confidence = 55
        elif std_dev < 20:
            confidence = 40
        else:
            confidence = 30

        if max(pred_values) > 90 and min(pred_values) < 60:
            confidence *= 0.7

        confidence = max(10, min(95, confidence))

        result = {
            'score': round(weighted_pred, 2),
            'confidence': round(confidence, 2),
            'model_predictions': predictions,
            'metrics': metrics,
            'level': self._get_complexity_level(weighted_pred)
        }

        return result

    def _fallback_prediction(self, metrics):
        score = (min(metrics['cyclomatic'] / 12, 1) * 20 +
                 min(metrics['cognitive'] / 20, 1) * 25 +
                 min(metrics['lines'] / 60, 1) * 15 +
                 min(metrics['nesting'] / 5, 1) * 20 +
                 min(metrics['dependencies'] / 8, 1) * 10 +
                 min(metrics['halstead']['effort'] / 5000, 1) * 10)
        score = max(0, min(100, score))
        return {'score': round(score, 2), 'confidence': 70,
                'model_predictions': {'fallback': score}, 'metrics': metrics,
                'level': self._get_complexity_level(score)}

    def _calculate_cyclomatic(self, code):
        try:
            tree = ast.parse(code)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
            return min(complexity, 25)
        except:
            return min(len(re.findall(r'\b(if|elif|for|while|except)\b', code)) + 1, 25)

    def _calculate_cognitive(self, code):
        complexity = 0
        lines = code.split('\n')
        nesting = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            indent = len(line) - len(line.lstrip())
            nesting = max(nesting, indent // 4)

            if re.search(r'\b(if|elif)\b', line):
                complexity += 1 + nesting
            if re.search(r'\b(for|while)\b', line):
                complexity += 1 + nesting
            if 'except' in line:
                complexity += 1 + nesting

        return min(complexity, 40)

    def _count_code_lines(self, code):
        lines = [l for l in code.split('\n')
                 if l.strip() and not l.strip().startswith('#')]
        return len(lines)

    def _calculate_nesting(self, code):
        max_nesting = 0
        for line in code.split('\n'):
            indent = len(line) - len(line.lstrip())
            if line.strip() and not line.strip().startswith('#'):
                max_nesting = max(max_nesting, indent // 4)
        return max_nesting

    def _count_dependencies(self, code):
        imports = re.findall(r'^import\s+(\w+)|^from\s+(\w+)\s+import', code, re.MULTILINE)
        return len(imports)

    def _count_operators(self, code):
        operators = r'[+\-*/%=<>!&|^~]+'
        return len(re.findall(operators, code))

    def _count_identifiers(self, code):
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        keywords = {'if', 'else', 'elif', 'for', 'while', 'def', 'class',
                    'return', 'import', 'from', 'as', 'try', 'except', 'in',
                    'is', 'not', 'and', 'or', 'True', 'False', 'None'}
        return len([i for i in identifiers if i not in keywords])

    def _calculate_halstead(self, code):
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
        keywords = {'if', 'else', 'elif', 'for', 'while', 'def', 'class',
                    'return', 'import', 'from', 'as', 'try', 'except',
                    'in', 'is', 'not', 'and', 'or'}

        operators = set()
        operands = set()

        for token in tokens:
            if token in keywords or (not token.isalpha() and token.strip()):
                operators.add(token)
            elif token.isalpha() and token not in keywords:
                operands.add(token)
            elif token.isdigit():
                operands.add(token)

        n1 = max(len(operators), 1)
        n2 = max(len(operands), 1)
        N1 = sum(1 for t in tokens if t in operators)
        N2 = sum(1 for t in tokens if t in operands)

        if n1 > 0 and n2 > 0:
            volume = (N1 + N2) * np.log2(n1 + n2)
            difficulty = (n1 / 2) * (N2 / n2)
            effort = difficulty * volume
        else:
            volume = difficulty = effort = 0

        return {'volume': round(volume, 2), 'difficulty': round(difficulty, 2),
                'effort': round(effort, 2)}

    def _calculate_comment_ratio(self, code):
        lines = code.split('\n')
        if not lines:
            return 0
        code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith('#'))
        comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
        total = code_lines + comment_lines
        return comment_lines / total if total > 0 else 0

    def _count_functions(self, code):
        return len(re.findall(r'\bdef\s+(\w+)\s*\(', code))

    def _count_classes(self, code):
        return len(re.findall(r'\bclass\s+(\w+)\s*[:\(]', code))

    def _count_lambdas(self, code):
        return len(re.findall(r'\blambda\s+[\w\s,]*:', code))

    def _get_complexity_level(self, score):
        if score < 25:
            return "Низкая"
        elif score < 45:
            return "Ниже среднего"
        elif score < 60:
            return "Средняя"
        elif score < 75:
            return "Выше среднего"
        else:
            return "Высокая"

class AICodeAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Нейросетевой анализатор сложности кода")
        self.root.geometry("1400x800")

        self.analyzer = NeuralCodeAnalyzer()
        self.analysis_history = []
        self.current_result = None

        self.setup_styles()
        self.create_widgets()
        self.show_test_metrics()

    def setup_styles(self):
        self.bg_color = "#1e1e2e"
        self.fg_color = "#cdd6f4"
        self.primary = "#89b4fa"
        self.secondary = "#f38ba8"
        self.success = "#a6e3a1"
        self.root.configure(bg=self.bg_color)

    def show_test_metrics(self):
        if self.analyzer.test_metrics:
            print("\n" + "=" * 50)
            print("📊 МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ")
            print("=" * 50)
            for model_name, metrics in self.analyzer.test_metrics.items():
                print(f"\n🤖 {model_name.upper()}:")
                print(f"   MAE: {metrics['MAE']}")
                print(f"   R²: {metrics['R2']}")
                print(f"   RMSE: {metrics['RMSE']}")
            print("=" * 50 + "\n")

    def create_widgets(self):
        # Верхняя панель
        top_frame = tk.Frame(self.root, bg=self.bg_color)
        top_frame.pack(fill=tk.X, padx=20, pady=10)

        title = tk.Label(top_frame, text="🧠 Нейросетевой анализ сложности кода",
                         font=("Arial", 18, "bold"), bg=self.bg_color, fg=self.primary)
        title.pack()

        # Панель кнопок
        btn_frame = tk.Frame(top_frame, bg=self.bg_color)
        btn_frame.pack(fill=tk.X, pady=10)

        # Кнопки
        buttons = [
            ("🔍 Анализировать", self.analyze_code, self.primary),
            ("📁 Загрузить файл", self.load_file, self.success),
            ("📋 Вставить код", self.paste_code, "#fab387"),
            ("🗑️ Очистить", self.clear_code, self.secondary),
            ("💾 Сохранить", self.save_results, self.primary),
            ("📊 Метрики модели", self.show_model_stats, self.success),
            ("📜 История", self.show_history, "#fab387")
        ]

        for text, cmd, color in buttons:
            btn = tk.Button(btn_frame, text=text, command=cmd,
                            bg=color, fg="white", font=("Arial", 10, "bold"),
                            padx=15, pady=8, bd=0, cursor="hand2")
            btn.pack(side=tk.LEFT, padx=5)
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#45475a"))
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.configure(bg=c))

        # Основная область
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Левая панель
        left_frame = tk.Frame(main_paned, bg=self.bg_color)
        main_paned.add(left_frame, weight=1)

        tk.Label(left_frame, text="📝 Введите код для анализа:",
                 font=("Arial", 11, "bold"), bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)

        # Текстовое поле с поддержкой вставки
        self.code_text = scrolledtext.ScrolledText(
            left_frame, wrap=tk.NONE,
            font=("Consolas", 11),
            bg="#313244", fg=self.fg_color,
            insertbackground=self.fg_color,
            height=25,
            undo=True
        )
        self.code_text.pack(fill=tk.BOTH, expand=True)

        # Привязываем Ctrl+V для вставки
        self.code_text.bind('<Control-v>', self.paste_event)

        # Пример кода
        self.insert_example()

        # Правая панель
        right_frame = tk.Frame(main_paned, bg=self.bg_color)
        main_paned.add(right_frame, weight=1)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка результатов
        self.ai_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.ai_frame, text="🤖 Результаты ИИ")
        self.create_results_frame()

        # Вкладка метрик
        self.metrics_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.metrics_frame, text="📊 Детальные метрики")
        self.create_metrics_frame()

        # Вкладка графиков
        self.plots_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.plots_frame, text="📈 Визуализация")

        # Вкладка рекомендаций
        self.recommendations_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.recommendations_frame, text="💡 Рекомендации")

        self.recommendations_text = scrolledtext.ScrolledText(
            self.recommendations_frame, wrap=tk.WORD,
            font=("Arial", 11), bg="#313244", fg=self.fg_color
        )
        self.recommendations_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def paste_event(self, event):
        """Обработка Ctrl+V"""
        try:
            text = self.root.clipboard_get()
            self.code_text.insert(tk.INSERT, text)
            return "break"
        except:
            pass

    def paste_code(self):
        """Вставка кода из буфера обмена"""
        try:
            text = self.root.clipboard_get()
            self.code_text.delete(1.0, tk.END)
            self.code_text.insert(1.0, text)
            messagebox.showinfo("Успех", "Код вставлен из буфера обмена")
        except:
            messagebox.showerror("Ошибка", "Не удалось вставить код из буфера обмена")

    def load_file(self):
        """Загрузка кода из файла"""
        filename = filedialog.askopenfilename(
            title="Выберите файл с кодом",
            filetypes=[("Python files", "*.py"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    code = f.read()
                self.code_text.delete(1.0, tk.END)
                self.code_text.insert(1.0, code)
                messagebox.showinfo("Успех", f"Файл {os.path.basename(filename)} загружен")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")

    def clear_code(self):
        """Очистка поля ввода"""
        self.code_text.delete(1.0, tk.END)

    def create_results_frame(self):
        result_frame = tk.Frame(self.ai_frame, bg="#313244")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Оценка сложности
        score_frame = tk.Frame(result_frame, bg="#45475a")
        score_frame.pack(fill=tk.X, pady=10)
        tk.Label(score_frame, text="Оценка сложности (ИИ):",
                 font=("Arial", 12), bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=20, pady=15)
        self.score_label = tk.Label(score_frame, text="—",
                                    font=("Arial", 24, "bold"), bg="#45475a", fg=self.primary)
        self.score_label.pack(side=tk.RIGHT, padx=20, pady=15)

        # Уровень сложности
        level_frame = tk.Frame(result_frame, bg="#45475a")
        level_frame.pack(fill=tk.X, pady=10)
        tk.Label(level_frame, text="Уровень:",
                 font=("Arial", 12), bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=20, pady=15)
        self.level_label = tk.Label(level_frame, text="—",
                                    font=("Arial", 18, "bold"), bg="#45475a", fg=self.secondary)
        self.level_label.pack(side=tk.RIGHT, padx=20, pady=15)

        # Уверенность
        confidence_frame = tk.Frame(result_frame, bg="#45475a")
        confidence_frame.pack(fill=tk.X, pady=10)
        tk.Label(confidence_frame, text="Уверенность модели:",
                 font=("Arial", 12), bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=20, pady=15)
        self.confidence_label = tk.Label(confidence_frame, text="—",
                                         font=("Arial", 18, "bold"), bg="#45475a", fg=self.success)
        self.confidence_label.pack(side=tk.RIGHT, padx=20, pady=15)

        # Прогресс
        self.progress = ttk.Progressbar(result_frame, length=400, mode='determinate')
        self.progress.pack(pady=20)

        # Предсказания моделей
        models_frame = tk.LabelFrame(result_frame, text="Предсказания моделей",
                                     bg="#313244", fg=self.fg_color, font=("Arial", 11, "bold"))
        models_frame.pack(fill=tk.X, pady=10)

        self.model_predictions = {}
        model_names = {'random_forest': '🌲 Random Forest',
                       'gradient_boost': '📈 Gradient Boost',
                       'neural_network': '🧠 Нейросеть'}

        for model_key, model_label in model_names.items():
            frame = tk.Frame(models_frame, bg="#45475a")
            frame.pack(fill=tk.X, padx=10, pady=5)
            tk.Label(frame, text=model_label, font=("Arial", 10),
                     bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=10, pady=5)
            pred_label = tk.Label(frame, text="—", font=("Arial", 10, "bold"),
                                  bg="#45475a", fg=self.primary)
            pred_label.pack(side=tk.RIGHT, padx=10, pady=5)
            self.model_predictions[model_key] = pred_label

    def create_metrics_frame(self):
        canvas = tk.Canvas(self.metrics_frame, bg="#313244", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.metrics_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#313244")

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        tk.Label(scrollable_frame, text="📊 Детальный анализ метрик кода",
                 font=("Arial", 14, "bold"), bg="#313244", fg=self.primary).pack(pady=20)

        self.metric_widgets = {}

        # Базовые метрики
        basic_frame = tk.LabelFrame(scrollable_frame, text="📈 Базовые метрики",
                                    bg="#313244", fg=self.fg_color, font=("Arial", 12, "bold"))
        basic_frame.pack(fill=tk.X, padx=20, pady=10)

        basic_metrics = [
            ("🔄 Цикломатическая сложность", "cyclomatic", "Норма: < 10"),
            ("🧠 Когнитивная сложность", "cognitive", "Норма: < 15"),
            ("📏 Количество строк кода", "lines", "Норма: < 50"),
            ("📦 Макс. глубина вложенности", "nesting", "Норма: < 4"),
            ("🔗 Количество зависимостей", "dependencies", "Норма: < 10"),
            ("⚙️ Количество операторов", "operators", "Индикатор насыщенности"),
            ("🏷️ Количество идентификаторов", "identifiers", "Имена переменных/функций")
        ]

        for i, (label, key, tip) in enumerate(basic_metrics):
            self._create_metric_row(basic_frame, label, key, tip, i)

        # Метрики Холстеда
        halstead_frame = tk.LabelFrame(scrollable_frame, text="📚 Метрики Холстеда",
                                       bg="#313244", fg=self.fg_color, font=("Arial", 12, "bold"))
        halstead_frame.pack(fill=tk.X, padx=20, pady=10)

        halstead_metrics = [
            ("📊 Объем программы", "halstead_volume", "halstead", "Норма: < 1000"),
            ("📈 Сложность", "halstead_difficulty", "halstead", "Норма: < 20"),
            ("⏱️ Трудоемкость", "halstead_effort", "halstead", "Норма: < 5000")
        ]

        for i, (label, key, category, tip) in enumerate(halstead_metrics):
            self._create_metric_row(halstead_frame, label, key, tip, i, category)

        # Дополнительные метрики
        extra_frame = tk.LabelFrame(scrollable_frame, text="📊 Дополнительные метрики",
                                    bg="#313244", fg=self.fg_color, font=("Arial", 12, "bold"))
        extra_frame.pack(fill=tk.X, padx=20, pady=10)

        extra_metrics = [
            ("💬 Соотношение комментариев", "comment_ratio", "Рекомендуется: 10-20%"),
            ("🔧 Количество функций", "function_count", "Функции в коде"),
            ("🏭 Количество классов", "class_count", "Классы в коде"),
            ("λ Количество лямбд", "lambda_count", "Анонимные функции")
        ]

        for i, (label, key, tip) in enumerate(extra_metrics):
            self._create_metric_row(extra_frame, label, key, tip, i)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _create_metric_row(self, parent, label, key, tooltip, row, category=None):
        frame = tk.Frame(parent, bg="#45475a")
        frame.grid(row=row, column=0, padx=10, pady=5, sticky="ew")
        parent.grid_columnconfigure(0, weight=1)

        tk.Label(frame, text=label, font=("Arial", 10),
                 bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=10, pady=8)

        dict_key = f"{category}_{key}" if category else key
        value_label = tk.Label(frame, text="—", font=("Arial", 10, "bold"),
                               bg="#45475a", fg=self.primary)
        value_label.pack(side=tk.RIGHT, padx=10, pady=8)
        self.metric_widgets[dict_key] = value_label

    def insert_example(self):
        example = '''def fibonacci(n):
    """Вычисление числа Фибоначчи"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def quicksort(arr):
    """Быстрая сортировка"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed = []

    def process(self):
        try:
            for item in self.data:
                if isinstance(item, (int, float)):
                    if item > 0:
                        self.processed.append(item * 2)
                    elif item < 0:
                        self.processed.append(abs(item))
                    else:
                        self.processed.append(0)
            return self.processed
        except Exception as e:
            print(f"Error: {e}")
            return None'''

        self.code_text.delete(1.0, tk.END)
        self.code_text.insert(1.0, example)

    def analyze_code(self):
        code = self.code_text.get(1.0, tk.END).strip()
        if not code:
            messagebox.showwarning("Предупреждение", "Введите код для анализа!")
            return

        try:
            self.root.config(cursor="watch")
            self.root.update()

            result = self.analyzer.predict_complexity(code)
            self.current_result = result

            self.update_results(result)
            self.update_metrics_display(result)

            self.analysis_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'code': code[:100] + '...',
                'result': result})

            self.create_visualization(result)
            self.generate_recommendations(result)

            self.notebook.select(0)
            messagebox.showinfo("Успех", "Анализ завершен!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")
        finally:
            self.root.config(cursor="")

    def update_results(self, result):
        self.score_label.config(text=f"{result['score']}/100")
        self.level_label.config(text=result['level'])
        self.confidence_label.config(text=f"{result['confidence']}%")
        self.progress['value'] = result['score']

        for model, pred in result['model_predictions'].items():
            if model in self.model_predictions:
                self.model_predictions[model].config(text=f"{pred:.1f}")

        level_colors = {"Низкая": self.success, "Ниже среднего": "#94e2d5",
                        "Средняя": "#fab387", "Выше среднего": self.secondary,
                        "Высокая": "#f38ba8"}
        self.level_label.config(fg=level_colors.get(result['level'], self.primary))

    def update_metrics_display(self, result):
        metrics = result['metrics']

        for key, widget in self.metric_widgets.items():
            if key.startswith('halstead_'):
                halstead_key = key.replace('halstead_', '')
                if halstead_key in metrics['halstead']:
                    value = metrics['halstead'][halstead_key]
                    widget.config(text=self._format_value(value, halstead_key))

                    # Цветовая индикация
                    if (halstead_key == 'volume' and value > 1000) or \
                            (halstead_key == 'difficulty' and value > 20) or \
                            (halstead_key == 'effort' and value > 5000):
                        widget.config(fg=self.secondary)
                    else:
                        widget.config(fg=self.success)

            elif key in metrics:
                value = metrics[key]
                widget.config(text=self._format_value(value, key))

                # Цветовая индикация
                if (key == 'cyclomatic' and value > 10) or \
                        (key == 'cognitive' and value > 15) or \
                        (key == 'nesting' and value > 4) or \
                        (key == 'lines' and value > 50):
                    widget.config(fg=self.secondary)
                else:
                    widget.config(fg=self.success)

    def _format_value(self, value, key):
        if isinstance(value, float):
            if key == 'comment_ratio':
                return f"{value:.1%}"
            elif key in ['halstead_volume', 'halstead_effort']:
                return f"{value:,.0f}"
            else:
                return f"{value:.1f}"
        return str(value)

    def create_visualization(self, result):
        """Создание визуализации с аналитическими графиками"""
        for widget in self.plots_frame.winfo_children():
            widget.destroy()

        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('#313244')

        # Получаем данные из анализатора
        analyzer = self.analyzer

        # 1. Прогноз vs Факт (Scatter plot)
        ax1 = fig.add_subplot(2, 2, 1)
        if hasattr(analyzer, 'y_test_actual') and hasattr(analyzer, 'y_test_pred_ensemble'):
            ax1.scatter(analyzer.y_test_actual, analyzer.y_test_pred_ensemble,
                        alpha=0.6, color='#89b4fa', s=30)
            ax1.plot([0, 100], [0, 100], '--', color='white', linewidth=1, alpha=0.5)
            ax1.set_xlabel('Фактическое значение', color='white')
            ax1.set_ylabel('Предсказанное значение', color='white')
            ax1.set_title('Прогноз vs Факт (Ансамбль)', color='white')
            ax1.tick_params(colors='white')
            ax1.set_facecolor('#45475a')
            ax1.set_xlim(0, 105)
            ax1.set_ylim(0, 105)

        # 2. Важность признаков
        ax2 = fig.add_subplot(2, 2, 2)
        if hasattr(analyzer, 'feature_importance') and analyzer.feature_importance:
            importance = analyzer.feature_importance
            sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
            names = list(sorted_imp.keys())
            values = list(sorted_imp.values())

            # Переименовываем для читаемости
            name_mapping = {
                'cyclomatic': 'Цикломатическая',
                'cognitive': 'Когнитивная',
                'lines': 'Строки кода',
                'nesting': 'Вложенность',
                'dependencies': 'Зависимости',
                'operators': 'Операторы',
                'identifiers': 'Идентификаторы',
                'halstead_volume': 'Объем (Холстед)',
                'halstead_difficulty': 'Сложность (Холстед)',
                'halstead_effort': 'Трудоемкость (Холстед)',
                'comment_ratio': 'Комментарии',
                'function_count': 'Функции',
                'class_count': 'Классы',
                'lambda_count': 'Лямбды'
            }
            names_display = [name_mapping.get(n, n) for n in names]

            bars = ax2.barh(names_display, values, color='#a6e3a1')
            ax2.set_xlabel('Важность', color='white')
            ax2.set_title('Важность признаков (Feature Importance)', color='white')
            ax2.tick_params(colors='white')
            ax2.set_facecolor('#45475a')
            # Добавляем проценты
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax2.text(val + 0.01, i, f'{val * 100:.1f}%', va='center', color='white', fontsize=9)

        # 3. Кривые обучения
        ax3 = fig.add_subplot(2, 2, 3)
        if hasattr(analyzer, 'X_train') and hasattr(analyzer, 'y_train'):
            try:
                train_sizes, train_scores, test_scores = learning_curve(
                    analyzer.models['random_forest'],
                    np.vstack([analyzer.X_train, analyzer.X_val]),
                    np.concatenate([analyzer.y_train, analyzer.y_val]),
                    cv=5,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='neg_mean_absolute_error',
                    n_jobs=1
                )
                train_mean = -np.mean(train_scores, axis=1)
                test_mean = -np.mean(test_scores, axis=1)

                ax3.plot(train_sizes, train_mean, 'o-', color='#89b4fa', label='Обучающая')
                ax3.plot(train_sizes, test_mean, 'o-', color='#f38ba8', label='Валидационная')
                ax3.set_xlabel('Размер обучающей выборки', color='white')
                ax3.set_ylabel('MAE', color='white')
                ax3.set_title('Кривые обучения (Learning Curves)', color='white')
                ax3.legend(facecolor='#45475a', labelcolor='white')
                ax3.tick_params(colors='white')
                ax3.set_facecolor('#45475a')
            except:
                ax3.text(0.5, 0.5, 'Недостаточно данных\nдля построения кривых обучения',
                         ha='center', va='center', color='white')
                ax3.set_facecolor('#45475a')

        # 4. Анализ остатков
        ax4 = fig.add_subplot(2, 2, 4)
        if hasattr(analyzer, 'y_test_actual') and hasattr(analyzer, 'y_test_pred_ensemble'):
            residuals = analyzer.y_test_actual - analyzer.y_test_pred_ensemble
            ax4.scatter(analyzer.y_test_pred_ensemble, residuals, alpha=0.6, color='#fab387', s=30)
            ax4.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
            ax4.set_xlabel('Предсказанное значение', color='white')
            ax4.set_ylabel('Остатки', color='white')
            ax4.set_title('Анализ остатков (Residuals Plot)', color='white')
            ax4.tick_params(colors='white')
            ax4.set_facecolor('#45475a')
            ax4.axhline(y=np.std(residuals), color='#f38ba8', linestyle=':', alpha=0.5)
            ax4.axhline(y=-np.std(residuals), color='#f38ba8', linestyle=':', alpha=0.5)
            ax4.text(0.95, 0.95, f'Стд. отклонение: {np.std(residuals):.2f}',
                     transform=ax4.transAxes, ha='right', va='top', color='white', fontsize=9)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.plots_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_recommendations(self, result):
        self.recommendations_text.delete(1.0, tk.END)
        metrics = result['metrics']

        self.recommendations_text.insert(tk.END, "🤖 РЕКОМЕНДАЦИИ ОТ ИИ:\n\n", "title")

        if metrics['cyclomatic'] > 10:
            self.recommendations_text.insert(tk.END,
                                             f"⚠️ Цикломатическая сложность ({metrics['cyclomatic']}) > 10\n"
                                             "   → Разбейте сложные условия на функции\n\n", "warning")

        if metrics['cognitive'] > 15:
            self.recommendations_text.insert(tk.END,
                                             f"🧠 Когнитивная сложность ({metrics['cognitive']}) > 15\n"
                                             "   → Упростите логику, уменьшите вложенность\n\n", "warning")

        if metrics['nesting'] > 4:
            self.recommendations_text.insert(tk.END,
                                             f"📦 Глубина вложенности ({metrics['nesting']}) > 4\n"
                                             "   → Используйте ранний возврат\n\n", "warning")

        if metrics['halstead']['effort'] > 5000:
            self.recommendations_text.insert(tk.END,
                                             f"📊 Трудоемкость по Холстеду ({metrics['halstead']['effort']:.0f}) > 5000\n"
                                             "   → Упростите выражения\n\n", "info")

        if metrics['comment_ratio'] < 0.05:
            self.recommendations_text.insert(tk.END,
                                             "💬 Мало комментариев\n   → Добавьте комментарии\n\n", "info")

        if result['score'] > 70:
            self.recommendations_text.insert(tk.END,
                                             "🚨 Высокая сложность! Требуется рефакторинг\n\n", "warning")
        elif result['score'] > 50:
            self.recommendations_text.insert(tk.END,
                                             "📈 Средняя сложность. Есть потенциал для улучшения\n\n", "info")
        else:
            self.recommendations_text.insert(tk.END,
                                             "✅ Низкая сложность. Отличная работа!\n\n", "success")

        self.recommendations_text.tag_config("title", font=("Arial", 14, "bold"), foreground='#89b4fa')
        self.recommendations_text.tag_config("warning", foreground='#f38ba8', spacing1=5)
        self.recommendations_text.tag_config("info", foreground='#fab387', spacing1=5)
        self.recommendations_text.tag_config("success", foreground='#a6e3a1', spacing1=5)

    def show_model_stats(self):
        """Показать статистику моделей с валидационными и тестовыми метриками"""
        if not self.analyzer.test_metrics:
            messagebox.showinfo("Статистика", "Нет данных")
            return

        win = tk.Toplevel(self.root)
        win.title("Метрики моделей")
        win.geometry("750x600")
        win.configure(bg='#1e1e2e')

        text = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Arial", 11),
                                         bg='#313244', fg='#cdd6f4')
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text.insert(tk.END, "📊 МЕТРИКИ МОДЕЛЕЙ\n", "header")
        text.insert(tk.END, "=" * 60 + "\n\n", "separator")

        # --- ВАЛИДАЦИОННАЯ ВЫБОРКА ---
        text.insert(tk.END, "📈 ВАЛИДАЦИОННАЯ ВЫБОРКА (для расчёта весов)\n", "subheader")
        text.insert(tk.END, "-" * 50 + "\n", "separator")
        for name, m in self.analyzer.validation_metrics.items():
            text.insert(tk.END, f"🤖 {name.upper()}\n", "model")
            text.insert(tk.END, f"   MAE: {m['MAE']}\n")
            text.insert(tk.END, f"   R²: {m['R2']}\n")
            text.insert(tk.END, f"   RMSE: {m['RMSE']}\n\n")

        # --- ВЕСА АНСАМБЛЯ ---
        text.insert(tk.END, "⚖️ ВЕСА АНСАМБЛЯ\n", "subheader")
        text.insert(tk.END, "-" * 50 + "\n", "separator")
        if hasattr(self.analyzer, 'ensemble_weights'):
            for name, w in self.analyzer.ensemble_weights.items():
                text.insert(tk.END, f"   {name}: {w}\n")
        text.insert(tk.END, "\n")

        # --- ТЕСТОВАЯ ВЫБОРКА ---
        text.insert(tk.END, "📊 ТЕСТОВАЯ ВЫБОРКА (финальная оценка)\n", "subheader")
        text.insert(tk.END, "-" * 50 + "\n", "separator")
        for name, m in self.analyzer.test_metrics.items():
            if name == 'ensemble':
                text.insert(tk.END, f"🏆 {name.upper()}\n", "best")
            else:
                text.insert(tk.END, f"🤖 {name.upper()}\n", "model")
            text.insert(tk.END, f"   MAE: {m['MAE']}\n")
            text.insert(tk.END, f"   R²: {m['R2']}\n")
            text.insert(tk.END, f"   RMSE: {m['RMSE']}\n\n")

        # --- BASELINE МЕТОДЫ ---
        text.insert(tk.END, "📉 БАЗОВЫЕ МЕТОДЫ (BASELINE)\n", "subheader")
        text.insert(tk.END, "-" * 50 + "\n", "separator")
        if hasattr(self.analyzer, 'baseline_metrics'):
            for name, m in self.analyzer.baseline_metrics.items():
                name_display = {
                    'mean': 'Среднее значение',
                    'linear_loc': 'Линейная регрессия (LOC)',
                    'linear_all': 'Линейная регрессия (все признаки)'
                }.get(name, name)
                text.insert(tk.END, f"   {name_display}: MAE={m['MAE']}, R²={m['R2']}\n")
        text.insert(tk.END, "\n")

        # --- ВАЖНОСТЬ ПРИЗНАКОВ ---
        text.insert(tk.END, "📈 ВАЖНОСТЬ ПРИЗНАКОВ\n", "subheader")
        text.insert(tk.END, "-" * 50 + "\n", "separator")
        if hasattr(self.analyzer, 'feature_importance') and self.analyzer.feature_importance:
            sorted_imp = dict(sorted(self.analyzer.feature_importance.items(),
                                     key=lambda x: x[1], reverse=True))
            for name, val in sorted_imp.items():
                name_display = {
                    'cyclomatic': 'Цикломатическая',
                    'cognitive': 'Когнитивная',
                    'lines': 'Строки кода',
                    'nesting': 'Вложенность',
                    'dependencies': 'Зависимости',
                    'operators': 'Операторы',
                    'identifiers': 'Идентификаторы',
                    'halstead_volume': 'Объем (Холстед)',
                    'halstead_difficulty': 'Сложность (Холстед)',
                    'halstead_effort': 'Трудоемкость (Холстед)',
                    'comment_ratio': 'Комментарии',
                    'function_count': 'Функции',
                    'class_count': 'Классы',
                    'lambda_count': 'Лямбды'
                }.get(name, name)
                text.insert(tk.END, f"   {name_display}: {val * 100:.1f}%\n")

        # --- ИТОГОВЫЙ ВЫВОД ---
        text.insert(tk.END, "\n" + "=" * 60 + "\n", "separator")
        ensemble_mae = self.analyzer.test_metrics.get('ensemble', {}).get('MAE', '—')
        ensemble_r2 = self.analyzer.test_metrics.get('ensemble', {}).get('R2', '—')
        text.insert(tk.END, f"🏆 ИТОГОВАЯ ТОЧНОСТЬ АНСАМБЛЯ: MAE = {ensemble_mae}, R² = {ensemble_r2}\n", "final")

        # Настройка тегов
        text.tag_config("header", font=("Arial", 14, "bold"), foreground='#89b4fa')
        text.tag_config("subheader", font=("Arial", 12, "bold"), foreground='#fab387')
        text.tag_config("model", font=("Arial", 11, "bold"), foreground='#f38ba8')
        text.tag_config("best", font=("Arial", 11, "bold"), foreground='#a6e3a1')
        text.tag_config("final", font=("Arial", 12, "bold"), foreground='#89b4fa')
        text.tag_config("separator", foreground='#45475a')
        text.configure(state=tk.DISABLED)

    def save_results(self):
        if not self.current_result:
            messagebox.showwarning("Предупреждение", "Нет результатов для сохранения")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'result': self.current_result,
                    'code': self.code_text.get(1.0, tk.END).strip()
                }
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("Успех", f"Сохранено в {filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def show_history(self):
        if not self.analysis_history:
            messagebox.showinfo("История", "История пуста")
            return

        win = tk.Toplevel(self.root)
        win.title("История анализов")
        win.geometry("650x400")
        win.configure(bg='#1e1e2e')

        tree = ttk.Treeview(win, columns=('time', 'score', 'level', 'conf'), show='headings')
        tree.heading('time', text='Время')
        tree.heading('score', text='Оценка')
        tree.heading('level', text='Уровень')
        tree.heading('conf', text='Уверенность')
        tree.column('time', width=160)
        tree.column('score', width=80)
        tree.column('level', width=120)
        tree.column('conf', width=100)

        for item in self.analysis_history:
            tree.insert('', tk.END, values=(
                item['timestamp'],
                f"{item['result']['score']}/100",
                item['result']['level'],
                f"{item['result']['confidence']}%"
            ))

        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def view_details():
            sel = tree.selection()
            if sel:
                idx = tree.index(sel[0])
                self.show_history_details(self.analysis_history[idx])

        btn = tk.Button(win, text="Просмотреть детали", command=view_details,
                        bg='#89b4fa', fg='white', font=("Arial", 10, "bold"))
        btn.pack(pady=10)

    def show_history_details(self, item):
        win = tk.Toplevel(self.root)
        win.title("Детали анализа")
        win.geometry("500x400")
        win.configure(bg='#1e1e2e')

        text = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Arial", 10),
                                         bg='#313244', fg='#cdd6f4')
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        result = item['result']
        text.insert(tk.END, f"📊 Анализ от {item['timestamp']}\n\n", "title")
        text.insert(tk.END, f"Оценка: {result['score']}/100\n")
        text.insert(tk.END, f"Уровень: {result['level']}\n")
        text.insert(tk.END, f"Уверенность: {result['confidence']}%\n\n")
        text.insert(tk.END, "Предсказания моделей:\n", "sub")
        for model, pred in result['model_predictions'].items():
            text.insert(tk.END, f"  • {model}: {pred:.1f}\n")

        text.tag_config("title", font=("Arial", 12, "bold"), foreground='#89b4fa')
        text.tag_config("sub", font=("Arial", 11, "bold"), foreground='#f38ba8')
        text.configure(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = AICodeAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
