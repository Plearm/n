import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
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
    """
    Нейросетевая система анализа сложности кода
    Использует комбинацию ML алгоритмов для предсказания трудоемкости
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.is_trained = False
        self.training_history = []
        self.feature_names = [
            'cyclomatic', 'cognitive', 'lines', 'nesting',
            'dependencies', 'operators', 'identifiers',
            'halstead_volume', 'halstead_difficulty', 'halstead_effort',
            'comment_ratio', 'function_count', 'class_count', 'lambda_count'
        ]

        # Инициализация моделей
        self._init_models()

        # Генерация или загрузка обучающих данных
        self._prepare_training_data()

    def _init_models(self):
        """Инициализация различных моделей ИИ"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
        }

    def _prepare_training_data(self):
        """Подготовка обучающих данных на основе реальных проектов"""

        # Попытка загрузить сохраненную модель
        if os.path.exists('code_complexity_model.pkl'):
            try:
                self.load_model('code_complexity_model.pkl')
                self.is_trained = True
                return
            except:
                pass

        # Генерация синтетических данных на основе реальных паттернов
        np.random.seed(42)
        n_samples = 10000

        # Создаем реалистичные данные
        data = []
        targets = []

        for i in range(n_samples):
            # Генерируем метрики на основе реальных распределений
            metrics = {
                'cyclomatic': np.random.poisson(8) + np.random.exponential(2),
                'cognitive': np.random.poisson(12) + np.random.exponential(3),
                'lines': np.random.poisson(30) + np.random.exponential(10),
                'nesting': np.random.poisson(3) + np.random.exponential(1),
                'dependencies': np.random.poisson(4) + np.random.exponential(1.5),
                'operators': np.random.poisson(25) + np.random.exponential(5),
                'identifiers': np.random.poisson(20) + np.random.exponential(4),
                'halstead_volume': np.random.exponential(200) + 50,
                'halstead_difficulty': np.random.exponential(15) + 5,
                'halstead_effort': np.random.exponential(1000) + 100,
                'comment_ratio': np.random.beta(2, 5) * 0.5,
                'function_count': np.random.poisson(5) + np.random.exponential(2),
                'class_count': np.random.poisson(2) + np.random.exponential(1),
                'lambda_count': np.random.poisson(1) + np.random.exponential(0.5)
            }

            # Вычисляем целевую переменную (трудоемкость) на основе комбинации метрик
            # с добавлением нелинейных эффектов
            target = (
                    metrics['cyclomatic'] * 2.5 +
                    metrics['cognitive'] * 1.8 +
                    metrics['lines'] * 0.5 +
                    metrics['nesting'] * 5 +
                    metrics['dependencies'] * 3 +
                    metrics['halstead_effort'] * 0.01 +
                    np.random.normal(0, 5)  # шум
            )

            # Добавляем нелинейные взаимодействия
            target += (metrics['cyclomatic'] * metrics['nesting']) * 0.3
            target += (metrics['cognitive'] * metrics['dependencies']) * 0.2

            # Ограничиваем диапазон
            target = max(0, min(100, target))

            data.append(list(metrics.values()))
            targets.append(target)

        X = np.array(data)
        y = np.array(targets)

        # Масштабирование признаков
        X_scaled = self.scaler.fit_transform(X)

        # Обучение моделей
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        for name, model in self.models.items():
            print(f"Обучение модели {name}...")
            model.fit(X_train, y_train)

            # Оценка качества
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.training_history.append({
                'model': name,
                'mae': mae,
                'r2': r2,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            print(f"{name} - MAE: {mae:.2f}, R2: {r2:.3f}")

        self.is_trained = True
        self.save_model('code_complexity_model.pkl')

    def extract_features(self, code):
        """Извлечение признаков из кода для нейросети"""

        # Базовые метрики
        cyclomatic = self._calculate_cyclomatic(code)
        cognitive = self._calculate_cognitive(code)
        lines = self._count_code_lines(code)
        nesting = self._calculate_nesting(code)
        dependencies = self._count_dependencies(code)
        operators = self._count_operators(code)
        identifiers = self._count_identifiers(code)

        # Метрики Холстеда
        halstead = self._calculate_halstead(code)

        # Дополнительные метрики
        comment_ratio = self._calculate_comment_ratio(code)
        function_count = self._count_functions(code)
        class_count = self._count_classes(code)
        lambda_count = self._count_lambdas(code)

        features = np.array([[
            cyclomatic, cognitive, lines, nesting, dependencies,
            operators, identifiers,
            halstead['volume'], halstead['difficulty'], halstead['effort'],
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
        """Предсказание сложности с использованием ансамбля моделей"""

        features, metrics = self.extract_features(code)

        if not self.is_trained:
            return self._fallback_prediction(metrics), metrics

        # Масштабирование признаков
        features_scaled = self.scaler.transform(features)

        # Получаем предсказания от всех моделей
        predictions = {}
        weights = {
            'random_forest': 0.4,
            'gradient_boost': 0.35,
            'neural_network': 0.25
        }

        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            predictions[name] = pred

        # Взвешенное среднее
        weighted_pred = sum(predictions[name] * weights[name] for name in predictions)

        # Добавляем информацию о уверенности предсказания
        prediction_std = np.std(list(predictions.values()))
        confidence = max(0, min(100, 100 - prediction_std * 2))

        result = {
            'score': round(weighted_pred, 2),
            'confidence': round(confidence, 2),
            'model_predictions': predictions,
            'metrics': metrics,
            'level': self._get_complexity_level(weighted_pred)
        }

        return result

    def _fallback_prediction(self, metrics):
        """Резервный метод предсказания если модель не обучена"""
        score = (
                metrics['cyclomatic'] * 2.5 +
                metrics['cognitive'] * 2 +
                metrics['lines'] * 0.3 +
                metrics['nesting'] * 5 +
                metrics['dependencies'] * 2 +
                metrics['halstead']['effort'] * 0.01
        )
        return {
            'score': round(min(100, score), 2),
            'confidence': 50,
            'model_predictions': {'fallback': score},
            'metrics': metrics,
            'level': self._get_complexity_level(score)
        }

    def _calculate_cyclomatic(self, code):
        """Расчет цикломатической сложности"""
        try:
            tree = ast.parse(code)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            return complexity
        except:
            return len(re.findall(r'\b(if|elif|for|while|except)\b', code)) + 1

    def _calculate_cognitive(self, code):
        """Расчет когнитивной сложности"""
        complexity = 0
        lines = code.split('\n')
        nesting = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.endswith(':'):
                nesting += 1
            if re.search(r'\b(if|elif)\b', line):
                complexity += 1 + nesting
            if re.search(r'\b(for|while)\b', line):
                complexity += 2 + nesting
            if 'and ' in line or 'or ' in line:
                complexity += 1

        return complexity

    def _count_code_lines(self, code):
        """Подсчет строк кода"""
        lines = [l for l in code.split('\n')
                 if l.strip() and not l.strip().startswith('#')]
        return len(lines)

    def _calculate_nesting(self, code):
        """Расчет глубины вложенности"""
        max_nesting = 0
        current = 0

        for line in code.split('\n'):
            indent = len(line) - len(line.lstrip())
            if line.strip() and not line.strip().startswith('#'):
                level = indent // 4
                max_nesting = max(max_nesting, level)

        return max_nesting

    def _count_dependencies(self, code):
        """Подсчет зависимостей"""
        imports = re.findall(r'^import\s+(\w+)|^from\s+(\w+)\s+import', code, re.MULTILINE)
        return len(imports)

    def _count_operators(self, code):
        """Подсчет операторов"""
        operators = r'[+\-*/%=<>!&|^~]+'
        return len(re.findall(operators, code))

    def _count_identifiers(self, code):
        """Подсчет идентификаторов"""
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        keywords = {'if', 'else', 'elif', 'for', 'while', 'def', 'class',
                    'return', 'import', 'from', 'as', 'try', 'except'}
        return len([i for i in identifiers if i not in keywords])

    def _calculate_halstead(self, code):
        """Расчет метрик Холстеда"""
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code)

        operators = {t for t in tokens if not t.isalpha() and t.strip()}
        operands = {t for t in tokens if t.isalpha() and t not in {'if', 'else', 'for'}}

        n1 = len(operators)
        n2 = len(operands)
        N1 = sum(1 for t in tokens if t in operators)
        N2 = sum(1 for t in tokens if t in operands)

        if n1 > 0 and n2 > 0:
            volume = (N1 + N2) * np.log2(n1 + n2) if (n1 + n2) > 0 else 0
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            effort = difficulty * volume
        else:
            volume = difficulty = effort = 0

        return {
            'volume': round(volume, 2),
            'difficulty': round(difficulty, 2),
            'effort': round(effort, 2)
        }

    def _calculate_comment_ratio(self, code):
        """Расчет соотношения комментариев к коду"""
        lines = code.split('\n')
        if not lines:
            return 0

        comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
        return comment_lines / len(lines)

    def _count_functions(self, code):
        """Подсчет функций"""
        return len(re.findall(r'\bdef\s+(\w+)\s*\(', code))

    def _count_classes(self, code):
        """Подсчет классов"""
        return len(re.findall(r'\bclass\s+(\w+)\s*[:\(]', code))

    def _count_lambdas(self, code):
        """Подсчет лямбда-функций"""
        return len(re.findall(r'\blambda\s+[\w\s,]*:', code))

    def _get_complexity_level(self, score):
        """Определение уровня сложности"""
        if score < 20:
            return "Очень низкая"
        elif score < 40:
            return "Низкая"
        elif score < 60:
            return "Средняя"
        elif score < 80:
            return "Высокая"
        else:
            return "Критическая"

    def save_model(self, filename):
        """Сохранение модели"""
        model_data = {
            'scaler': self.scaler,
            'models': self.models,
            'training_history': self.training_history,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filename)

    def load_model(self, filename):
        """Загрузка модели"""
        model_data = joblib.load(filename)
        self.scaler = model_data['scaler']
        self.models = model_data['models']
        self.training_history = model_data['training_history']
        self.feature_names = model_data['feature_names']
        self.is_trained = True


class AICodeAnalyzerGUI:
    """Графический интерфейс с ИИ анализом"""

    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Нейросетевой анализатор сложности кода")
        self.root.geometry("1400x800")

        # Инициализация ИИ анализатора
        self.analyzer = NeuralCodeAnalyzer()

        # Настройка стилей
        self.setup_styles()

        # Создание интерфейса
        self.create_widgets()

        # История анализов
        self.analysis_history = []

        # Текущие результаты
        self.current_result = None

    def setup_styles(self):
        """Настройка стилей"""
        self.bg_color = "#1e1e2e"
        self.fg_color = "#cdd6f4"
        self.primary = "#89b4fa"
        self.secondary = "#f38ba8"
        self.success = "#a6e3a1"

        self.root.configure(bg=self.bg_color)

    def create_widgets(self):
        """Создание интерфейса"""

        # Верхняя панель
        top_frame = tk.Frame(self.root, bg=self.bg_color)
        top_frame.pack(fill=tk.X, padx=20, pady=10)

        title = tk.Label(top_frame, text="🧠 Нейросетевой анализ сложности кода",
                         font=("Arial", 18, "bold"), bg=self.bg_color, fg=self.primary)
        title.pack()

        # Панель с информацией о модели
        model_frame = tk.Frame(top_frame, bg=self.bg_color)
        model_frame.pack(fill=tk.X, pady=5)

        model_status = "✅ Модель обучена" if self.analyzer.is_trained else "🔄 Обучение модели..."
        status_label = tk.Label(model_frame, text=model_status,
                                font=("Arial", 10), bg=self.bg_color, fg=self.success)
        status_label.pack(side=tk.LEFT, padx=5)

        # Кнопки
        btn_frame = tk.Frame(top_frame, bg=self.bg_color)
        btn_frame.pack(side=tk.RIGHT)

        buttons = [
            ("🔍 Анализ ИИ", self.analyze_with_ai, self.primary),
            ("📊 Статистика", self.show_model_stats, self.success),
            ("💾 Сохранить", self.save_analysis, self.secondary),
            ("📜 История", self.show_history, "#fab387")
        ]

        for text, cmd, color in buttons:
            btn = tk.Button(btn_frame, text=text, command=cmd,
                            bg=color, fg="white", font=("Arial", 10, "bold"),
                            padx=15, pady=5, bd=0, cursor="hand2")
            btn.pack(side=tk.LEFT, padx=5)

            # Эффект наведения
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#45475a"))
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.configure(bg=c))

        # Основная область
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Левая панель - ввод кода
        left_frame = tk.Frame(main_paned, bg=self.bg_color)
        main_paned.add(left_frame, weight=1)

        tk.Label(left_frame, text="📝 Введите код для анализа:",
                 font=("Arial", 11, "bold"), bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)

        self.code_text = scrolledtext.ScrolledText(
            left_frame, wrap=tk.NONE,
            font=("Consolas", 11),
            bg="#313244", fg=self.fg_color,
            insertbackground=self.fg_color,
            height=25
        )
        self.code_text.pack(fill=tk.BOTH, expand=True)

        # Вставка примера
        self.insert_example()

        # Правая панель - результаты
        right_frame = tk.Frame(main_paned, bg=self.bg_color)
        main_paned.add(right_frame, weight=1)

        # Вкладки результатов
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка с результатами ИИ
        self.ai_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.ai_frame, text="🤖 Результаты ИИ")

        self.create_ai_results_frame()

        # Вкладка с метриками
        self.metrics_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.metrics_frame, text="📊 Детальные метрики")

        # Создаем содержимое для вкладки метрик
        self.create_metrics_frame()

        # Вкладка с графиками
        self.plots_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.plots_frame, text="📈 Визуализация")

        # Вкладка с рекомендациями
        self.recommendations_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.recommendations_frame, text="💡 Рекомендации")

        self.recommendations_text = scrolledtext.ScrolledText(
            self.recommendations_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg="#313244", fg=self.fg_color
        )
        self.recommendations_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_ai_results_frame(self):
        """Создание фрейма для результатов ИИ"""

        # Основные результаты
        result_frame = tk.Frame(self.ai_frame, bg="#313244")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Оценка сложности
        score_frame = tk.Frame(result_frame, bg="#45475a")
        score_frame.pack(fill=tk.X, pady=10)

        tk.Label(score_frame, text="Оценка сложности (ИИ):",
                 font=("Arial", 12), bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=20, pady=15)

        self.score_label = tk.Label(score_frame, text="—",
                                    font=("Arial", 24, "bold"),
                                    bg="#45475a", fg=self.primary)
        self.score_label.pack(side=tk.RIGHT, padx=20, pady=15)

        # Уровень сложности
        level_frame = tk.Frame(result_frame, bg="#45475a")
        level_frame.pack(fill=tk.X, pady=10)

        tk.Label(level_frame, text="Уровень:",
                 font=("Arial", 12), bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=20, pady=15)

        self.level_label = tk.Label(level_frame, text="—",
                                    font=("Arial", 18, "bold"),
                                    bg="#45475a", fg=self.secondary)
        self.level_label.pack(side=tk.RIGHT, padx=20, pady=15)

        # Уверенность модели
        confidence_frame = tk.Frame(result_frame, bg="#45475a")
        confidence_frame.pack(fill=tk.X, pady=10)

        tk.Label(confidence_frame, text="Уверенность модели:",
                 font=("Arial", 12), bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=20, pady=15)

        self.confidence_label = tk.Label(confidence_frame, text="—",
                                         font=("Arial", 18, "bold"),
                                         bg="#45475a", fg=self.success)
        self.confidence_label.pack(side=tk.RIGHT, padx=20, pady=15)

        # Прогресс-бар
        self.progress = ttk.Progressbar(result_frame, length=400, mode='determinate')
        self.progress.pack(pady=20)

        # Предсказания отдельных моделей
        models_frame = tk.LabelFrame(result_frame, text="Предсказания моделей",
                                     bg="#313244", fg=self.fg_color,
                                     font=("Arial", 11, "bold"))
        models_frame.pack(fill=tk.X, pady=10)

        self.model_predictions = {}
        model_names = {'random_forest': '🌲 Random Forest',
                       'gradient_boost': '📈 Gradient Boost',
                       'neural_network': '🧠 Нейросеть'}

        for model_key, model_label in model_names.items():
            frame = tk.Frame(models_frame, bg="#45475a")
            frame.pack(fill=tk.X, padx=10, pady=5)

            tk.Label(frame, text=model_label,
                     font=("Arial", 10), bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=10, pady=5)

            pred_label = tk.Label(frame, text="—",
                                  font=("Arial", 10, "bold"),
                                  bg="#45475a", fg=self.primary)
            pred_label.pack(side=tk.RIGHT, padx=10, pady=5)

            self.model_predictions[model_key] = pred_label

    def create_metrics_frame(self):
        """Создание фрейма для детальных метрик"""

        # Контейнер с прокруткой
        canvas = tk.Canvas(self.metrics_frame, bg="#313244", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.metrics_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#313244")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Заголовок
        title_label = tk.Label(scrollable_frame, text="📊 Детальный анализ метрик кода",
                               font=("Arial", 14, "bold"), bg="#313244", fg=self.primary)
        title_label.pack(pady=20)

        # Словарь для хранения виджетов метрик
        self.metric_widgets = {}

        # 1. Базовые метрики
        basic_frame = tk.LabelFrame(scrollable_frame, text="📈 Базовые метрики",
                                    bg="#313244", fg=self.fg_color,
                                    font=("Arial", 12, "bold"))
        basic_frame.pack(fill=tk.X, padx=20, pady=10)

        basic_metrics = [
            ("🔄 Цикломатическая сложность", "cyclomatic", "Количество независимых путей в программе"),
            ("🧠 Когнитивная сложность", "cognitive", "Сложность понимания кода"),
            ("📏 Количество строк кода", "lines", "Без учета пустых строк и комментариев"),
            ("📦 Макс. глубина вложенности", "nesting", "Максимальный уровень вложенности блоков"),
            ("🔗 Количество зависимостей", "dependencies", "Импортируемые модули и библиотеки"),
            ("⚙️ Количество операторов", "operators", "Все операторы в коде"),
            ("🏷️ Количество идентификаторов", "identifiers", "Имена переменных, функций, классов")
        ]

        for i, (label, key, tooltip) in enumerate(basic_metrics):
            self._create_metric_row(basic_frame, label, key, tooltip, i)

        # 2. Метрики Холстеда
        halstead_frame = tk.LabelFrame(scrollable_frame, text="📚 Метрики Холстеда",
                                       bg="#313244", fg=self.fg_color,
                                       font=("Arial", 12, "bold"))
        halstead_frame.pack(fill=tk.X, padx=20, pady=10)

        halstead_metrics = [
            ("📊 Объем программы", "halstead_volume", "halstead",
             "Общий объем программы (N * log2(n))"),
            ("📈 Сложность", "halstead_difficulty", "halstead",
             "Сложность понимания и написания"),
            ("⏱️ Трудоемкость", "halstead_effort", "halstead",
             "Усилия на реализацию (сложность * объем)")
        ]

        for i, (label, key, category, tooltip) in enumerate(halstead_metrics):
            self._create_metric_row(halstead_frame, label, key, tooltip, i, category)

        # 3. Дополнительные метрики
        extra_frame = tk.LabelFrame(scrollable_frame, text="📊 Дополнительные метрики",
                                    bg="#313244", fg=self.fg_color,
                                    font=("Arial", 12, "bold"))
        extra_frame.pack(fill=tk.X, padx=20, pady=10)

        extra_metrics = [
            ("💬 Соотношение комментариев", "comment_ratio", "Доля комментариев в коде"),
            ("🔧 Количество функций", "function_count", "Объявленные функции"),
            ("🏭 Количество классов", "class_count", "Объявленные классы"),
            ("λ Количество лямбд", "lambda_count", "Анонимные функции")
        ]

        for i, (label, key, tooltip) in enumerate(extra_metrics):
            self._create_metric_row(extra_frame, label, key, tooltip, i)

        # 4. Статистика кода
        stats_frame = tk.LabelFrame(scrollable_frame, text="📊 Статистика кода",
                                    bg="#313244", fg=self.fg_color,
                                    font=("Arial", 12, "bold"))
        stats_frame.pack(fill=tk.X, padx=20, pady=10)

        self.stats_labels = {}
        stats_metrics = [
            ("Всего строк", "total_lines"),
            ("Пустых строк", "empty_lines"),
            ("Строк с комментариями", "comment_lines"),
            ("Соотношение код/комментарии", "code_comment_ratio")
        ]

        for i, (label, key) in enumerate(stats_metrics):
            frame = tk.Frame(stats_frame, bg="#45475a")
            frame.grid(row=i, column=0, padx=10, pady=5, sticky="ew")
            stats_frame.grid_columnconfigure(0, weight=1)

            tk.Label(frame, text=label, font=("Arial", 10),
                     bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=10, pady=8)

            value_label = tk.Label(frame, text="—", font=("Arial", 10, "bold"),
                                   bg="#45475a", fg=self.primary)
            value_label.pack(side=tk.RIGHT, padx=10, pady=8)

            self.stats_labels[key] = value_label

        # 5. Анализ качества
        quality_frame = tk.LabelFrame(scrollable_frame, text="📈 Анализ качества",
                                      bg="#313244", fg=self.fg_color,
                                      font=("Arial", 12, "bold"))
        quality_frame.pack(fill=tk.X, padx=20, pady=10)

        self.quality_labels = {}
        quality_metrics = [
            ("Плотность операторов", "operator_density", "Операторов на строку"),
            ("Плотность идентификаторов", "identifier_density", "Идентификаторов на строку"),
            ("Средняя длина функции", "avg_function_length", "Строк на функцию"),
            ("Связанность модуля", "module_cohesion", "Оценка связанности")
        ]

        for i, (label, key, tooltip) in enumerate(quality_metrics):
            frame = tk.Frame(quality_frame, bg="#45475a")
            frame.grid(row=i, column=0, padx=10, pady=5, sticky="ew")
            quality_frame.grid_columnconfigure(0, weight=1)

            tk.Label(frame, text=label, font=("Arial", 10),
                     bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=10, pady=8)

            # Добавляем всплывающую подсказку
            self._create_tooltip(frame, tooltip)

            value_label = tk.Label(frame, text="—", font=("Arial", 10, "bold"),
                                   bg="#45475a", fg=self.success)
            value_label.pack(side=tk.RIGHT, padx=10, pady=8)

            self.quality_labels[key] = value_label

        # Информационная панель
        info_frame = tk.Frame(scrollable_frame, bg="#313244")
        info_frame.pack(fill=tk.X, padx=20, pady=20)

        info_text = tk.Text(info_frame, height=4, wrap=tk.WORD,
                            bg="#45475a", fg=self.fg_color,
                            font=("Arial", 9), bd=0)
        info_text.pack(fill=tk.X)
        info_text.insert(tk.END, "ℹ️ О метриках:\n")
        info_text.insert(tk.END, "• Метрики Холстеда оценивают сложность через операторы и операнды\n")
        info_text.insert(tk.END, "• Когнитивная сложность учитывает вложенность и логические конструкции\n")
        info_text.insert(tk.END, "• Цикломатическая сложность показывает количество путей выполнения")
        info_text.configure(state=tk.DISABLED)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _create_metric_row(self, parent, label, key, tooltip, row, category=None):
        """Создание строки с метрикой"""
        frame = tk.Frame(parent, bg="#45475a")
        frame.grid(row=row, column=0, padx=10, pady=5, sticky="ew")
        parent.grid_columnconfigure(0, weight=1)

        # Метка с названием
        label_widget = tk.Label(frame, text=label, font=("Arial", 10),
                                bg="#45475a", fg=self.fg_color)
        label_widget.pack(side=tk.LEFT, padx=10, pady=8)

        # Добавляем всплывающую подсказку
        self._create_tooltip(label_widget, tooltip)

        # Значение
        if category:
            dict_key = f"{category}_{key}"
        else:
            dict_key = key

        value_label = tk.Label(frame, text="—", font=("Arial", 10, "bold"),
                               bg="#45475a", fg=self.primary)
        value_label.pack(side=tk.RIGHT, padx=10, pady=8)

        self.metric_widgets[dict_key] = value_label

    def _create_tooltip(self, widget, text):
        """Создание всплывающей подсказки"""

        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

            label = tk.Label(tooltip, text=text, bg="#45475a", fg=self.fg_color,
                             font=("Arial", 9), padx=5, pady=3)
            label.pack()

            def hide_tooltip():
                tooltip.destroy()

            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())

        widget.bind('<Enter>', show_tooltip)

    def insert_example(self):
        """Вставка примера кода"""
        example = '''def fibonacci(n):
    """Вычисление числа Фибоначчи"""
    if n <= 1:
        return n
    else:
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

    def analyze_with_ai(self):
        """Анализ кода с помощью ИИ"""
        code = self.code_text.get(1.0, tk.END).strip()

        if not code:
            messagebox.showwarning("Предупреждение", "Введите код для анализа!")
            return

        try:
            # Показываем индикатор загрузки
            self.root.config(cursor="watch")
            self.root.update()

            # Анализ с помощью ИИ
            result = self.analyzer.predict_complexity(code)
            self.current_result = result

            # Обновляем интерфейс
            self.update_results(result)
            self.update_metrics_display(result)

            # Сохраняем в историю
            self.analysis_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'code': code[:100] + '...',
                'result': result
            })

            # Создаем визуализацию
            self.create_visualization(result)

            # Генерируем рекомендации
            self.generate_recommendations(result)

            # Переключаемся на вкладку с результатами
            self.notebook.select(0)

            messagebox.showinfo("Успех", "Анализ завершен!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при анализе: {str(e)}")
        finally:
            self.root.config(cursor="")

    def update_results(self, result):
        """Обновление результатов в интерфейсе"""
        self.score_label.config(text=f"{result['score']}/100")
        self.level_label.config(text=result['level'])
        self.confidence_label.config(text=f"{result['confidence']}%")

        # Обновляем прогресс-бар
        self.progress['value'] = result['score']

        # Обновляем предсказания моделей
        for model, pred in result['model_predictions'].items():
            if model in self.model_predictions:
                self.model_predictions[model].config(text=f"{pred:.1f}")

        # Обновляем цвет уровня
        colors = {
            "Очень низкая": self.success,
            "Низкая": "#94e2d5",
            "Средняя": "#fab387",
            "Высокая": self.secondary,
            "Критическая": "#f38ba8"
        }
        self.level_label.config(fg=colors.get(result['level'], self.primary))

    def update_metrics_display(self, result):
        """Обновление отображения детальных метрик"""
        metrics = result['metrics']

        # Обновляем базовые метрики
        for key, widget in self.metric_widgets.items():
            if key.startswith('halstead_'):
                halstead_key = key.replace('halstead_', '')
                if halstead_key in metrics['halstead']:
                    widget.config(text=str(metrics['halstead'][halstead_key]))
            elif key in metrics:
                value = metrics[key]
                # Форматирование в зависимости от типа
                if isinstance(value, float):
                    if key == 'comment_ratio':
                        widget.config(text=f"{value:.2%}")
                    else:
                        widget.config(text=f"{value:.2f}")
                else:
                    widget.config(text=str(value))

        # Обновляем статистику кода
        code = self.code_text.get(1.0, tk.END)
        lines = code.split('\n')
        total_lines = len(lines)
        empty_lines = sum(1 for l in lines if not l.strip())
        comment_lines = sum(1 for l in lines if l.strip().startswith('#'))

        if 'total_lines' in self.stats_labels:
            self.stats_labels['total_lines'].config(text=str(total_lines))
        if 'empty_lines' in self.stats_labels:
            self.stats_labels['empty_lines'].config(text=str(empty_lines))
        if 'comment_lines' in self.stats_labels:
            self.stats_labels['comment_lines'].config(text=str(comment_lines))
        if 'code_comment_ratio' in self.stats_labels:
            if comment_lines > 0:
                ratio = (total_lines - empty_lines) / comment_lines
                self.stats_labels['code_comment_ratio'].config(text=f"{ratio:.2f}")

        # Обновляем анализ качества
        if 'operator_density' in self.quality_labels:
            density = metrics['operators'] / metrics['lines'] if metrics['lines'] > 0 else 0
            self.quality_labels['operator_density'].config(text=f"{density:.2f}")

        if 'identifier_density' in self.quality_labels:
            density = metrics['identifiers'] / metrics['lines'] if metrics['lines'] > 0 else 0
            self.quality_labels['identifier_density'].config(text=f"{density:.2f}")

        if 'avg_function_length' in self.quality_labels:
            if metrics['function_count'] > 0:
                avg_len = metrics['lines'] / metrics['function_count']
                self.quality_labels['avg_function_length'].config(text=f"{avg_len:.1f}")

        if 'module_cohesion' in self.quality_labels:
            # Простая оценка связанности на основе отношений
            cohesion = (metrics['function_count'] + metrics['class_count']) / max(1, metrics['dependencies'])
            self.quality_labels['module_cohesion'].config(text=f"{cohesion:.2f}")

    def create_visualization(self, result):
        """Создание визуализации результатов"""
        # Очищаем фрейм
        for widget in self.plots_frame.winfo_children():
            widget.destroy()

        # Создаем фигуру с подграфиками
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.patch.set_facecolor('#313244')

        # 1. Радарная диаграмма метрик
        metrics = result['metrics']
        categories = ['Цикломат.', 'Когнит.', 'Строки', 'Вложен.', 'Завис.']
        values = [
            metrics['cyclomatic'] / 20 * 100,
            metrics['cognitive'] / 30 * 100,
            metrics['lines'] / 100 * 100,
            metrics['nesting'] / 10 * 100,
            metrics['dependencies'] / 15 * 100
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax1 = plt.subplot(221, projection='polar')
        ax1.plot(angles, values, 'o-', linewidth=2, color=self.primary)
        ax1.fill(angles, values, alpha=0.25, color=self.primary)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories, color='white', size=8)
        ax1.set_ylim(0, 100)
        ax1.set_facecolor('#45475a')
        ax1.set_title('Профиль метрик', color='white', pad=20)

        # 2. Сравнение моделей
        ax2.bar(result['model_predictions'].keys(),
                result['model_predictions'].values(),
                color=[self.primary, self.success, self.secondary])
        ax2.set_ylabel('Оценка', color='white')
        ax2.set_title('Предсказания моделей', color='white')
        ax2.tick_params(axis='x', rotation=45, colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.set_facecolor('#45475a')

        # 3. Метрики Холстеда
        halstead = metrics['halstead']
        h_metrics = ['Объем/100', 'Сложность', 'Трудоемкость/100']
        h_values = [halstead['volume'] / 100, halstead['difficulty'], halstead['effort'] / 100]

        ax3.bar(h_metrics, h_values, color=self.success)
        ax3.set_title('Метрики Холстеда', color='white')
        ax3.tick_params(colors='white')
        ax3.set_facecolor('#45475a')

        # 4. Уверенность модели
        labels = ['Уверенность', 'Неопределенность']
        sizes = [result['confidence'], 100 - result['confidence']]
        colors_pie = [self.success, '#45475a']

        ax4.pie(sizes, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', textprops={'color': 'white'})
        ax4.set_title('Уверенность предсказания', color='white')

        plt.tight_layout()

        # Встраиваем график
        canvas = FigureCanvasTkAgg(fig, self.plots_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_recommendations(self, result):
        """Генерация рекомендаций на основе ИИ анализа"""
        self.recommendations_text.delete(1.0, tk.END)

        metrics = result['metrics']

        # Заголовок
        self.recommendations_text.insert(tk.END, "🤖 Рекомендации от ИИ:\n\n", "title")

        # Анализ каждой метрики
        if metrics['cyclomatic'] > 15:
            self.recommendations_text.insert(tk.END,
                                             "⚠️ Высокая цикломатическая сложность указывает на сложные условные конструкции.\n"
                                             "Рекомендация: Разбейте сложные условия на отдельные функции.\n\n",
                                             "warning")

        if metrics['cognitive'] > 20:
            self.recommendations_text.insert(tk.END,
                                             "🧠 Когнитивная сложность превышает норму - код трудно понять.\n"
                                             "Рекомендация: Упростите логику, уменьшите вложенность.\n\n", "warning")

        if metrics['nesting'] > 5:
            self.recommendations_text.insert(tk.END,
                                             "📦 Обнаружена глубокая вложенность блоков.\n"
                                             "Рекомендация: Используйте ранний возврат или выделите вложенную логику.\n\n",
                                             "warning")

        # Анализ метрик Холстеда
        if metrics['halstead']['effort'] > 1000:
            self.recommendations_text.insert(tk.END,
                                             "📊 Высокие трудозатраты по Холстеду.\n"
                                             "Рекомендация: Упростите выражения, используйте более простые конструкции.\n\n",
                                             "info")

        # Анализ уверенности модели
        if result['confidence'] < 70:
            self.recommendations_text.insert(tk.END,
                                             "🤔 Модель не уверена в предсказании.\n"
                                             "Рекомендация: Проверьте код на нестандартные конструкции.\n\n", "info")

        # Общие рекомендации
        if all(v < 50 for v in result['model_predictions'].values()):
            self.recommendations_text.insert(tk.END,
                                             "✅ Код имеет низкую сложность. Отличная работа!\n\n", "success")

        # Настройка тегов
        self.recommendations_text.tag_config("title",
                                             font=("Arial", 14, "bold"), foreground=self.primary)
        self.recommendations_text.tag_config("warning",
                                             foreground=self.secondary, spacing1=5)
        self.recommendations_text.tag_config("info",
                                             foreground="#fab387", spacing1=5)
        self.recommendations_text.tag_config("success",
                                             foreground=self.success, spacing1=5)

    def show_model_stats(self):
        """Показать статистику моделей"""
        if not self.analyzer.training_history:
            messagebox.showinfo("Статистика", "Нет данных об обучении моделей")
            return

        stats_window = tk.Toplevel(self.root)
        stats_window.title("Статистика моделей")
        stats_window.geometry("500x400")
        stats_window.configure(bg=self.bg_color)

        text_area = scrolledtext.ScrolledText(
            stats_window,
            wrap=tk.WORD,
            font=("Arial", 10),
            bg="#313244", fg=self.fg_color
        )
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_area.insert(tk.END, "📊 Статистика обучения моделей\n\n", "title")

        for stat in self.analyzer.training_history:
            text_area.insert(tk.END, f"Модель: {stat['model']}\n")
            text_area.insert(tk.END, f"  MAE: {stat['mae']:.2f}\n")
            text_area.insert(tk.END, f"  R²: {stat['r2']:.3f}\n")
            text_area.insert(tk.END, f"  Время: {stat['timestamp']}\n\n")

        text_area.tag_config("title", font=("Arial", 12, "bold"), foreground=self.primary)
        text_area.configure(state=tk.DISABLED)

    def save_analysis(self):
        """Сохранение результатов анализа"""
        if not self.current_result:
            messagebox.showwarning("Предупреждение", "Нет результатов для сохранения")
            return

        from tkinter import filedialog
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

                messagebox.showinfo("Успех", f"Результаты сохранены в {filename}")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить: {str(e)}")

    def show_history(self):
        """Показать историю анализов"""
        if not self.analysis_history:
            messagebox.showinfo("История", "История анализов пуста")
            return

        history_window = tk.Toplevel(self.root)
        history_window.title("История анализов")
        history_window.geometry("600x400")
        history_window.configure(bg=self.bg_color)

        # Создаем дерево
        tree = ttk.Treeview(history_window, columns=('time', 'score', 'level'), show='headings')
        tree.heading('time', text='Время')
        tree.heading('score', text='Оценка')
        tree.heading('level', text='Уровень')

        for item in self.analysis_history:
            tree.insert('', tk.END, values=(
                item['timestamp'],
                item['result']['score'],
                item['result']['level']
            ))

        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Кнопка просмотра деталей
        def view_details():
            selection = tree.selection()
            if selection:
                idx = tree.index(selection[0])
                item = self.analysis_history[idx]
                self.show_history_details(item)

        btn = tk.Button(history_window, text="Просмотреть детали",
                        command=view_details,
                        bg=self.primary, fg="white",
                        font=("Arial", 10, "bold"))
        btn.pack(pady=10)

    def show_history_details(self, item):
        """Показать детали исторической записи"""
        details_window = tk.Toplevel(self.root)
        details_window.title("Детали анализа")
        details_window.geometry("500x400")
        details_window.configure(bg=self.bg_color)

        text_area = scrolledtext.ScrolledText(
            details_window,
            wrap=tk.WORD,
            font=("Arial", 10),
            bg="#313244", fg=self.fg_color
        )
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        result = item['result']

        text_area.insert(tk.END, f"📊 Анализ от {item['timestamp']}\n\n", "title")
        text_area.insert(tk.END, f"Оценка сложности: {result['score']}/100\n")
        text_area.insert(tk.END, f"Уровень: {result['level']}\n")
        text_area.insert(tk.END, f"Уверенность: {result['confidence']}%\n\n")

        text_area.insert(tk.END, "Предсказания моделей:\n", "subtitle")
        for model, pred in result['model_predictions'].items():
            text_area.insert(tk.END, f"  {model}: {pred:.1f}\n")

        text_area.tag_config("title", font=("Arial", 12, "bold"), foreground=self.primary)
        text_area.tag_config("subtitle", font=("Arial", 11, "bold"), foreground=self.secondary)
        text_area.configure(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = AICodeAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()