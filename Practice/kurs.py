
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
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
        self.feature_names = [
            'cyclomatic', 'cognitive', 'lines', 'nesting',
            'dependencies', 'operators', 'identifiers',
            'halstead_volume', 'halstead_difficulty', 'halstead_effort',
            'comment_ratio', 'function_count', 'class_count', 'lambda_count'
        ]

        self._init_models()
        self._load_or_init_model()

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

    def _load_or_init_model(self):
        """Загрузка сохраненной модели или инициализация без обучения"""
        if os.path.exists('code_complexity_model.pkl'):
            try:
                self.load_model('code_complexity_model.pkl')
                self.is_trained = True
                print("✅ Модель загружена из code_complexity_model.pkl")
                return
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
        
        # Если модели нет, работаем в режиме fallback
        print("⚠️ Модель не найдена. Работа в режиме fallback (без машинного обучения)")
        self.is_trained = False
        self.test_metrics = {}

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

        # Ограничиваем значения для стабильности
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

        # Если модель не обучена - используем fallback
        if not self.is_trained:
            return self._fallback_prediction(metrics)

        try:
            features_scaled = self.scaler.transform(features)

            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(features_scaled)[0]
                predictions[name] = max(0, min(100, pred))

            # Взвешенное среднее
            weights = {'random_forest': 0.33, 'gradient_boost': 0.34, 'neural_network': 0.33}
            weighted_pred = sum(predictions[name] * weights[name] for name in predictions)
            weighted_pred = max(0, min(100, weighted_pred))

            # Расчет уверенности
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

        except Exception as e:
            print(f"⚠️ Ошибка при предсказании: {e}")
            return self._fallback_prediction(metrics)

    def _fallback_prediction(self, metrics):
        """Резервный метод предсказания без ML"""
        score = (min(metrics['cyclomatic'] / 12, 1) * 20 +
                 min(metrics['cognitive'] / 20, 1) * 25 +
                 min(metrics['lines'] / 60, 1) * 15 +
                 min(metrics['nesting'] / 5, 1) * 20 +
                 min(metrics['dependencies'] / 8, 1) * 10 +
                 min(metrics['halstead']['effort'] / 5000, 1) * 10)
        score = max(0, min(100, score))
        return {
            'score': round(score, 2),
            'confidence': 60,
            'model_predictions': {'fallback': score},
            'metrics': metrics,
            'level': self._get_complexity_level(score)
        }

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

    def save_model(self, filename):
        joblib.dump({
            'scaler': self.scaler,
            'models': self.models,
            'training_history': self.training_history,
            'test_metrics': self.test_metrics,
            'feature_names': self.feature_names
        }, filename)

    def load_model(self, filename):
        data = joblib.load(filename)
        self.scaler = data['scaler']
        self.models = data['models']
        self.training_history = data.get('training_history', [])
        self.test_metrics = data.get('test_metrics', {})
        self.feature_names = data['feature_names']
        self.is_trained = True


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
        self.show_status()

    def setup_styles(self):
        self.bg_color = "#1e1e2e"
        self.fg_color = "#cdd6f4"
        self.primary = "#89b4fa"
        self.secondary = "#f38ba8"
        self.success = "#a6e3a1"
        self.root.configure(bg=self.bg_color)

    def show_status(self):
        """Отображение статуса модели"""
        status = "✅ Модель загружена" if self.analyzer.is_trained else "⚠️ Режим fallback (без ML)"
        print(f"\n{'='*50}")
        print(f"📊 СТАТУС: {status}")
        if self.analyzer.test_metrics:
            for model_name, metrics in self.analyzer.test_metrics.items():
                print(f"\n🤖 {model_name.upper()}:")
                print(f"   MAE: {metrics['MAE']}")
                print(f"   R²: {metrics['R2']}")
                print(f"   RMSE: {metrics['RMSE']}")
        print("="*50 + "\n")

    def create_widgets(self):
        # Верхняя панель
        top_frame = tk.Frame(self.root, bg=self.bg_color)
        top_frame.pack(fill=tk.X, padx=20, pady=10)

        title = tk.Label(top_frame, text="🧠 Нейросетевой анализ сложности кода",
                         font=("Arial", 18, "bold"), bg=self.bg_color, fg=self.primary)
        title.pack()

        # Статус модели
        status_frame = tk.Frame(top_frame, bg=self.bg_color)
        status_frame.pack(fill=tk.X, pady=5)

        status_text = "✅ Модель обучена" if self.analyzer.is_trained else "⚠️ Режим без ML (fallback)"
        status_color = self.success if self.analyzer.is_trained else "#fab387"
        status_label = tk.Label(status_frame, text=status_text,
                                font=("Arial", 10), bg=self.bg_color, fg=status_color)
        status_label.pack(side=tk.LEFT, padx=5)

        # Панель кнопок
        btn_frame = tk.Frame(top_frame, bg=self.bg_color)
        btn_frame.pack(fill=tk.X, pady=10)

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

        self.code_text = scrolledtext.ScrolledText(
            left_frame, wrap=tk.NONE,
            font=("Consolas", 11),
            bg="#313244", fg=self.fg_color,
            insertbackground=self.fg_color,
            height=25,
            undo=True
        )
        self.code_text.pack(fill=tk.BOTH, expand=True)
        self.code_text.bind('<Control-v>', self.paste_event)

        self.insert_example()

        # Правая панель
        right_frame = tk.Frame(main_paned, bg=self.bg_color)
        main_paned.add(right_frame, weight=1)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.ai_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.ai_frame, text="🤖 Результаты ИИ")
        self.create_results_frame()

        self.metrics_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.metrics_frame, text="📊 Детальные метрики")
        self.create_metrics_frame()

        self.plots_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.plots_frame, text="📈 Визуализация")

        self.recommendations_frame = tk.Frame(self.notebook, bg="#313244")
        self.notebook.add(self.recommendations_frame, text="💡 Рекомендации")

        self.recommendations_text = scrolledtext.ScrolledText(
            self.recommendations_frame, wrap=tk.WORD,
            font=("Arial", 11), bg="#313244", fg=self.fg_color
        )
        self.recommendations_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def paste_event(self, event):
        try:
            text = self.root.clipboard_get()
            self.code_text.insert(tk.INSERT, text)
            return "break"
        except:
            pass

    def paste_code(self):
        try:
            text = self.root.clipboard_get()
            self.code_text.delete(1.0, tk.END)
            self.code_text.insert(1.0, text)
            messagebox.showinfo("Успех", "Код вставлен из буфера обмена")
        except:
            messagebox.showerror("Ошибка", "Не удалось вставить код из буфера обмена")

    def load_file(self):
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
        self.code_text.delete(1.0, tk.END)

    def create_results_frame(self):
        result_frame = tk.Frame(self.ai_frame, bg="#313244")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        score_frame = tk.Frame(result_frame, bg="#45475a")
        score_frame.pack(fill=tk.X, pady=10)
        tk.Label(score_frame, text="Оценка сложности (ИИ):",
                 font=("Arial", 12), bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=20, pady=15)
        self.score_label = tk.Label(score_frame, text="—",
                                    font=("Arial", 24, "bold"), bg="#45475a", fg=self.primary)
        self.score_label.pack(side=tk.RIGHT, padx=20, pady=15)

        level_frame = tk.Frame(result_frame, bg="#45475a")
        level_frame.pack(fill=tk.X, pady=10)
        tk.Label(level_frame, text="Уровень:",
                 font=("Arial", 12), bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=20, pady=15)
        self.level_label = tk.Label(level_frame, text="—",
                                    font=("Arial", 18, "bold"), bg="#45475a", fg=self.secondary)
        self.level_label.pack(side=tk.RIGHT, padx=20, pady=15)

        confidence_frame = tk.Frame(result_frame, bg="#45475a")
        confidence_frame.pack(fill=tk.X, pady=10)
        tk.Label(confidence_frame, text="Уверенность модели:",
                 font=("Arial", 12), bg="#45475a", fg=self.fg_color).pack(side=tk.LEFT, padx=20, pady=15)
        self.confidence_label = tk.Label(confidence_frame, text="—",
                                         font=("Arial", 18, "bold"), bg="#45475a", fg=self.success)
        self.confidence_label.pack(side=tk.RIGHT, padx=20, pady=15)

        self.progress = ttk.Progressbar(result_frame, length=400, mode='determinate')
        self.progress.pack(pady=20)

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
                'result': result
            })

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

                    if (halstead_key == 'volume' and value > 1000) or \
                            (halstead_key == 'difficulty' and value > 20) or \
                            (halstead_key == 'effort' and value > 5000):
                        widget.config(fg=self.secondary)
                    else:
                        widget.config(fg=self.success)

            elif key in metrics:
                value = metrics[key]
                widget.config(text=self._format_value(value, key))

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
        for widget in self.plots_frame.winfo_children():
            widget.destroy()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.patch.set_facecolor('#313244')

        metrics = result['metrics']
        categories = ['Цикломат.', 'Когнит.', 'Строки', 'Вложен.', 'Завис.']
        values = [
            min(metrics['cyclomatic'] / 12, 1) * 100,
            min(metrics['cognitive'] / 20, 1) * 100,
            min(metrics['lines'] / 60, 1) * 100,
            min(metrics['nesting'] / 5, 1) * 100,
            min(metrics['dependencies'] / 8, 1) * 100
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax1 = plt.subplot(221, projection='polar')
        ax1.plot(angles, values, 'o-', linewidth=2, color='#89b4fa')
        ax1.fill(angles, values, alpha=0.25, color='#89b4fa')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories, color='white', size=8)
        ax1.set_ylim(0, 100)
        ax1.set_facecolor('#45475a')
        ax1.set_title('Профиль метрик (нормализовано)', color='white', pad=20)

        ax2.bar(result['model_predictions'].keys(),
                result['model_predictions'].values(),
                color=['#89b4fa', '#a6e3a1', '#f38ba8'])
        ax2.set_ylabel('Оценка (0-100)', color='white')
        ax2.set_title('Предсказания моделей', color='white')
        ax2.tick_params(axis='x', rotation=45, colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.set_ylim(0, 105)
        ax2.set_facecolor('#45475a')

        halstead = metrics['halstead']
        h_metrics = ['Объем', 'Сложность', 'Трудоемкость']
        h_values = [
            min(halstead['volume'] / 1500, 1) * 100,
            min(halstead['difficulty'] / 30, 1) * 100,
            min(halstead['effort'] / 8000, 1) * 100
        ]

        ax3.bar(h_metrics, h_values, color='#a6e3a1')
        ax3.set_title('Метрики Холстеда (нормализовано)', color='white')
        ax3.set_ylabel('Нормализованное значение (%)', color='white')
        ax3.tick_params(colors='white')
        ax3.set_ylim(0, 105)
        ax3.set_facecolor('#45475a')

        labels = ['Уверенность', 'Неопределенность']
        sizes = [result['confidence'], 100 - result['confidence']]
        colors_pie = ['#a6e3a1', '#45475a']

        ax4.pie(sizes, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', textprops={'color': 'white'})
        ax4.set_title('Уверенность предсказания', color='white')

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
        if not self.analyzer.test_metrics:
            messagebox.showinfo("Статистика", "Нет данных о метриках модели")
            return

        win = tk.Toplevel(self.root)
        win.title("Метрики моделей")
        win.geometry("550x450")
        win.configure(bg='#1e1e2e')

        text = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Arial", 11),
                                         bg='#313244', fg='#cdd6f4')
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text.insert(tk.END, "📊 МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ\n\n", "header")

        for name, m in self.analyzer.test_metrics.items():
            text.insert(tk.END, f"🤖 {name.upper()}\n", "model")
            text.insert(tk.END, f"   MAE: {m['MAE']}\n")
            text.insert(tk.END, f"   R²: {m['R2']}\n")
            text.insert(tk.END, f"   RMSE: {m['RMSE']}\n\n")

        best = min(self.analyzer.test_metrics.items(), key=lambda x: x[1]['MAE'])
        text.insert(tk.END, f"🏆 Лучшая: {best[0].upper()} (MAE: {best[1]['MAE']})\n\n", "best")

        text.insert(tk.END, "📖 Пояснение:\n", "expl")
        text.insert(tk.END, "   MAE - средняя ошибка (меньше = лучше)\n")
        text.insert(tk.END, "   R² - качество модели (1 = идеально)\n")
        text.insert(tk.END, "   RMSE - среднеквадратичная ошибка\n")

        text.tag_config("header", font=("Arial", 14, "bold"), foreground='#89b4fa')
        text.tag_config("model", font=("Arial", 12, "bold"), foreground='#f38ba8')
        text.tag_config("best", font=("Arial", 12, "bold"), foreground='#a6e3a1')
        text.tag_config("expl", font=("Arial", 11, "bold"), foreground='#fab387')
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
