import sys
import os
import tempfile
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFormLayout, QSpinBox, QTableWidget, QTableWidgetItem,
                             QDialog, QScrollArea, QCheckBox, QComboBox, QHeaderView,
                             QTabWidget, QMessageBox, QGroupBox, QProgressDialog,
                             QFileDialog, QSizePolicy)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import math

class SolutionTableDialog(QDialog):
    def __init__(self, solution_data, control_solution, time_points, x_values, parent=None):
                # инициализация диалогового окна с основной сеткой, контрольной сеткой и отклонениями
        super().__init__(parent)
        self.setWindowTitle("Таблица решений и отклонений")
        self.setMinimumSize(1200, 800)
        
        self.tabs = QTabWidget()
        
        self.main_table = QTableWidget()
        self.main_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.main_table.verticalHeader().setDefaultSectionSize(25)
        self.main_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.control_table = QTableWidget()
        self.control_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.control_table.verticalHeader().setDefaultSectionSize(25)
        self.control_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.diff_table = QTableWidget()
        self.diff_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.diff_table.verticalHeader().setDefaultSectionSize(25)
        self.diff_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # заполнение таблиц данными решений с форматированием чисел 
        self.fill_table(self.main_table, solution_data, time_points, x_values, "Основная сетка")
        self.fill_table(self.control_table, control_solution, time_points, x_values, "Контрольная сетка")

        # заполнение таблиц отклонений, подсвечивание максимальных отклонений желтым цветом
        self.fill_diff_table(self.diff_table, solution_data, control_solution, time_points, x_values)

        self.tabs.addTab(self.main_table, "Основная сетка")
        self.tabs.addTab(self.control_table, "Контрольная сетка")
        self.tabs.addTab(self.diff_table, "Отклонения")

        # экспорт данных в таблицу формата .csv, кнопка 
        self.export_btn = QPushButton("Экспорт в CSV")
        self.export_btn.clicked.connect(self.export_to_csv)
        
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(self.export_btn)
        self.setLayout(layout)

        self.resize(QApplication.primaryScreen().availableSize() * 0.9)

    # функция заполнения таблиц данными решения задачи конвекции-диффузии
    def fill_table(self, table, solution_data, time_points, x_values, title):
        num_layers = len(solution_data)
        num_x = len(x_values)
        table.setRowCount(num_layers)
        table.setColumnCount(num_x + 1)

        table.setHorizontalHeaderItem(0, QTableWidgetItem("t"))
        for col in range(num_x):
            table.setHorizontalHeaderItem(col+1, QTableWidgetItem(f"x={x_values[col]:.4f}"))

        for row in range(num_layers):
            time_item = QTableWidgetItem()
            time_item.setData(Qt.DisplayRole, f"{time_points[row]:.6g}")
            table.setItem(row, 0, time_item)
            
            for col in range(num_x):
                val = solution_data[row][col]
                item = QTableWidgetItem()
                
                if abs(val) < 1e-4 and abs(val) > 1e-10:
                    item.setData(Qt.DisplayRole, f"{val:.4e}")
                elif abs(val) < 1e-10:
                    item.setData(Qt.DisplayRole, "0")
                else:
                    item.setData(Qt.DisplayRole, f"{val:.6f}")
                
                table.setItem(row, col+1, item)
    
    # заполнение таблицы отклонений между основной и контрольной сеткой
    def fill_diff_table(self, table, main_data, control_data, time_points, x_values):
        num_layers = len(main_data)
        num_x = len(x_values)
        table.setRowCount(num_layers)
        table.setColumnCount(num_x + 2)

        table.setHorizontalHeaderItem(0, QTableWidgetItem("t"))
        table.setHorizontalHeaderItem(1, QTableWidgetItem("max|u2 - u|"))
        for col in range(num_x):
            table.setHorizontalHeaderItem(col+2, QTableWidgetItem(f"x={x_values[col]:.4f}"))

        for row in range(num_layers):
            time_item = QTableWidgetItem()
            time_item.setData(Qt.DisplayRole, f"{time_points[row]:.6g}")
            table.setItem(row, 0, time_item)

            diff = np.abs(control_data[row] - main_data[row])
            max_diff = np.max(diff)
            max_idx = np.argmax(diff)
            
            max_item = QTableWidgetItem()
            max_item.setData(Qt.DisplayRole, f"{max_diff:.6f}")
            table.setItem(row, 1, max_item)
            
            for col in range(num_x):
                val = diff[col]
                item = QTableWidgetItem()

                if abs(val) < 1e-4 and abs(val) > 1e-10:
                    item.setData(Qt.DisplayRole, f"{val:.4e}")
                elif abs(val) < 1e-10:
                    item.setData(Qt.DisplayRole, "0")
                else:
                    item.setData(Qt.DisplayRole, f"{val:.6f}")

                if col == max_idx:
                    item.setBackground(Qt.yellow)
                
                table.setItem(row, col+2, item)
    
    "функция, позволяющая портировать все три таблицы данных в файл формата .csv"
    def export_to_csv(self):
        try:
            options = QFileDialog.Options()
            base_name, _ = QFileDialog.getSaveFileName(self, "Сохранить базовый CSV", "", "CSV Files (*.csv)", options=options)
            if not base_name:
                return
            self.save_table_to_csv(self.main_table, base_name + "_main.csv")
            self.save_table_to_csv(self.control_table, base_name + "_control.csv")
            self.save_table_to_csv(self.diff_table, base_name + "_diff.csv")

            QMessageBox.information(self, "Успех", "Таблицы успешно экспортированы в CSV файлы!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при экспорте: {str(e)}")
    
    "сохранение отдельной таблицы в файл с форматом .csv"
    def save_table_to_csv(self, table, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            headers = []
            for col in range(table.columnCount()):
                headers.append(table.horizontalHeaderItem(col).text())
            f.write(",".join(headers) + "\n")

            for row in range(table.rowCount()):
                row_data = []
                for col in range(table.columnCount()):
                    item = table.item(row, col)
                    if item:
                        row_data.append(item.text().replace(',', '.'))
                    else:
                        row_data.append("")
                f.write(",".join(row_data) + "\n")

"диалоговое окно для сравнения характеристик основной и контрольной сетки"
class ComparisonTableDialog(QDialog):

    "инициализация таблицы сравнения"
    def __init__(self, main_solution, control_solution, time_points, x_values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Таблица сравнения сеток")
        self.setMinimumSize(1000, 800)
        
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(25)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        num_layers = len(main_solution)
        num_x = len(x_values)
        self.table.setRowCount(num_layers)
        self.table.setColumnCount(7)
        
        # заголовки
        headers = [
            "Слой", "t", "max|u2 - u|", 
            "Узел i max", "Среднее отклонение",
            "Основная сетка (n x m)", "Контрольная сетка (n2 x m2)"
        ]
        for col, header in enumerate(headers):
            self.table.setHorizontalHeaderItem(col, QTableWidgetItem(header))
        
        # заполнение данных
        n_main = len(x_values)
        n_control = len(control_solution[0]) if control_solution else 0
        
        for row in range(num_layers):
            # основные данные слоя
            self.table.setItem(row, 0, QTableWidgetItem(f"{row}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{time_points[row]:.4f}"))
            
            # расчет отклонений
            diff = np.abs(control_solution[row] - main_solution[row])
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            max_idx = np.argmax(diff)
            
            self.table.setItem(row, 2, QTableWidgetItem(f"{max_diff:.6f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{max_idx}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{mean_diff:.6f}"))
            
            # размеры сеток
            m_main = len(main_solution)
            m_control = len(control_solution) if control_solution else 0
            self.table.setItem(row, 5, QTableWidgetItem(f"{n_main} x {m_main}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{n_control} x {m_control}"))
        
        # экспорт
        self.export_btn = QPushButton("Экспорт в CSV")
        self.export_btn.clicked.connect(self.export_to_csv)
        
        # макет
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.export_btn)
        self.setLayout(layout)
        
        # размеры
        self.resize(QApplication.primaryScreen().availableSize() * 0.8)
    
    "экспорт таблицы в файл формата .csv"
    def export_to_csv(self):

        try:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(self, "Сохранить таблицу сравнения", "", "CSV Files (*.csv)", options=options)
            if not filename:
                return
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Заголовки
                headers = []
                for col in range(self.table.columnCount()):
                    headers.append(self.table.horizontalHeaderItem(col).text())
                f.write(",".join(headers) + "\n")
                
                # Данные
                for row in range(self.table.rowCount()):
                    row_data = []
                    for col in range(self.table.columnCount()):
                        item = self.table.item(row, col)
                        if item:
                            row_data.append(item.text().replace(',', '.'))
                        else:
                            row_data.append("")
                    f.write(",".join(row_data) + "\n")
            
            QMessageBox.information(self, "Успех", "Таблица успешно экспортирована в CSV файл!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при экспорте: {str(e)}")

# главное окно приложения для решения уравнения конвекции-диффузии с визуалом
class BurgersEquationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.solution_history = []  # основная сетка
        self.control_solution_history = []  # контрольная сетка (2x)
        self.time_points = []
        self.x_values = []
        self.current_params = {}
        self.need_recalculate = True
        self.cmap = 'viridis'  # цветовая гамма
        self.global_max_diff = 0.0  # Максимальное отклонение по всей сетке
        self.y_min, self.y_max = -1.5, 1.5
        self.initUI()
    
    # инициализация 
    def initUI(self):
        self.setWindowTitle("Моделирование уравнения Бюргерса с источником")
        self.setGeometry(100, 100, 1600, 900)
        
        self.heatmap_fig = Figure(figsize=(10, 6))
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        
        self.layer_fig = Figure(figsize=(7, 5))
        self.layer_canvas = FigureCanvas(self.layer_fig)

        graphics_container = QWidget()
        graphics_layout = QHBoxLayout()
        graphics_layout.addWidget(self.heatmap_canvas, 3)
        graphics_layout.addWidget(self.layer_canvas, 2)
        graphics_container.setLayout(graphics_layout)

        control_container = QWidget()
        control_layout = QVBoxLayout()

        self.T_input = QLineEdit("2.0")
        self.Nx_input = QLineEdit("200")
        self.Nt_input = QLineEdit("5000")
        self.viscosity_input = QLineEdit("0.01")  # Вязкость
        self.delta_input = QLineEdit("0.1")      # Параметр δ для массового оператора
        self.layer_input = QSpinBox()
        self.time_input = QLineEdit("0.0")       # Поле для ввода времени
        self.save_every_input = QLineEdit("100")
        self.source_amp_input = QLineEdit("9.0")  # Амплитуда источника
        self.source_freq_input = QLineEdit("5.0") # Частота источника

        self.form_layout = QFormLayout()
        self.form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.form_layout.addRow(QLabel("Шаг сохр:"), self.save_every_input)
        self.form_layout.addRow(QLabel("Время (T):"), self.T_input)
        self.form_layout.addRow(QLabel("Узлы (Nx):"), self.Nx_input)
        self.form_layout.addRow(QLabel("Шаги (Nt):"), self.Nt_input)
        self.form_layout.addRow(QLabel("Вязкость (v):"), self.viscosity_input)
        self.form_layout.addRow(QLabel("δ:"), self.delta_input)
        self.form_layout.addRow(QLabel("Амплитуда f:"), self.source_amp_input)
        self.form_layout.addRow(QLabel("Частота f:"), self.source_freq_input)
        self.form_layout.addRow(QLabel("Слой:"), self.layer_input)
        self.form_layout.addRow(QLabel("Время (t):"), self.time_input)
        
        self.run_btn = QPushButton("Запустить")
        self.table_btn = QPushButton("Таблица решений")
        self.compare_btn = QPushButton("Сравнить сетки")
        self.animate_btn = QPushButton("Создать анимацию")
        self.save_heatmap_btn = QPushButton("Сохранить тепл. карту")
        self.save_layer_btn = QPushButton("Сохранить график слоя")
        
        self.grid_cb = QCheckBox("Показать контрольную сетку")
        self.show_ic_cb = QCheckBox("Показать начальное условие")
        self.show_ic_cb.setChecked(True)

        self.form_layout.addRow(self.grid_cb)
        self.form_layout.addRow(self.show_ic_cb)
        
        self.init_initial_conditions_ui()
        
        # статистика
        stats_group = QGroupBox("Статистика слоя")
        stats_layout = QFormLayout()
        
        self.stats_ic = QLabel("Не задано")
        self.stats_max_diff = QLabel("0")
        self.stats_global_max_diff = QLabel("0")
        self.stats_max_diff_loc = QLabel("-")
        self.stats_mean_diff = QLabel("0")
        self.stats_source_val = QLabel("0")
        self.stats_grid_size = QLabel("-")
        self.stats_time_point = QLabel("-")
        
        stats_layout.addRow(QLabel("Нач. условие:"), self.stats_ic)
        stats_layout.addRow(QLabel("Макс. отклонение (слой):"), self.stats_max_diff)
        stats_layout.addRow(QLabel("Макс. отклонение (вся сетка):"), self.stats_global_max_diff)
        stats_layout.addRow(QLabel("Место макс. откл. (слой, узел):"), self.stats_max_diff_loc)
        stats_layout.addRow(QLabel("Среднее отклонение:"), self.stats_mean_diff)
        stats_layout.addRow(QLabel("Значение источника:"), self.stats_source_val)
        stats_layout.addRow(QLabel("Размер сетки:"), self.stats_grid_size)
        stats_layout.addRow(QLabel("Время слоя:"), self.stats_time_point)
        
        stats_group.setLayout(stats_layout)
        stats_group.setMaximumHeight(250)
        
        btn_layout = QVBoxLayout()
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.table_btn)
        btn_layout.addWidget(self.compare_btn)
        btn_layout.addWidget(self.animate_btn)
        btn_layout.addWidget(self.save_heatmap_btn)
        btn_layout.addWidget(self.save_layer_btn)

        form_container = QWidget()
        form_container.setLayout(self.form_layout)
        form_container.setMaximumWidth(400)
        
        control_layout.addWidget(form_container)
        control_layout.addWidget(stats_group)
        control_layout.addLayout(btn_layout)
        control_container.setLayout(control_layout)

        container = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(control_container, stretch=1)
        layout.addWidget(graphics_container, stretch=5)
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.run_btn.clicked.connect(self.run_simulation)
        self.table_btn.clicked.connect(self.show_solution_table)
        self.compare_btn.clicked.connect(self.show_comparison_table)
        self.animate_btn.clicked.connect(self.create_animation)
        self.layer_input.valueChanged.connect(self.draw_layer)
        self.save_heatmap_btn.clicked.connect(self.save_heatmap)
        self.save_layer_btn.clicked.connect(self.save_layer_plot)
        self.time_input.returnPressed.connect(self.find_layer_by_time)
        self.grid_cb.stateChanged.connect(self.redraw_current_layer)
        self.show_ic_cb.stateChanged.connect(self.redraw_current_layer)

        self.draw_empty_layer()

    "настройка выбора начальных узлов"
    def init_initial_conditions_ui(self):
        self.ic_combo = QComboBox()
        self.ic_combo.addItems([
            "Ударная волна", 
            "Гауссов пакет", 
            "Синусоидальный",
            "Пилообразный",
            "Линейное",
            "Квадратичное",
            "Кубическое",
            "Экспоненциальное"
        ])
        self.form_layout.addRow(QLabel("Начальное условие:"), self.ic_combo)
        self.ic_combo.currentTextChanged.connect(self._handle_param_change)

    def _handle_param_change(self):
        self.need_recalculate = True

    def get_initial_condition(self, x):
        ic_type = self.ic_combo.currentText()
        L = x[-1]
        
        if ic_type == "Ударная волна":
            # Резкий скачок (разрыв) в середине области
            return np.where(x < L/2, 1.0, -1.0)
        
        elif ic_type == "Гауссов пакет":
            # Колоколообразное распределение (гауссов пакет)
            return np.exp(-50 * (x - L/2)**2)
        
        elif ic_type == "Синусоидальный":
            # Периодическое распределение
            return np.sin(2 * np.pi * x / L)
        
        elif ic_type == "Пилообразный":
            # Линейно возрастающий сигнал с резкими падениями
            return 2 * (x / L - np.floor(0.5 + x / L))
        
        elif ic_type == "Линейное":
            # Линейное изменение от 1 до 0
            return 1.0 - x/L
        
        elif ic_type == "Квадратичное":
            # Квадратичное распределение
            return 1.0 - (x/L)**2
        
        elif ic_type == "Кубическое":
            # Кубическое распределение
            return 1.0 - (x/L)**3
        
        elif ic_type == "Экспоненциальное":
            # Экспоненциальное затухание
            return np.exp(-5*x/L)
        
        else:
            return np.sin(2 * np.pi * x / L)
    
    def source_function(self, t):
        """Источник f = A * sin(ωt)"""
        try:
            A = float(self.source_amp_input.text())
            ω = float(self.source_freq_input.text())
            return A * math.sin(ω * t)
        except:
            return 0.0
    
    def find_layer_by_time(self):
        """Находит слой по введенному времени"""
        try:
            target_time = float(self.time_input.text())
            
            if not self.time_points:
                QMessageBox.warning(self, "Ошибка", "Данные не рассчитаны!")
                return

            layer_index = 0
            min_diff = float('inf')
            found_index = -1
            
            for i, t in enumerate(self.time_points):
                diff = target_time - t
                if diff >= 0 and diff < min_diff:
                    min_diff = diff
                    found_index = i
                    
            if found_index >= 0:
                self.layer_input.setValue(found_index)
                self.draw_layer(found_index)
            else:
                QMessageBox.warning(self, "Предупреждение", 
                                   f"Слой с временем ≤ {target_time} не найден. Используется начальный слой.")
                self.layer_input.setValue(0)
                self.draw_layer(0)
                
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Некорректное значение времени!")
    
    def redraw_current_layer(self):
        """Перерисовывает текущий слой при изменении настроек"""
        if self.solution_history:
            self.draw_layer(self.layer_input.value())

    def run_simulation(self):
        try:
            T = float(self.T_input.text())
            Nx = int(self.Nx_input.text())
            Nt = int(self.Nt_input.text())
            viscosity = float(self.viscosity_input.text())
            delta = float(self.delta_input.text())
            
            # Проверка условий для предотвращения осцилляций
            L = 1.0
            h = L / (Nx - 1)
            x = np.linspace(0, L, Nx)
            u0 = self.get_initial_condition(x)
            u_max = np.max(np.abs(u0))
            
            # Расчет сеточного числа Рейнольдса
            Re_c = u_max * h / viscosity if viscosity > 0 else float('inf')

            if Re_c > 2:
                min_Nx = int(np.ceil(u_max * L / (2 * viscosity)) + 1) if viscosity > 0 else Nx * 4
                min_viscosity = u_max * h / 2

                advice = []
                if viscosity < 1e-5:
                    advice.append("1. Увеличьте вязкость (viscosity) до хотя бы 0.001")
                elif h > 0.01:
                    advice.append("1. Уменьшите шаг сетки (увеличьте Nx) до {} или более".format(min_Nx))
                else:
                    advice.append("1. Увеличьте вязкость до {:.4f} или более".format(min_viscosity))
                
                advice.append("2. Рассмотрите использование схемы 'upwind' вместо центральных разностей")
                advice.append("3. Увеличьте параметр δ для массового оператора")
                
                # вывод предупреждения об осцилляциях
                msg = (
                    "Внимание! Параметры сетки могут вызвать нефизичные осцилляции!\n"
                    "Сеточное число Рейнольдса Re_c = {:.2f} > 2\n\n"
                    "Рекомендации для текущих параметров:\n{}"
                ).format(Re_c, "\n".join(advice))
                
                msg_box = QMessageBox(QMessageBox.Warning, "Предупреждение об осцилляциях", msg, 
                                     QMessageBox.Ok | QMessageBox.Cancel, self)
                
                if msg_box.exec_() == QMessageBox.Cancel:
                    return
            
            # расчет основной сетки
            x, solution, time_points = self.solve_equation(T, Nx, Nt, viscosity, delta)
            
            # расчет контрольной сетки
            self.solve_control_grid(T, Nx, Nt, viscosity, delta)

            self.layer_input.setMaximum(len(self.solution_history)-1)
            self.layer_input.setValue(0)
            self.draw_heatmap()
            self.draw_layer(0)
            self.need_recalculate = False
            
            # статистика
            n_main = len(self.x_values)
            m_main = len(self.solution_history)
            n_control = len(self.control_solution_history[0]) if self.control_solution_history else 0
            m_control = len(self.control_solution_history) if self.control_solution_history else 0
            self.stats_grid_size.setText(f"Основная: {n_main}×{m_main}, Контрольная: {n_control}×{m_control}")

            self.calculate_global_max_diff()
            
        except Exception as e:
            print(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при расчете: {str(e)}")

    def solve_equation(self, T, Nx, Nt, viscosity, delta):
        L = 1.0
        h = L / (Nx - 1)
        tau = T / Nt
        save_every = max(1, int(self.save_every_input.text()))
        
        # сетка
        x = np.linspace(0, L, Nx)
        u_n = self.get_initial_condition(x)
        
        # ГУ
        u_env = 2/7  # Температура окружающей среды
        H = 7        # Коэффициент теплообмена
        
        # НУ
        u_n[0] = u_n[1]  # Левая граница: ∂u/∂x = 0 (2-й род)
        # Правая граница: ∂u/∂x + (H/viscosity)*u = (H/viscosity)*u_env (3-й род)
        u_n[-1] = (viscosity * u_n[-2] + H * h * u_env) / (viscosity + H * h)
        
        self.solution_history = [u_n.copy()]
        self.time_points = [0.0]
        
        # прогонка
        a = np.zeros(Nx)   # Нижняя диагональ (i-1)
        b = np.zeros(Nx)   # Главная диагональ (i)
        c = np.zeros(Nx)   # Верхняя диагональ (i+1)
        d = np.zeros(Nx)   # Правая часть

        sigma = viscosity * tau / (2 * h**2)
        gamma = tau / (4 * h)
        
        for step in range(1, Nt+1):
            t_current = step * tau
            f_val = self.source_function(t_current)
            
            # линеаризация по Тейлору
            for i in range(1, Nx-1):
                a[i] = gamma * u_n[i-1] - sigma
                b[i] = 1 + 2 * sigma
                c[i] = -gamma * u_n[i+1] - sigma
                M = delta * u_n[i-1] + (1 - 2 * delta) * u_n[i] + delta * u_n[i+1]
                d[i] = M + gamma * (u_n[i+1]**2 - u_n[i-1]**2) - sigma * (u_n[i-1] - 2 * u_n[i] + u_n[i+1])
                d[i] += tau * f_val
            
            # применение ГУ
            # Левая граница (i=0): ∂u/∂x = 0 (2-й род)
            a[0] = 0
            b[0] = 1
            c[0] = -1
            d[0] = 0
            
            # Правая граница (i=Nx-1): ∂u/∂x + (H/viscosity)*u = (H/viscosity)*u_env (3-й род)
            a[Nx-1] = -viscosity
            b[Nx-1] = viscosity + H * h
            c[Nx-1] = 0
            d[Nx-1] = H * h * u_env
            
            # 2. Решение системы методом прогонки
            u_new = self.run_through_method(a, b, c, d)
            
            # 3. Обновление граничных условий
            u_new[0] = u_new[1]  # Левая граница: ∂u/∂x = 0
            # Правая граница: ∂u/∂x + (H/viscosity)*u = (H/viscosity)*u_env
            u_new[-1] = (viscosity * u_new[-2] + H * h * u_env) / (viscosity + H * h)
            
            u_n = u_new
            
            # 4. Сохранение данных
            if step % save_every == 0:
                self.solution_history.append(u_n.copy())
                self.time_points.append(t_current)
        
        self.x_values = x
        return x, u_n, self.time_points

    def solve_control_grid(self, T, Nx, Nt, viscosity, delta):
        """Решение на контрольной сетке (2x по пространству и времени)"""
        # Удваиваем количество узлов
        Nx2 = 2 * Nx
        Nt2 = 2 * Nt
        save_every = max(1, int(self.save_every_input.text()))
        
        L = 1.0
        h2 = L / (Nx2 - 1)
        tau2 = T / Nt2
        
        # Инициализация сетки
        x2 = np.linspace(0, L, Nx2)
        u_n2 = self.get_initial_condition(x2)
        
        # Параметры граничных условий
        u_env = 2/7  # Температура окружающей среды
        H = 7        # Коэффициент теплообмена
        
        # Применение начальных граничных условий
        u_n2[0] = u_n2[1]  # Левая граница: ∂u/∂x = 0 (2-й род)
        # Правая граница: ∂u/∂x + (H/viscosity)*u = (H/viscosity)*u_env (3-й род)
        u_n2[-1] = (viscosity * u_n2[-2] + H * h2 * u_env) / (viscosity + H * h2)
        
        control_solution_history = [u_n2.copy()]
        control_time_points = [0.0]
        
        # Массивы для метода прогонки
        a = np.zeros(Nx2)   # Нижняя диагональ (i-1)
        b = np.zeros(Nx2)   # Главная диагональ (i)
        c = np.zeros(Nx2)   # Верхняя диагональ (i+1)
        d = np.zeros(Nx2)   # Правая часть
        
        # Коэффициенты для схемы Кранка-Николсона
        sigma = viscosity * tau2 / (2 * h2**2)
        gamma = tau2 / (4 * h2)
        
        for step in range(1, Nt2+1):
            # Текущее время
            t_current = step * tau2
            
            # Значение источника в текущий момент времени
            f_val = self.source_function(t_current)
            
            # 1. Линеаризация по Тейлору
            for i in range(1, Nx2-1):
                # Коэффициенты согласно схеме
                a[i] = gamma * u_n2[i-1] - sigma
                b[i] = 1 + 2 * sigma
                c[i] = -gamma * u_n2[i+1] - sigma
                
                # Правая часть с массовым оператором и источником
                M = delta * u_n2[i-1] + (1 - 2 * delta) * u_n2[i] + delta * u_n2[i+1]
                d[i] = M + gamma * (u_n2[i+1]**2 - u_n2[i-1]**2) - sigma * (u_n2[i-1] - 2 * u_n2[i] + u_n2[i+1])
                
                # Добавляем источник (явная схема)
                d[i] += tau2 * f_val
            
            # Граничные условия
            # Левая граница (i=0): ∂u/∂x = 0 (2-й род)
            a[0] = 0
            b[0] = 1
            c[0] = -1
            d[0] = 0
            
            # Правая граница (i=Nx2-1): ∂u/∂x + (H/viscosity)*u = (H/viscosity)*u_env (3-й род)
            a[Nx2-1] = -viscosity
            b[Nx2-1] = viscosity + H * h2
            c[Nx2-1] = 0
            d[Nx2-1] = H * h2 * u_env
            
            # выполнение прогонки
            u_new2 = self.run_through_method(a, b, c, d)
            
            # обновление ГУ
            u_new2[0] = u_new2[1] 
            u_new2[-1] = (viscosity * u_new2[-2] + H * h2 * u_env) / (viscosity + H * h2)
            
            u_n2 = u_new2
            
            # сохранение данных с контр.сетки 
            if step % (2 * save_every) == 0:
                control_solution_history.append(u_n2.copy())
                control_time_points.append(t_current)
        
        self.control_solution_history = control_solution_history
        return x2, u_n2, control_time_points

    def calculate_global_max_diff(self):
        """Вычисляет максимальное отклонение по всей сетке"""
        if not self.solution_history or not self.control_solution_history:
            self.global_max_diff = 0.0
            return
            
        max_diff = 0.0
        x_control = np.linspace(0, 1, len(self.control_solution_history[0]))
        
        for i in range(len(self.solution_history)):
            u_control_interp = np.interp(
                self.x_values, 
                x_control, 
                self.control_solution_history[i]
            )
            
            diff = np.abs(u_control_interp - self.solution_history[i])
            layer_max = np.max(diff)
            if layer_max > max_diff:
                max_diff = layer_max
        
        self.global_max_diff = max_diff
        self.stats_global_max_diff.setText(f"{max_diff:.6f}")

    def run_through_method(self, a, b, c, d):
        n = len(d)
        # проверка на трехдиагональность
        if n < 3:
            # прямое решение для маленьких СЛАУ
            return np.linalg.solve(np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1), d)
        
        cp = np.zeros(n)
        dp = np.zeros(n)
        x = np.zeros(n)
        
        # Прямой ход прогонки
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        
        for i in range(1, n):
            denom = b[i] - a[i] * cp[i-1]
            if abs(denom) < 1e-10:
                denom = 1e-10
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i-1]) / denom
        
        # Обратный ход прогонки
        x[-1] = dp[-1]
        for i in range(n-2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i+1]
            
        return x
    
    "функция построения тепловой карты"
    def draw_heatmap(self):
        self.heatmap_fig.clear()
        if not self.solution_history: return
        
        ax = self.heatmap_fig.add_subplot(111)
        U = np.array(self.solution_history)
        X = self.x_values
        T = np.array(self.time_points)
        
        ax.invert_yaxis()
        abs_max = np.nanmax(np.abs(U))
        norm = Normalize(vmin=-abs_max, vmax=abs_max) if abs_max != 0 else Normalize()
        
        im = ax.imshow(U, aspect='auto', cmap=self.cmap,
                      extent=[X[0], X[-1], T[-1], T[0]],
                      interpolation='bilinear',
                      norm=norm)
        
        ax.set_title(f"Тепловая карта уравнения конвекции-диффузии")
        ax.set_xlabel("Пространство, x [м]")
        ax.set_ylabel("Время, t [с]")
        self.heatmap_fig.colorbar(im, label="Теплота, u")
        self.heatmap_canvas.draw()

    "функция построения шаблона графика слоя"
    def draw_empty_layer(self):
        self.layer_fig.clear()
        ax = self.layer_fig.add_subplot(111)
        ax.set_title("График слоя")
        ax.set_xlabel("Пространство, x [м]")
        ax.set_ylabel("Теплота, u")
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlim(0, 1)
        ax.grid(True)
        self.layer_canvas.draw()

    "функция построения графика слоя"
    def draw_layer(self, layer_index):
        try:
            self.layer_fig.clear()
            ax = self.layer_fig.add_subplot(111)
            
            if not self.solution_history or layer_index >= len(self.solution_history):
                ax.set_title("Данные недоступны")
                ax.grid(True)
                self.layer_canvas.draw()
                return
                
            t = self.time_points[layer_index]
            x = self.x_values
            u = self.solution_history[layer_index]
            
            ax.plot(x, u, 'b-', label="Основная сетка")
            
            if self.grid_cb.isChecked() and self.control_solution_history:
                u_control = self.control_solution_history[layer_index]
                
                x_control = np.linspace(0, 1, len(u_control))
                u_interp = np.interp(x, x_control, u_control)
                
                ax.plot(x, u_interp, 'r--', linewidth=1, label="Контрольная сетка")
            
            if self.show_ic_cb.isChecked():
                ic = self.get_initial_condition(self.x_values)
                ax.plot(self.x_values, ic, 'g--', label="Начальное условие")
            
            ax.set_title(f"Слой {layer_index}, t = {t:.4f} с")
            ax.set_xlabel("Пространство, x [м]")
            ax.set_ylabel("Теплота, u")
            ax.set_ylim(self.y_min, self.y_max)
            ax.set_xlim(0, 1)
            ax.legend()
            ax.grid(True)
            self.layer_canvas.draw()
            self.update_stats(layer_index)
            
        except Exception as e:
            print(f"Ошибка при отрисовке слоя: {e}")

    def create_statistics_text(self, stats):
        """Форматирует статистику в текстовый блок"""
        return (
            f"Нач. условие: {stats['ic']}\n"
            f"Время слоя: {stats['time_point']:.6f}\n"
            f"Макс. отклонение (слой): {stats['max_diff']:.6f}\n"
            f"Макс. отклонение (вся сетка): {stats['global_max_diff']:.6f}\n"
            f"Место макс. откл. (слой, узел): {stats['max_diff_loc']}\n"
            f"Среднее отклонение: {stats['mean_diff']:.6f}\n"
            f"Значение источника: {stats['source_val']:.4f}\n"
            f"Размер сетки: {stats['grid_size']}"
        )


    def update_stats(self, layer_index):
        """Обновление статистики для текущего слоя"""
        stats = {
            'ic': self.ic_combo.currentText(),
            'time_point': self.time_points[layer_index] if self.time_points else 0.0,
            'max_diff': 0.0,
            'global_max_diff': self.global_max_diff,
            'max_diff_loc': '-',
            'mean_diff': 0.0,
            'source_val': 0.0,
            'grid_size': f"Основная: {len(self.x_values)}×{len(self.solution_history)}, "
                        f"Контрольная: {len(self.control_solution_history[0]) if self.control_solution_history else 0}×"
                        f"{len(self.control_solution_history) if self.control_solution_history else 0}"
        }
        
        if self.solution_history and self.control_solution_history and layer_index < len(self.solution_history):
            u_control = self.control_solution_history[layer_index]
            x_control = np.linspace(0, 1, len(u_control))
            u_interp = np.interp(self.x_values, x_control, u_control)
            
            u_main = self.solution_history[layer_index]
            diff = np.abs(u_interp - u_main)
            max_diff = np.max(diff)
            max_idx = np.argmax(diff)
            mean_diff = np.mean(diff)
            
            source_val = self.source_function(self.time_points[layer_index])
            
            stats.update({
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'max_diff_loc': f"{layer_index}, {max_idx}",
                'source_val': source_val
            })
        
        self.stats_ic.setText(stats['ic'])
        self.stats_time_point.setText(f"{stats['time_point']:.6f}")
        self.stats_max_diff.setText(f"{stats['max_diff']:.6f}")
        self.stats_global_max_diff.setText(f"{stats['global_max_diff']:.6f}")
        self.stats_max_diff_loc.setText(stats['max_diff_loc'])
        self.stats_mean_diff.setText(f"{stats['mean_diff']:.6f}")
        self.stats_source_val.setText(f"{stats['source_val']:.4f}")
        self.stats_grid_size.setText(stats['grid_size'])
        
        return stats
    
    "функция показа таблицы решений"
    def show_solution_table(self):
        if not self.solution_history or not self.control_solution_history:
            QMessageBox.warning(self, "Ошибка", "Сначала запустите расчет с включенной контрольной сеткой!")
            return
            
        control_interp = []
        x_control = np.linspace(0, 1, len(self.control_solution_history[0]))
        
        for i in range(len(self.solution_history)):
            u_control_interp = np.interp(
                self.x_values, 
                x_control, 
                self.control_solution_history[i]
            )
            control_interp.append(u_control_interp)
        
        dialog = SolutionTableDialog(
            self.solution_history,
            control_interp,
            self.time_points,
            self.x_values,
            self
        )
        dialog.exec_()
        
    
    def show_comparison_table(self):
        if not self.solution_history or not self.control_solution_history:
            QMessageBox.warning(self, "Ошибка", "Сначала запустите расчет с включенной контрольной сеткой!")
            return
            
        control_interp = []
        x_control = np.linspace(0, 1, len(self.control_solution_history[0]))
        
        for i in range(len(self.solution_history)):
            u_control_interp = np.interp(
                self.x_values, 
                x_control, 
                self.control_solution_history[i]
            )
            control_interp.append(u_control_interp)
        
        dialog = ComparisonTableDialog(
            self.solution_history,
            control_interp,
            self.time_points,
            self.x_values,
            self
        )
        dialog.exec_()
    
    def create_animation(self):
        """Создание GIF-анимации из всех слоев"""
        if not self.solution_history:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анимации!")
            return
            
        try:
            import imageio
            from PIL import Image
        except ImportError:
            QMessageBox.critical(self, "Ошибка", 
                                "Для создания анимации необходимо установить библиотеки:\n"
                                "pip install imageio pillow")
            return
            
        try:
            temp_dir = tempfile.mkdtemp(prefix='burgers_animation_')
            filenames = []
            
            progress = QProgressDialog("Создание анимации...", "Отмена", 0, len(self.solution_history), self)
            progress.setWindowTitle("Создание GIF")
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(0)

            for i in range(len(self.solution_history)):
                if progress.wasCanceled():
                    break
                    
                filename = os.path.join(temp_dir, f"frame_{i:05d}.png")
                filenames.append(filename)
                fig = Figure(figsize=(7, 5))
                ax = fig.add_subplot(111)
                t = self.time_points[i]
                x = self.x_values
                u = self.solution_history[i]
                ax.plot(x, u, 'b-', label="Основная сетка")
                if self.grid_cb.isChecked() and self.control_solution_history:
                    u_control = self.control_solution_history[i]
                    x_control = np.linspace(0, 1, len(u_control))
                    u_interp = np.interp(x, x_control, u_control)
                    ax.plot(x, u_interp, 'r--', linewidth=1, label="Контрольная сетка")
                
                if self.show_ic_cb.isChecked():
                    ic = self.get_initial_condition(self.x_values)
                    ax.plot(x, ic, 'g--', label="Начальное условие")
                
                ax.set_title(f"Слой {i}, t = {t:.4f} с")
                ax.set_xlabel("Пространство, x [м]")
                ax.set_ylabel("Теплота, u")
                ax.set_ylim(self.y_min, self.y_max)
                ax.set_xlim(0, 1)
                ax.legend()
                ax.grid(True)
                
                fig.savefig(filename, dpi=100)
                plt.close(fig)
                progress.setValue(i+1)
           
            if filenames:
                gif_filename = os.path.join(os.getcwd(), "Послойное решение.gif")
                with imageio.get_writer(gif_filename, mode='I', duration=0.1) as writer:
                    for filename in filenames:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                
                for filename in filenames:
                    os.remove(filename)
                os.rmdir(temp_dir)
                
                QMessageBox.information(self, "Готово", f"Анимация сохранена как {gif_filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при создании анимации: {str(e)}")
            if 'filenames' in locals():
                for filename in filenames:
                    if os.path.exists(filename):
                        os.remove(filename)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    def save_heatmap(self):
        """Сохраняет тепловую карту в файл"""
        if not self.solution_history:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения!")
            return
            
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить тепловую карту", "", 
                                                "PNG (*.png);;JPEG (*.jpg *.jpeg);;PDF (*.pdf);;SVG (*.svg)", 
                                                options=options)
        if filename:
            try:
                self.heatmap_fig.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Успех", "Тепловая карта успешно сохранена!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении: {str(e)}")
    
    def save_layer_plot(self):
        """Сохраняет график слоя со статистикой в файл"""
        if not self.solution_history:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения!")
            return
            
        layer_index = self.layer_input.value()
        stats = self.update_stats(layer_index)
        
        save_fig = plt.figure(figsize=(12, 6), tight_layout=True)
        gs = save_fig.add_gridspec(1, 2, width_ratios=[2, 1])
        
        ax1 = save_fig.add_subplot(gs[0])
        t = self.time_points[layer_index]
        x = self.x_values
        u = self.solution_history[layer_index]
        
        ax1.plot(x, u, 'b-', label="Основная сетка")
        
        if self.grid_cb.isChecked() and self.control_solution_history:
            u_control = self.control_solution_history[layer_index]
            x_control = np.linspace(0, 1, len(u_control))
            u_interp = np.interp(x, x_control, u_control)
            ax1.plot(x, u_interp, 'r--', linewidth=1, label="Контрольная сетка")
        
        if self.show_ic_cb.isChecked():
            ic = self.get_initial_condition(self.x_values)
            ax1.plot(x, ic, 'g--', label="Начальное условие")
        
        ax1.set_title(f"Слой {layer_index}, t = {t:.4f} с")
        ax1.set_xlabel("Пространство, x [м]")
        ax1.set_ylabel("Теплота, u")
        ax1.set_ylim(self.y_min, self.y_max)
        ax1.set_xlim(0, 1)
        ax1.legend()
        ax1.grid(True)
        
        ax2 = save_fig.add_subplot(gs[1])
        ax2.axis('off')
        
        stats_text = self.create_statistics_text(stats)
        ax2.text(0.05, 0.95, stats_text, 
                transform=ax2.transAxes, 
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", 
                        facecolor='lightgray', 
                        edgecolor='gray', 
                        alpha=0.5))
        
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить график слоя", "", 
                                                "PNG (*.png);;JPEG (*.jpg *.jpeg);;PDF (*.pdf);;SVG (*.svg)", 
                                                options=options)
        if filename:
            try:
                save_fig.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Успех", "График слоя успешно сохранен!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении: {str(e)}")
            finally:
                plt.close(save_fig)
        else:
            plt.close(save_fig)

            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BurgersEquationApp()
    window.show()
    sys.exit(app.exec_())