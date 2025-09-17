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

class HeatmapWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Тепловая карта решения")
        self.setGeometry(100, 100, 800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        self.save_btn = QPushButton("Сохранить тепловую карту (в формате .png)")
        self.save_btn.clicked.connect(self.save_heatmap)
        layout.addWidget(self.save_btn)
        
    def draw_heatmap(self, x_values, time_points, solution_history, cmap='viridis'):
        self.fig.clear()
        if not solution_history: 
            self.canvas.draw()
            return
            
        ax = self.fig.add_subplot(111)
        U = np.array(solution_history)
        X = x_values
        T = np.array(time_points)
        
        ax.invert_yaxis()
        abs_max = np.nanmax(np.abs(U))
        norm = Normalize(vmin=-abs_max, vmax=abs_max) if abs_max != 0 else Normalize()
        
        im = ax.imshow(U, aspect='auto', cmap=cmap,
                      extent=[X[0], X[-1], T[-1], T[0]],
                      interpolation='bilinear',
                      norm=norm)
        
        ax.set_title(f"Тепловая карта уравнения конвекции-диффузии")
        ax.set_xlabel("Пространство, x [м]")
        ax.set_ylabel("Время, t [с]")
        self.fig.colorbar(im, label="Теплота, u")
        self.canvas.draw()
        
    def save_heatmap(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить тепловую карту", "", 
                                                "PNG (*.png);;JPEG (*.jpg *.jpeg);;PDF (*.pdf);;SVG (*.svg)", 
                                                options=options)
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Успех", "Тепловая карта успешно сохранена!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении: {str(e)}")

class SolutionTableDialog(QDialog):
    def __init__(self, solution_data, control_solution, time_points, x_values, Nx, Nt, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Таблица решений и отклонений")
        self.setMinimumSize(1400, 800)
        self.Nx = Nx
        self.Nt = Nt
        
        self.tabs = QTabWidget()
        
        # Основная сетка
        self.main_table = QTableWidget()
        self.main_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.main_table.verticalHeader().setVisible(False)
        self.main_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # Контрольная сетка
        self.control_table = QTableWidget()
        self.control_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.control_table.verticalHeader().setVisible(False)
        self.control_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # Таблица сравнения
        self.comparison_table = QTableWidget()
        self.comparison_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.comparison_table.verticalHeader().setVisible(False)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.fill_main_table(self.main_table, solution_data, time_points, x_values)
        self.fill_main_table(self.control_table, control_solution, time_points, x_values)
        self.fill_comparison_table(self.comparison_table, solution_data, control_solution, time_points, x_values)

        self.tabs.addTab(self.main_table, "Основная сетка")
        self.tabs.addTab(self.control_table, "Контрольная сетка")
        self.tabs.addTab(self.comparison_table, "Сравнение сеток")

        self.export_btn = QPushButton("Экспорт в CSV")
        self.export_btn.clicked.connect(self.export_to_csv)
        
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(self.export_btn)
        self.setLayout(layout)

        self.resize(QApplication.primaryScreen().availableSize() * 0.9)

    def fill_main_table(self, table, solution_data, time_points, x_values):
        num_layers = len(solution_data)
        num_x = len(x_values)
        table.setRowCount(num_layers + 1)  # +1 для строки с номерами узлов
        table.setColumnCount(num_x + 2)  # +2 для номера слоя и времени

        # Заголовки
        table.setHorizontalHeaderItem(0, QTableWidgetItem("№ слоя"))
        table.setHorizontalHeaderItem(1, QTableWidgetItem("t"))
        for col in range(num_x):
            table.setHorizontalHeaderItem(col+2, QTableWidgetItem(f"x={x_values[col]:.4f}"))

        # Строка с номерами узлов
        table.setItem(0, 0, QTableWidgetItem("Узел"))
        table.setItem(0, 1, QTableWidgetItem(""))
        for col in range(num_x):
            node_item = QTableWidgetItem()
            node_item.setData(Qt.DisplayRole, f"{col}")
            table.setItem(0, col+2, node_item)

        for row in range(num_layers):
            # Номер слоя
            layer_item = QTableWidgetItem()
            layer_item.setData(Qt.DisplayRole, f"{row}")
            table.setItem(row+1, 0, layer_item)
            
            # Время
            time_item = QTableWidgetItem()
            time_item.setData(Qt.DisplayRole, f"{time_points[row]:.6g}")
            table.setItem(row+1, 1, time_item)
            
            # Значения решения
            for col in range(num_x):
                val = solution_data[row][col]
                item = QTableWidgetItem()
                
                if abs(val) < 1e-4 and abs(val) > 1e-10:
                    item.setData(Qt.DisplayRole, f"{val:.4e}")
                elif abs(val) < 1e-10:
                    item.setData(Qt.DisplayRole, "0")
                else:
                    item.setData(Qt.DisplayRole, f"{val:.6f}")
                
                table.setItem(row+1, col+2, item)
    
    def fill_comparison_table(self, table, main_data, control_data, time_points, x_values):
        num_layers = len(main_data)
        num_x = len(x_values)
        table.setRowCount(num_layers + 1)  # +1 для строки с номерами узлов
        table.setColumnCount(num_x + 5)  # +5 для номера слоя, t, max|u2-u|, i_max, среднее отклонение

        # Заголовки
        headers = ["№ слоя", "t", "max|u2 - u|", "i_max", "Среднее отклонение"]
        for col, header in enumerate(headers):
            table.setHorizontalHeaderItem(col, QTableWidgetItem(header))
        for col in range(num_x):
            table.setHorizontalHeaderItem(col+5, QTableWidgetItem(f"x={x_values[col]:.4f}"))

        # Строка с номерами узлов
        table.setItem(0, 0, QTableWidgetItem("Узел"))
        table.setItem(0, 1, QTableWidgetItem(""))
        table.setItem(0, 2, QTableWidgetItem(""))
        table.setItem(0, 3, QTableWidgetItem(""))
        table.setItem(0, 4, QTableWidgetItem(""))
        for col in range(num_x):
            node_item = QTableWidgetItem()
            node_item.setData(Qt.DisplayRole, f"{col}")
            table.setItem(0, col+5, node_item)

        for row in range(num_layers):
            # Номер слоя
            layer_item = QTableWidgetItem()
            layer_item.setData(Qt.DisplayRole, f"{row}")
            table.setItem(row+1, 0, layer_item)
            
            # Время
            time_item = QTableWidgetItem()
            time_item.setData(Qt.DisplayRole, f"{time_points[row]:.6g}")
            table.setItem(row+1, 1, time_item)

            # Расчет отклонений
            diff = np.abs(control_data[row] - main_data[row])
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            max_idx = np.argmax(diff)
            
            # Максимальное отклонение
            max_item = QTableWidgetItem()
            max_item.setData(Qt.DisplayRole, f"{max_diff:.6f}")
            table.setItem(row+1, 2, max_item)
            
            # Индекс максимального отклонения (правильная нумерация узлов)
            idx_item = QTableWidgetItem()
            idx_item.setData(Qt.DisplayRole, f"{max_idx}")
            table.setItem(row+1, 3, idx_item)
            
            # Среднее отклонение
            mean_item = QTableWidgetItem()
            mean_item.setData(Qt.DisplayRole, f"{mean_diff:.6f}")
            table.setItem(row+1, 4, mean_item)
            
            # Отклонения по узлам
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
                
                table.setItem(row+1, col+5, item)
            
            # Размеры сеток
            table.setItem(row+1, num_x+5, QTableWidgetItem(f"{self.Nx} x {self.Nt}"))
            table.setItem(row+1, num_x+6, QTableWidgetItem(f"{2*self.Nx} x {2*self.Nt}"))
    
    def export_to_csv(self):
        try:
            options = QFileDialog.Options()
            base_name, _ = QFileDialog.getSaveFileName(self, "Сохранить базовый CSV", "", "CSV Files (*.csv)", options=options)
            if not base_name:
                return
            
            self.save_table_to_csv(self.main_table, base_name + "_main.csv")
            self.save_table_to_csv(self.control_table, base_name + "_control.csv")
            self.save_table_to_csv(self.comparison_table, base_name + "_comparison.csv")

            QMessageBox.information(self, "Успех", "Таблицы успешно экспортированы в CSV файлы!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при экспорте: {str(e)}")
    
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

class ConvectionDiffusionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.solution_history = []
        self.control_solution_history = []
        self.time_points = []
        self.x_values = []
        self.current_params = {}
        self.need_recalculate = True
        self.cmap = 'viridis'
        self.global_max_diff = 0.0
        self.y_min, self.y_max = -1.5, 1.5
        self.heatmap_window = None
        self.Nx = 0
        self.Nt = 0
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Моделирование уравнения конвекции-диффузии с источником")
        self.setGeometry(100, 100, 1600, 900)
        
        self.layer_fig = Figure(figsize=(10, 6))
        self.layer_canvas = FigureCanvas(self.layer_fig)
        self.layer_canvas.setFocusPolicy(Qt.ClickFocus)
        self.layer_canvas.setFocus()

        main_container = QWidget()
        main_layout = QHBoxLayout(main_container)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_container.setMinimumWidth(450)

        self.T_input = QLineEdit("2.0")
        self.Nx_input = QLineEdit("200")
        self.Nt_input = QLineEdit("5000")
        self.viscosity_input = QLineEdit("0.01")
        self.delta_input = QLineEdit("0.1")
        self.layer_input = QSpinBox()
        self.time_input = QLineEdit("0.0")
        self.save_every_input = QLineEdit("100")
        self.source_amp_input = QLineEdit("9.0")
        self.source_freq_input = QLineEdit("5.0")

        self.form_layout = QFormLayout()
        self.form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.form_layout.addRow(QLabel("Шаг просмотра числ. решения:"), self.save_every_input)
        self.form_layout.addRow(QLabel("Горизонт расчета моделирования по времени:"), self.T_input)
        self.form_layout.addRow(QLabel("Число участков разбиения по оси x (Nx):"), self.Nx_input)
        self.form_layout.addRow(QLabel("Число участков разбиения по оси t (Nt):"), self.Nt_input)
        self.form_layout.addRow(QLabel("Вязкость V, параметр модели:"), self.viscosity_input)
        self.form_layout.addRow(QLabel("δ (параметр сглаживающего оператора):"), self.delta_input)
        self.form_layout.addRow(QLabel("Амплитуда А источника:"), self.source_amp_input)
        self.form_layout.addRow(QLabel("Частота (ω) источника:"), self.source_freq_input)
        self.form_layout.addRow(QLabel("№ слоя для показа на графике:"), self.layer_input)
        self.form_layout.addRow(QLabel("Время t, для которого подбирается ближний слой:"), self.time_input)
        
        self.run_btn = QPushButton("Запустить расчет")
        self.table_btn = QPushButton("Таблица решений")
        self.animate_btn = QPushButton("Сохранить графики слоев в GIF-анимации")
        self.show_heatmap_btn = QPushButton("Показать тепловую карту")
        self.save_layer_btn = QPushButton("Сохранить график слоя (в .png)")
        
        self.grid_cb = QCheckBox("Показать контрольную сетку")
        self.show_ic_cb = QCheckBox("Показать начальное условие")
        self.show_ic_cb.setChecked(True)

        self.form_layout.addRow(self.grid_cb)
        self.form_layout.addRow(self.show_ic_cb)
        
        self.init_initial_conditions_ui()
        
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
        btn_layout.addWidget(self.animate_btn)
        btn_layout.addWidget(self.show_heatmap_btn)
        btn_layout.addWidget(self.save_layer_btn)

        left_layout.addLayout(self.form_layout)
        left_layout.addWidget(stats_group)
        left_layout.addLayout(btn_layout)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        
        self.equation_label = QLabel(
            "Уравнение: u_t + u * u_x = V * u_xx + A * sin(ω*t)\n"
            "x ∈ [0, 1]\n"
            "Граничные условия:\n"
            "  При x=0: u_x = 0\n"
            "  При x=1: u_x + (7/V)*u = (7/V)*(2/7)\n"
            "Начальное условие: не задано"
        )
        self.equation_label.setWordWrap(True)
        self.equation_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc;")
        self.equation_label.setMaximumHeight(150)
        
        right_layout.addWidget(self.equation_label)
        right_layout.addWidget(self.layer_canvas)

        main_layout.addWidget(left_container)
        main_layout.addWidget(right_container)
        self.setCentralWidget(main_container)

        self.run_btn.clicked.connect(self.run_simulation)
        self.table_btn.clicked.connect(self.show_solution_table)
        self.animate_btn.clicked.connect(self.create_animation)
        self.layer_input.valueChanged.connect(self.draw_layer)
        self.show_heatmap_btn.clicked.connect(self.show_heatmap)
        self.save_layer_btn.clicked.connect(self.save_layer_plot)
        self.time_input.returnPressed.connect(self.find_layer_by_time)
        self.grid_cb.stateChanged.connect(self.redraw_current_layer)
        self.show_ic_cb.stateChanged.connect(self.redraw_current_layer)

        self.draw_empty_layer()

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
            return np.where(x < L/2, 1.0, -1.0)
        
        elif ic_type == "Гауссов пакет":
            return np.exp(-50 * (x - L/2)**2)
        
        elif ic_type == "Синусоидальный":
            return np.sin(2 * np.pi * x / L)
        
        elif ic_type == "Пилообразный":
            return 2 * (x / L - np.floor(0.5 + x / L))
        
        elif ic_type == "Линейное":
            return 1.0 - x/L
        
        elif ic_type == "Квадратичное":
            return 1.0 - (x/L)**2
        
        elif ic_type == "Кубическое":
            return 1.0 - (x/L)**3
        
        elif ic_type == "Экспоненциальное":
            return np.exp(-5*x/L)
        
        else:
            return np.sin(2 * np.pi * x / L)
    
    def source_function(self, t):
        try:
            A = float(self.source_amp_input.text())
            ω = float(self.source_freq_input.text())
            return A * math.sin(ω * t)
        except:
            return 0.0
    
    def find_layer_by_time(self):
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
        if self.solution_history:
            self.draw_layer(self.layer_input.value())

    def run_simulation(self):
        try:
            # Проверка шага сохранения
            save_every_text = self.save_every_input.text()
            if not save_every_text.isdigit() or int(save_every_text) <= 0:
                QMessageBox.warning(self, "Ошибка", "Шаг сохранения должен быть целым положительным числом!")
                return
            
            T = float(self.T_input.text())
            self.Nx = int(self.Nx_input.text())
            self.Nt = int(self.Nt_input.text())
            viscosity = float(self.viscosity_input.text())
            delta = float(self.delta_input.text())
            save_every = int(save_every_text)
            
            L = 1.0
            h = L / (self.Nx - 1)
            x = np.linspace(0, L, self.Nx)
            u0 = self.get_initial_condition(x)
            u_max = np.max(np.abs(u0))
            
            Re_c = u_max * h / viscosity if viscosity > 0 else float('inf')

            if Re_c > 2:
                min_Nx = int(np.ceil(u_max * L / (2 * viscosity)) + 1) if viscosity > 0 else self.Nx * 4
                min_viscosity = u_max * h / 2

                advice = []
                if viscosity < 1e-5:
                    advice.append("1. Увеличьте вязкость (viscosity) до хотя бы 0.001")
                elif h > 0.01:
                    advice.append("1. Уменьшите шаг сетки (увеличьте Nx) до {} или более".format(min_Nx))
                else:
                    advice.append("1. Увеличьте вязкость до {:.4f} или более".format(min_viscosity))
                
                advice.append("2. Рассмотрите использование схемя 'upwind' вместо центральных разностей")
                advice.append("3. Увеличьте параметр δ для массового оператора")
                
                msg = (
                    "Внимание! Параметры сетки могут вызвать нефизичные осцилляции!\n"
                    "Сеточное число Рейнольдса Re_c = {:.2f} > 2\n\n"
                    "Рекомендации для текущих параметров:\n{}"
                ).format(Re_c, "\n".join(advice))
                
                msg_box = QMessageBox(QMessageBox.Warning, "Предуреждание об осцилляциях", msg, 
                                     QMessageBox.Ok | QMessageBox.Cancel, self)
                
                if msg_box.exec_() == QMessageBox.Cancel:
                    return
            
            x, solution, time_points = self.solve_equation(T, self.Nx, self.Nt, viscosity, delta, save_every)
            
            self.solve_control_grid(T, self.Nx, self.Nt, viscosity, delta, save_every)

            self.layer_input.setMaximum(len(self.solution_history)-1)
            self.layer_input.setValue(0)
            self.draw_layer(0)
            self.need_recalculate = False
            
            # Правильный расчет размеров сеток
            self.stats_grid_size.setText(f"Основная: {self.Nx}×{self.Nt}, Контрольная: {2*self.Nx}×{2*self.Nt}")

            # Обновляем отображение уравнения с начальным условием
            ic_type = self.ic_combo.currentText()
            self.equation_label.setText(
                "Уравнение: u_t + u * u_x = V * u_xx + A * sin(ω*t)\n"
                "x ∈ [0, 1]\n"
                "Граничные условия:\n"
                "  При x=0: u_x = 0\n"
                "  При x=1: u_x + (7/V)*u = (7/V)*(2/7)\n"
                f"Начальное условие: {ic_type}"
            )

            self.calculate_global_max_diff()
            
            if self.heatmap_window and self.heatmap_window.isVisible():
                self.heatmap_window.draw_heatmap(self.x_values, self.time_points, self.solution_history, self.cmap)
            
        except Exception as e:
            print(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при расчете: {str(e)}")

    def solve_equation(self, T, Nx, Nt, viscosity, delta, save_every):
        L = 1.0
        h = L / (Nx - 1)
        tau = T / Nt
        
        x = np.linspace(0, L, Nx)
        u_n = self.get_initial_condition(x)
        
        u_env = 2/7
        H = 7
        
        u_n[0] = u_n[1]
        u_n[-1] = (viscosity * u_n[-2] + H * h * u_env) / (viscosity + H * h)
        
        self.solution_history = [u_n.copy()]
        self.time_points = [0.0]
        
        a = np.zeros(Nx)
        b = np.zeros(Nx)
        c = np.zeros(Nx)
        d = np.zeros(Nx)

        sigma = viscosity * tau / (2 * h**2)
        gamma = tau / (4 * h)
        
        for step in range(1, Nt+1):
            t_current = step * tau
            f_val = self.source_function(t_current)
            
            for i in range(1, Nx-1):
                a[i] = gamma * u_n[i-1] - sigma
                b[i] = 1 + 2 * sigma
                c[i] = -gamma * u_n[i+1] - sigma
                M = delta * u_n[i-1] + (1 - 2 * delta) * u_n[i] + delta * u_n[i+1]
                d[i] = M + gamma * (u_n[i+1]**2 - u_n[i-1]**2) - sigma * (u_n[i-1] - 2 * u_n[i] + u_n[i+1])
                d[i] += tau * f_val
            
            a[0] = 0
            b[0] = 1
            c[0] = -1
            d[0] = 0
            
            a[Nx-1] = -viscosity
            b[Nx-1] = viscosity + H * h
            c[Nx-1] = 0
            d[Nx-1] = H * h * u_env
            
            u_new = self.run_through_method(a, b, c, d)
            
            u_new[0] = u_new[1]
            u_new[-1] = (viscosity * u_new[-2] + H * h * u_env) / (viscosity + H * h)
            
            u_n = u_new
            
            if step % save_every == 0:
                self.solution_history.append(u_n.copy())
                self.time_points.append(t_current)
        
        self.x_values = x
        return x, u_n, self.time_points

    def solve_control_grid(self, T, Nx, Nt, viscosity, delta, save_every):
        Nx2 = 2 * Nx
        Nt2 = 2 * Nt
        
        L = 1.0
        h2 = L / (Nx2 - 1)
        tau2 = T / Nt2
        
        x2 = np.linspace(0, L, Nx2)
        u_n2 = self.get_initial_condition(x2)
        
        u_env = 2/7
        H = 7
        
        u_n2[0] = u_n2[1]
        u_n2[-1] = (viscosity * u_n2[-2] + H * h2 * u_env) / (viscosity + H * h2)
        
        control_solution_history = [u_n2.copy()]
        control_time_points = [0.0]
        
        a = np.zeros(Nx2)
        b = np.zeros(Nx2)
        c = np.zeros(Nx2)
        d = np.zeros(Nx2)
        
        sigma = viscosity * tau2 / (2 * h2**2)
        gamma = tau2 / (4 * h2)
        
        for step in range(1, Nt2+1):
            t_current = step * tau2
            f_val = self.source_function(t_current)
            
            for i in range(1, Nx2-1):
                a[i] = gamma * u_n2[i-1] - sigma
                b[i] = 1 + 2 * sigma
                c[i] = -gamma * u_n2[i+1] - sigma
                
                M = delta * u_n2[i-1] + (1 - 2 * delta) * u_n2[i] + delta * u_n2[i+1]
                d[i] = M + gamma * (u_n2[i+1]**2 - u_n2[i-1]**2) - sigma * (u_n2[i-1] - 2 * u_n2[i] + u_n2[i+1])
                
                d[i] += tau2 * f_val
            
            a[0] = 0
            b[0] = 1
            c[0] = -1
            d[0] = 0
            
            a[Nx2-1] = -viscosity
            b[Nx2-1] = viscosity + H * h2
            c[Nx2-1] = 0
            d[Nx2-1] = H * h2 * u_env
            
            u_new2 = self.run_through_method(a, b, c, d)
            
            u_new2[0] = u_new2[1]
            u_new2[-1] = (viscosity * u_new2[-2] + H * h2 * u_env) / (viscosity + H * h2)
            
            u_n2 = u_new2
            
            if step % (2 * save_every) == 0:
                control_solution_history.append(u_n2.copy())
                control_time_points.append(t_current)
        
        self.control_solution_history = control_solution_history
        return x2, u_n2, control_time_points

    def calculate_global_max_diff(self):
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
        if n < 3:
            return np.linalg.solve(np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1), d)
        
        cp = np.zeros(n)
        dp = np.zeros(n)
        x = np.zeros(n)
        
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        
        for i in range(1, n):
            denom = b[i] - a[i] * cp[i-1]
            if abs(denom) < 1e-10:
                denom = 1e-10
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i-1]) / denom
        
        x[-1] = dp[-1]
        for i in range(n-2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i+1]
            
        return x
    
    def show_heatmap(self):
        if not self.solution_history:
            QMessageBox.warning(self, "Ошибка", "Нет данных для отображения!")
            return
            
        if self.heatmap_window is None:
            self.heatmap_window = HeatmapWindow(self)
        self.heatmap_window.draw_heatmap(self.x_values, self.time_points, self.solution_history, self.cmap)
        self.heatmap_window.show()

    def draw_empty_layer(self):
        self.layer_fig.clear()
        ax = self.layer_fig.add_subplot(111)
        ax.set_title("График слоя")
        ax.set_xlabel("Пространство, x [м]")
        ax.set_ylabel("Теплота, u")
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlim(0, 1)
        ax.grid(True)
        ax.axvline(x=0, color='k', linestyle='-')
        ax.axvline(x=1, color='k', linestyle='-')
        self.layer_canvas.draw()

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
            
            # Автоматическое масштабирование
            y_min = np.min(u) - 0.1
            y_max = np.max(u) + 0.1
            
            ax.plot(x, u, 'b-', label="Основная сетка")
            
            if self.grid_cb.isChecked() and self.control_solution_history:
                u_control = self.control_solution_history[layer_index]
                
                x_control = np.linspace(0, 1, len(u_control))
                u_interp = np.interp(x, x_control, u_control)
                
                ax.plot(x, u_interp, 'r--', linewidth=1, label="Контрольная сетка")
            
            if self.show_ic_cb.isChecked():
                ic = self.get_initial_condition(self.x_values)
                ax.plot(self.x_values, ic, 'g--', label="Начальное условие")
            
            # Вычисление номера слоя при шаге сохранения = 1
            save_every = int(self.save_every_input.text())
            actual_layer = layer_index * save_every
            
            ax.set_title(f"График показательного слоя № {layer_index}, график слоя № {actual_layer}, t = {t:.4f} с")
            ax.set_xlabel("Пространство, x [м]")
            ax.set_ylabel("Теплота, u")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(0, 1)
            ax.axvline(x=0, color='k', linestyle='-')
            ax.axvline(x=1, color='k', linestyle='-')
            ax.legend()
            ax.grid(True)
            self.layer_canvas.draw()
            self.update_stats(layer_index)
            
        except Exception as e:
            print(f"Ошибка при отрисовке слоя: {e}")

    def create_statistics_text(self, stats):
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
        stats = {
            'ic': self.ic_combo.currentText(),
            'time_point': self.time_points[layer_index] if self.time_points else 0.0,
            'max_diff': 0.0,
            'global_max_diff': self.global_max_diff,
            'max_diff_loc': '-',
            'mean_diff': 0.0,
            'source_val': 0.0,
            'grid_size': f"Основная: {self.Nx}×{self.Nt}, Контрольная: {2*self.Nx}×{2*self.Nt}"
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
            self.Nx,
            self.Nt,
            self
        )
        dialog.exec_()
    
    def create_animation(self):
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
            temp_dir = tempfile.mkdtemp(prefix='convection_diffusion_animation_')
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
                
                # Автоматическое масштабирование
                y_min = np.min(u) - 0.1
                y_max = np.max(u) + 0.1
                
                ax.plot(x, u, 'b-', label="Основная сетка")
                if self.grid_cb.isChecked() and self.control_solution_history:
                    u_control = self.control_solution_history[i]
                    x_control = np.linspace(0, 1, len(u_control))
                    u_interp = np.interp(x, x_control, u_control)
                    ax.plot(x, u_interp, 'r--', linewidth=1, label="Контрольная сетка")
                
                if self.show_ic_cb.isChecked():
                    ic = self.get_initial_condition(self.x_values)
                    ax.plot(x, ic, 'g--', label="Начальное условие")
                
                # Вычисление номера слоя при шаге сохранения = 1
                save_every = int(self.save_every_input.text())
                actual_layer = i * save_every
                
                ax.set_title(f"График показательного слоя № {i}, график слоя № {actual_layer}, t = {t:.4f} с")
                ax.set_xlabel("Пространство, x [м]")
                ax.set_ylabel("Теплота, u")
                ax.set_ylim(y_min, y_max)
                ax.set_xlim(0, 1)
                ax.axvline(x=0, color='k', linestyle='-')
                ax.axvline(x=1, color='k', linestyle='-')
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
    
    def save_layer_plot(self):
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
        
        # Автоматическое масштабирование
        y_min = np.min(u) - 0.1
        y_max = np.max(u) + 0.1
        
        ax1.plot(x, u, 'b-', label="Основная сетка")
        
        if self.grid_cb.isChecked() and self.control_solution_history:
            u_control = self.control_solution_history[layer_index]
            x_control = np.linspace(0, 1, len(u_control))
            u_interp = np.interp(x, x_control, u_control)
            ax1.plot(x, u_interp, 'r--', linewidth=1, label="Контрольная сетка")
        
        if self.show_ic_cb.isChecked():
            ic = self.get_initial_condition(self.x_values)
            ax1.plot(x, ic, 'g--', label="Начальное условие")
        
        # Вычисление номера слоя при шаге сохранения = 1
        save_every = int(self.save_every_input.text())
        actual_layer = layer_index * save_every
        
        ax1.set_title(f"График показательного слоя № {layer_index}, график слоя № {actual_layer}, t = {t:.4f} с")
        ax1.set_xlabel("Пространство, x [м]")
        ax1.set_ylabel("Теплота, u")
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlim(0, 1)
        ax1.axvline(x=0, color='k', linestyle='-')
        ax1.axvline(x=1, color='k', linestyle='-')
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
    window = ConvectionDiffusionApp()
    window.show()
    sys.exit(app.exec_())