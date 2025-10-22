import sys
import os
import tempfile
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFormLayout, QSpinBox, QTableWidget, QTableWidgetItem,
                             QDialog, QScrollArea, QCheckBox, QComboBox, QHeaderView,
                             QTabWidget, QMessageBox, QGroupBox, QProgressDialog,
                             QFileDialog, QSizePolicy, QTextEdit)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import math
import base64
from io import BytesIO

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка по программе")
        self.setGeometry(100, 100, 1100, 800)
        
        # Получаем абсолютный путь к папке с изображениями
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.initial_conditions_dir = os.path.join(self.base_dir, "initial_conditions")
        
        layout = QVBoxLayout()
        
        self.tabs = QTabWidget()
        
        # Создаем вкладки
        self.setup_contents_tab()
        self.setup_problem_tab()
        self.setup_physics_tab()
        self.setup_initial_conditions_tab()
        self.setup_boundary_conditions_tab()
        self.setup_convective_tab()
        self.setup_source_tab()
        self.setup_method_tab()
        
        layout.addWidget(self.tabs)
        
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def setup_contents_tab(self):
        """Вкладка с содержанием и навигацией"""
        tab = QTextEdit()
        tab.setReadOnly(True)
        
        html_content = """
        <h1>Содержание справки</h1>
        
        <p>Добро пожаловать в справку по программе моделирования уравнения конвекции-диффузии. 
        Здесь вы найдете всю необходимую информацию для работы с программой.</p>
        
        <h2>Быстрая навигация</h2>
        
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
        <h3>📋 Основные разделы:</h3>
        <ul style="font-size: 14px;">
            <li><a href="#problem">Постановка задачи</a> - основное уравнение и условия</li>
            <li><a href="#physics">Физический смысл</a> - объяснение компонентов уравнения</li>
            <li><a href="#initial">Начальные условия</a> - доступные варианты начального распределения</li>
            <li><a href="#boundary">Граничные условия</a> - условия на границах области</li>
            <li><a href="#convective">Конвективный член</a> - типы конвективного переноса</li>
            <li><a href="#source">Источник тепла</a> - варианты внешних воздействий</li>
            <li><a href="#method">Метод решения</a> - численный метод и параметры</li>
        </ul>
        </div>
        
        <h2>Краткое описание программы</h2>
        
        <p>Эта программа предназначена для численного решения уравнения конвекции-диффузии с источником тепла. 
        Она позволяет исследовать различные физические сценарии, изменяя начальные и граничные условия, 
        параметры модели и характеристики источников.</p>
        
        <p>Программа использует современные численные методы для обеспечения точности и устойчивости решения, 
        а также предоставляет разнообразные инструменты для визуализации и анализа результатов.</p>
        """
        
        tab.setHtml(html_content)
        self.tabs.addTab(tab, "Содержание")
    
    def setup_problem_tab(self):
        tab = QTextEdit()
        tab.setReadOnly(True)
        tab.setHtml("""
        <h1><a name="problem">Постановка задачи</a></h1>
        
        <p>В программе решается уравнение конвекции-диффузии с источником тепла - одно из фундаментальных уравнений 
        математической физики, описывающее процессы переноса. Это уравнение объединяет два основных механизма 
        распространения: конвективный перенос и диффузионное выравнивание.</p>
        
        <h2>Основное уравнение</h2>
        
        <p style="text-align: center; font-size: 18px; font-weight: bold;">
        u<sub>t</sub> + C(u) · u<sub>x</sub> = V · u<sub>xx</sub> + f(x,t)
        </p>
        
        <p>Здесь u(x,t) представляет собой температуру в точке x в момент времени t. Уравнение описывает 
        как температура изменяется под влиянием различных физических процессов.</p>
        
        <h2>Область решения</h2>
        
        <p>Расчеты проводятся в одномерной пространственной области от x=0 до x=1 и во временном интервале 
        от t=0 до заданного пользователем времени T. Такая нормированная область удобна для анализа, 
        а при необходимости может быть масштабирована на реальные физические размеры.</p>
        
        <h2>Начальное условие</h2>
        
        <p>В начальный момент времени t=0 задается распределение температуры u(x,0) = u₀(x). 
        Программа предоставляет разнообразные варианты начальных условий, от простых ступенчатых распределений 
        до сложных нелинейных профилей.</p>
        
        <h2>Граничные условия</h2>
        
        <p>На границах расчетной области задаются условия, определяющие взаимодействие системы с окружающей средой. 
        На левой границе установлено условие теплоизоляции, а на правой - условие теплообмена с внешней средой.</p>
        """)
        self.tabs.addTab(tab, "Постановка задачи")
    
    def setup_physics_tab(self):
        tab = QTextEdit()
        tab.setReadOnly(True)
        tab.setHtml("""
        <h1><a name="physics">Физический смысл уравнения</a></h1>
        
        <p>Уравнение конвекции-диффузии описывает широкий класс физических явлений, связанных с переносом 
        вещества или энергии. Каждый член уравнения имеет четкий физический смысл и соответствует 
        определенному механизму переноса.</p>
        
        <h2>Конвективный перенос</h2>
        
        <p>Член C(u)·u<sub>x</sub> описывает конвективный перенос - направленное движение температуры вместе 
        со средой. Это может быть вызвано различными причинами:</p>
        
        <ul>
        <li><strong>Градиентом температуры</strong> - естественное стремление тепла перемещаться из областей 
        с высокой температурой в области с низкой температурой</li>
        <li><strong>Внешними силами</strong> - воздействие гравитации, электрического поля или других внешних факторов</li>
        <li><strong>Движением среды</strong> - макроскопическое перемещение жидкости или газа, увлекающее за собой тепло</li>
        <li><strong>Самовоздействием</strong> - когда сама температура влияет на характер своего переноса</li>
        </ul>
        
        <h2>Диффузионный процесс</h2>
        
        <p>Член V·u<sub>xx</sub> отвечает за диффузионное выравнивание - процесс, при котором температура 
        стремится равномерно распределиться по всему объему. Этот механизм работает благодаря:</p>
        
        <ul>
        <li><strong>Молекулярной диффузии</strong> - хаотическому тепловому движению молекул, 
        приводящему к постепенному перемешиванию</li>
        <li><strong>Теплопроводности</strong> - способности материала проводить тепло через непосредственный контакт</li>
        <li><strong>Турбулентному перемешиванию</strong> - в случае turbulent flows, когда хаотические 
        движения вихрей эффективно перемешивают среду</li>
        </ul>
        
        <p>Коэффициент диффузии V определяет, насколько быстро происходит выравнивание температуры. 
        Большие значения соответствуют быстрому распространению тепла, малые - медленному.</p>
        
        <h2>Источник тепла</h2>
        
        <p>Функция f(x,t) описывает внешние воздействия на систему, которые могут добавлять или отнимать энергию:</p>
        
        <ul>
        <li><strong>Тепловыделение</strong> - химические реакции, ядерные процессы, электрический нагрев</li>
        <li><strong>Охлаждение</strong> - испарение, излучение, контакт с холодными поверхностями</li>
        <li><strong>Периодические воздействия</strong> - циклические процессы, сезонные изменения</li>
        <li><strong>Локальные источники</strong> - точечные или распределенные тепловые воздействия</li>
        </ul>
        
        <p>Сочетание этих трех механизмов позволяет описывать сложные физические процессы, 
        встречающиеся в природе и технике.</p>
        """)
        self.tabs.addTab(tab, "Физический смысл")
    
    def setup_initial_conditions_tab(self):
        tab = QTextEdit()
        tab.setReadOnly(True)
        
        # Формируем HTML с изображениями в base64
        html_content = """
        <h1><a name="initial">Начальные условия</a></h1>
        
        <p>Начальное условие определяет распределение температуры в начальный момент времени t=0. 
        От выбора начального условия сильно зависит поведение решения и его эволюция во времени. 
        Программа предоставляет восемь различных вариантов начальных распределений.</p>
        
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
        """
        
        # Список начальных условий с описаниями
        ics = [
            ("step.png", "Ступенчатое распределение", 
             "u(x,0) = 1 при x < 0.5, -1 при x ≥ 0.5", 
             "Характеризуется резким скачком температуры в центре области. Такое распределение моделирует два различных состояния, разделенных четкой границей."),
            
            ("gaussian.png", "Гауссов пакет", 
             "u(x,0) = exp(-50·(x-0.5)²)", 
             "Колоколообразное распределение с максимумом в центре. Моделирует локализованный тепловой импульс."),
            
            ("sinusoidal.png", "Синусоидальное распределение", 
             "u(x,0) = sin(2πx)", 
             "Периодическое распределение с одной полной волной на всей области. Полезно для анализа волновых процессов."),
            
            ("sawtooth.png", "Пилообразное распределение", 
             "u(x,0) = 2·(x - floor(0.5 + x))", 
             "Линейное нарастание температуры с резким сбросом. Интересно для изучения разрывов."),
            
            ("linear.png", "Линейное распределение", 
             "u(x,0) = 1 - x", 
             "Постепенное линейное уменьшение температуры от левой границы к правой."),
            
            ("quadratic.png", "Квадратичное распределение", 
             "u(x,0) = 1 - x²", 
             "Параболическое распределение с максимальной температурой на левой границе."),
            
            ("cubic.png", "Кубическое распределение", 
             "u(x,0) = 1 - x³", 
             "Более крутое нелинейное убывание температуры, чем при квадратичном распределении."),
            
            ("exponential.png", "Экспоненциальное распределение", 
             "u(x,0) = exp(-5x)", 
             "Быстрое экспоненциальное затухание температуры от левой границы.")
        ]
        
        for filename, title, formula, description in ics:
            img_path = os.path.join(self.initial_conditions_dir, filename)
            if os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode()
                    html_content += f"""
                    <div style="width: 48%; margin-bottom: 25px; text-align: center; border: 1px solid #ddd; padding: 12px; border-radius: 5px;">
                        <h3 style="margin-top: 0;">{title}</h3>
                        <p style="font-family: monospace; background: #f8f8f8; padding: 8px; border-radius: 3px;">{formula}</p>
                        <img src="data:image/png;base64,{image_data}" alt="{title}" style="max-width: 100%; height: 180px; border: 1px solid #eee;">
                        <p style="color: #555; font-style: italic; text-align: left; margin-top: 10px;">{description}</p>
                    </div>
                    """
                except Exception as e:
                    html_content += f"""
                    <div style="width: 48%; margin-bottom: 25px; text-align: center; border: 1px solid #ddd; padding: 12px; border-radius: 5px;">
                        <h3 style="margin-top: 0;">{title}</h3>
                        <p style="font-family: monospace; background: #f8f8f8; padding: 8px; border-radius: 3px;">{formula}</p>
                        <p style="color: red;">Ошибка загрузки изображения: {str(e)}</p>
                        <p style="color: #555; font-style: italic; text-align: left; margin-top: 10px;">{description}</p>
                    </div>
                    """
            else:
                html_content += f"""
                <div style="width: 48%; margin-bottom: 25px; text-align: center; border: 1px solid #ddd; padding: 12px; border-radius: 5px;">
                    <h3 style="margin-top: 0;">{title}</h3>
                    <p style="font-family: monospace; background: #f8f8f8; padding: 8px; border-radius: 3px;">{formula}</p>
                    <p style="color: red;">Файл не найден: {img_path}</p>
                    <p style="color: #555; font-style: italic; text-align: left; margin-top: 10px;">{description}</p>
                </div>
                """
        
        html_content += """
        </div>
        
        <div style="background-color: #f0f8f0; padding: 15px; border-radius: 5px; margin-top: 20px;">
        <h3>💡 Рекомендации по выбору:</h3>
        <p>Для начала работы рекомендуется использовать ступенчатое или гауссово распределение - они демонстрируют 
        характерные особенности уравнения. Синусоидальное распределение хорошо подходит для изучения волновых процессов, 
        а линейное и экспоненциальное - для анализа установившихся режимов.</p>
        </div>
        """
        
        tab.setHtml(html_content)
        self.tabs.addTab(tab, "Начальные условия")
    
    def setup_boundary_conditions_tab(self):
        tab = QTextEdit()
        tab.setReadOnly(True)
        tab.setHtml("""
        <h1><a name="boundary">Граничные условия</a></h1>
        
        <p>Граничные условия определяют взаимодействие расчетной области с внешней средой. 
        Правильный выбор граничных условий очень важен для получения физически корректных решений.</p>
        
        <h2>Левая граница (x = 0)</h2>
        
        <p style="text-align: center; font-size: 16px; font-weight: bold;">
        u<sub>x</sub> = 0
        </p>
        
        <p>Это условие Неймана, которое означает отсутствие потока тепла через границу. 
        Физически это соответствует теплоизолированной стенке - тепло не может ни уходить через эту границу, 
        ни приходить извне. Температурный профиль у такой границы имеет нулевой наклон.</p>
        
        <p>Такое условие часто используется для моделирования симметричных систем или случаев, 
        когда одна из границ действительно является теплоизолированной.</p>
        
        <h2>Правая граница (x = 1)</h2>
        
        <p style="text-align: center; font-size: 16px; font-weight: bold;">
        u<sub>x</sub> + (7/V)·u = (7/V)·(2/7)
        </p>
        
        <p>Это смешанное граничное условие (условие Робина), которое описывает теплообмен с окружающей средой. 
        Оно учитывает как поток тепла через границу, так и разность температур между границей и окружающей средой.</p>
        
        <p>Параметр 7/V определяет интенсивность теплообмена:</p>
        <ul>
        <li>При больших значениях V (сильная диффузия) условие приближается к условию Дирихле u = 2/7</li>
        <li>При малых значениях V (слабая диффузия) преобладает условие Неймана u<sub>x</sub> = 0</li>
        </ul>
        
        <p>Температура окружающей среды установлена равной 2/7. Это значение выбрано для обеспечения 
        определенных свойств решения и удобства анализа.</p>
        
        <h2>Физическая интерпретация</h2>
        
        <p>Выбранные граничные условия моделируют реальную физическую ситуацию: с одной стороны система 
        теплоизолирована, с другой - происходит теплообмен с окружающей средой заданной температуры. 
        Такая конфигурация часто встречается в технических приложениях, например, в задачах теплозащиты зданий 
        или проектировании теплообменных аппаратов.</p>
        
        <p>Граничные условия обеспечивают единственность решения задачи и его физическую корректность 
        во всем диапазоне параметров.</p>
        """)
        self.tabs.addTab(tab, "Граничные условия")
    
    def setup_convective_tab(self):
        tab = QTextEdit()
        tab.setReadOnly(True)
        tab.setHtml("""
        <h1><a name="convective">Конвективный член</a></h1>
        
        <p>Конвективный член C(u)·u<sub>x</sub> определяет характер направленного переноса температуры. 
        В программе реализовано пять различных типов конвективного члена, каждый из которых соответствует 
        определенному физическому механизму.</p>
        
        <h2>Const - постоянная конвекция</h2>
        
        <p><strong>C(u) = C</strong> (постоянная величина, задаваемая пользователем)</p>
        
        <p>Это простейший случай, когда скорость переноса постоянна и не зависит от температуры. 
        Такая модель подходит для описания вынужденной конвекции, когда движение среды создается внешними силами 
        (насосом, вентилятором) и не зависит от температурного поля.</p>
        
        <h2>0 - отсутствие конвекции</h2>
        
        <p><strong>C(u) = 0</strong></p>
        
        <p>При этом конвективный член полностью отсутствует, и уравнение превращается в уравнение теплопроводности. 
        Это чисто диффузионный процесс, при котором температура выравнивается за счет теплопроводности, 
        без направленного переноса.</p>
        
        <h2>u - линейная конвекция</h2>
        
        <p><strong>C(u) = u</strong></p>
        
        <p>Скорость переноса пропорциональна самой температуре. Это соответствует случаю, когда более нагретые 
        участки среды движутся быстрее. Такая зависимость часто возникает в задачах естественной конвекции, 
        где движение вызвано архимедовой силой, пропорциональной разности температур.</p>
        
        <h2>u² - квадратичная конвекция</h2>
        
        <p><strong>C(u) = u²</strong></p>
        
        <p>Сильно нелинейная зависимость, при которой скорость переноса растет как квадрат температуры. 
        Такие зависимости могут возникать в турбулентных потоках или в средах с особой реологией. 
        Квадратичная конвекция может приводить к образованию ударных волн и других нелинейных эффектов.</p>
        
        <h2>u³ - кубическая конвекция</h2>
        
        <p><strong>C(u) = u³</strong></p>
        
        <p>Еще более сильная нелинейность, когда скорость переноса пропорциональна кубу температуры. 
        Такие зависимости характерны для некоторых химических процессов и экстремальных условий. 
        Кубическая конвекция может вызывать очень сложное и богатое динамическое поведение.</p>
        
        <h2>Рекомендации по выбору</h2>
        
        <div style="background-color: #fff8e1; padding: 15px; border-radius: 5px;">
        <p>Для начала работы рекомендуется использовать линейную конвекцию (u) - она демонстрирует основные 
        нелинейные эффекты, но при этом решение остается достаточно регулярным. Постоянная конвекция (Const) 
        хороша для изучения базовых механизмов, а квадратичная и кубическая конвекции представляют интерес 
        для исследования сложных нелинейных явлений.</p>
        
        <p>Отсутствие конвекции (0) полезно для сравнения и изучения вклада чистой диффузии в общий процесс.</p>
        </div>
        """)
        self.tabs.addTab(tab, "Конвективный член")
    
    def setup_source_tab(self):
        tab = QTextEdit()
        tab.setReadOnly(True)
        tab.setHtml("""
        <h1><a name="source">Источник тепла</a></h1>
        
        <p>Источник тепла f(x,t) описывает внешние воздействия на систему, которые могут добавлять 
        или отнимать энергию. В программе реализовано семь различных типов источников, 
        охватывающих широкий спектр физических ситуаций.</p>
        
        <h2>Постоянный источник</h2>
        
        <p><strong>f(t) = A</strong></p>
        
        <p>Простейший тип источника - постоянная во времени и пространстве мощность нагрева или охлаждения. 
        Моделирует ситуации, когда тепло выделяется или поглощается равномерно по всему объему с постоянной интенсивностью.</p>
        
        <h2>Синусоидальный по времени</h2>
        
        <p><strong>f(t) = A·sin(ω·t)</strong></p>
        
        <p>Источник, мощность которого изменяется по синусоидальному закону во времени. 
        Частота ω определяет скорость колебаний. Такие источники моделируют циклические процессы, 
        например, суточные или сезонные изменения солнечной радиации.</p>
        
        <h2>Косинусоидальный по времени</h2>
        
        <p><strong>f(t) = A·cos(ω·t)</strong></p>
        
        <p>Аналогичен синусоидальному, но со сдвигом фазы. Начинается с максимальной мощности в момент t=0. 
        Удобен для моделирования процессов, которые начинаются с максимальной интенсивности.</p>
        
        <h2>Синусоидальный по пространству</h2>
        
        <p><strong>f(x) = A·sin(ω·x)</strong></p>
        
        <p>Источник с периодическим распределением мощности в пространстве. 
        Моделирует неоднородный нагрев, например, от системы параллельных нагревателей 
        или неравномерное поглощение излучения.</p>
        
        <h2>Косинусоидальный по пространству</h2>
        
        <p><strong>f(x) = A·cos(ω·x)</strong></p>
        
        <p>Пространственно-периодический источник с максимумом на левой границе. 
        Подходит для моделирования ситуаций, когда наиболее интенсивный нагрев происходит у одной из границ.</p>
        
        <h2>Синусоидальный по пространству и времени</h2>
        
        <p><strong>f(x,t) = A·sin(ω_x·x + ω_t·t)</strong></p>
        
        <p>Источник в виде бегущей волны - периодический и в пространстве, и во времени. 
        Моделирует перемещающиеся источники тепла или распространение тепловых волн.</p>
        
        <h2>Косинусоидальный по пространству и времени</h2>
        
        <p><strong>f(x,t) = A·cos(ω_x·x + ω_t·t)</strong></p>
        
        <p>Бегущая волна источника со сдвигом фазы относительно синусоидального варианта.</p>
        
        <h2>Параметры источника</h2>
        
        <ul>
        <li><strong>A</strong> - амплитуда источника, определяет максимальную мощность нагрева или охлаждения</li>
        <li><strong>ω_t</strong> - временная частота, задает скорость изменения источника во времени</li>
        <li><strong>ω_x</strong> - пространственная частота, определяет периодичность источника в пространстве</li>
        </ul>
        
        
        <p>Бегущие волны (комбинация пространственных и временных вариаций) демонстрируют наиболее сложное 
        и интересное поведение, но требуют более тонкой настройки параметров.</p>
        </div>
        """)
        self.tabs.addTab(tab, "Источник тепла")
    
    def setup_method_tab(self):
        tab = QTextEdit()
        tab.setReadOnly(True)
        tab.setHtml("""
        <h1><a name="method">Метод решения и параметры</a></h1>
        
        <p>Программа использует современные численные методы для решения уравнения конвекции-диффузии, 
        обеспечивающие высокую точность и устойчивость даже для сложных нелинейных случаев.</p>
        
        <h2>Численная схема</h2>
        
        <p>Основой решения является <strong>неявная разностная схема</strong>, которая обладает важным 
        преимуществом - безусловной устойчивостью. Это означает, что решение остается стабильным 
        при любых значениях шагов по времени и пространству, что особенно важно для задач с большими градиентами 
        и сильной нелинейностью.</p>
        
        <p>Для решения возникающих систем линейных уравнений используется <strong>метод прогонки</strong> 
        (алгоритм Томаса) - эффективный алгоритм для трехдиагональных систем со сложностью O(N), 
        где N - количество узлов сетки.</p>
        
        <h2>Контроль точности</h2>
        
        <p>Для оценки точности решения программа использует <strong>контрольную сетку</strong> 
        с удвоенным количеством узлов как по пространству, так и по времени. Сравнение решений 
        на основной и контрольной сетках позволяет оценить погрешность и убедиться в корректности результатов.</p>
        
        <p>Этот подход основан на принципе Рунге - если решение на сгущенной сетке мало отличается 
        от решения на основной сетке, то можно считать, что достигнута достаточная точность.</p>
        
        <h2>Массовый оператор</h2>
        
        <p>Для повышения устойчивости схемы при больших градиентах и в нелинейных случаях 
        используется <strong>массовый оператор</strong>. Параметр δ определяет степень его влияния:</p>
        
        <ul>
        <li>δ = 0 - массовый оператор отключен</li>
        <li>δ > 0 - массовый оператор включен (рекомендуемое значение δ = 0.125)</li>
        </ul>
        
        <p>Массовый оператор особенно полезен при работе с разрывными начальными условиями 
        и в случаях сильной нелинейности.</p>
        
        <h2>Основные параметры модели</h2>
        
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
        <h3>📊 Параметры расчета:</h3>
        
        <ul>
        <li><strong>T</strong> - время расчета, определяет продолжительность моделируемого процесса</li>
        <li><strong>Nx</strong> - количество узлов по пространству, влияет на разрешение пространственных деталей</li>
        <li><strong>Nt</strong> - количество шагов по времени, определяет детальность временной развертки</li>
        <li><strong>V</strong> - коэффициент диффузии, характеризует интенсивность выравнивания температуры</li>
        <li><strong>δ</strong> - параметр массового оператора, повышает устойчивость схемы</li>
        <li><strong>C</strong> - параметр конвективного члена (для типа Const)</li>
        <li><strong>A, ω_t, ω_x</strong> - параметры источника тепла</li>
        </ul>
        </div>
        
        <h2>Визуализация результатов</h2>
        
        <p>Программа предоставляет несколько способов анализа и визуализации результатов:</p>
        
        <h3>График слоя</h3>
        <p>Показывает распределение температуры в выбранный момент времени. Позволяет сравнивать 
        решение с начальным условием и анализировать эволюцию температурного профиля.</p>
        
        <h3>Тепловая карта</h3>
        <p>Двумерное представление u(x,t), где по осям отложены пространственная координата и время, 
        а температура кодируется цветом. Очень наглядно показывает общую динамику процесса.</p>
        
        <h3>Таблица решений</h3>
        <p>Содержит численные значения температуры во всех узлах сетки для всех моментов времени. 
        Позволяет проводить количественный анализ и оценивать погрешность.</p>
        
        <h3>Анимация</h3>
        <p>Последовательность графиков, показывающая эволюцию решения во времени. 
        Особенно полезна для демонстрации динамических процессов.</p>
        
        <h2>Рекомендации по настройке параметров</h2>
        
        <div style="background-color: #f0f8f0; padding: 15px; border-radius: 5px;">
        <p>Для большинства задач хорошим стартом являются следующие значения:</p>
        <ul>
        <li>Nx = 100-200 (обеспечивает достаточное пространственное разрешение)</li>
        <li>Nt = 1000-5000 (гарантирует устойчивость и точность временной дискретизации)</li>
        <li>V = 0.01-0.1 (типичные значения для задач теплопереноса)</li>
        <li>Шаг сохранения = 10-20 (баланс между детализацией и объемом данных)</li>
        </ul>
        
        <p>При появлении нефизичных осцилляций рекомендуется увеличить коэффициент диффузии V 
        или включить массовый оператор с δ = 0.125.</p>
        </div>
        """)
        self.tabs.addTab(tab, "Метод решения")

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
        
        self.save_btn = QPushButton("Сохранить тепловую карту")
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
        self.fig.colorbar(im, label="Температура, u")
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
    def __init__(self, solution_data, control_solution, time_points, x_values, x_control_values, Nx, Nt, parent=None):
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
        self.fill_control_table(self.control_table, control_solution, time_points, x_control_values)
        self.fill_comparison_table(self.comparison_table, solution_data, control_solution, time_points, x_values, x_control_values)

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
    
    def fill_control_table(self, table, solution_data, time_points, x_values):
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
    
    def fill_comparison_table(self, table, main_data, control_data, time_points, x_values, x_control_values):
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

            # Интерполяция контрольной сетки на основную для сравнения
            u_control_interp = np.interp(x_values, x_control_values, control_data[row])
            
            # Расчет отклонений
            diff = np.abs(u_control_interp - main_data[row])
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
        self.x_control_values = []
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

        # Поля ввода с новыми значениями по умолчанию
        self.save_every_input = QLineEdit("10")
        self.T_input = QLineEdit("5.0")
        self.Nx_input = QLineEdit("200")
        self.Nt_input = QLineEdit("5000")
        self.diffusion_coeff_input = QLineEdit("0.01")
        self.layer_input = QSpinBox()
        self.time_input = QLineEdit("0.0")
        
        # Флажки для массового оператора и источника
        self.mass_operator_cb = QCheckBox("Решение задачи с массовым оператором")
        self.mass_operator_cb.setChecked(True)
        self.delta_input = QLineEdit("0.125")  # Новое значение по умолчанию
        
        self.source_cb = QCheckBox("Использовать источник")
        self.source_cb.setChecked(True)
        
        # Комбобокс для выбора типа источника
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems([
            "Постоянный источник",
            "Синусоидальный по времени",
            "Косинусоидальный по времени", 
            "Синусоидальный по пространству",
            "Косинусоидальный по пространству",
            "Синусоидальный по пространству и времени",
            "Косинусоидальный по пространству и времени"
        ])
        self.source_type_combo.setCurrentText("Синусоидальный по времени")  # Новое значение по умолчанию
        
        # Поля для параметров источника
        self.source_amp_input = QLineEdit("3.0")  # Новое значение по умолчанию
        self.source_freq_input = QLineEdit("1.0")  # Новое значение по умолчанию
        self.source_freq_x_input = QLineEdit("5.0")
        self.source_freq_t_input = QLineEdit("5.0")
        
        # Комбобокс для выбора типа конвективного члена
        self.convective_type_combo = QComboBox()
        self.convective_type_combo.addItems(["Const", "0", "u", "u^2", "u^3"])
        self.convective_type_combo.setCurrentText("u")  # Значение по умолчанию
        
        # Поле для постоянной величины конвективного члена
        self.convective_const_input = QLineEdit("1.0")

        self.form_layout = QFormLayout()
        self.form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        
        # Основные параметры
        self.form_layout.addRow(QLabel("Шаг просмотра числ. решения:"), self.save_every_input)
        self.form_layout.addRow(QLabel("Горизонт расчета моделирования по времени:"), self.T_input)
        self.form_layout.addRow(QLabel("Число участков разбиения по оси x (Nx):"), self.Nx_input)
        self.form_layout.addRow(QLabel("Число участков разбиения по оси t (Nt):"), self.Nt_input)
        self.form_layout.addRow(QLabel("Коэффициент диффузии V:"), self.diffusion_coeff_input)
        
        # Массовый оператор (будет скрываться)
        self.mass_operator_row = self.form_layout.rowCount()
        self.form_layout.addRow(self.mass_operator_cb)
        self.delta_row = self.form_layout.rowCount()
        self.form_layout.addRow(QLabel("δ (параметр массового оператора):"), self.delta_input)
        
        # Конвективный член
        self.form_layout.addRow(QLabel("Тип конвективного члена C(u):"), self.convective_type_combo)
        self.convective_const_row = self.form_layout.rowCount()
        self.form_layout.addRow(QLabel("Величина C (для типа Const):"), self.convective_const_input)
        
        # Источник (будет скрываться)
        self.source_row = self.form_layout.rowCount()
        self.form_layout.addRow(self.source_cb)
        self.source_type_row = self.form_layout.rowCount()
        self.form_layout.addRow(QLabel("Тип источника:"), self.source_type_combo)
        self.source_amp_row = self.form_layout.rowCount()
        self.form_layout.addRow(QLabel("Амплитуда А источника:"), self.source_amp_input)
        self.source_freq_row = self.form_layout.rowCount()
        self.form_layout.addRow(QLabel("Частота (ω) источника:"), self.source_freq_input)
        self.source_freq_x_row = self.form_layout.rowCount()
        self.form_layout.addRow(QLabel("Пространственная частота (ω_x):"), self.source_freq_x_input)
        self.source_freq_t_row = self.form_layout.rowCount()
        self.form_layout.addRow(QLabel("Временная частота (ω_t):"), self.source_freq_t_input)
        
        # Остальные параметры
        self.form_layout.addRow(QLabel("№ слоя для показа на графике:"), self.layer_input)
        self.form_layout.addRow(QLabel("Время t, для которого подбирается ближний слой:"), self.time_input)
        
        self.run_btn = QPushButton("Запустить расчет")
        self.table_btn = QPushButton("Таблица решений")
        self.animate_btn = QPushButton("Сохранить графики слоев в GIF-анимации")
        self.show_heatmap_btn = QPushButton("Показать тепловую карту")
        self.save_layer_btn = QPushButton("Сохранить график слоя (в .png)")
        self.help_btn = QPushButton("Справка")
        
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
        self.stats_mass_operator = QLabel("Нет")
        self.stats_source = QLabel("Нет")
        self.stats_scheme = QLabel("Неявная схема")
        self.stats_convective = QLabel("u")
        
        stats_layout.addRow(QLabel("Нач. условие:"), self.stats_ic)
        stats_layout.addRow(QLabel("Конвективный член:"), self.stats_convective)
        stats_layout.addRow(QLabel("Макс. отклонение (слой):"), self.stats_max_diff)
        stats_layout.addRow(QLabel("Макс. отклонение (вся сетка):"), self.stats_global_max_diff)
        stats_layout.addRow(QLabel("Место макс. откл. (слой, узел):"), self.stats_max_diff_loc)
        stats_layout.addRow(QLabel("Среднее отклонение:"), self.stats_mean_diff)
        stats_layout.addRow(QLabel("Значение источника:"), self.stats_source_val)
        stats_layout.addRow(QLabel("Размер сетки:"), self.stats_grid_size)
        stats_layout.addRow(QLabel("Время слоя:"), self.stats_time_point)
        stats_layout.addRow(QLabel("Массовый оператор:"), self.stats_mass_operator)
        stats_layout.addRow(QLabel("Источник:"), self.stats_source)
        stats_layout.addRow(QLabel("Схема решения:"), self.stats_scheme)
        
        stats_group.setLayout(stats_layout)
        stats_group.setMaximumHeight(350)
        
        btn_layout = QVBoxLayout()
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.table_btn)
        btn_layout.addWidget(self.animate_btn)
        btn_layout.addWidget(self.show_heatmap_btn)
        btn_layout.addWidget(self.save_layer_btn)
        btn_layout.addWidget(self.help_btn)

        left_layout.addLayout(self.form_layout)
        left_layout.addWidget(stats_group)
        left_layout.addLayout(btn_layout)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        
        # Область для постановки задачи
        problem_group = QGroupBox("Постановка задачи")
        problem_layout = QVBoxLayout()
        
        self.problem_label = QLabel(
            "Уравнение: u_t + C(u) * u_x = V * u_xx + f(x,t)\n"
            "x ∈ [0, 1]\n"
            "Граничные условия:\n"
            "  При x=0: u_x = 0\n"
            "  При x=1: u_x + (7/V)*u = (7/V)*(2/7)\n"
            "Начальное условие: не задано\n"
            "Схема решения: неявная"
        )
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet("padding: 10px;")
        
        problem_layout.addWidget(self.problem_label)
        problem_group.setLayout(problem_layout)
        problem_group.setMaximumHeight(200)
        
        right_layout.addWidget(problem_group)
        right_layout.addWidget(self.layer_canvas)

        main_layout.addWidget(left_container)
        main_layout.addWidget(right_container)
        self.setCentralWidget(main_container)

        # Подключение сигналов
        self.run_btn.clicked.connect(self.run_simulation)
        self.table_btn.clicked.connect(self.show_solution_table)
        self.animate_btn.clicked.connect(self.create_animation)
        self.layer_input.valueChanged.connect(self.draw_layer)
        self.show_heatmap_btn.clicked.connect(self.show_heatmap)
        self.save_layer_btn.clicked.connect(self.save_layer_plot)
        self.time_input.returnPressed.connect(self.find_layer_by_time)
        self.grid_cb.stateChanged.connect(self.redraw_current_layer)
        self.show_ic_cb.stateChanged.connect(self.redraw_current_layer)
        self.mass_operator_cb.stateChanged.connect(self.update_mass_operator_visibility)
        self.source_cb.stateChanged.connect(self.update_source_visibility)
        self.source_type_combo.currentTextChanged.connect(self.update_source_fields_visibility)
        self.convective_type_combo.currentTextChanged.connect(self.update_convective_const_visibility)
        self.help_btn.clicked.connect(self.show_help)

        # Инициализация видимости полей
        self.update_mass_operator_visibility()
        self.update_source_visibility()
        self.update_source_fields_visibility()
        self.update_convective_const_visibility()

        self.draw_empty_layer()

    def update_mass_operator_visibility(self):
        """Обновление видимости полей массового оператора"""
        visible = self.mass_operator_cb.isChecked()
        
        # Показываем/скрываем поле параметра δ
        label = self.form_layout.itemAt(self.delta_row, QFormLayout.LabelRole)
        field = self.form_layout.itemAt(self.delta_row, QFormLayout.FieldRole)
        
        if label:
            label.widget().setVisible(visible)
        if field:
            field.widget().setVisible(visible)

    def update_source_visibility(self):
        """Обновление видимости полей источника"""
        visible = self.source_cb.isChecked()
        
        # Скрываем/показываем все строки, связанные с источником
        for row in range(self.source_type_row, self.source_freq_t_row + 1):
            label = self.form_layout.itemAt(row, QFormLayout.LabelRole)
            field = self.form_layout.itemAt(row, QFormLayout.FieldRole)
            
            if label:
                label.widget().setVisible(visible)
            if field:
                field.widget().setVisible(visible)
        
        # Обновляем видимость конкретных полей в зависимости от типа источника
        self.update_source_fields_visibility()

    def update_source_fields_visibility(self):
        """Обновление видимости конкретных полей источника в зависимости от типа"""
        if not self.source_cb.isChecked():
            return
            
        source_type = self.source_type_combo.currentText()
        
        # Определяем, какие поля должны быть видны
        show_amp = True  # Амплитуда всегда видна
        show_freq = source_type in ["Синусоидальный по времени", "Косинусоидальный по времени", 
                                  "Синусоидальный по пространству", "Косинусоидальный по пространству"]
        show_freq_x = source_type in ["Синусоидальный по пространству и времени", "Косинусоидальный по пространству и времени"]
        show_freq_t = source_type in ["Синусоидальный по пространству и времени", "Косинусоидальный по пространству и времени"]
        
        # Обновляем подписи полей в зависимости от типа источника
        if source_type == "Постоянный источник":
            self.form_layout.itemAt(self.source_amp_row, QFormLayout.LabelRole).widget().setText("Величина источника:")
        else:
            self.form_layout.itemAt(self.source_amp_row, QFormLayout.LabelRole).widget().setText("Амплитуда А источника:")
        
        if source_type in ["Синусоидальный по времени", "Косинусоидальный по времени"]:
            self.form_layout.itemAt(self.source_freq_row, QFormLayout.LabelRole).widget().setText("Частота (ω) источника:")
        elif source_type in ["Синусоидальный по пространству", "Косинусоидальный по пространству"]:
            self.form_layout.itemAt(self.source_freq_row, QFormLayout.LabelRole).widget().setText("Пространственная частота (ω):")
        else:
            self.form_layout.itemAt(self.source_freq_row, QFormLayout.LabelRole).widget().setText("Частота источника:")
        
        # Обновляем видимость полей
        self.form_layout.itemAt(self.source_amp_row, QFormLayout.LabelRole).widget().setVisible(show_amp)
        self.form_layout.itemAt(self.source_amp_row, QFormLayout.FieldRole).widget().setVisible(show_amp)
        
        self.form_layout.itemAt(self.source_freq_row, QFormLayout.LabelRole).widget().setVisible(show_freq)
        self.form_layout.itemAt(self.source_freq_row, QFormLayout.FieldRole).widget().setVisible(show_freq)
        
        self.form_layout.itemAt(self.source_freq_x_row, QFormLayout.LabelRole).widget().setVisible(show_freq_x)
        self.form_layout.itemAt(self.source_freq_x_row, QFormLayout.FieldRole).widget().setVisible(show_freq_x)
        
        self.form_layout.itemAt(self.source_freq_t_row, QFormLayout.LabelRole).widget().setVisible(show_freq_t)
        self.form_layout.itemAt(self.source_freq_t_row, QFormLayout.FieldRole).widget().setVisible(show_freq_t)

    def update_convective_const_visibility(self):
        """Обновление видимости поля для постоянной величины конвективного члена"""
        convective_type = self.convective_type_combo.currentText()
        visible = (convective_type == "Const")
        
        label = self.form_layout.itemAt(self.convective_const_row, QFormLayout.LabelRole)
        field = self.form_layout.itemAt(self.convective_const_row, QFormLayout.FieldRole)
        
        if label:
            label.widget().setVisible(visible)
        if field:
            field.widget().setVisible(visible)

    def show_help(self):
        """Показать диалог справки"""
        help_dialog = HelpDialog(self)
        help_dialog.exec_()

    def init_initial_conditions_ui(self):
        self.ic_combo = QComboBox()
        self.ic_combo.addItems([
            "Ступенчатое распределение",
            "Гауссов пакет", 
            "Синусоидальное распределение",
            "Пилообразное распределение",
            "Линейное распределение",
            "Квадратичное распределение",
            "Кубическое распределение",
            "Экспоненциальное распределение"
        ])
        self.form_layout.addRow(QLabel("Начальное условие:"), self.ic_combo)
        self.ic_combo.currentTextChanged.connect(self._handle_param_change)

    def _handle_param_change(self):
        self.need_recalculate = True

    def get_initial_condition(self, x):
        ic_type = self.ic_combo.currentText()
        L = x[-1]
        
        if ic_type == "Ступенчатое распределение":
            return np.where(x < L/2, 1.0, -1.0)
        
        elif ic_type == "Гауссов пакет":
            return np.exp(-50 * (x - L/2)**2)
        
        elif ic_type == "Синусоидальное распределение":
            return np.sin(2 * np.pi * x / L)
        
        elif ic_type == "Пилообразное распределение":
            return 2 * (x / L - np.floor(0.5 + x / L))
        
        elif ic_type == "Линейное распределение":
            return 1.0 - x/L
        
        elif ic_type == "Квадратичное распределение":
            return 1.0 - (x/L)**2
        
        elif ic_type == "Кубическое распределение":
            return 1.0 - (x/L)**3
        
        elif ic_type == "Экспоненциальное распределение":
            return np.exp(-5*x/L)
        
        else:
            return np.sin(2 * np.pi * x / L)
    
    def get_initial_condition_equation(self):
        ic_type = self.ic_combo.currentText()
        
        if ic_type == "Ступенчатое распределение":
            return "u(x,0) = 1.0 при x < 0.5, -1.0 при x ≥ 0.5"
        
        elif ic_type == "Гауссов пакет":
            return "u(x,0) = exp(-50*(x-0.5)^2)"
        
        elif ic_type == "Синусоидальное распределение":
            return "u(x,0) = sin(2πx)"
        
        elif ic_type == "Пилообразное распределение":
            return "u(x,0) = 2*(x - floor(0.5 + x))"
        
        elif ic_type == "Линейное распределение":
            return "u(x,0) = 1.0 - x"
        
        elif ic_type == "Квадратичное распределение":
            return "u(x,0) = 1.0 - x^2"
        
        elif ic_type == "Кубическое распределение":
            return "u(x,0) = 1.0 - x^3"
        
        elif ic_type == "Экспоненциальное распределение":
            return "u(x,0) = exp(-5x)"
        
        else:
            return "u(x,0) = sin(2πx)"
    
    def get_convective_coefficient(self, u_value):
        """Вычисление коэффициента конвективного члена"""
        convective_type = self.convective_type_combo.currentText()
        
        if convective_type == "Const":
            try:
                return float(self.convective_const_input.text())
            except:
                return 1.0
        elif convective_type == "0":
            return 0.0
        elif convective_type == "u":
            return u_value
        elif convective_type == "u^2":
            return u_value ** 2
        elif convective_type == "u^3":
            return u_value ** 3
        else:
            return u_value
    
    def get_convective_equation(self):
        """Получение уравнения для конвективного члена"""
        convective_type = self.convective_type_combo.currentText()
        
        if convective_type == "Const":
            try:
                C = self.convective_const_input.text()
                return f"C(u) = {C}"
            except:
                return "C(u) = 1.0"
        elif convective_type == "0":
            return "C(u) = 0"
        elif convective_type == "u":
            return "C(u) = u"
        elif convective_type == "u^2":
            return "C(u) = u²"
        elif convective_type == "u^3":
            return "C(u) = u³"
        else:
            return "C(u) = u"
    
    def source_function(self, t, x=None):
        if not self.source_cb.isChecked():
            return 0.0
            
        try:
            A = float(self.source_amp_input.text())
            source_type = self.source_type_combo.currentText()
            
            if source_type == "Постоянный источник":
                return A
                
            elif source_type == "Синусоидальный по времени":
                ω = float(self.source_freq_input.text())
                return A * math.sin(ω * t)
                
            elif source_type == "Косинусоидальный по времени":
                ω = float(self.source_freq_input.text())
                return A * math.cos(ω * t)
                
            elif source_type == "Синусоидальный по пространству":
                if x is None:
                    return 0.0
                ω = float(self.source_freq_input.text())
                return A * math.sin(ω * x)
                
            elif source_type == "Косинусоидальный по пространству":
                if x is None:
                    return 0.0
                ω = float(self.source_freq_input.text())
                return A * math.cos(ω * x)
                
            elif source_type == "Синусоидальный по пространству и времени":
                if x is None:
                    return 0.0
                ω_x = float(self.source_freq_x_input.text())
                ω_t = float(self.source_freq_t_input.text())
                return A * math.sin(ω_x * x + ω_t * t)
                
            elif source_type == "Косинусоидальный по пространству и времени":
                if x is None:
                    return 0.0
                ω_x = float(self.source_freq_x_input.text())
                ω_t = float(self.source_freq_t_input.text())
                return A * math.cos(ω_x * x + ω_t * t)
                
            else:
                return 0.0
        except:
            return 0.0
    
    def get_source_equation(self):
        if not self.source_cb.isChecked():
            return "f(x,t) = 0"
            
        source_type = self.source_type_combo.currentText()
        A = self.source_amp_input.text()
        
        if source_type == "Постоянный источник":
            return f"f(x,t) = {A}"
            
        elif source_type == "Синусоидальный по времени":
            ω = self.source_freq_input.text()
            return f"f(x,t) = {A}·sin({ω}·t)"
            
        elif source_type == "Косинусоидальный по времени":
            ω = self.source_freq_input.text()
            return f"f(x,t) = {A}·cos({ω}·t)"
            
        elif source_type == "Синусоидальный по пространству":
            ω = self.source_freq_input.text()
            return f"f(x,t) = {A}·sin({ω}·x)"
            
        elif source_type == "Косинусоидальный по пространству":
            ω = self.source_freq_input.text()
            return f"f(x,t) = {A}·cos({ω}·x)"
            
        elif source_type == "Синусоидальный по пространству и времени":
            ω_x = self.source_freq_x_input.text()
            ω_t = self.source_freq_t_input.text()
            return f"f(x,t) = {A}·sin({ω_x}·x + {ω_t}·t)"
            
        elif source_type == "Косинусоидальный по пространству и времени":
            ω_x = self.source_freq_x_input.text()
            ω_t = self.source_freq_t_input.text()
            return f"f(x,t) = {A}·cos({ω_x}·x + {ω_t}·t)"
            
        else:
            return "f(x,t) = 0"
    
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
            diffusion_coeff = float(self.diffusion_coeff_input.text())
            
            # Определение параметра массового оператора
            if self.mass_operator_cb.isChecked():
                delta = float(self.delta_input.text())
            else:
                delta = 0.0
                
            save_every = int(save_every_text)
            
            L = 1.0
            h = L / self.Nx
            x = np.linspace(0, L, self.Nx + 1)
            u0 = self.get_initial_condition(x)
            u_max = np.max(np.abs(u0))
            
            Re_c = u_max * h / diffusion_coeff if diffusion_coeff > 0 else float('inf')

            if Re_c > 2:
                min_Nx = int(np.ceil(u_max * L / (2 * diffusion_coeff)) + 1) if diffusion_coeff > 0 else self.Nx * 4
                min_diffusion_coeff = u_max * h / 2

                advice = []
                if diffusion_coeff < 1e-5:
                    advice.append("1. Увеличьте коэффициент диффузии (V) до хотя бы 0.001")
                elif h > 0.01:
                    advice.append("1. Уменьшите шаг сетки (увеличьте Nx) до {} или более".format(min_Nx))
                else:
                    advice.append("1. Увеличьте коэффициент диффузии до {:.4f} или более".format(min_diffusion_coeff))
                
                advice.append("2. Рассмотрите использование схемы 'upwind' вместо центральных разностей")
                if self.mass_operator_cb.isChecked():
                    advice.append("3. Увеличьте параметр δ для массового оператора")
                
                msg = (
                    "Внимание! Параметры сетки могут вызвать нефизичные осцилляции!\n"
                    "Сеточное число Рейнольдса Re_c = {:.2f} > 2\n\n"
                    "Рекомендации для текущих параметров:\n{}"
                ).format(Re_c, "\n".join(advice))
                
                msg_box = QMessageBox(QMessageBox.Warning, "Предупреждение об осцилляциях", msg, 
                                     QMessageBox.Ok | QMessageBox.Cancel, self)
                
                if msg_box.exec_() == QMessageBox.Cancel:
                    return
            
            x, solution, time_points = self.solve_equation(T, self.Nx, self.Nt, diffusion_coeff, delta, save_every)
            
            self.solve_control_grid(T, self.Nx, self.Nt, diffusion_coeff, delta, save_every)

            self.layer_input.setMaximum(len(self.solution_history)-1)
            self.layer_input.setValue(0)
            self.draw_layer(0)
            self.need_recalculate = False
            
            # Обновляем отображение уравнения с начальным условием
            ic_type = self.ic_combo.currentText()
            ic_equation = self.get_initial_condition_equation()
            source_equation = self.get_source_equation()
            convective_equation = self.get_convective_equation()
            
            # Формируем текст постановки задачи
            problem_text = f"Уравнение: u_t + {convective_equation} * u_x = V * u_xx"
            if self.source_cb.isChecked():
                problem_text += " + " + source_equation.split('=')[1].strip()
            problem_text += "\nx ∈ [0, 1]\n"
            problem_text += "Граничные условия:\n"
            problem_text += "  При x=0: u_x = 0\n"
            problem_text += "  При x=1: u_x + (7/V)*u = (7/V)*(2/7)\n"
            problem_text += f"Начальное условие: {ic_type}\n"
            problem_text += f"{ic_equation}\n"
            problem_text += "Схема решения: неявная"
            
            self.problem_label.setText(problem_text)

            self.calculate_global_max_diff()
            
            if self.heatmap_window and self.heatmap_window.isVisible():
                self.heatmap_window.draw_heatmap(self.x_values, self.time_points, self.solution_history, self.cmap)
            
        except Exception as e:
            print(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при расчете: {str(e)}")

    def solve_equation(self, T, Nx, Nt, diffusion_coeff, delta, save_every):
        L = 1.0
        h = L / Nx
        tau = T / Nt
        
        x = np.linspace(0, L, Nx + 1)
        u_n = self.get_initial_condition(x)
        
        u_env = 2/7
        H = 7
        
        # Граничные условия
        u_n[0] = u_n[1]
        u_n[-1] = (diffusion_coeff * u_n[-2] + H * h * u_env) / (diffusion_coeff + H * h)
        
        self.solution_history = [u_n.copy()]
        self.time_points = [0.0]
        self.x_values = x
        
        a = np.zeros(Nx + 1)
        b = np.zeros(Nx + 1)
        c = np.zeros(Nx + 1)
        d = np.zeros(Nx + 1)

        sigma = diffusion_coeff * tau / (2 * h**2)
        gamma = tau / (4 * h)
        
        for step in range(1, Nt + 1):
            t_current = step * tau
            
            # Коэффициенты для внутренних узлов
            for i in range(1, Nx):
                # Вычисление конвективного члена с учетом выбранного типа
                convective_coeff = self.get_convective_coefficient(u_n[i])
                a[i] = gamma * convective_coeff - sigma
                b[i] = 1 + 2 * sigma
                c[i] = -gamma * convective_coeff - sigma
                
                # Массовый оператор
                if self.mass_operator_cb.isChecked():
                    M = delta * u_n[i-1] + (1 - 2 * delta) * u_n[i] + delta * u_n[i+1]
                else:
                    M = u_n[i]
                    
                # Вычисление источника в точке x[i]
                f_val = self.source_function(t_current, x[i])
                
                # Конвективный член в правой части
                convective_term = gamma * (self.get_convective_coefficient(u_n[i+1]) * u_n[i+1] - 
                                         self.get_convective_coefficient(u_n[i-1]) * u_n[i-1])
                    
                d[i] = M + convective_term - sigma * (u_n[i-1] - 2 * u_n[i] + u_n[i+1])
                d[i] += tau * f_val
            
            # Граничные условия
            a[0] = 0
            b[0] = 1
            c[0] = -1
            d[0] = 0
            
            a[Nx] = -diffusion_coeff
            b[Nx] = diffusion_coeff + H * h
            c[Nx] = 0
            d[Nx] = H * h * u_env
            
            # Решение системы
            u_new = self.run_through_method(a, b, c, d)
            
            # Применение граничных условий к решению
            u_new[0] = u_new[1]
            u_new[-1] = (diffusion_coeff * u_new[-2] + H * h * u_env) / (diffusion_coeff + H * h)
            
            u_n = u_new
            
            if step % save_every == 0:
                self.solution_history.append(u_n.copy())
                self.time_points.append(t_current)
        
        return x, u_n, self.time_points

    def solve_control_grid(self, T, Nx, Nt, diffusion_coeff, delta, save_every):
        Nx2 = 2 * Nx
        Nt2 = 2 * Nt
        
        L = 1.0
        h2 = L / Nx2
        tau2 = T / Nt2
        
        x2 = np.linspace(0, L, Nx2 + 1)
        u_n2 = self.get_initial_condition(x2)
        
        u_env = 2/7
        H = 7
        
        # Граничные условия для контрольной сетки
        u_n2[0] = u_n2[1]
        u_n2[-1] = (diffusion_coeff * u_n2[-2] + H * h2 * u_env) / (diffusion_coeff + H * h2)
        
        control_solution_history = [u_n2.copy()]
        control_time_points = [0.0]
        self.x_control_values = x2
        
        a = np.zeros(Nx2 + 1)
        b = np.zeros(Nx2 + 1)
        c = np.zeros(Nx2 + 1)
        d = np.zeros(Nx2 + 1)
        
        sigma = diffusion_coeff * tau2 / (2 * h2**2)
        gamma = tau2 / (4 * h2)
        
        for step in range(1, Nt2 + 1):
            t_current = step * tau2
            
            # Коэффициенты для внутренних узлов
            for i in range(1, Nx2):
                # Вычисление конвективного члена с учетом выбранного типа
                convective_coeff = self.get_convective_coefficient(u_n2[i])
                a[i] = gamma * convective_coeff - sigma
                b[i] = 1 + 2 * sigma
                c[i] = -gamma * convective_coeff - sigma
                
                # Массовый оператор
                if self.mass_operator_cb.isChecked():
                    M = delta * u_n2[i-1] + (1 - 2 * delta) * u_n2[i] + delta * u_n2[i+1]
                else:
                    M = u_n2[i]
                
                # Вычисление источника в точке x2[i]
                f_val = self.source_function(t_current, x2[i])
                
                # Конвективный член в правой части
                convective_term = gamma * (self.get_convective_coefficient(u_n2[i+1]) * u_n2[i+1] - 
                                         self.get_convective_coefficient(u_n2[i-1]) * u_n2[i-1])
                    
                d[i] = M + convective_term - sigma * (u_n2[i-1] - 2 * u_n2[i] + u_n2[i+1])
                d[i] += tau2 * f_val
            
            # Граничные условия
            a[0] = 0
            b[0] = 1
            c[0] = -1
            d[0] = 0
            
            a[Nx2] = -diffusion_coeff
            b[Nx2] = diffusion_coeff + H * h2
            c[Nx2] = 0
            d[Nx2] = H * h2 * u_env
            
            # Решение системы
            u_new2 = self.run_through_method(a, b, c, d)
            
            # Применение граничных условий к решению
            u_new2[0] = u_new2[1]
            u_new2[-1] = (diffusion_coeff * u_new2[-2] + H * h2 * u_env) / (diffusion_coeff + H * h2)
            
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
        
        for i in range(len(self.solution_history)):
            u_control_interp = np.interp(
                self.x_values, 
                self.x_control_values, 
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
        ax.set_ylabel("Температура, u")
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
            y_min = min(np.min(u) - 0.1, self.y_min)
            y_max = max(np.max(u) + 0.1, self.y_max)
            
            ax.plot(x, u, 'b-', label="Основная сетка")
            
            if self.grid_cb.isChecked() and self.control_solution_history:
                u_control = self.control_solution_history[layer_index]
                
                x_control = self.x_control_values
                ax.plot(x_control, u_control, 'r--', linewidth=1, label="Контрольная сетка")
            
            if self.show_ic_cb.isChecked():
                ic = self.get_initial_condition(self.x_values)
                ax.plot(self.x_values, ic, 'g--', label="Начальное условие")
            
            # Вычисление номеров слоев
            save_every = int(self.save_every_input.text())
            actual_layer = layer_index * save_every
            
            # Новый формат заголовка
            title = f"График расчетного слоя № {actual_layer}, визуальный слой № {layer_index}, t = {t:.4f} с"
            
            ax.set_title(title)
            ax.set_xlabel("Пространство, x [м]")
            ax.set_ylabel("Температура, u")
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
            f"Конвективный член: {stats['convective']}\n"
            f"Время слоя: {stats['time_point']:.6f}\n"
            f"Макс. отклонение (слой): {stats['max_diff']:.6f}\n"
            f"Макс. отклонение (вся сетка): {stats['global_max_diff']:.6f}\n"
            f"Место макс. откл. (слой, узел): {stats['max_diff_loc']}\n"
            f"Среднее отклонение: {stats['mean_diff']:.6f}\n"
            f"Значение источника: {stats['source_val']:.4f}\n"
            f"Размер сетки: {stats['grid_size']}\n"
            f"Массовый оператор: {stats['mass_operator']}\n"
            f"Источник: {stats['source']}\n"
            f"Схема решения: {stats['scheme']}"
        )

    def update_stats(self, layer_index):
        # Определение статуса массового оператора
        if self.mass_operator_cb.isChecked():
            delta = float(self.delta_input.text())
            mass_operator_status = f"Да (δ={delta})"
        else:
            mass_operator_status = "Нет"
            
        # Определение статуса источника
        if self.source_cb.isChecked():
            source_type = self.source_type_combo.currentText()
            A = self.source_amp_input.text()
            source_status = f"Да ({source_type}, A={A})"
        else:
            source_status = "Нет"
            
        # Определение типа конвективного члена
        convective_type = self.convective_type_combo.currentText()
        if convective_type == "Const":
            try:
                C = self.convective_const_input.text()
                convective_status = f"Const (C={C})"
            except:
                convective_status = "Const (C=1.0)"
        else:
            convective_status = convective_type
        
        stats = {
            'ic': self.ic_combo.currentText(),
            'convective': convective_status,
            'time_point': self.time_points[layer_index] if self.time_points else 0.0,
            'max_diff': 0.0,
            'global_max_diff': self.global_max_diff,
            'max_diff_loc': '-',
            'mean_diff': 0.0,
            'source_val': 0.0,
            'grid_size': f"Основная: {self.Nx}×{self.Nt}, Контрольная: {2*self.Nx}×{2*self.Nt}",
            'mass_operator': mass_operator_status,
            'source': source_status,
            'scheme': "Неявная схема"
        }
        
        if self.solution_history and self.control_solution_history and layer_index < len(self.solution_history):
            u_control_interp = np.interp(
                self.x_values, 
                self.x_control_values, 
                self.control_solution_history[layer_index]
            )
            
            u_main = self.solution_history[layer_index]
            diff = np.abs(u_control_interp - u_main)
            max_diff = np.max(diff)
            max_idx = np.argmax(diff)
            mean_diff = np.mean(diff)
            
            # Вычисление значения источника в средней точке
            x_mid = self.x_values[len(self.x_values)//2]
            source_val = self.source_function(self.time_points[layer_index], x_mid)
            
            stats.update({
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'max_diff_loc': f"{layer_index}, {max_idx}",
                'source_val': source_val
            })
        
        self.stats_ic.setText(stats['ic'])
        self.stats_convective.setText(stats['convective'])
        self.stats_time_point.setText(f"{stats['time_point']:.6f}")
        self.stats_max_diff.setText(f"{stats['max_diff']:.6f}")
        self.stats_global_max_diff.setText(f"{stats['global_max_diff']:.6f}")
        self.stats_max_diff_loc.setText(stats['max_diff_loc'])
        self.stats_mean_diff.setText(f"{stats['mean_diff']:.6f}")
        self.stats_source_val.setText(f"{stats['source_val']:.4f}")
        self.stats_grid_size.setText(stats['grid_size'])
        self.stats_mass_operator.setText(stats['mass_operator'])
        self.stats_source.setText(stats['source'])
        self.stats_scheme.setText(stats['scheme'])
        
        return stats
    
    def show_solution_table(self):
        if not self.solution_history or not self.control_solution_history:
            QMessageBox.warning(self, "Ошибка", "Сначала запустите расчет с включенной контрольной сеткой!")
            return
        
        dialog = SolutionTableDialog(
            self.solution_history,
            self.control_solution_history,
            self.time_points,
            self.x_values,
            self.x_control_values,
            self.Nx,
            self.Nt,
            self
        )
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()
    
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
                y_min = min(np.min(u) - 0.1, self.y_min)
                y_max = max(np.max(u) + 0.1, self.y_max)
                
                ax.plot(x, u, 'b-', label="Основная сетка")
                if self.grid_cb.isChecked() and self.control_solution_history:
                    u_control = self.control_solution_history[i]
                    x_control = self.x_control_values
                    ax.plot(x_control, u_control, 'r--', linewidth=1, label="Контрольная сетка")
                
                if self.show_ic_cb.isChecked():
                    ic = self.get_initial_condition(self.x_values)
                    ax.plot(x, ic, 'g--', label="Начальное условие")
                
                # Вычисление номеров слоев
                save_every = int(self.save_every_input.text())
                actual_layer = i * save_every
                
                # Новый формат заголовка
                title = f"График расчетного слоя № {actual_layer}, визуальный слой № {i}, t = {t:.4f} с"
                
                ax.set_title(title)
                ax.set_xlabel("Пространство, x [м]")
                ax.set_ylabel("Температура, u")
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
        y_min = min(np.min(u) - 0.1, self.y_min)
        y_max = max(np.max(u) + 0.1, self.y_max)
        
        ax1.plot(x, u, 'b-', label="Основная сетка")
        
        if self.grid_cb.isChecked() and self.control_solution_history:
            u_control = self.control_solution_history[layer_index]
            x_control = self.x_control_values
            ax1.plot(x_control, u_control, 'r--', linewidth=1, label="Контрольная сетка")
        
        if self.show_ic_cb.isChecked():
            ic = self.get_initial_condition(self.x_values)
            ax1.plot(x, ic, 'g--', label="Начальное условие")
        
        # Вычисление номеров слоев
        save_every = int(self.save_every_input.text())
        actual_layer = layer_index * save_every
        
        # Новый формат заголовка
        title = f"График расчетного слоя № {actual_layer}, визуальный слой № {layer_index}, t = {t:.4f} с"
        
        ax1.set_title(title)
        ax1.set_xlabel("Пространство, x [м]")
        ax1.set_ylabel("Температура, u")
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