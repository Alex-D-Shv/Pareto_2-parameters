# Pareto_2-parameters

Данный репозиторий содержит исходный код программного комплекса для численного решения задач взаимодействия двух геометрических параметров постоянного магнита на величину магнитного поля и магнитного коэффициента через многокритериальную оптимизацию с построением фронта Парето. Разработано в рамках научной статьи по специальности ВАК 1.2.2.

This repository contains the source code for a software package for the numerical solution of problems involving the interaction of two geometric parameters of a permanent magnet, determining the magnetic field magnitude and magnetic coefficient, using multicriteria optimization with Pareto frontier construction. Developed as part of a research paper in the Higher Attestation Commission (HAC) specialty 1.2.2.

Описание модели и методов Язык реализации: Python 3.10+ 
Численный метод: Многокритериальная оптимизация по двум параметрам с использованием фронта Парето
Основные библиотеки: NumPy, Matplotlib

Установка и запуск

Клонируйте репозиторий:
git clone https://github.com
cd Pareto_2-parameters

Установите зависимости:
pip install -r requirements.txt

Запустите расчет эксперимента:
python main.py

Воспроизведение результатов статьи При запуске main.py программа рассчитывает модель и генерирует графики, которые соответствуют Рисункам в тексте опубликованной статьи.
