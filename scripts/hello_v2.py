from kfp import dsl
from kfp import compiler

# --- КОМПОНЕНТ 1: Генератор ---
# Декоратор @dsl.component превращает функцию в шаг пайплайна.
# base_image - какой образ использовать (python:3.9 подойдет для простых скриптов)
@dsl.component(base_image='python:3.9')
def say_hello(name: str) -> str:
    message = f'Hello, {name} from KFP V2!'
    print(message)
    return message

# --- КОМПОНЕНТ 2: Потребитель ---
@dsl.component(base_image='python:3.9')
def print_message(msg: str):
    print(f"Received message: {msg}")

# --- ПАЙПЛАЙН ---
# Здесь мы связываем компоненты
@dsl.pipeline(
    name='hello-world-v2',
    description='A simple intro to KFP V2 syntax'
)
def hello_pipeline(recipient: str = 'World'):
    # Шаг 1: Запуск генератора
    task1 = say_hello(name=recipient)
    
    # Шаг 2: Запуск потребителя
    # Мы передаем выход (output) первого шага на вход второму
    task2 = print_message(msg=task1.output)

if __name__ == '__main__':
    # Компилируем в YAML
    compiler.Compiler().compile(
        pipeline_func=hello_pipeline,
        package_path='hello_v2.yaml'
    )
    print("✅ Compiled to hello_v2.yaml")