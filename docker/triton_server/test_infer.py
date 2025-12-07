# test_infer.py (исправленный)
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

MODEL = "simple_identity"
URL = "localhost:8000"

def normalize_shape(shape):
    # заменяем динамические (-1) на 1 (минимально возможный положительный размер)
    return tuple(1 if (isinstance(d, int) and d < 0) else d for d in shape)

def make_bytes_array(shape, sample=b"hello"):
    # Создаём numpy массив dtype=object заполненный байтовыми строками
    # shape ожидается tuple положительных int
    arr = np.empty(shape, dtype=object)
    # заполнение: каждую ячейку делаем sample
    it = np.nditer(np.zeros(shape), flags=['multi_index'])
    while not it.finished:
        arr[it.multi_index] = sample
        it.iternext()
    return arr

def main():
    client = InferenceServerClient(URL)

    meta = client.get_model_metadata(MODEL)
    print("Model metadata:", meta)

    # Взять первый input (в большинстве простых моделей их 1)
    input_meta = meta['inputs'][0]
    print("Input meta:", input_meta)

    # Подставляем конкретную форму (заменяем -1 на 1)
    desired_shape = normalize_shape(tuple(input_meta['shape']))
    print("Using shape:", desired_shape)

    # Для BYTES — надо массив dtype=object с байтовыми элементами
    if input_meta['datatype'].upper().startswith("BYTES"):
        # можно положить любую байтовую строку, модель simple_identity вернёт её обратно
        np_input = make_bytes_array(desired_shape, sample=b"hello_from_client")
    else:
        # fallback: numeric zeros float32
        np_input = np.zeros(desired_shape, dtype=np.float32)

    # Создаём InferInput
    infer_input = InferInput(input_meta['name'], desired_shape, input_meta['datatype'])
    # set_data_from_numpy корректно обработает dtype=object для BYTES
    infer_input.set_data_from_numpy(np_input)

    # Запросим весь первый output
    output_meta = meta['outputs'][0]
    out = InferRequestedOutput(output_meta['name'])

    resp = client.infer(MODEL, inputs=[infer_input], outputs=[out])
    # Выведем результат (как numpy). Для BYTES вернёт dtype=object
    result = resp.as_numpy(output_meta['name'])
    print("Response numpy dtype:", type(result), result.dtype if isinstance(result, np.ndarray) else None)
    print("Response array shape:", None if result is None else result.shape)
    print("First element:", result.flatten()[0] if result is not None else None)

if __name__ == "__main__":
    main()

