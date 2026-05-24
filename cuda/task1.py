import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    device = torch.device('cuda')
    # pin_memory + non_blocking transfers keep the GPU input pipeline asynchronous
    dataloader = DataLoader(
        prepare_data(),
        batch_size=256,
        shuffle=True,
        pin_memory=True
    )

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).to(device)
    model.train()

    # Было:
    # model = nn.Sequential(
    #     nn.Linear(128, 512), nn.ReLU(),
    #     nn.Linear(512, 128), nn.ReLU(),
    #     nn.Linear(128, 2)
    # ).cuda().train()
    # Ошибка: при измерениях времени дальше использовались CPU-таймеры без синхронизации,
    # а перенос данных делался синхронно — это искажало метрики и тормозило конвейер.

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    for batch_idx, (data, target) in enumerate(dataloader):
        # Перенос на GPU без блокировки, чтобы не ломать асинхронный конвейер.
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Генерируем шум сразу на GPU, чтобы не гонять лишние данные CPU->GPU.
        noise = torch.randn_like(data, device=device)
        data = data + noise

        # Более дешёвое обнуление градиентов (избегаем лишних записей в память).
        optimizer.zero_grad(set_to_none=True)

        # Было:
        # noise = torch.randn(data.shape).to('cuda')
        # data = data.to('cuda') + noise
        # target = target.to('cuda')
        # optimizer.zero_grad()
        # Ошибка: синхронный перенос и создание шума на CPU ломали конвейер и увеличивали задержки.

        # Используем CUDA events — CPU-таймер без синхронизации даёт «нечестные» метрики.
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        fwd_start.record()
        output = model(data)
        loss = criterion(output, target)
        fwd_end.record()

        bwd_start.record()
        loss.backward()
        bwd_end.record()
        optimizer.step()

        torch.cuda.synchronize()  # ждём окончания GPU-работы перед чтением времени
        forward_times.append(fwd_start.elapsed_time(fwd_end) / 1000.0)
        backward_times.append(bwd_start.elapsed_time(bwd_end) / 1000.0)

        # Сохраняем число, а не тензор с графом — иначе будет утечка памяти.
        losses_history.append(loss.item())
        print(f"Batch {batch_idx} loss: {losses_history[-1]:.4f}")

        # Было:
        # time_start = time.time()
        # output = model(data)
        # loss = criterion(output, target)
        # time_end = time.time()
        # forward_times.append(time_end - time_start)
        # ...
        # time_start_bwd = time.time()
        # loss.backward()
        # time_end_bwd = time.time()
        # backward_times.append(time_end_bwd - time_start_bwd)
        # ...
        # losses_history.append(loss)
        # torch.cuda.empty_cache()
        # Ошибки:
        # 1) CPU-таймеры без синхронизации → неверные метрики (GPU работает асинхронно).
        # 2) losses_history.append(loss) держит граф → утечка памяти.
        # 3) empty_cache() внутри цикла ломает кэш-локальность и тормозит.

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times)}, "
          f"avg backward time is {statistics.mean(backward_times)}")

if __name__ == '__main__':
    train()